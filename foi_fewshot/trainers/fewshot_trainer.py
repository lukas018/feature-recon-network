import collections
import math
from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.optim as optim
from learn2learn.data import MetaDataset, TaskDataset

from ..utils import (
    SummaryGroup,
    compute_metrics,
    LogEntry,
)

from .trainer_utils import (
    TrainerState,
    TrainerControl,
    MetabatchWrapper,
    create_taskloader,
)

from .callbacks import (
    CallbackHandler,
    ProgressCallback,
    DefaultFlowCallback,
    WriterCallback,
)

from ..data import fast_metadataset

from ..data import fast_metadataset


DEFAULT_CALLBACKS = [DefaultFlowCallback, ProgressCallback]
DEFAULT_MODEL_PREFIX = "checkpoint"


def default_lr_scheduler(model, optimizer, args):

    def get_lr(epoch):
        return 1.0

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    return lr_scheduler


class FewshotTrainer:
    """FewshotTrainer

    General trainer wrapper for few-shot trainers
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_task_generator=None,
        optimizers=(None, None),
        callbacks=None,
    ):
        """
        :param model: Input model
        :param train_ds: Training dataset
        :param args: Training arguments
        :param eval_args: Optional special argument for evaluation
        :param base_eval_dataset: Dataset of base classes with novel samples
        :param novel_eval_dataset: Dataset with novel classes
        """

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.train_meta_dataset = fast_metadataset(train_dataset)
        self.eval_task_generator = eval_task_generator
        self.optimizer, self.lr_scheduler = optimizers
        self.create_optimizer_and_scheduler()

        callbacks = (
            DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        )

        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        self.state = TrainerState()
        self.control = TrainerControl()

    def create_optimizer_and_scheduler(self):
        if self.optimizer is None:
            lr = self.args.learning_rate
            momentum = self.args.momentum
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        if self.lr_scheduler is None:
            self.lr_scheduler = default_lr_scheduler(
                self.model, self.optimizer, self.args
            )

    def _envelop_model(self, model, fewshot_mode=True):
        if fewshot_mode:
            model = MetabatchWrapper(model)

        data_parallel = (
            DistributedDataParallel if self.args.distributed else DataParallel
        )
        learner = data_parallel(
            model,
            device_ids=self.args.device_ids,
            output_device=torch.device("cpu"),
        )
        return learner

    def save_checkpoint(self, modeldir=None, metrics=None):
        """Save the current state of the model"""

        modeldir = modeldir if modeldir is not None else self.args.modeldir
        modeldir = Path(modeldir).expanduser()
        modeldir.mkdir(exist_ok=True)
        prefix = self.args.modeldir_prefix if self.args.modeldir_prefix is not None else DEFAULT_MODEL_PREFIX
        prefix = (
            f"{self.args.modeldir_prefix}-{self.state.global_step}"
        )

        checkpointdir = Path(modeldir, prefix)
        checkpointdir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), Path(checkpointdir, f"model.pkl"))
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), Path(checkpointdir, "optimizer.pkl"))

        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), Path(checkpointdir, "scheduler.pkl"))

        torch.save(self.args, Path(checkpointdir, "train_arguments.pkl"))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = str(checkpointdir)

        state_path = Path(checkpointdir, f"state.json")
        with open(state_path, "w") as fh:
            fh.write(self.state.to_json())

    def load_checkpoint(self, modeldir):
        """Load the trainer state from checkpoint"""

        modeldir = Path(modeldir).expanduser()
        self.model.load_state_dict(torch.load(Path(modeldir, f"model.pkl")))

        optimizer_path = Path(modeldir, f"optimizer.pkl")
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        scheduler_path = Path(modeldir, f"scheduler.pkl")
        if scheduler_path.is_file():
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path))

        state_path = Path(modeldir, f"state.json"), "w"
        if state_path.is_file():
            with open(state_path, "w") as fh:
                self.state = TrainerState.fromdict(json.load(fh))

    def scheduler_step(self):
        """Runs scheduler at the current update step"""
        self.lr_scheduler.step()

    def training_step(self, model, inputs):
        """"""

        model.train()
        outputs = model(**inputs)
        loss = self.compute_loss(model, inputs, outputs)

        if 'loss' in outputs:
            for x, y in zip(loss, outputs['loss']):
                x += y

        if hasattr(loss, "__len__") and len(loss) > 0:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def train(self, modeldir=None):

        if modeldir is not None:
            self.load_checkpoint(modeldir)

        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0

        train_dataloader = self.get_train_dataloader()
        model = self._envelop_model(
            self.model, isinstance(train_dataloader.dataset, TaskDataset)
        )

        num_update_steps_per_epoch = (
            len(train_dataloader) // self.args.gradient_accumulation_steps
        )
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        max_steps = math.ceil(self.args.num_epochs * num_update_steps_per_epoch)
        if self.state.max_steps == 0:
            self.state.max_steps = max_steps

        for _ in range(epochs_trained):
            epoch_iterator = self.get_train_dataloader()
            for _ in epoch_iterator:
                break

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        for epoch in range(epochs_trained, self.args.num_epochs):
            epoch_iterator = iter(self.get_train_dataloader())

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            )
            num_update_steps_per_epoch = (
                len(epoch_iterator) // self.args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = self.state.global_step % (
                num_update_steps_per_epoch
            )
            steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps

            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )
            tr_loss = torch.tensor(0.0)

            for step, inputs in enumerate(epoch_iterator):
                self.control = self.callback_handler.on_step_begin(
                    self.args, self.state, self.control
                )

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1

                tr_loss += self.training_step(model, inputs)
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    self.optimizer_step(tr_loss)
                    self.scheduler_step()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(
                        self.args,
                        self.state,
                        self.control,
                    )

                    self._maybe_log_save_evaluate(tr_loss, model, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_log_save_evaluate(tr_loss, model, epoch)

            if self.control.should_training_stop:
                break

        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            self.load_checkpoint(self.state.best_model_checkpoint)

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

    def optimizer_step(self, loss):
        # loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()

        if self.control.should_save:
            self.save_checkpoint(metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def prediction_step(
        self,
        model,
        inputs,
    ):

        with torch.no_grad():
            outputs = model(**inputs)
            labels = inputs["query_labels"]
            logits = outputs["logits"]

            loss = self.compute_loss(model, inputs, outputs)

            if 'loss' in outputs:
                for x, y in zip(loss, outputs['loss']):
                    x += y

            return loss, logits, labels

    def prediction_loop(self, model, dataloader):

        model = self._envelop_model(model, isinstance(dataloader.dataset, TaskDataset))

        total_loss, total_logits, total_labels = [], [], []
        for step, batch in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, batch)
            total_loss.extend(loss)
            total_logits.extend(logits)
            total_labels.extend(labels)

        metrics = self.compute_metrics(total_logits, total_labels)
        metrics['loss'] = np.mean(total_loss)
        return metrics

    def compute_metrics(self, logits, labels):
        logits = list(itertools.chain(*logits))
        labels = list(itertools.chain(*labels))
        def _acc(logits, labels):
            idx = torch.argmax(logits)
            size = 1. if len(labels.shape) == 0 else labels.shape[0]
            acc = ((idx == labels).float().sum() / size).detach().numpy()
            return acc

        accs = list(itertools.starmap(_acc, zip(logits, labels)))
        accs = np.mean(accs)
        return {'acc': accs}

    def compute_loss(self, model, inputs, outputs):
        labels = inputs["query_labels"]
        logits = outputs["logits"]

        if isinstance(logits, torch.Tensor):
            loss = F.cross_entropy(logits, labels)
        else:
            losses = tuple(itertools.starmap(F.cross_entropy, zip(logits, labels)))
            loss = torch.stack(losses)

        return loss

    def evaluate(self, eval_task_generator=None):
        metrics = {}

        model = self.model
        eval_task_generator = (
            self.eval_task_generator
            if eval_task_generator is None
            else eval_task_generator
        )

        metrics = dict()
        for prefix, dl in self.eval_task_generator:
            _metrics = self.prediction_loop(model, dl)
            _metrics = {f"eval-{prefix}-{key}": float(metric) for key, metric in _metrics.items()}
            self.log(_metrics)
            metrics = {
                **metrics,
                **_metrics
            }

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics=metrics
        )
        return metrics

    def get_train_dataloader(self):
        """Returns the train dataloader"""
        dl = create_taskloader(
            self.train_meta_dataset,
            self.args,
        )
        return dl

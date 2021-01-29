import collections
import copy
import functools
import inspect
import itertools
import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from ..data import fast_metadataset
from .callbacks import CallbackHandler
from .callbacks import DefaultFlowCallback
from .callbacks import ProgressCallback
from .hyperparameter_search import run_hp_search
from .trainer_arguments import SchedulerUpdateStrategy
from .trainer_utils import create_dataloader
from .trainer_utils import create_taskloader
from .trainer_utils import MetabatchWrapper
from .trainer_utils import TrainerControl
from .trainer_utils import TrainerState
from learn2learn.data import TaskDataset

DEFAULT_CALLBACKS = [DefaultFlowCallback, ProgressCallback]
DEFAULT_MODEL_PREFIX = "checkpoint"
logger = logging.getLogger(__name__)


def _default_lr_scheduler(model, optimizer, args):
    """A learning rate scheduler which always output the same result"""

    def get_lr(epoch):
        return 1.0

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    return lr_scheduler


def _default_compute_objective(metrics, keyword="eval-loss"):
    metrics = copy.deepcopy(metrics)
    loss = (
        metrics.pop(
            "eval_loss",
            None,
        )
        if keyword is None
        else metrics.pop(keyword, None)
    )
    _ = metrics.pop("epoch", None)
    return loss if len(metrics) == 0 else sum(metrics.values())


class FewshotTrainer:
    """FewshotTrainer

    General trainer wrapper for few-shot training
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_task_generator=None,
        optimizers=(None, None),
        callbacks=None,
        model_init=None,
    ):
        """
        :param model: Input model
        :param train_ds: Training dataset
        :param args: Training arguments
        :param eval_task_generator: Task generator for evaluation tasks
        :param optimizers: Tuple of optimizer and lr_scheduler
        :param callbacks: List of callback object that will be called during
            training
        :param model_init: Optional function for initializing the model
            argument.  Required when performing hyper-parameter search.
        """

        self.model = model.to(args.device)
        self.args = args
        self.train_dataset = train_dataset
        self.train_meta_dataset = fast_metadataset(train_dataset)
        self.eval_task_generator = eval_task_generator
        self.optimizer, self.lr_scheduler = optimizers
        self.create_optimizer_and_scheduler()
        self.model_init = model_init

        callbacks = (
            DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        )

        self.callback_handler = CallbackHandler(
            callbacks,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        self.state = TrainerState()
        self.control = TrainerControl()

    def create_optimizer_and_scheduler(self):
        """Initializes optimizer and scheduler if not already done"""

        if self.optimizer is None:
            lr = self.args.learning_rate
            momentum = self.args.momentum
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = _default_lr_scheduler(
                self.model,
                self.optimizer,
                self.args,
            )

    def _envelop_model(self, model, fewshot_mode=True):
        """Wraps and returns the specified model such that it can perform either
        fewshot or standard classification

        :param model: Input model to wrap
        :param fewshot_mode: If true the model is wrapped in a meta-batch
            wrapping class to allow for accepting multiple tasks simultaniously

        :returns: The wrapped model
        """

        if fewshot_mode:
            model = MetabatchWrapper(model)

        if torch.cuda.is_available():
            data_parallel = (
                DistributedDataParallel if self.args.distributed else DataParallel
            )

            main_device = (
                self.args.device_ids[0] if self.args.device_ids is not None else 0
            )
            device = torch.device(main_device if torch.cuda.is_available() else "cpu")
            devices = (
                [device] if self.args.device_ids is not None else self.args.device_ids
            )

            model = data_parallel(
                model,
                device_ids=devices,
            )

        return model

    def save_checkpoint(self, modeldir=None, trial=None, metrics=None):
        """Save the current state of the model as checkpoint

        :param modeldir: Directory to save model to
        :param trial: Optuna trial object if hyperparameter search is performed
        :param metrics: If metrics are provided and
            self.args.metric_for_best_model is set to true, then the best
            performing model is saved to the trainer state.
        """

        modeldir = modeldir if modeldir is not None else self.args.modeldir
        modeldir = Path(modeldir).expanduser()
        modeldir.mkdir(exist_ok=True)
        prefix = (
            self.args.modeldir_prefix
            if self.args.modeldir_prefix is not None
            else DEFAULT_MODEL_PREFIX
        )
        prefix = f"{self.args.modeldir_prefix}-{self.state.global_step}"

        if trial is not None:
            run_id = trial.number
            run_name = f"run-{run_id}"
            checkpointdir = Path(modeldir, run_name, prefix)
        else:
            checkpointdir = Path(modeldir, prefix)

        checkpointdir.mkdir(exist_ok=True, parents=True)

        torch.save(self.model.state_dict(), Path(checkpointdir, "model.pkl"))
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                Path(checkpointdir, "optimizer.pkl"),
            )

        if self.lr_scheduler is not None:
            torch.save(
                self.lr_scheduler.state_dict(),
                Path(checkpointdir, "scheduler.pkl"),
            )

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

        state_path = Path(checkpointdir, "state.json")
        with open(state_path, "w") as fh:
            fh.write(self.state.to_json())

    def _hp_search_setup(self, trial):
        """Initializes the trainer and the model using an optuna hyperparameter trial object

        :param trial: Trial object used to sample learning parameters
        """
        self._trial = trial
        if trial is None:
            return

        params = self.hp_space(trial)
        for key, value in params.items():
            if not hasattr(self.args, key):
                raise AttributeError(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`.",
                )
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)
            logger.info(f"Trial: {trial.params}")

    def load_checkpoint(self, checkpoint_dir):
        """Load the trainer state from checkpoint
        :param checkpoint_dir: directory where the checkpoint is located
        """

        self.model = self.model.cuda()

        checkpoint_dir = Path(checkpoint_dir).expanduser()
        optimizer_path = Path(checkpoint_dir, "optimizer.pkl")
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=torch.device("cpu"),
                ),
            )
            # NOTE This fixes a weird bug where the momentum is stored on cpu while parameters are loaded on cuda
            # See https://github.com/pytorch/pytorch/issues/8741 for more information
            self.optimizer.load_state_dict(self.optimizer.state_dict())

        scheduler_path = Path(checkpoint_dir, "scheduler.pkl")
        if scheduler_path.is_file():
            self.lr_scheduler.load_state_dict(
                torch.load(
                    scheduler_path,
                    map_location=torch.device("cpu"),
                ),
            )

        state_path = Path(checkpoint_dir, "state.json")

        if state_path.is_file():
            with open(state_path) as fh:
                self.state = TrainerState.from_dict(json.load(fh))

    def scheduler_step(self):
        """Runs scheduler at the current update step"""

        if self.args.scheduler_update_strategy == SchedulerUpdateStrategy.STEPS or (
            self.args.scheduler_update_strategy == SchedulerUpdateStrategy.EPOCH
            and self.state.num_update_steps_per_epoch != 0
            and (self.state.global_step + 1) % self.state.num_update_steps_per_epoch
            == 0
        ):
            self.lr_scheduler.step()

    def training_step(self, model, inputs):
        """Perform a single training step

        This includes doing a forward step using the model, computing the loss,
        scaling the loss, and then backpropagating the gradient

        :param model: model used for forward pass
        :param inputs: dictionary with input tensors
        """

        model.train()
        outputs = model(**inputs)
        loss = self.compute_loss(model, inputs, outputs)

        if "loss" in outputs:
            for x, y in zip(loss, outputs["loss"]):
                x += y

        if len(loss.size()) > 0:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def train(self, modeldir=None, trial=None):
        """Train the model

        :param modeldir: If specified the trainer will load model checkpoints, trainer state etc from the modeldir.
        :param trial: optuna trial object used during hyperparameter search.
        :returns:
        """

        self._hp_search_setup(trial)

        if modeldir is not None:
            self.load_checkpoint(modeldir)

        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self.state.is_hyper_param_search = trial is not None

        train_dataloader = self.get_train_dataloader()
        model = self._envelop_model(
            self.model,
            isinstance(train_dataloader.dataset, TaskDataset),
        )

        num_update_steps_per_epoch = (
            len(train_dataloader) // self.args.gradient_accumulation_steps
        )
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        max_steps = math.ceil(self.args.num_epochs * num_update_steps_per_epoch)
        if self.state.max_steps == 0:
            self.state.max_steps = max_steps

        # for _ in range(epochs_trained):
        #     epoch_iterator = self.get_train_dataloader()
        #     for _ in epoch_iterator:
        #         break

        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,
        )

        tr_loss = torch.tensor(0.0)
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
                self.args,
                self.state,
                self.control,
            )

            for step, inputs in enumerate(epoch_iterator):
                self.control = self.callback_handler.on_step_begin(
                    self.args,
                    self.state,
                    self.control,
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

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args,
                self.state,
                self.control,
            )
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.control.should_training_stop:
                break

        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            best_model_state_dict = torch.load(
                Path(self.state.best_model_checkpoint, "model.pkl"),
            )
            self.model.load_state_dict(best_model_state_dict)

        self.control = self.callback_handler.on_train_end(
            self.args,
            self.state,
            self.control,
        )

    def optimizer_step(self, loss):
        """Peform a step using the trainer's optimizer

        :param loss: The current loss

        """
        # loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def log(self, logs):
        """Save the logs to the current state

        :param logs: Log dict
        """

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.control = self.callback_handler.on_log(
            self.args,
            self.state,
            self.control,
            logs=logs,
        )
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        """Maybe log, maybe save, maybe evaluate depending on the state of the
        traners.control object

        :param tr_loss: The accumulated loss since last save
        :param model: The model
        :param trial: Optional optuna hyperparameter search trial
        :param epoch: The current epoch, specified as a float
        """

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
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self.save_checkpoint(metrics=metrics, trial=trial)
            self.control = self.callback_handler.on_save(
                self.args,
                self.state,
                self.control,
            )

    def prediction_step(
        self,
        model,
        inputs,
    ):
        """Performs a single prediction step using a prediction model

        :param model: Model used to training
        :param inputs: Dictionary of input tensors

        :returns: loss, logits, labels
        """

        # TODO(Lukas) Make a less ugly hack
        @contextmanager
        def dummy_cm():
            yield

        def detach(output):
            """Detach item from tensor so that cuda memory can be released"""
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy()
            return [detach(x) for x in output]

        with torch.no_grad() if self.args.disable_gradient_during_eval else dummy_cm():
            outputs = model(**inputs)
            labels = inputs["query_labels"]
            logits = outputs["logits"]

            loss = self.compute_loss(model, inputs, outputs)

            if "loss" in outputs:
                for x, y in zip(loss, outputs["loss"]):
                    x += y

            # Detach the values from its torch tensors such that we can free cuda memory
            loss = detach(loss)
            logits = detach(logits)
            labels = detach(labels)

            return loss, logits, labels

    def prediction_loop(self, model, dataloader):
        """Perform continous predictions over the samples from the dataloader

        :param model: prediction model
        :param dataloader: dataloader (either a standard dataloader or a task
            loader which generates samples)

        :returns: metrics dict
        """

        model = self._envelop_model(model, isinstance(dataloader.dataset, TaskDataset))
        model.eval()

        total_loss, total_logits, total_labels = [], [], []
        for step, batch in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, batch)
            if len(loss.shape) == 0:
                total_loss.append(loss)
            else:
                total_loss.extend(loss)

            total_logits.extend(logits)
            total_labels.extend(labels)

        if isinstance(dataloader.dataset, TaskDataset):
            total_logits = list(itertools.chain(*total_logits))
            total_labels = list(itertools.chain(*total_labels))

        metrics = self.compute_metrics(total_logits, total_labels)
        metrics["loss"] = np.mean(total_loss)
        return metrics

    def compute_metrics(self, logits, labels):
        """Compute the metrics given the logits and labels

        Override this method or specifiy compute_metrics in the constructor to change what metrics to compute

        :param logits:
        :param labels:
        :returns: Dictionary containing the computed metrics
        """

        def _acc(logits, labels):
            # logits = logits.cpu()
            # labels = labels.long().cpu()
            idx = np.argmax(logits, axis=-1)

            # idx = torch.argmax(logits)
            size = 1.0 if len(labels.shape) == 0 else labels.shape[0]
            # acc = ((idx == labels).float().sum() / size).detach().numpy()
            acc = np.sum(idx == labels) / size
            return acc

        accs = list(itertools.starmap(_acc, zip(logits, labels)))
        accs = np.mean(accs)
        return {"acc": accs}

    def compute_loss(self, model, inputs, outputs):
        """Computes the loss given

        :param model: Model such that model parameters can be used to compute the loss
        :param inputs: Inputs dict
        :param outputs: Output dicts
        """

        labels = inputs["query_labels"]
        logits = outputs["logits"]

        def _compute_loss(logits, labels):
            labels = labels.long().cpu()
            logits = logits.float().cpu()
            loss = F.cross_entropy(logits, labels)
            return loss

        if isinstance(logits, torch.Tensor):
            loss = _compute_loss(logits, labels)
        else:
            losses = list(itertools.starmap(_compute_loss, zip(logits, labels)))
            loss = torch.stack(losses)

        return loss

    def evaluate(self, eval_task_generator=None):
        """Evaluate the current model on the different tasks specified by a
        EvalTaskGenerator

        :param eval_task_generator: Task generator

        :returns: Metrics dictionary
        """
        metrics = {}

        model = self.model
        eval_task_generator = (
            self.eval_task_generator
            if eval_task_generator is None
            else eval_task_generator
        )

        metrics = dict()
        for prefix, dl in eval_task_generator:
            _metrics = self.prediction_loop(model, dl)
            _metrics = {
                f"eval-{prefix}-{key}": float(metric)
                for key, metric in _metrics.items()
            }
            self.log(_metrics)
            metrics = {**metrics, **_metrics}

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics=metrics,
        )
        return metrics

    def get_train_dataloader(self):
        """Returns the train dataloader"""
        dl = create_taskloader(
            self.train_meta_dataset,
            self.args,
        )
        return dl

    def call_model_init(self, trial=None):
        """Call the model_init function

        :param trial: If the model_init takes a trial argument, this trial will
            be used to initialize the model

        :returns: An intiailized model
        """

        model_init_argcount = len(inspect.signature(self.model_init).parameters)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def hyperparameter_search(
        self,
        hp_space,
        n_trials,
        compute_objective=None,
        direction="minimize",
        **kwargs,
    ):
        """Perform hyper parameter search using Optuna backend

        :param hp_space: Optuna trial object used to sample hyperparameters
        :param n_trials: Number of trials to perform hp search
        :param compute_objective: Objective to minimize during hp search
        :param direction: If objective should use `minimize` or `maximize` to

            determine the optimal setting.

        :returns: the best parameters found during hp search
        """

        self.hp_space = hp_space

        if compute_objective is None:
            self.compute_objective = compute_objective
        elif isinstance(compute_objective, str):
            self.compute_objective = functools.partial(
                _default_compute_objective,
                key=compute_objective,
            )
        else:
            self.compute_objective = _default_compute_objective

        best_run = run_hp_search(self, n_trials, direction, **kwargs)
        return best_run

    def _report_to_hp_search(self, trial, epoch, metrics):
        """Reports the results of the curren trial to the optuna study such that
        the sampling may be optimized

        :param trial: Optional optuna trial object
        :param epoch: The current epoch
        :param metrics: Metrics dictionary
        """
        if trial is None:
            return
        self.objective = self.compute_objective(metrics.copy())
        import optuna

        trial.report(self.objective, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


class PreTrainer(FewshotTrainer):
    """Trainer wrapper for performing image classifiaction pre-training"""

    def get_train_dataloader(self):
        """Returns the DataLoader used during training of the model
        :returns: DataLoader for the training objective
        """
        return create_dataloader(self.train_dataset, self.args)

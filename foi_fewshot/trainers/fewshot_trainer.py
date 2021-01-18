from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ..utils import (
    fewshot_episode,
    SummaryGroup,
    compute_metrics,
    LogEntry,
)
from ..data import initialize_taskdataset
from .trainer_utils import TrainingState, MetabatchWrapper


class FewshotTrainer:
    """FewshotTrainer

    General trainer wrapper for few-shot trainers
    """

    def __init__(
        self,
        model,
        train_dataset,
        args,
        eval_args=None,
        base_eval_dataset=None,
        novel_eval_dataset=None,
        optimizer=None,
        scheduler=None,
        distributed=False,
        on_epoch_begin_callback=None,
        on_epoch_end_callback=None,
        on_update_callback=None,
    ):
        """
        :param model: Input model
        :param train_ds: Training dataset
        :param args: Training arguments
        :param eval_args: Optional special argument for evaluation
        :param base_eval_dataset: Dataset of base classes with novel samples
        :param novel_eval_dataset: Dataset with novel classes
        :param optimizer: optimizer to run at each update step
        :param scheduler: Scheduler that runs at each step
        :param on_epoch_begin_callback: Callback that takes the trainer as parameters
        :param on_epoch_end_callback: Callback that takes the trainer as parameters
        :param on_update_callback: Callback that takes the trainer as parameters
        :param distributed: The trainer uses DistributedDataParallel instead of DataParallel
        This can cause problems with gradient based update methods, such as MAML but is significantly
        faster than DataParallel
        """

        self.model = model
        self.distributed = distributed
        self.train_dataset = train_dataset
        self.args = args
        self.eval_args = eval_args
        self.logs = []

        # Move the model to the correct device if possible
        if self.args.device_ids is not None and len(self.args.device_ids) > 0:
            self.model = model.to(torch.device(self.args.device_ids[0]))
        else:
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        self._envelop_model(model)

        self.optimizer = optimizer
        self.optimizer = self._get_optimizer()
        self.scheduler = scheduler

        self.base_eval_dataset = base_eval_dataset
        self.novel_eval_dataset = novel_eval_dataset

        self.on_epoch_begin_callback = on_epoch_begin_callback
        self.on_epoch_end_callback = on_epoch_end_callback
        self.on_update_callback = on_update_callback
        self.state = TrainingState()

    def _get_optimizer(
        self,
    ):
        if self.args.optimizer:
            return self.args.optimizer
        return torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

    def _envelop_model(self, model):
        self.model = model
        self.mb_wrapper = MetabatchWrapper(self.model, self.args)
        data_parallel = DistributedDataParallel if self.distributed else DataParallel
        self.learner = data_parallel(
            self.mb_wrapper,
            device_ids=self.args.device_ids,
            output_device=torch.device("cpu"),
        )

    @classmethod
    def latest_modeldir(cls, path):
        """Returns the most recent folder from the given directory
        :param path: Path to checkpoints
        :return: A Path object or None
        """

        dirs = list(filter(lambda x: x.is_dir(), Path(path).glob("*")))
        if len(dirs) == 0:
            return None

        times = list(map(dirs.stat().st_mtime))
        i = np.argmax(times)
        return dirs[i]

    def save_checkpoint(self, modeldir=None):
        """Save the current state of the model"""

        modeldir = modeldir if modeldir is not None else self.args.modeldir
        modeldir = Path(modeldir).expanduser()
        modeldir.mkdir(exist_ok=True)

        if self.args.checkpoint_namegen is not None:
            prefix = self.args.checkpoint_namegen(self)
        else:
            prefix = f"{self.args.modeldir_prefix}-{self.state.epoch}-{self.state.current_step}"

        checkpointdir = Path(modeldir, prefix)
        checkpointdir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), Path(checkpointdir, f"model.pkl"))
        if self.optimizer is not None:
            torch.save(self.optimizer, Path(checkpointdir, "optimizer.pkl"))

        if self.scheduler is not None:
            torch.save(self.scheduler, Path(checkpointdir, "scheduler.pkl"))

        state_path = Path(checkpointdir, f"state.json")
        with open(state_path, "w") as fh:
            fh.write(self.state.to_json())

    def load_checkpoint(self, modeldir):
        """Load the trainer state from checkpoint"""

        modeldir = Path(modeldir).expanduser()
        self.model.load_state_dict(torch.load(Path(modeldir, f"model.pkl")))
        self._envelop_model(self.model)

        optimizer_path = Path(modeldir, f"optimizer.pkl")
        if optimizer_path.is_file():
            self.optimizer.load(optimizer_path)

        scheduler_path = Path(modeldir, f"scheduler.pkl")
        if scheduler_path.is_file():
            self.scheduler.load(optimizer_path)

        state_path = Path(modeldir, f"state.json"), "w"
        if state_path.is_file():
            with open(state_path, "w") as fh:
                self.state = TrainingState.fromdict(json.load(fh))

    def train(self, modeldir=None):
        """Trains the model using standard fewshot training

        :param modeldir: Checkpoint path to load model from
        """

        if modeldir:
            self.load_checkpoint(modeldir)

        self.model.train()

        for epoch in range(
            self.state.epoch, self.args.num_episodes * self.args.epoch_length
        ):

            dl_train = self.get_train_dataloader()
            dl_iter = tqdm(dl_train, desc="Training", initial=self.state.current_step)

            if self.state.current_step > 0:
                for i in range(0, self.state_current_step):
                    next(dl_iter)

            if self.on_epoch_begin_callback:
                self.on_epoch_begin_callback(self)

            for batch in dl_iter:

                losses, metrics = self.learner(batch)
                self.add_logs(metrics, prefix="train")

                self.update_step(losses)
                self.scheduler_step()

                if self.state.current_step % self.args.log_step == 0:
                    self.save_logs()

                if (
                    self.state.current_step % self.args.save_step == 0
                    and self.state.current_step > 0
                ):
                    self.save_checkpoint()

                self.state.global_step += 1
                self.state.current_step += 1

                if self.on_update_callback:
                    self.on_update_callback(self)

            if self.on_epoch_end_callback:
                self.on_epoch_end_callback(self)

            self.state.epoch += 1
            self.state.current_step = 0

            # Save the checkpoint
            self.save_checkpoint()

            # Save some of the metrics
            self.evalutate(logging=True)

        return self.model

    def update_step(self, losses):
        """Runs the update for the model
        :param losses: A list of loss functions
        """

        loss = losses.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def scheduler_step(self):
        """Runs scheduler at the current update step"""

        if self.scheduler is None:
            return

        # Check if the scheduler is a torch lr_scheduler
        # TODO(Lukas) We should find a better way to check this
        if hasattr("step", self.scheduler):
            self.scheduler.step()

        # Check if scheduler is a callable
        elif callable(self.scheduler):
            self.scheduler(self)

    def _logs_to_writer(self, writer=None, log_entries=None):
        log_entries = (
            self._get_logs(before=self.state.global_step)
            if log_entries is None
            else log_entries
        )
        writer = self.args.writer if writer is None else writer
        if writer is not None:
            for entry in log_entries:
                entry.write(writer)

    def _logs_to_file(self, log_entries=None):
        log_file = self.args.log_file
        if log_file is None:
            return

        log_entries = (
            self._get_logs(before=self.state.global_step)
            if log_entries is None
            else log_entries
        )

        with open(log_file, "a") as fh:
            for entry in log_entries:
                fh.write(str(entry))

    def _get_logs(self, before=None):
        logs = self.logs
        if before:
            logs = list(filter(lambda x: x.global_step <= before, logs))
        return logs

    def add_logs(self, metrics, prefix=None):
        """Adds metrics to the list of unsaved logs
        :param metrics: A list of dictionaries
        :param prefix: Prefix to add to the record (used when writing to SummaryWriter)
        """
        if not self.args.do_logging:
            return

        entry = LogEntry(
            self.state.global_step, prefix, SummaryGroup.from_dicts(metrics)
        )
        self.logs.append(entry)
        return entry

    def discard_logs(
        self,
    ):
        self.logs = []

    def save_logs(self):
        """Save the stores logs to desired targets"""
        entries = self._get_logs(before=self.state.global_step)
        self._logs_to_writer(log_entries=entries)
        self._logs_to_file(log_entries=entries)
        self.discard_logs()

    def _fewshot_eval(self, ds, num_episodes, args, prefix=None):

        training = self.model.training
        if training:
            self.model.eval()

        # Start a data loader
        dl = initialize_taskdataset(ds, args.nways, args.kshot, args.num_workers)

        args = self.args if self.eval_args is None else self.eval_args
        metrics = []
        for i in range(num_episodes):
            batch = next(dl)
            _, _res = self.learner(batch, args=args)
            metrics.extend(_res)

        if training:
            self.model.train()

        return metrics

    def fewshot_eval(
        self,
        datasets: Optional[Dict[str, Dataset]] = None,
        num_episodes: int = None,
        logging=False,
    ):
        """Performs fewshot evaluation on the given datasets or the evaluation datasets

        :param datasets: Dictionary of dataset
        :param num_episodes: Number of episodes to evaluate
        :param logging: Log the results
        """

        if datasets is None:
            datasets = dict()
            if self.base_eval_dataset is not None:
                datasets.update("base", self.base_eval_dataset)

            if self.novel_eval_dataset is not None:
                datasets.update("novel", self.novel_eval_dataset)

        num_episodes = (
            self.args.eval_episodes if num_episodes is None else num_episodes,
        )
        res = {
            key: self._fewshot_evaluate(
                ds, num_episodes, self.args, prefix=key + "-eval"
            )
            for key, ds in datasets.items()
        }

        if logging:
            for prefix, metrics in res.items():
                self.add_logs(metrics, prefix=prefix)
                self.save_logs()

        return res

    def evaluate(self, dataset=None, prefix="eval", logging=False):
        """Evaluates the model and return the results

        :param dataset: Dataset to evaluate
        :param prefix: Prefix to add when logging
        :param logging: Whether to log the results or not
        """

        self.model.eval()

        # Get the few-shot evaluation
        datasets = {prefix: dataset} if dataset is not None else None
        fewshot_metrics = self.fewshot_eval(datasets, logging=False)

        if logging:
            for _prefix, metrics in fewshot_metrics.items():
                self.add_logs(metrics, prefix=f"{prefix}-{_prefix}")
                self.save_logs()

        return fewshot_metrics

    def get_train_dataloader(self):
        """Returns the train dataloader"""

        args = self.args
        dl = initialize_taskdataset(
            self.train_dataset,
            self.args.nways,
            self.args.kshots,
            self.args.num_episodes,
            self.args.num_workers,
            self.args.batch_size,
        )
        return dl

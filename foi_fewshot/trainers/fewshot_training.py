import json

import torch
import torch.nn as nn
from torch import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

from ..utils import fewshot_episode, initialize_taskdataset, classes_split, split_dataset, SummaryGroup, compute_metrics

from typing import List


@dataclass_json
@dataclass
class TrainingState():
    """State represetntation for  the training process
    """

    epoch :int = 0
    global_step: int = 0
    current_step: int = 0

    #TODO(Lukas) maybe this should be somewhere else
    metrics: List[float] = []


class FewshotTrainer():
    """FewshotTrainer

    General trainer wrapper for few-shot trainers
    """

    def __init__(self,
                 model,
                 train_ds,
                 args,
                 eval_args=None,
                 base_eval_dataset=None,
                 novel_eval_dataset=None,
                 optimizer=None,
                 scheduler=None,
                 on_epoch_end_callback = None,
                 on_update_callback = None
        ):
        """
        :param model:
        :param train_ds: Training dataset
        :param args:
        :param eval_args:
        :param optimizer:
        :param scheduler:
        :param
        """

        self.model = model
        self.train_ds = train_ds
        self.args = args
        self.eval_args = eval_args

        if optimizer is None:
            self.optimzer = torch.optim.SGD(self.model.parameters(), momentum=0.9)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        self.base_eval_dataset = base_eval_dataset
        self.novel_eval_dataset = novel_eval_dataset

        self.on_epoch_end_callback = on_epoch_end_callback
        self.on_update_callback = on_update_callback
        self.validate = self.args.do_eval

        self.state = TrainingState()

    def _get_optimizer(self, ):
        if self.args.optimizer:
            return self.args.optimizer
        return torch.optim.SGD(self.model.parameters(), momentum=0.9)


    @classmethod
    def latest_modeldir(cls, path):
        dirs = list(filter(lambda x: x.isdir(), Path(path).glob('*')))
        if len(dirs) == 0:
            return None

        times = list(map(dirs.stat().st_mtime))
        i = np.argmax(times)
        return dirs[i]

    def save_checkpoint(self, modeldir=None):
        """Save the current state of the model
        """

        modeldir = modeldir if modeldir is not None else self.args.modeldir
        modeldir.mkdir(exist_ok=True)

        if self.args.checkpoint_namegen is not None:
            prefix = self.args.checkpoint_namegen(self)
        else:
            prefix = f"{self.args.prefix}-{self.state.epoch}"


        checkpointdir = Path(modeldir, prefix)
        checkpointdir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), Path(checkpointdir, f"model.pkl"))
        if self.optimizer is not None:
            torch.save(self.optimizer, Path(modeldir, "optimizer.pkl"))

        if self.scheduler is not None:
            torch.save(self.scheduler, Path(modeldir, "scheduler.pkl"))

        state_path = Path(modeldir, f"{prefix}-state.json")
        with open(state_path, 'w') as fh:
            fh.write(self.state.tojson())

    def load_checkpoint(self, modeldir):
        """Load the trainer state from checkpoint
        """

        self.model.load_state_dict(torch.load(Path(modeldir, f"model.pkl")))

        optimizer_path = Path(modeldir, f"optimizer.pkl")
        if optimizer_path.isfile():
            self.optimizer.load(optimizer_path)

        scheduler_path = Path(modeldir, f"scheduler.pkl")
        if scheduler_path.isfile():
            self.scheduler.load(optimizer_path)

        state_path = Path(modeldir, f"state.json"), 'w'
        if state_path.isfile():
            with open(state_path, 'w') as fh:
                self.state = TrainingState.fromdict(json.load(fh))


    def fewshot_training(self, model_dir=None):
        """Trains the model using standard fewshot training
        """

        if model_dir:
            self.load_checkpoint(model_dir)

        self.model.train()

        dl_train = self.get_train_dataloader()
        loss = 0

        for epoch in range(self.state.epoch,
                           self.args.max_episodes * self.args.batch_size // self.args.epoch_length):

            for step in range(self.args.epoch_length * self.args.batch_size):
                self.state.current_step += 1

                batch = next(dl_train)
                _loss, metrics = self.fewshot_episode(self.model,
                                                      batch,
                                                      self.args.kquery,
                                                      self.args.device,
                                                      training=True)

                loss += _loss
                self.state.metrics.append(metrics)

                if step % self.args.batch_size == 0 or step == self.args.epoch_length - 1:
                    self.update_step(loss)
                    loss = 0

                    self.state.global_step += 1
                    self.state.current_step += 1

                    if self.on_update_callback:
                        self.on_update_callback(self)

                    if self.scheduler is not None:
                        scheduler.step()

                    sgs = SummaryGroup.from_dicts(self.state.metrics[-self.args.batch_size:])
                    _write_sgs({"train": sgs})


            if self.on_epoch_end_callback:
                self.on_epoch_end_callback(self)

            self.state.epoch += 1

            self.save_checkpoint()

            # Save some of the metrics
            results = self.fewshot_eval()
            self._write_sgs(results)


    def update_step(self, loss):
        loss.backward()
        self.optimzer.step()
        self.optimizer.zero_grad()


    def _write_sgs(self, sgs):
        """Writes a dictionary of SummaryGroups to the trainers SummaryWritter
        """

        if self.args.writer is not None:
            for key, sg in sgs.items():
                sg.write(self.args.writer, self.state.step, key)


    def _fewshot_eval(self, ds, num_episodes, args):
        self.model.eval()

        # Start a data loader
        dl  = initialize_taskdataset(ds, args.nways, args.kshot, args.num_workers)

        kquery = self.val_args.kquery if self.val_args is not None else self.args.kquery
        device = self.val_args.device if self.val_args is not None else self.args.device

        metrics = []
        for i in range(num_episodes):
            batch = next(dl)
            _, _res = fewshot_episode(self.learner, batch, kquery, device, args.metric_fn)
            metrics.append(_res)

        sg = SummaryGroup.from_dicts(metrics)
        return sg

    def fewshot_eval(self,
                     datasets: Optional[Dict[str, Dataset]] = None,
                     num_episodes: int =None):
        """Performs fewshot evaluation on the given datasets or the evaluation datasets
        """


        if datasets is None:
            if self.base_eval_dataset is not None:
                datasets.update("base", self.base_eval_dataset)

            if self.novel_eval_dataset is not None:
                datasets.update("novel", self.novel_eval_dataset)

        num_episodes = self.args.eval_episodes if num_episodes is None else num_episodes,
        res = {key: self._fewshot_eval(ds, num_episodes, self.args) for key, ds in datasets.items()}
        return res

    def get_train_dataloader(self):
        args = self.args
        dl = initialize_taskdataset(self.train_dataset, self.args.nways, self.args.kways, self.args.num_workers)
        return dl


    def fewshot_episode(self, batch, *args, **kwargs):
        return fewshot_episode(self.model, batch, *args, **kwargs)


class PreTrainer(FewshotTrainer):
    """Trainer wrapper-class for few-shot laerner's pre-training stage
    """

    def get_train_dataloader(self):
        """
        """

        args = self.args
        dl = DataLoader(self.train_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
        return dl

    def get_eval_dataloader(self, dataset):
        """
        """

        args = self.eval_args if self.eval_args is not None else self.args
        dataset = self.eval_base_dataset if not dataset else dataset
        dl = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
        return dl


    def predict(self, images):
        """Predicts labels of input images

        logits are returned on the device specified in args.device

        :param images: tensor of n images, i.e. with shape [n, h, w, c]
        :return: Logits
        """
        images = images.to(self.args.device)
        return self.model(images)


    def forward_step(self, batch):
        """Perform a single forward step
        """

        images, labels = batch
        labels = labels.to(self.args.device)
        logits = self.predict(images)

        loss = F.cross_entropy(logits, labels)
        metrics = compute_metrics(logits, labels, loss, self.arg.metric_fn)

        return loss, metrics

    def evaluate(self, dataset):
        """Evaluate the model
        """

        self.model.eval()

        # Get the few-shot evaluation
        fewshot_metrics = self.fewshot_eval()

        # Get the standard classifiation evaluation
        dl = self.get_eval_dataloader(dataset)

        metrics = []
        for i, batch in enumerate(dl):
            _, _metrics = self.forward_step(batch)
            metrics.append(metrics)

        self.model.train()

        # Combine results and return
        return {**fewshot_metrics, "standard": metrics}


    def train(self, checkpoint=None):
        """Runs training process specified by the training arguments

        :param checkpoint: modeldir with checkpoints
        """

        # Load existing checkpoints
        if checkpoint:
            self.load_checkpoint(checkpoint)


        dl_train = self.get_train_dataloader()
        loss = 0

        for epoch in range(self.args.max_peoch):
            self.model.train()
            dl_train = self.get_train_dataloader()

            for i, batch in dl_train:

                self.state.global_step += 1
                self.state.curret_step += 1

                self.optimizer.zero_grad()
                loss, acc = self.forward_step(batch)
                loss.backward()
                self.optimizer.step()

                self.on_update_callback(self)

            self.on_epoch_end_callback(self)

            if self.args.do_eval:
                sgs = self.evaluate()
                self._write_sgs(sgs)

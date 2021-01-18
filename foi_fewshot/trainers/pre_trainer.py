from typing import List, Optional
import json

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel

import numpy as np
from pathlib import Path

from ..utils import compute_metrics
from .fewshot_trainer import FewshotTrainer


class PreTrainer(FewshotTrainer):
    """Trainer wrapper-class for few-shot laerner's pre-training stage"""

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
        super().__init__(
            model,
            train_dataset,
            args,
            eval_args,
            base_eval_dataset,
            novel_eval_dataset,
            optimizer,
            scheduler,
            distributed,
            on_epoch_begin_callback,
            on_epoch_end_callback,
            on_update_callback,
        )


    def _envelop_model(self, model):
        self.model = model
        data_parallel = DistributedDataParallel if self.distributed else DataParallel
        self.data_parallel = data_parallel(
            self.model,
            device_ids=self.args.device_ids,
            output_device=torch.device("cpu"),
        )

    def get_train_dataloader(self):
        """"""

        args = self.args
        dl = DataLoader(
            self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        return dl

    def get_eval_dataloader(self, dataset):
        """"""
        args = self.eval_args if self.eval_args is not None else self.args
        dataset = self.eval_base_dataset if not dataset else dataset
        dl = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        return dl

    def predict(self, images):
        """Predicts labels of input images

        logits are returned on the device specified in args.device

        :param images: tensor of n images, i.e. with shape [n, h, w, c]
        :return: Logits
        """
        return self.data_parallel(images)

    def forward_step(self, batch):
        """Perform a single forward step"""

        images, labels = batch
        logits = self.predict(images)

        loss = F.cross_entropy(logits, labels.long())
        metrics = compute_metrics(logits, labels, loss, self.args.metric_fn)

        return loss, metrics

    def evaluate(self, dataset=None, prefix="eval", logging=False):
        """Evaluate the model"""

        # Get the few-shot evaluation
        fewshot_metrics = self.fewshot_eval(logging=False)

        training = self.model.training
        if training:
            self.model.eval()

        dataset = dataset if dataset is not None else self.base_eval_dataset

        # Get the standard classifiation evaluation
        dl = self.get_eval_dataloader(dataset)

        metrics = []
        for i, batch in enumerate(dl):
            _, _metrics = self.forward_step(batch)
            metrics.extend(metrics)

        # Combine the two metrics
        metrics = {**fewshot_metrics, prefix: metrics}

        if training:
            self.model.training()

        if logging:
            for prefix, metric in metrics.items():
                self.add_logs(metric, prefix=prefix)
                self.save_logs()

        return

    def train(self, model_dir=None):
        """Trains the model using standard fewshot training"""

        if model_dir:
            self.load_checkpoint(model_dir)

        self.model.train()

        for epoch in range(
            self.state.epoch,
            self.args.max_epochs,
        ):

            dl_train = self.get_train_dataloader()
            dl_iter = tqdm(iter(dl_train), desc="Training model")

            if self.on_epoch_begin_callback:
                self.on_epoch_begin_callback(self)

            for batch in dl_iter:
                losses, metrics = self.forward_step(batch)
                self.update_step(losses)
                self.scheduler_step()

                if (
                    self.state.current_step % self.args.save_step == 0
                    and self.state.current_step > 0
                ):
                    self.save_checkpoint()

                self.state.global_step += 1
                self.state.current_step += 1

                if self.state.current_step % self.args.log_step == 0:
                    self.save_logs()

                if self.on_update_callback:
                    self.on_update_callback(self)

            if self.on_epoch_end_callback:
                self.on_epoch_end_callback(self)

            self.state.epoch += 1
            self.state.current_step = 0

            # Save the checkpoint
            self.save_checkpoint()

            # Save some of the metrics
            if self.args.do_eval:
                self.evalutate(logging=True)

        return self.model

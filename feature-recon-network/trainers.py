import os
from pathlib import Path
import torch
from dataclasses import asdict, dataclass
from torch.utils.data import DataLoader

from .utils import Averager
from .utils import Statistics
from .models import resnet12
from .training_arguments import TrainingArguments, PretrainArguments

import learn2learn2 as l2l
from torch import functional as F


@dataclass
class TrainerState():
    """Simple Wrapper for the trainer state
    """

    epoch: int
    global_step: int


class FewshotTrainer():

    def get_train_dataloader(self,):

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        task_dataset = initialize_task_dataset(self.train_dataset)
        return DataLoader(
            ds=task_dataset,
            num_workers=self.args.train_num_workers
        )


    def _get_eval_dataloader(self, ds, args):
        task_dataset = initialize_task_dataset(ds,
                                               args.eval_nways,
                                               args.eval_kshot)
        return DataLoader(
            ds=task_dataset,
            num_workers=args.eval_num_workers
        )

    def get_novel_eval_dataloader(self):
        if self.novel_eval_dataset is None:
            raise ValueError("Trainer: training requires a novel-eval dataset")
        return self._get_eval_dataloader(self, ds, args)

    def get_base_eval_dataloader(self):
        if self.base_eval_dataset is None:
            raise ValueError("Trainer: training requires a novel-eval dataset")
        return self._get_eval_dataloader(self, ds, args)

    def save_checkpoint(self, model_path):
        if model_path is not None:


    def load_checkpoint(self, model_path):

        if Path(model_path, "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )

        if Path(model_path, "scheduler.pt").exists():
            self.scheduler.load_state_dict(
                torch.load(os.path.join(model_path, "scheduler.pt"))
            )

    def train(self,):



class PreTrainer():
    """PreTrainer

    A trainer wrapper for the pretraining stage
    """

    def __init__(self, model, args, ds_train, ds_val, ds_novel):
        self.model = model
        self.model.init_pretraining(len(ds_train))

        self.stats = Statistics()
        self.args = args

        self.opt = self.arg.optimizer(model.parameters(),
                                      **self.args.optimizer_arguments)
        self.scheduler = self.args.scheduler(self.opt,
                                             **self.args.scheduler_arguments)


    def train_epoch(self, ds_train):
        dl_train = DataLoader(ds_train, batch_size=self.args.train_batch_size)

        for batch in dl_train:
            images, labels = batch
            self.opt.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy()
            loss.backward()
            self.opt.step()
            acc = (torch.max(logits)[1] == labels) / len(labels)
            self.stats["pretrain/loss/train"]

        if self.scheduler:
            self.scheduler.step()

    def eval(self, ds_val):
        self.model.eval()
        dl_val = DataLoader(ds_val, batch_size=self.args.val_batch_size)

        for batch in dl_val:
            images, labels = batch
            self.opt.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy()
            acc = (torch.max(logits)[1] == labels) / len(labels)

        base_dl = initialize_taskdataset(ds_val, self.args)

    def dump(self, epoch):
        modeldir = self.args.modeldir
        torch.save(self.model.state_dict(), Path(modeldir, f"pretrain-model-{epoch}.pkl"))
        torch.save(self.opt, Path(modeldir, f"pretrain-opt-{epoch}.pkl"))
        torch.save(self.scheduler, Path(modeldir, f"pretrain-scheduler-{epoch}.pkl"))

    def load(self, epoch):
        modeldir = self.args.modeldir
        self.model.load_state_dict(torch.load(Path(modeldir, f"pretrain-model-{epoch}.pkl")))
        self.opt.load(Path(modeldir, f"pretrain-opt-{epoch}.pkl"))
        self.scheduler.load(Path(modeldir, f"pretrain-scheduler-{epoch}.pkl"))


if __name__ == '__main__':
    ds = l2l.data.MiniImagenet(mode='train', download=True)
    model = resnet12()
    train_args = TrainingArguments("~/Downalod")
    pt_args = PretrainArguments(**asdict(train_args))

    trainer = PreTrainer(model, pt_args)
    trainer.train()

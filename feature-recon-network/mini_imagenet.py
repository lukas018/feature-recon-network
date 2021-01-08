from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torch.functional as F
from learn2learn.data.transforms import Nways, Kshot, LoadImage, ConsecutiveLabels, RemapLabels
from learn2learn.data import MetaDataset, TaskDataset

from .random_transformers import RandomNways, RandomKshot
from .fewshot_training import fewshot_episode


class Averager():

    def __init__(self, alpha):
        self.alpha = alpha
        self.v = None
        self.count

    def _update(self, v):
        if self.count > 0:
            self.v = (self.alpha * v) + ((1. *self.alpha) * self.v)
        else:
            self.v = v

    def __call__(self, v=None):
        if v is not None:
            self._update(v)
        return self.v


class Statistics():

    def __init__(self, writer, alpha=0.95):
        self.writer = writer
        self.averagers = defaultdict(partial(Averager, alpha=alpha))

    def __setitem__(self, key, value):
        self.averagres[key](value)

    def write(self, step):
        for key, averager in self.averages.items():
            writer.add_scalar(key, averager(), step)

    def __str__(self):
        return '\n'.join([f"{key}:{averager()}"
                          for key, averager
                          in self.averages.items()])


def initialize_taskdataset(ds, nways, kways, num_workers):
    task_transforms = [
        RandomNways(ds, nways) if isinstance(nways, tuple) else Nways(ds, nways),
        RandomKshot(ds, nways) if isinstance(kways, tuple) else Kshot(ds, kways),
        LoadImage(ds),
        ConsecutiveLabels(ds),
        RemapLabels(ds)
    ]
    meta_ds = MetaDataset(ds)
    task_ds = TaskDataset(ds, task_transforms)
    return DataLoader(task_ds, num_workers=num_workers)


def fewshot_eval(learner, ds, num_episodes, args):

    learner.eval()
    dl  = initialize_taskdataset(ds, args.nways, args.kshot, args.num_workers)

    for i in range(num_episodes):
        batch = next(dl)
        loss, acc = fewshot_episode(learner, batch, query_k, device)

        statistics[''] = loss.detach.cpu().numpy()
        statistics[''] = acc


def fewshot_trainig(learner, ds, num_episodes, args):
    dl = initialize_taskdataset(ds)
    for i in range(num_episodes):
        opt.zero_grad()
        loss, acc = fewshot_episode(frn, batch, query_k, device)
        err.backward()
        opt.update()

        writer.add_scalar('fewshot/loss/train', loss, i)
        writer.add_scalar('fewshot/acc/train', acc, i)



class PreTrainer():

    def __init__(self, model, args, ds_train, ds_val, ds_novel):
        self.model = model
        self.model.init_pretraining()
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
            acc = (torch.max(logits)[1] == labels) / len(labels)
            self.stats["pretrain/loss/train"]

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

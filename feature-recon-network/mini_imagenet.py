from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torch.functional as F
from learn2learn.data.transforms import Nways, Kshot, LoadImage, ConsecutiveLabels, RemapLabels
from learn2learn.data import MetaDataset, TaskDataset

from cached_property import cached_property

from .random_transformers import RandomNways, RandomKshot
from .fewshot_training import fewshot_episode
import numpy as np, scipy.stats as st


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


class Stats():

    def __init__(self, moving_avg_window=None):
        self._values = []
        self.moving_avg_window = moving_avg_window

    @property
    def array(self):
        return np.array(self._values)

    @property
    def mean(self):
        if self.moving_avg_window is not None:
            return np.mean(self.array[-self.moving_avg_window:])
        else:
            return np.mean(self.array)

    @property
    def std(self):
        return np.std(self.array)

    @property
    def c95(self):
        return self.conf_int(0.95)

    def conf_int(self, c):
        a = self.array
        interval = st.t.interval(c, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        return interval

    def __call__(self, *args):
        self._values.extend([*args])


class SummaryGroup():

    def __init__(self, writer):
        self.writer = writer
        self.stats = defaultdict(Statistics)

    def __setitem__(self, key, *values):
        self.stats[key](value)

    def write(self, step):
        for key, stats in self.stats.items():
            writer.add_scalar(key, stats.mean, step)

    def __str__(self):
        strs = [f"{key}: {stat.mean:.3f} ± {stat.std:.3f}"
                          for key, stat
                          in self.stats.items()]
        return '\n'.join(strs)

    def join(sg):
        new_sg = SummaryGroup(self.writer)
        new_sg.stats = copy.deepcopy(new_sg.stats)
        new_sg.stats.update(sg.stats)
        return new_sg


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
        loss, res = fewshot_episode(learner, batch, query_k, device)



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

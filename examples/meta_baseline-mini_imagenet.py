#!/usr/bin/env python3
from dataclasses import asdict

from tqdm import tqdm
from foi_fewshot.models import ResNet12
from foi_fewshot.data import mini_imagenet, split_dataset
from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.trainers import (
    PreTrainer,
    FewshotTrainer,
    TrainingArguments,
    FewshotArguments,
)
from foi_fewshot.data import initialize_taskdataset
from torch.nn.parallel import DataParallel, DistributedDataParallel

from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import (
    RandomNWays,
    RandomKShots,
    LoadData,
    ConsecutiveLabels,
    RemapLabels,
    NWays,
    KShots,
)

model = ResNet12()
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train)

train_arguments = TrainingArguments(
    modeldir="~/Downloads/models",
    logdir="~/Downloads/logdir",
    modeldir_prefix="pretrain",
    batch_size=8,
    max_epochs=100,
    save_step=5,
)

model = ResNet12()
meta_baseline = MetaBaseline(model)
meta_baseline.init_pretraining(640, 64)
pre_trainer = PreTrainer(model, ds_train, train_arguments, None, ds_base, ds_novel)
pre_trainer.train()

fs_arguments = FewshotArguments(
    **asdict(train_arguments),
    nways=5,
    ksupport=2,
    kquery=3,
    modeldir_prefix="pretrain",
)

fs_trainer = FewshotTrainer(
    meta_baseline, ds_train, fs_arguments, None, ds_base, ds_novel
)
dl = fs_trainer.get_train_dataloader()
fs_trainer.train()
results = fs_trainer.evaluate(ds_test)

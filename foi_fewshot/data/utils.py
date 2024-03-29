#!/usr/bin/env python3
import copy
import functools
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from torch.utils.data.distributed import DistributedSampler

import torch
from ..utils import prepare_fewshot_batch
from learn2learn.data import MetaDataset
from learn2learn.data import TaskDataset
from learn2learn.data.transforms import ConsecutiveLabels
from learn2learn.data.transforms import KShots
from learn2learn.data.transforms import LoadData
from learn2learn.data.transforms import NWays
from learn2learn.data.transforms import RandomKShots
from learn2learn.data.transforms import RandomNWays
from learn2learn.data.transforms import RemapLabels
from learn2learn.vision.datasets import CUBirds200
from learn2learn.vision.datasets import DescribableTextures
from learn2learn.vision.datasets import FC100
from learn2learn.vision.datasets import FGVCAircraft
from learn2learn.vision.datasets import FullOmniglot
from learn2learn.vision.datasets import MiniImagenet
from learn2learn.vision.datasets import TieredImagenet

DATASET_ATTRIBUTES = {
    MiniImagenet: ["x", "y"],
    FullOmniglot: ["dataset"],
    CUBirds200: ["data"],
    FC100: ["images", "labels"],
    TieredImagenet: ["images", "labels"],
    FGVCAircraft: ["data"],
    DescribableTextures: ["data"],
}


def prepare_task(batch, kquery):
    query_data, query_labels, support_data, support_labels = prepare_fewshot_batch(
        batch,
        kquery,
    )
    return {
        "query": query_data,
        "query_labels": query_labels,
        "support": support_data,
        "support_labels": support_labels,
    }


def task_collate(data, kquery):
    batch = collate.default_collate(data)
    batch = prepare_task(batch, kquery)
    return batch


def fewshot_metabatch_collate(tasks):
    """Collates"""

    # results = prepare_task(batch, kquery)
    keys = list(tasks[0].keys())
    meta_batch = {k: torch.stack([task[k] for task in tasks]) for k in keys}
    return meta_batch


class DeterministicTaskDataset(TaskDataset):

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        state = random.getstate()
        random.seed(self.seed + i)
        results = self.dataset[i]
        random.setstate(state)
        return results


def initialize_taskdataset(
    ds,
    nways,
    kshots,
    kquery,
    num_tasks,
    shuffle=False,
):
    """Returns a fewshot classification task data dataset

    :param ds: Dataset or Metadataset
    :param nways: Number of classes (range or int)
    :param kshots: Number of shots (support + query)
    :param kquery: Number of kquery elements (fixed)
    :param num_tasks: The amount of samples to use in the data-loader
    """

    if not isinstance(ds, MetaDataset):
        ds = fast_metadataset(ds)

    task_transforms = [
        RandomNWays(ds, nways) if isinstance(nways, tuple) else NWays(ds, nways),
        RandomKShots(ds, kshots) if isinstance(kshots, tuple) else KShots(ds, kshots),
        LoadData(ds),
        ConsecutiveLabels(ds),
        RemapLabels(ds, shuffle=shuffle),
    ]

    # def collate_fn(batch):
    #     return tuple(tuple(dp) for dp in batch)

    task_collate_fn = functools.partial(task_collate, kquery=kquery)

    # We only need this custom collator if we used variable task sizes
    # if not isinstance(nways, tuple) and not isinstance(kshots, tuple):
    # collate_fn = None

    task_ds = TaskDataset(
        ds,
        task_transforms,
        num_tasks=num_tasks,
        task_collate=task_collate_fn,
    )

    return task_ds


def initialize_taskloader(
    task_ds,
    num_workers=1,
    batch_size=1,
    deterministic=False,
    distributed=False,
    epoch=-1,
    seed=42,
):
    """Initialize a taskloader from a task-dataset

    :param task_ds: dataset to sample tasks from
    :param num_workers: workers in dataloader
    :param batch_size: meta-batch size (e.g. number of tasks in each batch)
    :param deterministic:
    :param distributed:
    :param epoch:
    :param seed:

    :returns: DataLoader for meta-learning tasks
    """
    shuffle = False
    if deterministic:
        task_ds = DeterministicTaskDataset(task_ds, seed=epoch+seed)

    # If we want to use a DataDistributedParallell we need a distributed sampler
    if distributed:
        sampler = DistributedSampler(task_ds, seed=seed)
        sampler.set_epoch(epoch)
    elif deterministic:
        # Create a generator with a fixed seed
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = torch.utils.data.RandomSampler(task_ds, generator=generator)
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        task_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=fewshot_metabatch_collate,
        sampler=sampler,
        shuffle=shuffle
    )


def _classes_split(y, frac):
    """Split a list of labels into two splits such that each label-type is equally
    represented in both splits

    :param y: List of labels
    :param frac: The fraction of examples to put in the first split

    :returns: Indices over the two splits
    """

    idx_groups = _idx_groupby(y, lambda x: x)

    def sampler(values):
        random.shuffle(values)
        pivot = int(len(values) * frac)
        return values[:pivot], values[pivot:]

    split1, split2 = (*zip(*map(sampler, idx_groups.values())),)  # type: ignore
    return np.array(list(chain(*split1))), np.array(list(chain(*split2)))


def _idx_groupby(items, key):
    group = defaultdict(list)
    for i, item in enumerate(items):
        group[key(item)].append(i)
    return group


def split_dataset(ds, frac=0.95, even_class_dist=False, custom_attrs=None):
    """Split dataset into two

    This function is supposed to be used to create train/validation splits
    of existing datasets.

    :param ds: Dataset
    :param frac: Fractions of data use in the first dataset
    :param even_class_dist: Enfore an even class distribution between splits
    :param custom_attrs: Name of the attributes to split along (otherwise DATASET_ATTRIBUTES is used to look up)
    """

    ds1 = copy.copy(ds)
    ds2 = copy.copy(ds1)

    if custom_attrs is not None:
        attrs = custom_attrs
    else:
        try:
            attrs = DATASET_ATTRIBUTES[type(ds)]
        except ValueError:
            ValueError(
                f"Can't split dataset of type: {type(ds)}\nPlease provide *custom_attrs*",
            )

    if even_class_dist:
        labels = (
            list(map(itemgetter(1), getattr(ds, attrs[0])))
            if len(attrs) == 1
            else getattr(ds, attrs[-1])
        )
        indx1, indx2 = _classes_split(labels, frac)
    else:
        indx = np.arange(len(ds))
        random.shuffle(indx)
        cutoff = int(frac * len(indx))
        indx1, indx2 = indx[:cutoff], indx[cutoff:]

    def _copy(src, target, attrs, indx):
        """Copy src.attr[indx] -> target.attr"""
        for attr in attrs:
            src_data = getattr(src, attr)
            target_data = (
                src_data[indx]
                if isinstance(src_data, np.ndarray)
                else [src_data[i] for i in indx]
            )
            setattr(target, attr, target_data)

    _copy(ds, ds1, attrs, indx1)
    _copy(ds, ds2, attrs, indx2)

    # Since we might mess up the bookkeeping we need to recompute it later
    def remove_bookkeeping(ds):
        if hasattr(ds, "_bookkeeping_path"):
            delattr(ds, "_bookkeeping_path")

    remove_bookkeeping(ds1)
    remove_bookkeeping(ds2)

    return ds1, ds2


def fast_metadataset(dataset):
    attrs = DATASET_ATTRIBUTES.get(type(dataset), None)
    if attrs is None:
        return MetaDataset(dataset)

    if len(attrs) > 1:
        x = getattr(dataset, attrs[-1])
    else:
        x = getattr(dataset, attrs[-1])
        x = list(map(itemgetter(1), x))

    indices_to_labels = {i: x for i, x in enumerate(x)}
    labels_to_indices = _idx_groupby(x, lambda x: x)
    return MetaDataset(dataset, labels_to_indices, indices_to_labels)

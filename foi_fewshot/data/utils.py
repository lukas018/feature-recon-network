#!/usr/bin/env python3
import copy
import numpy as np
from torch.utils.data import DataLoader
from learn2learn.data.transforms import (
    RandomNWays,
    RandomKShots,
    LoadData,
    ConsecutiveLabels,
    RemapLabels,
    NWays,
    KShots,
)

from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.vision.datasets import (
    MiniImagenet,
    FullOmniglot,
    CUBirds200,
    FC100,
    TieredImagenet,
    FGVCAircraft,
    DescribableTextures,
)


DATASET_ATTRIBUTES = {
    MiniImagenet: ["x", "y"],
    FullOmniglot: ["dataset"],
    CUBirds200: ["data"],
    FC100: ["images", "labels"],
    TieredImagenet: ["images", "labels"],
    FGVCAircraft: ["data"],
    DescribableTextures: ["data"],
}

from operator import itemgetter
from collections import defaultdict
import random
from itertools import chain
from itertools import starmap


def initialize_taskdataset(
    ds, nways, kshots, num_tasks, num_workers, batch_size=1, shuffle=False
):
    """Returns a fewshot classificatino task data loader

    :param ds: Dataset
    :param nways: Number of classes (range or int)
    :param kshow: Number of shots (support + query)
    :param num_workers: Number of workers in the dataloader
    """

    ds = MetaDataset(ds)
    task_transforms = [
        RandomNWays(ds, nways) if isinstance(nways, tuple) else NWays(ds, nways),
        RandomKShots(ds, kshots) if isinstance(kshots, tuple) else KShots(ds, kshots),
        LoadData(ds),
        ConsecutiveLabels(ds),
        RemapLabels(ds, shuffle=shuffle),
    ]

    def collate_fn(batch):
        return tuple(tuple(dp) for dp in batch)

    # We only need this custom collator if we used variable task sizes
    # if not isinstance(nways, tuple) and not isinstance(kshots, tuple):
    # collate_fn = None

    task_ds = TaskDataset(ds, task_transforms, num_tasks=num_tasks * batch_size)
    return DataLoader(
        task_ds, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn
    )


def _classes_split(y, frac):
    """Helper function for spliting items while retaining class balance
    Returns"""

    def groupby(items, key):
        group = defaultdict(list)
        for i, item in enumerate(items):
            group[key(item)].append(i)
        return group

    idx_groups = groupby(y, lambda x: x)

    def sampler(values):
        random.shuffle(values)
        pivot = int(len(values) * frac)
        return values[:pivot], values[pivot:]

    split1, split2 = (*zip(*map(sampler, idx_groups.values())),)
    return np.array(list(chain(*split1))), np.array(list(chain(*split2)))


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
        except:
            ValueError(
                f"Can't split dataset of type: {type(ds)}\nPlease provide *custom_attrs*"
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

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import copy
import numpy as np
import torch
import torch.nn as nn

from operator import itemgetter
from collections import defaultdict
import random
from itertools import chain
from itertools import starmap

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


def maml_episode(
    learner,
    batch,
    update_steps,
    kquery,
    device,
    loss_fn=F.cross_entropy,
    metric_fn=None,
):
    """MAML fewshot episode where the learner preforms *n=update_steps* gradient steps"""

    # Separate data into support/query sets
    query_data, query_labels, support_data, support_labels = _prepare_batch(
        batch,
        kquery,
    )

    # We don't need to keep track of the nway, kshot setup here
    support_data = support_data.flatten(0, 1)

    # Adapt the model
    for step in range(update_steps):
        error = loss_fn(learner(support_data), support_labels)
        learner.adapt(error)

    # Evaluate the adapted model
    logits = learner(query_data)

    # Compute loss
    loss = loss_fn(logits, query_labels)
    res = compute_metrics(logits, query_labels, loss, metric_fn)

    return loss, res


def reptile_episode(
    learner,
    batch,
    update_steps,
    kquery,
    optimizer,
    device,
    loss_fn=F.cross_entropy,
    metric_fn=None,
):
    """MAML fewshot episode where the learner preforms *n=update_steps* gradient steps"""

    # Separate data into support/query sets
    query_data, query_labels, support_data, support_labels = _prepare_batch(
        batch,
        kquery,
    )

    # We don't need to keep track of the nway, kshot setup here
    support_data = support_data.flatten(0, 1)

    # Adapt the model
    for step in range(update_steps):
        error = loss_fn(learner(support_data), support_labels)
        optimizer.step(error)

    # Evaluate the adapted model
    logits = learner(query_data)

    # Compute loss
    loss = loss_fn(logits, query_labels)
    res = compute_metrics(logits, query_labels, loss, metric_fn)

    return loss, res


def fewshot_episode(
    learner,
    batch,
    kquery,
    device=None,
    metric_fn=None,
    loss_fn=F.cross_entropy,
    **kwargs,
):
    """Perform a single fewshot epoch

    :param learner: The fewshot learner
    :param batch: The batch of images [n, ksupport + kquery, h, w, c]
    :param query_k: The number of samples to use as validation
    :param compute_metrics: Computes an additional set of matrices
    """

    # Separate data into support and query sets
    query_data, query_labels, support_data, support_labels = _prepare_batch(
        batch,
        kquery,
    )

    # TODO(Lukas) Currently we assume that support is ordered, this should be changed
    logits = learner(query_data, support_data)

    _loss = 0
    if isinstance(logits, tuple):
        if len(logits) == 2:
            logits, _loss = logits
        elif len(logits) == 1:
            logits = logits[0]
        else:
            raise ValueError("Learner returned logits tuple of size >2")

    loss = loss_fn(logits, query_labels)
    loss += _loss

    res = compute_metrics(logits, query_labels, loss, metric_fn)
    return loss, res


def _split_fewshot_batch(images, labels, nways, ktotal, kquery):
    """Separate a batch of images into support and query data"""

    # Separate data into support and query sets
    query_indices = np.zeros(nways * ktotal, dtype=bool)
    query_indices[(np.arange(nways * ktotal) % ktotal) >= (ktotal - kquery)] = True
    support_indices = torch.from_numpy(~query_indices)
    query_indices = torch.from_numpy(query_indices)

    query_data, query_labels = images[query_indices], labels[query_indices]
    support_data, support_labels = images[support_indices], labels[support_indices]

    # We reshape here to keep track of the number of nways, kshots
    support_data = support_data.reshape((nways, ktotal - kquery, *images.shape[1:]))

    return query_data, query_labels, support_data, support_labels


def _prepare_batch(batch, kquery):
    data, labels = batch

    # Figure out the number of samples and classes
    nways = len(torch.unique(labels))
    ktotal = len(labels) // nways
    return _split_fewshot_batch(data, labels, nways, ktotal, kquery)


def compute_metrics(logits, labels, loss=None, metric_fn=None):
    """Helper function for computing results from prediction

    :param logits: Distribution of labels
    :param labels:
    :param loss: Loss value (provided since the loss sometimes uses external information)
    :param metric_fn: Function that takes logits and labels and returns a dict of metrics
    """

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().data.numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().data.numpy()

    acc = np.sum(np.argmax(logits, axis=1) == labels) / len(logits)
    metrics = {"acc": acc}

    if loss is not None:
        if isinstance(loss, torch.Tensor):
            loss = float(loss.detach().data.numpy())
        metrics.update({"loss": loss})

    if metric_fn is not None:
        metrics = {**metrics, **metric_fn(logits, labels)}

    return metrics


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


def split_dataset(ds, frac, even_class_dist=False, custom_attrs=None):
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

    return ds1, ds2

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


def prepare_fewshot_batch(batch, kquery):
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

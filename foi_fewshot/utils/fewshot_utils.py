import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import torch
import torch.nn as nn




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



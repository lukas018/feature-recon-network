import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import copy
import numpy as np
import torch
import torch.nn as nn
from .random_transforms import RandomNWays, RandomKShot


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
        batch, device, kquery
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
        batch, device, kquery
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
    learner, batch, kquery, device, metric_fn=None, loss_fn=F.cross_entropy, **kwargs
):
    """Perform a single fewshot epoch

    :param learner: The fewshot learner
    :param batch: The batch of images [n, ksupport + kquery, h, w, c]
    :param query_k: The number of samples to use as validation
    :param compute_metrics: Computes an additional set of matrices
    """

    # Separate data into support and query sets
    query_data, query_labels, support_data, support_labels = _prepare_batch(
        batch, device, kquery
    )

    # TODO(Lukas) Currently we assume that support is ordered, this should be changed
    logits = learner(query_data, support_data)

    _loss = 0
    if isinstance(logits, tuple):
        logits, _loss = logits

    loss = loss_fn(logits, query_labels)
    loss += _loss

    res = compute_metrics(logits, query_labels, loss, metric_fn)
    return loss, res


def _split_fewshot_batch(images, labels, nways, total_k, query_k):
    """Create"""

    # Separate data into support and query sets
    query_indices = np.zeros((nways, total_k), dtype=bool)
    query_indices[(np.arange(nway * k_total) % total_k) >= (total_k - query_k)] = True
    query_indices = torch.from_numpy(query_indices)
    support_indices = torch.from_numpy(~query_indices)

    query_data, query_labels = images[query_indices], labels[query_indices]
    support_data, support_labels = images[support_indices], labels[support_indices]

    # We reshape here to keep track of the number of nways, kshots
    support_data = support_data.reshape((nways, total_k - query_k, images.shape[1:]))

    return query_data, query_labels, support_data, support_labels


def _prepare_batch(batch, device, kquery):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Figure out the number of samples and classes
    nways = len(set(labels))
    total_k = nways // len(labels)
    kshos = total_k - kquery
    img_shape = data.shape[1:]

    return _split_fewshot_batch(data, labels, nways, total_k, kquery)


def compute_metrics(logits, labels, loss=None, metric_fn=None):
    """Helper function for computing results from prediction

    :param logits: Distribution of labels
    :param labels:
    :param loss: Loss value (provided since the loss sometimes uses external information)
    :param metric_fn: Function that takes logits and labels and returns a dict of metrics
    """

    if isinstance(logits, nn.Tensor):
        logits = logits.detach().data.numpy()

    if isinstance(labels, nn.Tensor):
        labels = labels.detach().data.numpy()

    acc = np.sum(logits == labels) / len(logits)
    metrics = {"acc": acc}

    if loss is not None:
        if isinstance(logits, nn.Tensor):
            loss = loss.detach().data.numpy()
        metrics.update("loss", loss)

    if metric_fn is not None:
        metrcs = {**metrics, **metric_fn(logits, labels)}

    return metrics


def initialize_taskdataset(ds, nways, kways, num_workers):
    """Returns a fewshot classificatino task data loader

    :param ds: Dataset
    :param nways: Number of classes (range or int)
    :param kshow: Number of shots (support + query)
    :param num_workers: Number of workers in the dataloader
    """

    task_transforms = [
        RandomNways(ds, nways) if isinstance(nways, tuple) else Nways(ds, nways),
        RandomKshot(ds, nways) if isinstance(kways, tuple) else Kshot(ds, kways),
        LoadImage(ds),
        ConsecutiveLabels(ds),
        RemapLabels(ds),
    ]
    meta_ds = MetaDataset(ds)
    task_ds = TaskDataset(ds, task_transforms)
    return DataLoader(task_ds, num_workers=num_workers)


def classes_split(items, frac):
    """Helper function for spliting items while retaining class balance"""

    def groupby(items, key):
        group = defaultdict()
        for i, item in enumerate(items):
            group[key(item)].append(item)
        return group

    groups = idx_groupby(items, itemgetter(1))

    def sampler(values):
        random.shuffle(values)
        pivot = int(len(values) * frac)
        return values[:pivot], values[pivot:]

    split1, split2 = zip(*map(sampler, groups.values()))
    return np.array(chain(*split1)), np.array(chain(*split2))


def split_dataset(ds, frac):
    """Split dataset while retaining class balance"""

    ds1 = copy.deepcopy(ds)
    ds2 = copy.copy(ds1)

    ds1.data, ds2.data = class_split(ds1.data, frac)
    return ds1, ds2

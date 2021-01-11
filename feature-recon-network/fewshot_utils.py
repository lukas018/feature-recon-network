import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def fewshot_episode(learner, batch, query_k, device, metric_fn=None):
    """Perform a single fewshot epoch

    :param learner: The fewshot learner
    :param batch: The batch of images [n, ksupport + kquery, h, w, c]
    :param query_k: The number of samples to use as validation
    :param compute_metrics: Computes an additional set of matrices
    """

    data, labels = batch

    # Figure out the number of samples and classes
    nway = len(set(labels))
    total_k = nway // len(labels)
    k = total_k - query_k
    img_shape = data.shape[1:]

    # Send to gpu/cpu device
    data, labels = data.to(device), labels.to(device)

    # Separate data into support and query sets
    query_indices = np.zeros(data.size(0), dtype=bool)
    query_indices[(np.arange(nway*k_total) % total_k) >= (total_k - query_k)] = True
    query_indices = torch.from_numpy(query_indices)
    support_indicesj = torch.from_numpy(~query_indices)

    query_data, labels = data[support_indicesj], labels[support_indicesj]
    support_data, _ = data[query_indices], labels[query_indices]

    # TODO(Lukas) Currently we assume that support is ordered, this should be changed
    logits, aux_loss = learner(support_data, query_data.reshape(n, k *img_shape))
    loss = F.cross_entropy(logits, labels)
    loss += aux_loss

    res = compute_metrics(logits, labels, loss, metric_fn)

    return loss, res

def compute_metrics(logits, labels, loss=None, metric_fn=None):
    """Helper function for computing results from prediction

    :param logits: Distribution of labels
    :param labels:
    :param loss: Loss value (provided since the loss sometimes uses external information)
    :param metric_fn: Function that takes logits and labels and returns a dict of metrics
    """

    if isinstace(logits, nn.Tensor):
        logits = logits.detach().data.numpy()

    if isinstace(labels, nn.Tensor):
        labels = labels.detach().data.numpy()

    acc = np.sum(logits == labels) / len(logits)
    metrics = {'acc': acc}

    if loss is not None:
        if isinstance(logits, nn.Tensor):
            loss = loss.detach().data.numpy()
        metrics.update('loss', loss)

    if metrics_fn is not None:
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
        RemapLabels(ds)
    ]
    meta_ds = MetaDataset(ds)
    task_ds = TaskDataset(ds, task_transforms)
    return DataLoader(task_ds, num_workers=num_workers)


def classes_split(items, frac):
    """Split a given dataset in two
    """

    def groupby(items, key):
        group = defaultdict()
        for i, item in enumerate(items):
            group[key(item)].append(item)
        return group

    groups = idx_groupby(items, itemgetter(1))

    def sampler(values):
        random.shuffle(values)
        pivot = int(len(values)*frac)
        return values[:pivot], values[pivot:]

    split1, split2 = zip(*map(sampler, groups.values()))
    return np.array(chain(*split1)), np.array(chain(*split2))

def split_dataset(ds, frac):
    """Split dataset while retaining class balance
    """

    ds1 = copy.deepcopy(ds)
    ds2 = copy.copy(ds1)

    ds1.data, ds2.data = class_split(ds1.data, frac)
    return ds1, ds2

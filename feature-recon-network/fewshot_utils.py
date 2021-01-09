import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def fewshot_episode(learner, batch, query_k, device, compute_metrics=None):
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

    # TODO(Lukas) Currently we assuemt that support
    logits, aux_loss = learner(support_data, query_data.reshape(n, k *img_shape))
    loss = F.cross_entropy(logits, labels)
    loss += aux_loss

    acc = ((torch.max(logits)[1]) == labels) / (nway*total_k)

    res = {"loss": loss.detach().data.numpy(), "accuracy": acc}
    if compute_metrics is not None:
        res = {**res, **compute_metrics(logits, labels)}

    return loss, res

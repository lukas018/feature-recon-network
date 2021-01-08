import copy
from operator import itemgetter

import torch
import torch.utils.data as data
import torch.nn as nn
from torch import linalg as LA
from typing import Optional, Tuple

from functools import partial
ridge_reg = (num_channels / num_support * dimensions) * torch.exp(alpha)
recalibration_term = torch.exp(beta)
identity = torch.eye()


# num_images x height x width x channel
features = feature_extractor(images)

# class x num_imagesx height x width x channel
features = features.reshape(features.dim[0] // num_classes, -1, *features.dim[1:])


class FeatureReconNetwork():
    """Feature Reconstruction Network is an implementation
    """

    def __init__(self,
                 learner,
                 num_channels,
                 dimensions,
                 alpha=1,
                 beta=1,
                 temperature=1
                 ):
        """Feature Reconstruction Network

        :param learner: The feature extractor (any cnn of equivelent)
        :param num_channels: The number of output channels on the final layer of learner
        :param dimensions: Height x width of the feature maps outputed by learner
        :param alpha: Initial value for learnable parameter alpha
        :param beta: Initial value for learnable parameter beta
        """
        self.learner = learner
        self.alpha = torch.float(alpha)
        self.beta = torch.float(beta)
        self.num_channels = num_channels
        self.dimensions = dimensions
        self.temperature = torch.float(temperature)

    def init_pretraining(self, num_classes: int):
        """Initialize the FRN for pre-training

        :param num_classes: The number of classes in the pretraining dataset
        """
        self.class_matrices = torch.randn((num_classes, self.dimensions, self.num_channels))


    def forward(self,
                query: nn.Torch,
                support: Optional[nn.Torch]=None
    ) -> Tuple[nn.Torch, Tuple[nn.Torch, nn.Torch]]:
        """Predict labels using FRN.

        This function offerst two different modes: standard prediction and few-shot predictions.
        Standard prediction acts standard image classification with logits of n-classes.
        This mode is enabled by default and is meant to be used during the pre-training phase outlined in the original paper.

     
        Few-shot prediction is used when support-images are given as arguments.
        Support images are a set of n x k images (n classes, with k examples in each)
        and is used to compute the class representation matrices used for prediction.

        :param query: Set of images such that shape=(bsz x h x w x channels)
        :param support: Set of images such that shape=(n x k x h x w x channels)
        :output logits: Prediction for each imput image. Logit dimension depends on mode-used.
        """

        # Compute and flatten the input features
        # [bsz, r, d]
        query = learner(query).flatten(1,2)

        if support is not None:

            # Do few-shot prediction
            nway = support.shape[0]
            k = support.shape[1]

            # [nway, k*r, d]
            support = learner(support.flatten(0, 1)).reshape(nway, -1, self.dimensions)
            aux_loss = self._aux_loss(support)

            r = torch.exp(self.beta)
            lam = (self.num_channels / (nway * self.dimensions) * torch.exp(self.alpha))

            recons = self._reconstruct(query, support, r, lam)
            logits = (self._predictions(recons, query), aux_loss)
        else:
            # Standard predictions
            recons = self.reconstruct(query, self.class_matrix, 1, 1)
            logits = self._predictions(recons, query)

        return logits


    def _aux_loss(self, class_matrices) -> nn.Torch:
        """ Compute the auxiluary loss

        :param class_matrices: Class matrices
        :output: loss
        """

        # Row normalize
        class_matrices = class_matrices / LA.norm(class_matrices, 2)
        loss = 0
        for i in range(len(class_matrices)):
            for j in range(len(class_matrices)):
                if i == j:
                    continue
                loss += LA.norm(class_matrices[i, :] @ class_matrices[j,:].t(), 2)
        return loss

    def _reconstruct(self,
                     query,
                     support,
                     r,
                     lam):

        """ Compute reconstructions according to paper
        :param query: [bsz, r, d]
        :param support: [way, support_shot* r, d]
        :param r: rho
        :param lam: lambda

        """
        # Extract relenant shape information
        nway = support.shape[0]
        bsz = query.shape[0]
        dimensions = query.shape[1]

        # Flatten everything
        query = query.flatten((0, 1, 2))
        support = support.flatten((1, 2))

        reg = support.shape[1] / support.shape[2]
        st = support.permutate(0, 2, 1)
        xtx = st @ support
        m_inv = (xtx + torch.eye(xtx.shape[-1]).unsqueeze(0) * (reg * lam) ).inverse()
        hat = m_inv @ xtx

        # Reshape to [bsz, nway, r*d]
        return (query @ hat) * r

    def _predictions(self, recons, original):
        k = recons.shape[1]
        dists = -self.temperature * cdist(recons,  original.repeat((1, k, 1)), 2)
        return dists / dists.sum(axis=-1)


def split(ds, frac):

    ds1 = copy.deepcopy(ds)
    ds2 = copy.deepcopy(ds)

    idx = np.arange(len(ds))
    np.shuffle(idx)
    train_idx, val_idx = idx[:int(len(idx)*frac)], idx[int(len(idx)*frac):]

    ds1.data = ds1.data[train_idx]
    ds2.data = ds2.data[val_idx]
    return ds1, ds2



class BaseDataset(data.Dataset):

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, label = self.items[i]
        if self.transform:
            img = self.transform(img)
        return img, label


class Datasets(Enum):
    """List of datasets available for use
    """

    miniimagenet = 0
    cub200 = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Color[s]
        except KeyError:
            raise ValueError()

def init_transforms(opt=None):
    """Initialize transforms
    """
    if opt is None:
        return None

def init_dataset(ds_name, transform_opts=None):

    if ds_name == Datasets.miniimagenet:
        transforms = init_transforms(ds_name, transform_opts)
        init_func = partial(MiniImagenet, root=root, transform=transforms, download=True)
        splits = init_func('train'), init_func('val'), init_func('test')
        return splits

    if ds_name == Datasets.cub200:
        transforms = init_transforms(ds_name, transform_opts)
        init_func = partial(CUBirds200, root=root, transform=transforms, download=True)
        splits = init_func('train'), init_func('val'), init_func('test')
        return splits
    return None

def num_classes(ds):
    return len(set(map(itemgetter(1), ds.data)))


def train():

    parser = argparse.ArgumentParser("Initialize the data")
    parser.add_argument('-ds',
                        '--dataset',
                        type=lambda ds: Datasets[ds],
                        choices=list(Datasets))
    parser.add_argument('pretrain-frac', type=float, default=0.9)
    parser.add_argument('dataset-options', default=None)

    args = parser.parse_args()
    splits = init_dataset(args.dataset, args.opt)
    frac = args.pretrain_frac


    def pretrain_path(path: str, epoch: int):
        return Path(path, f"pre-training, epoch={epoch}.pkl")

    def save_pretrained(path: str, epoch: int, model: nn.Module):
        torch.save(model.state_dict(), pretrain_path(path, epoch))

    def load_pretrained(path: str, epoch: int, model: nn.Module):
        model.load_state_dict(torch.load(pretrain_path(path, epoch)))
        return model

    def fewshot_path(path: str, epoch: int):
        return Path(path, f"fewshot, epoch={epoch}.pkl")

    def save_fewshot(path: str, epoch: int, model: nn.Module):
        torch.save(model.state_dict(), fewshot_path(path, epoch))

    def load_fewshot(path: str, epoch: int, model: nn.Module):
        model.load_state_dict(torch.load(fewshot_path(path, epoch)))
        return model

    resnet12 = ResNet12()
    frn = FeatureReconNetwork(resnet12, 1024, 1, 1)
    opt = nn.optim.SGD(frn.parameters(), momentum=.9)

    # Start pre-training
    frn.init_pretraining()

    train_ds = splits[0]
    train_ds, val_ds = split(train_ds, 0.9)

    for epoch in range(epochs):
        for batch, labels in data:
            opt.zero_grad()
            logits = frn(batch)
            _, idx = torch.max(logits)
            acc = sum(idx == labels) / num_query_images
            loss = F_.cross_entropy(labels, logits)
            loss.backward()


        save_pretrained(path, epoch, frn)

    # Start meta-training
    opt = nn.optim.SGD(frn.parameters(), momentum=.9)
    meta_tr, meta_val, meta_test = *map(MetaDataset, splits),

    for i, (images, labels) in enumerate(train_data):

        opt.zero_grad()
        logits = frn(batch, support_images)
        _, idx = torch.max(logits)
        acc = sum(idx == labels) / num_query_images
        loss = F_.cross_entropy(labels, logits)
        loss.backward()

        if i % 1000 == 0:
            save_pretrained(path, epoch, frn)

import copy
from operator import itemgetter

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from typing import Optional, Tuple

from functools import partial

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
        self.class_matrices = None

    def init_pretraining(self, num_classes: int):
        """Initialize the FRN for pre-training

        :param num_classes: The number of classes in the pretraining dataset
        """
        if self.class_matrices is not None or self.class_matrices.shape[0] == num_classes:
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

        :return: Prediction for each imput image. Logit dimension depends on mode-used.
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
        n = recons.shape[0]
        dists = cdist(recons.repeat(len(original, 1, 1)),  original.repeat((1, n, 1)), 2)
        dists *= -self.temperature
        return F.softmax(dists, )


if __name__ == '__main__':
    model = models['resnet-12']

    ds = l2l.data.MiniImagenet(mode='train', download=True)
    train_arguments = TrainArguments()
    fs_arguments = PretrainArguments(**asdict(train_))
    trainer = FewshotTrainer(model, ds, fs_arguments)

    trainer.train()

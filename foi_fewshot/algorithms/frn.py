from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import cdist
from torch import linalg as LA
from torch import nn


class FeatureReconNetwork:
    """Feature Reconstruction Network"""

    def __init__(
        self,
        model: nn.Module,
        num_channels: int,
        dimensions: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        scale_factor: float = 1,
        temperature: float = 1,
    ):
        """Feature Reconstruction Network

        :param model: The feature extractor (any cnn of equivelent)
        :param num_channels: The number of output channels on the final layer of
            model
        :param dimensions: Height x width of the feature maps outputed by model
        :param alpha: Initial value for learnable parameter alpha
        :param beta: Initial value for learnable parameter beta
        """

        self.model = model
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)

        self.num_channels = num_channels
        self.dimensions = dimensions
        self.temperature = torch.tensor(temperature)

        self.class_matrices: Optional[torch.Tensor] = None
        self.cached_support: Optional[torch.Tensor] = None
        self.scale_factor = scale_factor

    def init_pretraining(self, num_classes: int):
        """Initialize the learner for pre-training

        :param num_classes: The number of classes ~n~ in the pretraining dataset
        """

        if self.class_matrices is None or self.class_matrices.shape[0] == num_classes:
            self.class_matrices = torch.randn(
                (num_classes, self.dimensions, self.num_channels),
            )

    def compute_support(
        self,
        support: torch.Tensor,
        cache: bool = False,
    ) -> torch.Tensor:
        """Compute and return the class representations based on the given
        support set.

        :param support: The set of support images [n,k]
        :param cache: Set to true to save the support representation.  This will
            make the model perform fewshot classification during the forward
            pass using the currently specified classes even if no support images
            are provided.
        """
        # Do few-shot prediction
        nway = support.shape[0]

        # [nway, k*r, d]
        support = (
            self.model(support.flatten(0, 1))
            .permute(0, 3, 1, 2)
            .reshape(nway, -1, self.dimensions)
            * self.scale_factor
        )

        if cache:
            self.cached_support = support

        return support

    def forward(
        self,
        query: torch.Tensor,
        support: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict labels using FRN.

        This function offerst two different modes: standard prediction and
        few-shot predictions.  Standard prediction acts standard image
        classification with logits of n-classes.  This mode is enabled by
        default and is meant to be used during the pre-training phase outlined
        in the original paper.

        Few-shot prediction is used when support-images are given as arguments
        or compute_support has been called with cache set to True.  Support
        images are a set of n x k images (n classes, with k examples in each)
        and is used to compute the class representation matrices used for
        prediction.

        :param query: Set of images such that shape=(bsz x h x w x channels)
        :param support: Set of images such that shape=(n x k x h x w x channels)

        :raises ValueError: If no support images are provided and the model
            neither have a cached set of class representations nor have called
            init_pretraining

        :returns: Prediction for each query image.  Logit dimension depends on
                 prediction mode-used.
        """

        # Compute and flatten the input features
        # [bsz, r, d]
        bsz = query.shape[0]
        query = self.model(query).permute(0, 3, 1, 2).flatten(1, 2) * self.scale_factor

        if support is not None or self.cached_support is not None:
            if support is not None:
                support = self.compute_support(support)
            else:
                support = self.cached_support

            assert support is not None
            nway = support.shape[0]

            r = torch.exp(self.beta)
            lam = self.num_channels / (nway * self.dimensions) * torch.exp(self.alpha)
            recons = self._reconstruct(query, support, r, lam)
            aux_loss = self._aux_loss(
                recons.reshape(bsz, nway, self.dimensions, self.num_channels),
            )
            logits = (self._predictions(recons, query), aux_loss)
        else:
            # Standard predictions
            if self.class_matrices is None:
                raise ValueError(
                    "Class matrices were not initialized, please run init_pretraining before calling method without support data",
                )
            recons = self._reconstruct(query, self.class_matrices, 1.0, 1.0)
            logits = self._predictions(recons, query)

        return logits

    def _reconstruct(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        r: Union[torch.Tensor, float],
        lam: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Compute feature reconstructions of the query images


        :param query: [bsz, r, d]
        :param support: [way, support_shot* r, d]
        :param r: rho
        :param lam: lambda
        """
        # this code snippet is copied more or less straight from the paper's appendix

        # Flatten everything
        query = query.flatten(0, 2)
        support = support.flatten(1, 2)

        reg = support.shape[1] / support.shape[2]
        st = support.permute(0, 2, 1)
        xtx = st @ support
        m_inv = (xtx + torch.eye(xtx.shape[-1]).unsqueeze(0) * (reg * lam)).inverse()
        hat = m_inv @ xtx

        # Reshape to [bsz, nway, r*d]
        return (query @ hat) * r

    def _predictions(self, recons, original):
        """Computes the (normalized) logits from the reconstruction and original features

        :param recons: The reconstructed features of the query set
        :param original: The original input images
        """

        n = recons.shape[0]
        original = original.unsqueeze(1)
        dists = cdist(recons, original.repeat((1, n, 1)), 2)
        dists *= -self.temperature
        return F.softmax(
            dists,
        )

    def _aux_loss(self, recons):
        """Auxiluary loss

        :param recons: [bsz, nway, dimension, channels]
        """

        def _loss(recons):
            recons_t = recons.permute(0, 2, 1)
            recons = recons.unsqueeze(1)
            res = recons @ recons_t
            nway = res.shape[0]
            mask = (~torch.eye(nway, dtype=bool)).reshape((nway, nway, 1, 1))

            res *= mask
            loss = LA.norm(res.flatten(0, 2), 2).sum()
            return loss

        loss = torch.stack([_loss(r) for r in recons]).mean()
        return loss

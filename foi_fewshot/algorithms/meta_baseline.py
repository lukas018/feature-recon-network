import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import starmap

from typing import Callable, Optional, Tuple


class MetaBaseline(nn.Module):
    """Implementation of ~A New  Meta-Maseline~ : https://arxiv.org/pdf/2003.04390.pdf

    A metric based fewshot learner which uses pre-training
    """

    def __init__(self, model, temperature=1, dist_fn: Callable = F.cosine_similarity):

        super().__init__()
        self.model = model
        self.temperature = torch.tensor(temperature)
        self.dist_fn = dist_fn
        self.class_matrix = None
        self.cached_centroids = None

    def init_pretraining(self, dimensions: int, num_classes: int):
        """Initialize the FRN for pre-training

        :param num_classes: The number of classes in the pretraining dataset
        """

        if self.class_matrix is None or self.class_matrix.shape[1] == num_classes:
            self.class_matrix = nn.Linear(dimensions, num_classes)

    def compute_centroids(self, support, cache=False):
        """Computes the centroids of the given support imgages


        :param support: Tensor of size [n, k, h, w, c]
            It is assumed that the images is grouped with regards to class the classes
        :param cache: Set to true to cache the centriods to use for later fewshot learning

        """

        nways = support.shape[0]
        kshots = support.shape[1]
        support_features = self.model(support.flatten(0, 1)).reshape((nways, kshots, -1))
        centroids = support_features.mean(axis=1)

        if cache:
            self.cached_centroids = centroids

        return centroids

    def forward(
            self, query: torch.Tensor, support: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict labels using FRN.

        This function offerst two different modes: standard prediction and few-shot predictions.
        Standard prediction acts standard image classification with logits of n-classes.
        This mode is enabled by default and is meant to be used during the pre-training phase outlined in the original paper.

        Few-shot prediction is used when support-images are given as arguments.
        Support images are a set of n x k images (n classes, with k examples in each)
        and is used to compute the class representation matrices used for prediction.

        :param query: Set of images such that shape=(meta_bsz, bsz, channels, h, w)
        :param support: Set of images such that shape=(bsz, channels, k, h,  w)

        :return: Prediction for each imput image. Logit dimension depends on mode-used.
        """

        bsz = query.shape[0]

        # flatten the input to bsz x dim_f
        features = self.model(query)
        if support is not None or self.cached_centroids is not None:

            if support is not None:
                centroids = self.compute_centroids(support)

            elif self.cached_centroids is not None:
                centroids = self.cached_centroids

            nways = centroids.shape[0]
            centroids = centroids.unsqueeze(0).repeat((bsz, 1, 1))
            features = features.unsqueeze(1).repeat((1, nways, 1))

            logits = self.dist_fn(features, centroids, dim=2)
            logits = F.softmax(self.temperature * logits, dim=1)

        else:

            # Do a normal forward pass
            features = self.model(query)
            if self.class_matrix is None:
                raise ValueError(
                    f"Final classification layer was not initialized, please run init_pretraining before calling"
                )

            logits = self.class_matrix(features)
            logits = F.softmax(logits, dim=1)

        return (logits,)

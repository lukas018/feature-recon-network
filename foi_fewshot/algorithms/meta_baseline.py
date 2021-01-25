from typing import Callable
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaBaseline(nn.Module):
    """Implementation of ~A New Meta-Maseline~ :
    https://arxiv.org/pdf/2003.04390.pdf

    A metric based fewshot learner which uses pre-training via standard image
    classification over the base classes to initialize feature extractor.  It
    then employs a second meta-learning phase, in which the model trains on
    various fewshot-learning tasks.
    """

    def __init__(self, model, temperature=1, dist_fn: Callable = F.cosine_similarity):
        """
        :param model: Feature extractor to wrap, e.g. a ResNet12 without a
            classification head.
        :param temperature: Initial temperature value for the softmax scaling
            function
        :param dist_fn: Distance function used to compute the distance
            between samples and class centriods.
        """

        super().__init__()
        self.model = model
        self.temperature = torch.tensor(temperature)
        self.dist_fn = dist_fn
        self.class_matrix: Optional[nn.Linear] = None
        self.cached_centroids: Optional[torch.Tensor] = None

    def init_pretraining(self, dimensions: int, num_classes: int):
        """Initialize the new meta-baseline network for pre-training

        The new meta-baseline network uses a standard linear layer (with bias)
        to perform predictions during the pretraining phase.  This method
        initializes that layer.

        :param dimensions: The expected output size of the model (specified in
            the constructor) which this model wraps.  This is needed to create
            the linear layer.
        :param num_classes: The number of classes in the pretraining dataset
        """

        if self.class_matrix is None or self.class_matrix.weight.shape[1] == num_classes:
            self.class_matrix = nn.Linear(dimensions, num_classes)

    def compute_centroids(self, support, cache=False):
        """Computes (and saves) the class centroids of the support images

        :param support: Tensor of size [n, k, h, w, c] It is assumed that the
            images is grouped with regards to class the classes
        :param cache: Set to true to cache/save the centriods to use for later
            fewshot clasification.  This is useful if one wants to save a model
            trained on a particular fewshot learning task.
        """

        nways = support.shape[0]
        kshots = support.shape[1]
        support_features = self.model(support.flatten(0, 1)).reshape(
            (nways, kshots, -1),
        )
        centroids = support_features.mean(axis=1)

        if cache:
            self.cached_centroids = centroids

        return centroids

    def forward(
        self, query: torch.Tensor, support: Optional[torch.Tensor] = None, **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a forward pass as specified by the given query- and support set.

        Two different forward-pass modes are available: ~standard
        classification~ and ~few-shot classification~.  Standard classification
        is enabled by first running the init_pretraining method, where the
        number of classes ~n~ is specified, and then by only providing the query
        images.  The query images are then classified task over ~n~ classes.

        Fewshot prediction is instead used when support set are specified or
        when compute_centriods have been called with cache=True.  The support
        set should consist of n x k images (n classes, with k examples in each).
        The query images are then classified over the ~n~ way task specified by
        the support set.

        :param query: Set of images such that shape=[meta_bsz,bsz,channels,h,w]
        :param support: Set of images such that shape=[bsz,channels, k, h, w]

        :return: Prediction for each imput image.  Logit dimension depends on
                 mode-used.
        """

        bsz = query.shape[0]

        # flatten the input to bsz x dim_f
        features = self.model(query)
        outputs = {}
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
            outputs['logits'] = logits

        else:

            # Do a normal forward pass
            features = self.model(query)
            if self.class_matrix is None:
                raise ValueError(
                    'Final classification layer was not initialized, please run init_pretraining before calling',
                )

            logits = self.class_matrix(features)
            outputs['logits'] = logits

        return outputs

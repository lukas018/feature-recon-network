import torch
import torch.nn as nn
import torch.nn.functional as F
from .frn import FeatureReconNetwork



class MetaBaseline():
    """Implementation of ~A New  Meta-Maseline~ : https://arxiv.org/pdf/2003.04390.pdf

    A metric based fewshot learner which uses pre-training
    """

    def __init__(self,
                 model,
                 temperature=1,
                 dist_fn: Callable = F.cosine_distance):

        self.model = model
        self.temperature = torch.tensor(temperature)
        self.dist_fn = dist_fn

    def init_pretraining(self, dimensions: int, num_classes: int):
        """Initialize the FRN for pre-training

        :param num_classes: The number of classes in the pretraining dataset
        """

        if self.class_matrix is not None or self.class_matrix.shape[1] == num_classes:
            self.class_matrix = nn.Linear(dimensions, num_classes)


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

        bsz = query.shape[0]

        # flatten the input to bsz x dim_f
        features = self.model(query).flatten(1, 3)

        if support is not None:
            # Do few-shot prediction
            nway = support.shape[0]
            k = support.shape[1]
            img = support.shape[2:]

            features = features.unflatten(0).repeat((nway, 1, 1))

            support_features = self.model(support.flatten(0, 1)).reshape((nway, k, -1))
            centroids = support_features.mean(axis=1)
            centroids = centroids.unflatten(1).repeat((1, bsz, 1))

            logits = self.dist_fn(features, support_features)
            logits = F.softmax(self.temperature*logits)
            logits = (logits, 0)

        else:
            # Do a normal forward pass
            features = self.model(query)
            logits = self.class_matrix(features)
            logits = F.softmax(logits)

            return logits

import torch

import torch.nn as nn
import torch.nn.functional as F


class MetaBaseLine(nn.Module):
    """Implementation of the paper *A new meta-baseline*

    source: https://arxiv.org/pdf/2012.01506.pdf
    """

    def __init__(self,
                 model: nn.Module,
                 temperature:float=1.0,
                 distance_function=F.cosine_simularity):
        """Initialize the

        :param model: Output model
        :param temperature: initial value for the start parameter
        :param distasnce_function
        """

        self.model = model
        self.temperature = torch.float(temperature)
        self.class_matrix = None
        self.distance_function = distance_function

    def init_pretraining(self,
                         input_size: int,
                         num_classes: int):
        """Initialize the parameters for pretraining

        :param input_size: The number of features outputed by model
        :param num_classes: The number classes to train with
        """

        self.class_matrices = nn.Linear(input_size, num_classes)


    def forward(self, query, support=None):
        """Perform a forward pass

        If the support images are provided, the meta-baseline classification
        is used to predict labels for query. Otherwise, standard classification is used

        :param query: [bsz, h, w, c]
        :param support: [n, k, h, w, c]
        """

        features = self.model(query)

        if support is None:
            n, k = support.shape[0], support.shape[1]
            support_features = self.model(support.flatten((0, 1))).reshape((n, k, -1))

            # [bsz, r] x [n, r]
            support_features = support_features.mean(axis=1).flatten((1))
            features = features.unflatten(1).repeat((1, n, 1)).flatten((0, 1))
            support_features = support_features.unflatten(0).repeat((len(query), 1, 1)).flatten(0, 1)

            sim_score = F.cosine_similarity(features, support, dim=2).reshape((len(query), -1))
            logits = self.temperature*sim_score
        else:
            if self.class_matrix is not None:
                logits = self.class_matrix(features)
            else:
                raise ValueError("Class matrix not initialized, please call init_pretraining first")

        logits = F.softmax(logits)
        return logits

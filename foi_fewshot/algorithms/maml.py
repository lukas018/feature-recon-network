import numpy as np
import torch
import torch.nn.functional as F
from learn2learn.algorithms import l2l_maml


class MAML:
    """Wrapper class for MAML"""

    def __init__(self, model, fast_lr, update_steps=5, loss=F.cross_entropy):

        self.model = l2l_maml(model, fast_lr)
        self.update_steps = update_steps
        self.loss = loss

    def forward(self, query_data, support_images=None, overwrite=False):
        maml = self.model

        if support_images is not None:
            maml = self.model.clone()
            n = support_images.shape[0]
            k = support_images.shape[1]
            labels = torch.tensor(np.arange(n)).unflatten(1).repeat((1, k))

            for _ in range(self.update_steps):
                logits = F.softmax(maml(support_images.flatten(0, 1)))
                error = self.loss(logits, labels)
                maml.adapt(error)

        if overwrite:
            self.model = maml

        logits = maml(support_images)
        logits = F.softmax(logits)
        return logits


class ANIL:
    """Wrapper class for ANIL:"""

    def __init__(
        self, feature_extractor, head, fast_lr, update_steps=5, loss=F.cross_entropy,
    ):
        self.feature_extractor = feature_extractor
        self.head = l2l_maml(head, fast_lr)
        self.update_steps = update_steps
        self.loss = loss

    def forward(self, query_data, support_images=None, overwrite=False):
        head = self.head

        if support_images is not None:
            head = self.head.clone()

            n = support_images.shape[0]
            k = support_images.shape[1]
            labels = torch.tensor(np.arange(n)).unflatten(1).repeat((1, k))
            support_features = self.feature_extractor(support_images.flatten(0, 1))

            for _ in range(self.update_steps):
                logits = F.softmax(head(support_features))
                error = self.loss(logits, labels)
                head.adapt(error)

        if overwrite:
            self.head = head

        query_features = self.feature_extractor(query_data)
        logits = F.softmax(head(query_features))
        return logits

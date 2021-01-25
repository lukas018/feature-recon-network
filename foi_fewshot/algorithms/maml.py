import numpy as np
import torch
import torch.nn.functional as F
from learn2learn.algorithms import MAML as l2l_maml
from torch import nn


class MAML(nn.Module):
    """Wrapper class for MAML"""

    def __init__(self, model, fast_lr, update_steps=5, loss_fn=F.cross_entropy):
        """
        :param model: Model to wrap. A torch module with a fixed number of output nodes.
        :param fast_lr: The learning rate to use during the inner MAML update steps
        :param update_steps: The number of update steps to perform during the inner step
        :param loss_fn: The loss function to used to update during the inner step
        """

        self.model = l2l_maml(model, fast_lr)
        self.update_steps = update_steps
        self.loss_fn = loss_fn


    def adapt(self, support, cache=False):
        """Adapt the model

        :param support: Set of support images with shape [n,k]
        :param cache: If true the updated model will overwrite the base model.
            Useful if one wants to save a model trained on a specific fewshot
            task

        :returns: Updated maml model
        """


        maml = self.model.clone()

        n = support.shape[0]
        k = support.shape[1]
        labels = torch.tensor(np.arange(n)).unflatten(1).repeat((1, k))
        for _ in range(self.update_steps):
            logits = F.softmax(maml(support.flatten(0, 1)))
            error = self.loss_fn(logits, labels)
            maml.adapt(error)

        if cache:
            self.model = maml

        return maml

    def forward(self, query, support=None):
        """Perform a single forward step

        :param query:
        :param support:
        """

        maml = self.model

        if support is not None:
            maml = self.adapt(support)

        logits = maml(query)
        logits = F.softmax(logits)
        return logits


class ANIL(nn.Module):
    """Wrapper class for ANIL:"""

    def __init__(
        self, feature_extractor, head, fast_lr, update_steps=5, loss_fn=F.cross_entropy,
    ):

        """
        :param feature_extractor: Feature extractor, e.g. a CNN without a final classification head
        :param head: Classification head that performs classification
        :param fast_lr: The learning rate to use during the inner MAML update steps
        :param update_steps: The number of update steps to perform during the inner step
        :param loss_fn: The loss function to used to update during the inner step
        """
        self.feature_extractor = feature_extractor
        self.head = l2l_maml(head, fast_lr)
        self.update_steps = update_steps
        self.loss_fn = loss_fn


    def adapt(self, support, cache=True):

        head = self.head.clone()

        n = support.shape[0]
        k = support.shape[1]
        labels = torch.tensor(np.arange(n)).unflatten(1).repeat((1, k))
        support_features = self.feature_extractor(support.flatten(0, 1))

        for _ in range(self.update_steps):
            logits = F.softmax(head(support_features))
            error = self.loss_fn(logits, labels)
            head.adapt(error)

        if cache:
            self.head = head

        return head


    def forward(self, query, support=None, overwrite=False):
        head = self.head

        if support is not None:
            head = self.adapt(support)

        query_features = self.feature_extractor(query)
        logits = F.softmax(head(query_features))
        return logits

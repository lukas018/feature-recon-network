from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from learn2learn.algorithms import MAML

from foi_fewshot.trainers.fewshot_utils import maml_episode, reptile_episode


class MAMLTrainer(FewshotTrainer):
    def update(self, loss):
        loss.backward()
        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / self.args.batch_size).add_(p.data)
        self.optimizer.step()


class ReptileTrainer(FewshotTrainer):
    """Trainer for the Reptile algorithmk

    Reptile is a first-order variant of the MAML algorithm descrbied in the paper:
    https://arxiv.org/pdf/1803.02999.pdf

    The input model here can by any classification network, no wrapper class is required
    """

    def __init__(self):
        self.adapt_opt_state = self.optimizer.state_dict()
        self.optimizer_fn = None

    def fewshot_episode(self, batch, training=False, *args, **kwargs):
        model = deepcopy(self.model)

        opt = self.optimizer_fn(self)
        opt.load_state_dict(self.optimizer.state_dict())
        reptile = reptile_episode(
            model, batch, optimizer=opt, update=self.args.update_steps, *args, **kwargs
        )

        # Construct the global gradient from the difference between the two models
        if training:
            self.adapt_opt_state = opt.state_dict()
            for p, l in zip(self.model.parameters(), model.parameters()):
                p.grad.data.add_(-1.0, l.data)

    def update(self, loss):
        # Here we dont' need to backpropogate the loss
        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / self.args.batch_size).add_(p.data)
        self.optimizer.step()

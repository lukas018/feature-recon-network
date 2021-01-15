from dataclasses_json import dataclass_json
from dataclasses import dataclass
import torch
import torch.nn as nn

from foi_fewshot.utils import fewshot_episode


@dataclass_json
@dataclass
class TrainingState:
    """Stateful representation of the training process"""

    epoch: int = 0
    global_step: int = 0
    current_step: int = 0


class MetabatchWrapper(nn.Module):
    """Simple Wrapper to allow for processing multiple
    tasks at the same time
    """

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, meta_batch, args=None):
        args = args if args is not None else self.args
        results = [
            fewshot_episode(
                self.model, task_batch, args.kquery, None, args.metric_fn, args.loss_fn
            )
            for task_batch in meta_batch
        ]
        loss, res = *zip(*results),
        return torch.stack(loss), list(res)

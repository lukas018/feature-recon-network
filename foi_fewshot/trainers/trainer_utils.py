from typing import Optional, List, Dict

from dataclasses_json import dataclass_json
from dataclasses import dataclass
import torch
import torch.nn as nn

from .trainer_arguments import FewshotArguments


def create_dataloader(dataset, args):
    """
    :param dataset: Dataset from which to sample
    :param arguments: TrainingArguments
    """

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return dl


def create_taskloader(dataset, args):
    """Create a Taskloader specified by the TrainingArguments

    :param dataset:
    :param arguments: FewshotArguments
    """

    dl = initialize_taskloader(
        dataset,
        args.nways,
        args.kshots,
        args.kquery,
        args.epoch_steps * args.gradient_accumulation_steps,
        args.num_workers,
    )
    return dl


class EvalTaskGenerator:
    """Iterator object for creating dataloaders and iterators"""

    def __init__(self, args=None):
        self.args = args
        self.entries = []

    def add(self, prefix, dataset, task_args=None):
        task_args = task_args if task_args is not None else self.args
        entry = (prefix, dataset, task_args)
        self.entries.append(entry)

    def __next__(self):
        for entry in self.entries:
            prefix, ds, ta = entry
            if isinstance(ta, FewshotArguments):
                yield prefix, create_taskloader(ds, ta)
            else:
                yield prefix, create_dataloader(ds, ta)

    def __iter__(self):
        iter(self)


@dataclass_json
@dataclass
class TrainerState:
    """Stateful representation of the training process"""

    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_hyper_param_search: bool = False

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []


@dataclass_json
@dataclass
class TrainerControl:
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """ Internal method that resets the variable for a new training. """
        self.should_training_stop = False

    def _new_epoch(self):
        """ Internal method that resets the variable for a new epoch. """
        self.should_epoch_stop = False

    def _new_step(self):
        """ Internal method that resets the variable for a new step. """
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class MetabatchWrapper(nn.Module):
    """Simple Wrapper to allow for processing multiple tasks at the same time"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, meta_batch):
        keys = list(meta_batch.keys())
        records = [{k: meta_batch[k][i]} for i in range(len(meta_batch[keys[0]]))]
        outputs = [model(**record) for record in records]
        return default_collate(outputs)

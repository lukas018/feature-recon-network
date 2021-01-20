from typing import Optional, List, Dict

from dataclasses_json import dataclass_json
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data._utils import collate

from .trainer_arguments import FewshotArguments
from ..data import initialize_taskloader


def _custom_collate(batches):
    """Simple collate function that wraps standard image, labels pairs
    to a dict: query -> images,  query_labels -> labels
    """
    records = [{"query": images, "query_labels": labels} for images, labels in batches]
    return collate.default_collate(records)


def create_dataloader(dataset, args):
    """Create a dataloader from a given dataset

    The dataloader will output data in the form of dicts with two labels "query" and "query_labels"
    This is to make it compatable with most of the fewshot-wrappers

    :param dataset: Dataset from which to sample
    :param arguments: TrainingArguments
    """

    batch_size = args.batch_size if args is not None else 1
    num_workers = args.num_workers if args is not None else 1

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
        shuffle=True
    )
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
        args.batch_size,
    )
    return dl


class EvalTaskGenerator:
    """Iterator object for creating dataloaders for both fewshot-tasks and standard learning"""

    def __init__(self, args=None):
        self.args = args
        self.entries = []

    def add(self, prefix, dataset, task_args=None):
        """Adds a new dataset to the generator

        :param prefix: Unique identifier to associate with the dataset
        :param dataset: Dataset to generate dataloader from
        :param task_args: An instance of TrainingArguments or derived class.
           Used to determine what kind of dataloader to create.
           If instance of FewshotArgument the dataloader will output fewshot-learning tasks.
           If instance of TrainingArguments it will generate standard classification batches.
           If task_args is None, self.args is used as default.
        """

        task_args = task_args if task_args is not None else self.args
        entry = (prefix, dataset, task_args)
        self.entries.append(entry)

    def _generator(self):
        for entry in self.entries:
            prefix, ds, ta = entry
            if isinstance(ta, FewshotArguments):
                yield prefix, create_taskloader(ds, ta)
            else:
                yield prefix, create_dataloader(ds, ta)

    def __iter__(self):
        return self._generator()


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


def multi_size_collate(batches):
    """Collate function to join tensors if the tensors with the same key have different shapes
    """

    keys = batches[0].keys()
    records = {k: [batch[k] for batch in batches] for k in keys}
    return records


class MetabatchWrapper(nn.Module):
    """Simple Wrapper to process Meta-batches, i.e. a collection of tasks"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        keys = list(kwargs.keys())
        records = [{k: kwargs[k][i] for k in keys} for i in range(len(kwargs[keys[0]]))]
        outputs = [self.model(**record) for record in records]

        # If the input was tensors, i.e. everything is of similar size
        # then return the result as tensors
        if isinstance(kwargs[keys[0]], torch.Tensor):
            return collate.default_collate(outputs)
        else:
            return multi_size_collate(outputs)

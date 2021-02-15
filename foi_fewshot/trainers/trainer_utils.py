from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from dataclasses_json import dataclass_json
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from torch.utils.data.distributed import DistributedSampler

from ..data import initialize_taskloader, initialize_taskdataset
from .trainer_arguments import FewshotArguments


def _custom_collate(batches):
    """Simple collate function that wraps standard image, labels pairs
    to a dict: query -> images,  query_labels -> labels
    """
    records = [{"query": images, "query_labels": labels} for images, labels in batches]
    return collate.default_collate(records)


def create_dataloader(dataset, args, epoch=0):
    """Create a dataloader from a given dataset

    The dataloader will output data in the form of dicts with two labels "query" and "query_labels"
    This is to make it compatable with most of the fewshot-wrappers

    :param dataset: Dataset from which to sample
    :param arguments: TrainingArguments
    """

    batch_size = args.batch_size if args is not None else 1
    num_workers = args.num_workers if args is not None else 1

    if args.distributed:
        sampler = DistributedSampler(dataset, seed=args.seed)
        sampler.set_epoch(epoch)
    else:
        sampler = None

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_custom_collate,
        shuffle=True,
        sampler=sampler,
    )
    return dl


def create_taskloader(dataset, args, epoch=0):
    """Create a Taskloader specified by the TrainingArguments

    :param dataset:
    :param arguments: FewshotArguments
    :param distributed
    :param seed:
    """

    task_ds = initialize_taskdataset(
        dataset,
        args.nways,
        args.kshots,
        args.kquery,
        args.epoch_steps * args.gradient_accumulation_steps * args.batch_size,
    )

    dl = initialize_taskloader(
        task_ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        deterministic=args.deterministic,
        distributed=args.distributed,
        seed=args.seed,
        epoch=epoch
    )
    return dl


class EvalTaskGenerator:
    """Iterator object for creating dataloaders for both fewshot-tasks and standard learning"""

    def __init__(self, args=None, seed=42):
        self.args = args
        self.entries = []
        self.seed = seed

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
                yield prefix, create_taskloader(ds, ta, seed=self.seed)
            else:
                yield prefix, create_dataloader(ds, ta, seed=self.seed)

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
    num_update_steps_per_epoch: int = 0
    log_history: Optional[List[Dict[str, float]]] = None
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
    """Collate function to join tensors if the tensors with the same key have different shapes"""

    keys = batches[0].keys()
    records = {k: [batch[k] for batch in batches] for k in keys}
    return records


class MetabatchWrapper(nn.Module):
    """
    Simple Wrapper to process meta-batches, i.e. collections of tasks/batches
    rather than a single batch
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        """Performs forward step over a set of tasks"""

        keys = list(kwargs.keys())
        records = [{k: kwargs[k][i] for k in keys} for i in range(len(kwargs[keys[0]]))]
        outputs = [self.model(**record) for record in records]

        # If the input was tensors, i.e. everything is of similar size
        # then return the result as tensors
        if isinstance(kwargs[keys[0]], torch.Tensor):
            return collate.default_collate(outputs)
        else:
            return multi_size_collate(outputs)


def _generic_eval_taskgen(
        datasets: List[Tuple[str, torch.utils.data.Dataset]],
        task_gen: Optional[EvalTaskGenerator]=None,
        n_samples: int=25,
        batch_size: int=4,
        nways: int =5,
        kquery:int =15,
        kshots:int =(5, 1),
):

    if task_gen is None:
        task_gen = EvalTaskGenerator()

    nways = nways if hasattr(nways, '__len__') else (nways,)
    kshots = kshots if hasattr(kshots, '__len__') else (kshots,)

    args = FewshotArguments(
        nways=1,
        ksupport=1,
        kquery=kquery,
        batch_size=batch_size,
        epoch_steps=n_samples,
    )

    def _adjust_arguments(args, n, k):
        args = asdict(args)
        args.update({"nways": n, "ksupport": k})
        args = FewshotArguments(**args)
        return args

    for prefix, ds in datasets:
        for n in nways:
            for k in kshots:
                task_gen.add(f"{prefix}:n={n}k={k}", ds, _adjust_arguments(args, n, k))

    return task_gen


def create_eval_taskgen(
    ds_base=None,
    ds_novel=None,
    n_samples=25,
    batch_size=4,
    nways=5,
    kquery=15,
    kshots = (5, 1),
    classification_task=False,
):
    """Helper function for creating a common evalution task generator

    This task generator will create five different tasks.
    Two nway=5, kshot=5, and two nway=5, kshot=1 for each datset and
    a classification task on the base dataset

    :param ds_base: Datset of baseclasses used to train the model
    :param ds_novel: Dataset of novel (unseen) classes
    :param n_samples: Number of meta-batches to evaluate on
    :param batch_size: Number of tasks in each meta-batch
    :param nways: Int or tuple of ints, the number of nway classification task to generate
    :param kquery: The number of evaluation samples in each task
    :param kshot: Int or tuple of ints, the number of kshot to use
    :param classification_task: Add a standard classification task using the ds_base
    :returns: EvalTaskGenerator that can be used for evlauation
    """

    datasets = []
    if ds_base is not None:
        datasets.append(('base', ds_base))

    if ds_novel is not None:
        datasets.append(('novel', ds_novel))

    if len(datasets) == 0:
        raise ValueError("At least one dataset of either ds_base and ds_novel needs to be provided")

    eval_taskgen = _generic_eval_taskgen(datasets, None, n_samples, batch_size, nways, kquery, kshots)

    if ds_base is not None and classification_task:
        eval_taskgen.add("base-class", ds_base, None)

    return eval_taskgen


def create_test_taskgen(
    ds_test,
    n_samples=800,
    batch_size=4,
    nways=5,
    kquery=15,
    kshots = (5, 1),
):

    """Helper function for creating a common testing task generator

    This task generator will create two different tasks.
    One nway=5, kshot=5, and one nway=5, kshot=1.

    :param ds_test: Dataset of novel (unseen) classes
    :param batch_size: Number of tasks in each meta-batch
    :param n_samples: Number of meta-batches to evaluate on
    :param nways: Int or tuple of ints, the number of nway classification task to generate
    :param kquery: The number of evaluation samples in each task
    :param kshot: Int or tuple of ints, the number of kshot to use

    :returns: EvalTaskGenerator that can be used for evlauation
    """

    return _generic_eval_taskgen([('test', ds_test)], None, n_samples, batch_size, nways, kquery, kshots)

from typing import Optional, Tuple, Union, Callable, List
from dataclasses import dataclass
from dataclasses import field
import torch
import torch.nn.functional as F
from enum import Enum


class EvaluationStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

class SchedulerUpdateStrategy(Enum):
    NP= "no"
    STEPS = "steps"
    EPOCH = "epoch"

@dataclass
class TrainingArguments:
    modeldir: str = field(
        default=None, metadata={"help": "Path to checkpoint models to"}
    )
    logdir: str = field(default=None, metadata={"help": "Path to write logs to"})

    batch_size: int = field(
        default=1, metadata={"help": "Batch size used during training"}
    )

    device_ids: List[int] = field(
        default=None, metadata={"help": "GPU devices ids to use during training"}
    )
    n_gpus: Optional[int] = field(
        default=None, metadata={"help": "Number of gpus to use"}
    )
    evaluation_strategy: EvaluationStrategy = field(
        default=EvaluationStrategy.NO, metadata={"help": "How to perform evaulation"}
    )
    eval_steps: int = field(
        default=100,
        metadata={
            "help": "Perform evaluation every n steps for EvalationStrategy.STEPS"
        },
    )
    scheduler_update_strategy: SchedulerUpdateStrategy  = field(
        default=SchedulerUpdateStrategy.STEPS,
        metadata={
            "help": "How the update scheduler should be updated"
        }
    )
    learning_rate: float = field(
        default=0.1, metadata={"help": "Initial learning rate"}
    )
    momentum: float = field(
        default=0.9, metadata={"help": "Momentum for optimizer"}
    )
    load_best_model_at_end: bool = field(default=True)
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=1)
    num_workers: int = field(
        default=8, metadata={"help": "Number of parallel workers for loading data"}
    )
    num_epochs: int = field(default=100, metadata={"help": "Maximum number of epochs"})

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of steps to perform between updates"}
    )
    seed: int = field(default=42)

    metric_for_best_model: str = field(
        default=None, metadata={"help": "Metric used to determine which model is best"}
    )
    greater_is_better: bool = field(
        default=False, metadata={"help": "If a greater metric is better"}
    )
    modeldir_prefix: Optional[str] = field(
        default=None, metadata={"help": "Checkpoint prefix"}
    )

    # TODO(Lukas) Create a better option for this
    distributed: bool = field(
        default=None,
        metadata={"help": "Whether to use DataParallel or DistributedDataParallel"},
    )

    def post_init(self):
        if self.n_gpus is None:
            self.n_gpus = len(self.device_ids)


@dataclass
class PretrainArguments(TrainingArguments):
    scheduler_update_stretagy: SchedulerUpdateStrategy = field(
        default=SchedulerUpdateStrategy.EPOCH,
        metadata={
            "help": "How/when the learning rate scheduler should be updated"
        }
    )


@dataclass
class FewshotArguments(TrainingArguments):
    nways: Union[int, Tuple[int, int]] = field(
        default=5, metadata={"help": "The number of classes in each task."}
    )
    ksupport: Union[int, Tuple[int, int]] = field(
        default=10, metadata={"help": "The number of samples per class"}
    )
    kquery: int = field(default=15, metadata={"help": "The number of data points"})

    epoch_steps: int = field(
        default=100, metadata={"Help": "The number of steps in each epoch"}
    )

    @property
    def kshots(self):
        if isinstance(self.ksupport, tuple):
            return self.ksupport[0] + self.kquery, self.ksupport[1] + self.kquery
        else:
            return self.ksupport + self.kquery

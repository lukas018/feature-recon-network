from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class EvaluationStrategy(Enum):
    """Enum over the various Evaluatin strategems"""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SchedulerUpdateStrategy(Enum):
    """Enum over the various scheduler updates strategems"""

    NP = "no"
    STEPS = "steps"
    EPOCH = "epoch"


@dataclass
class TrainingArguments:
    """Generic arguments for Trainer wrappers"""

    modeldir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint models to"},
    )
    logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to write logs to",
        },
    )

    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size used during training"},
    )

    multi_gpu: bool = field(
        default=False,
        metadata={"help": "If more than one GPU is available use all of them"},
    )
    device_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "GPU devices ids to use during training."},
    )
    n_gpus: Optional[int] = field(
        default=None,
        metadata={"help": "Number of gpus to use"},
    )
    evaluation_strategy: EvaluationStrategy = field(
        default=EvaluationStrategy.NO,
        metadata={"help": "How to perform evaulation"},
    )
    eval_steps: int = field(
        default=100,
        metadata={
            "help": "Perform evaluation every n steps for EvalationStrategy.STEPS",
        },
    )
    save_steps: int = field(
        default=1000,
        metadata={
            "help": "How often to save the model",
        },
    )
    scheduler_update_strategy: SchedulerUpdateStrategy = field(
        default=SchedulerUpdateStrategy.STEPS,
        metadata={
            "help": "How the update scheduler should be updated",
        },
    )
    learning_rate: float = field(
        default=0.1,
        metadata={"help": "Initial learning rate"},
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for optimizer"},
    )
    load_best_model_at_end: bool = field(default=True)
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=1)
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of parallel workers for loading data"},
    )
    num_epochs: int = field(default=100, metadata={"help": "Maximum number of epochs"})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to perform between updates"},
    )
    seed: int = field(default=42)

    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "Metric used to determine which model is best"},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "If a greater metric is better"},
    )

    modeldir_prefix: str = field(
        default="training",
        metadata={"help": "prefix to add to checkpoint-files"},
    )

    # TODO(Lukas) Create a better option for this
    distributed: bool = field(
        default=False,
        metadata={"help": "Whether to use DataParallel or DistributedDataParallel"},
    )

    def post_init(self):
        if self.n_gpus is None:
            self.n_gpus = len(self.device_ids)


@dataclass
class PretrainArguments(TrainingArguments):
    """Trainer arguments for Pretrainer"""

    scheduler_update_stretagy: SchedulerUpdateStrategy = field(
        default=SchedulerUpdateStrategy.EPOCH,
        metadata={
            "help": "How/when the learning rate scheduler should be updated",
        },
    )


@dataclass
class FewshotArguments(TrainingArguments):
    """Trainer arguments for fewshot trainer"""

    scheduler_update_stretagy: SchedulerUpdateStrategy = field(
        default=SchedulerUpdateStrategy.STEPS,
        metadata={
            "help": "How/when the learning rate scheduler should be updated",
        },
    )

    nways: Union[int, Tuple[int, int]] = field(
        default=5,
        metadata={"help": "The number of classes in each task."},
    )
    ksupport: Union[int, Tuple[int, int]] = field(
        default=10,
        metadata={"help": "The number of samples per class"},
    )
    kquery: int = field(default=15, metadata={"help": "The number of data points"})

    epoch_steps: int = field(
        default=100,
        metadata={"Help": "The number of steps in each epoch"},
    )

    @property
    def kshots(self):
        if isinstance(self.ksupport, tuple):
            return self.ksupport[0] + self.kquery, self.ksupport[1] + self.kquery
        else:
            return self.ksupport + self.kquery

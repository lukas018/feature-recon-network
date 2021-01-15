from typing import Optional ,Tuple ,Union ,Callable ,List
from dataclasses import dataclass
from dataclasses import field
import torch
import torch.nn.functional as F

@dataclass
class TrainingArguments:
    modeldir: str = field(metadata={"help": "Path to save model dir"})
    logdir: str = field(metadata={"help": "Path to logging dir"})

    do_eval: bool = field(
        default=True, metadata={"Help": "Evaluate model during training"}
    )
    batch_size: int = field(default=1, metadata={"help": ""})
    device_ids: List[int] = field(default=None, metadata={"help": "GPU devices ids"})
    num_workers: int = field(default=8)
    max_epochs: int = field(default=100, metadata={"help": ""})

    seed: int = field(default=42)
    modeldir_prefix: str = field(default="training-model", metadata={"help": ""})

    optimizer: Optional[object] = field(
        default=None, metadata={"help": "Optimizer to use during training"}
    )
    scheduler: Optional[object] = field(
        default=None, metadata={"help": "Scheduler to use during training"}
    )

    checkpoint_namegen: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "Callabe which takes the current trainer as input and outputs a suitable prefix for the checkpoint model"
        },
    )
    metric_fn: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "Callable which takes logits and labels and outputs a dict of numeral metrics"
        },
    )

    loss_fn: Callable = field(
        default=F.cross_entropy,
        metadata={
            "help": "Loss function"
        }
    )


@dataclass
class PretrainArguments(TrainingArguments):
    pass


@dataclass
class FewshotArguments(TrainingArguments):
    nways: Union[int, Tuple[int, int]] = field(default=5)
    ksupport: Union[int, Tuple[int, int]] = field(default=10)
    kquery: int = field(default=15)

    batch_size: int = field(default=1)
    num_workers: int = field(default=24)
    eval_episodes: int = field(default=100)

    @property
    def kshots(self):
        if not isinstance(self.ksupport, tuple) and not instance(self.kquery, tuple):
            return self.ksupport + kquery

        def tuple_wrap(v):
            if not isinstance(v, tuple):
                v = (v, v)
            return v

        kshots = *map(sum, zip(*map(tuple_wrap, (self.ksupport, kquery)))),
        return kshots

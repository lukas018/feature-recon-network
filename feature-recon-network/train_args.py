from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Tuple
from typing import Union

import torch


@dataclass
class TrainingArguments:

    model_dir: str = field(metadata={"help": "Path to save model dir"})
    log_dir: str = field(metadata={"help": "Path to logging dir"})

    path_to_cls: Optional[str] = field(
        default=None, metadata={"help": "Path to classes json"},
    )
    model: str = field(default="resnet-12", metadata={"help": ""})


    novel_frac: float = field(
        default=0.9,
        metadata={"help": "The fraction of classes to use in novel validaton."},
    )

    base_frac: float = field(
        default=0.9,
        metadata={"help": "The fraction of samples in each class to use in validation"},
    )

    use_cuda: bool = field(default=True)
    seed: int = field(default=42)


@dataclass
class PretrainArguments(TrainingArguments):
    batch_size: int = field(default=32, metadata={"help": "Batch size used during pretraining"})
    num_workers: int = field(default=8)
    max_epochs: int = field(default=100, metadata={"help": ""})

    optimizer: torch.optim = field(default=torch.optim.SGD)
    optimizer_args: dict = field(
        default={
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
        metadata={"help", ""}
    )

    scheduler: torch.optim.lr_scheduler.Scheduler = field(default=torch.optim.lr_scheduler.CosineAnnealingLR)
    scheduler_args: dict = field(
        default={
            "T_max": 1,
            "eta_min": 0.1,
        },
        metadata={"help": ""}
    )


@dataclass
class FewshotArguments(TrainingArguments):
    batch_size: int = field(default=4)
    max_batches: int = field(default=100)

    nways: Union[int, Tuple[int, int]] = field(default=5)
    nsupport: Union[int, Tuple[int, int]] = filed(default=10)
    nquery: int = field(default=15)

    weight_decay: float = 0.0001
    num_workers: int = field(default=24)

    optimizer: torch.optim = field(default=torch.optim.SGD)
    optimizer_args: dict = field(
        default={
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
        metadata={"help", ""}
    )

    scheduler: torch.optim.lr_scheduler.Scheduler = field(default=None)
    scheduler_args: dict = field(default=None)
    validation_step: int = field(default=100)

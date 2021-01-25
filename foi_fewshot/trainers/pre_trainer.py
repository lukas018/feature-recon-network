#!/usr/bin/env python3
from .fewshot_trainer import FewshotTrainer
from .trainer_utils import create_dataloader


class PreTrainer(FewshotTrainer):
    """Trainer wrapper for performing image classifiaction pre-training
    """

    def get_train_dataloader(self):
        return create_dataloader(self.train_dataset, self.args)

import copy
from operator import itemgetter

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from typing import Optional, Tuple

from functools import partial



if __name__ == '__main__':
    model = models['resnet-12']

    ds = l2l.data.MiniImagenet(mode='train', download=True)
    train_arguments = TrainArguments()
    fs_arguments = PretrainArguments(**asdict(train_))
    trainer = FewshotTrainer(model, ds, fs_arguments)

    trainer.train()

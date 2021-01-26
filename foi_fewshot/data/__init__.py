#!/usr/bin/env python3
from torchvision.transforms import ColorJitter
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

import learn2learn as l2l
from .utils import fast_metadataset
from .utils import initialize_taskloader
from .utils import split_dataset

__all__ = ["fast_metadataset", "initialize_taskloader", "split_dataset"]


def mini_imagenet(root):
    normalize = Normalize(
        mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
        std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
    )
    train_data_transforms = Compose(
        [
            ToPILImage(),
            RandomCrop(84, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ],
    )
    test_data_transforms = Compose(
        [
            normalize,
        ],
    )
    ds_train = l2l.vision.datasets.MiniImagenet(
        root,
        mode="train",
        transform=train_data_transforms,
        download=True,
    )
    ds_val = l2l.vision.datasets.MiniImagenet(
        root,
        mode="validation",
        transform=test_data_transforms,
        download=True,
    )
    ds_test = l2l.vision.datasets.MiniImagenet(
        root,
        mode="test",
        transform=test_data_transforms,
        download=True,
    )

    return ds_train, ds_val, ds_test

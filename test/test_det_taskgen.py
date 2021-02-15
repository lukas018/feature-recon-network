import torch
from foi_fewshot.data import mini_imagenet
from foi_fewshot.trainers.trainer_utils import create_taskloader
from foi_fewshot.trainers import FewshotArguments
from foi_fewshot.data.utils import DeterministicTaskDataset

def test_determinism():
    fs_args = FewshotArguments(
        nways=5,
        ksupport=1,
        kquery=15,
        batch_size=1,
        num_epochs=30,
        num_workers=10,
        deterministic=True,
    )

    ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
    ts1 = create_taskloader(ds_test, fs_args).dataset
    ts2 = create_taskloader(ds_test, fs_args).dataset

    for i in range(len(ts1)):
        assert torch.all(torch.eq(ts1[i]['query'], ts2[i]['query']))


def test_randomization():
    fs_args = FewshotArguments(
        nways=5,
        ksupport=1,
        kquery=15,
        batch_size=1,
        num_epochs=30,
        num_workers=10,
        deterministic=False,
    )

    ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
    ts1 = create_taskloader(ds_test, fs_args).dataset
    ts2 = create_taskloader(ds_test, fs_args).dataset

    for i in range(len(ts1)):
        assert not torch.all(torch.eq(ts1[i]['query'], ts2[i]['query']))

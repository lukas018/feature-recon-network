import pytest
import learn2learn as l2l

from operator import itemgetter
from functools import reduce

from foi_fewshot.utils import initialize_taskdataset, split_dataset

ds = None


def test_task_dataloader():
    global ds
    ds = l2l.vision.datasets.MiniImagenet(
        root="~/Downloads", mode="train", download=True
    )
    nways = (2, 10)
    kquery = 10
    kways = (kquery + 1, kquery + 5)

    dl = initialize_taskdataset(ds, nways, kways, num_tasks=100, num_workers=1)

    from tqdm import tqdm

    for batch in tqdm(iter(dl), total=100):
        images, labels = batch
        assert images.shape[1] <= 10 * 15
        assert images.shape[1] >= 2 * 11

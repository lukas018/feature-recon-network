import pytest
import learn2learn as l2l

from operator import itemgetter
from functools import foldl

from foi_fewshot.utils import initialize_taskdataset, split_dataset


@pytest.fixture
def init_dataset():
    ds = l2l.vision.datasets.MiniImagenet(mode='train', download=True)
    return ds

def test_task_dataloader(init_dataset):

    nways = (1, 10)
    kquery = 10
    kways =  (kquery + 1, 20)

    dl = initialize_taskdataset(init_dataset, nways, kways, kquery, num_workers=1)
    lens = []
    for batch in range(dl):
        images, labels = batch
        lens.append(len(images))
        assert lens[-1] < 5*20
        assert lens[-1] >= 4*20

    assert not foldl(lambda x, y: x == y, lens)


def test_ds_split(init_dataset):
    frac = 0.7
    ds1, ds2 = split_dataset(init_dataset, frac)

    assert len(ds1) == len(init_dataset) * frac
    assert len(ds2) == len(init_dataset) (1. - frac)

    init_labels = set(map(itemgetter(1), init_dataset.data))
    ds1_labels = set(map(itemgetter(1), ds1.data))
    ds2_labels = set(map(itemgetter(1), ds2.data))

    assert len(init_labels) == len(ds1_labels)
    assert len(init_labels) == len(ds2_labels)

from learn2learn.data.transforms import Nways, Kshot, LoadImage, ConsecutiveLabels, RemapLabels
from .random_transforms import RandomNways, RandomKshot


def initialize_taskdataset(ds, nways, kways, num_workers):
    task_transforms = [
        RandomNways(ds, nways) if isinstance(nways, tuple) else Nways(ds, nways),
        RandomKshot(ds, nways) if isinstance(kways, tuple) else Kshot(ds, kways),
        LoadImage(ds),
        ConsecutiveLabels(ds),
        RemapLabels(ds)
    ]
    meta_ds = MetaDataset(ds)
    task_ds = TaskDataset(ds, task_transforms)
    return DataLoader(task_ds, num_workers=num_workers)


def classes_split(items, frac):

    def groupby(items, key):
        group = defaultdict()
        for i, item in enumerate(items):
            group[key(item)].append(item)
        return group

    groups = idx_groupby(items, itemgetter(1))

    def sampler(values):
        random.shuffle(values)
        pivot = int(len(values)*frac)
        return values[:pivot], values[pivot:]

    split1, split2 = zip(*map(sampler, groups.values()))
    return np.array(chain(*split1)), np.array(chain(*split2))

def split_dataset(ds, frac):
    """Split dataset while retaining class balance
    """

    ds1 = copy.deepcopy(ds)
    ds2 = copy.copy(ds1)

    ds1.data, ds2.data = class_split(ds1.data, frac)
    return ds1, ds2

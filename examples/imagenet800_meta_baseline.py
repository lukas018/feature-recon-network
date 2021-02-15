import random
from pathlib import Path
import json

import torch
from torch import optim
from torchvision import models

from learn2learn.data import FilterMetaDataset, SubsetMetaDataset, MetaDataset
from torchvision.transforms import (
    Compose,
    ToPILImage,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ColorJitter,
)

from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.data.imagenet_lmdb import ImageNet
from foi_fewshot.trainers import (
    create_eval_taskgen,
    create_test_taskgen,
    EvaluationStrategy,
    FewshotArguments,
    FewshotTrainer,
    PretrainArguments,
    PreTrainer,
    SchedulerUpdateStrategy,
)
from foi_fewshot.trainers.callbacks import WriterCallback


def initialize_miniimagenet800(train, validation):
    """Initialize miniimagenet by performing the correct split

    This method also caches the split so it can be reused later
    """
    ds_train = ImageNet(train)
    ds_val = ImageNet(validation)

    def _prepare_metadataset(ds, bookkeeping_path):
        if Path(bookkeeping_path).exist():
            ds._bookkeeping_path = bookkeeping_path
            ds = MetaDataset(ds)
        else:
            # This will take some time
            ds = MetaDataset(ds_train)
            ds.serialize_bookkeeping(bookkeeping_path)
        return ds

    bk_train = Path((Path(train).parent, "train_bookkeeping.pkl"))
    bk_val = Path((Path(train).parent, "validation_bookkeeping.pkl"))
    ds_train = _prepare_metadataset(ds_train, bk_train)
    ds_val = _prepare_metadataset(ds_val, bk_val)

    # Randomly select indices for train and test
    _random = random.Random(seed=10)
    idx = _random.choice(list(range(1000)), 1000)
    train_classes, val_classes = idx[:800], idx[800:]
    ds_train = FilterMetaDataset(ds_train, train_classes)
    ds_val = FilterMetaDataset(ds_train, val_classes)

    train_transform = Compose(
        [
            ToPILImage(),
            RandomCrop(126, padding=8),
            Resize(128),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )
    test_transform = Compose([ToPILImage(), Resize(128), ToTensor()])

    ds_train.transform = train_transform
    ds_val.transform = test_transform
    ds_train, ds_base = SubsetMetaDataset.create_intra_class_split(ds_train)
    # We just use novel ds as a way of plotting validation error without having
    # to evaluate on the entire dataset every epoch.
    # Since we don't use it for model-selection we should not introduce any
    # bias into the finished model
    _, ds_novel = SubsetMetaDataset.create_intra_class_split(ds_test, 0.9)

    return ds_train, ds_base, ds_novel, ds_val


path_to_ds_train = "/mnt/ssd2/ImageNet/train.lmdb"
path_to_ds_val = "/mnt/ssd2/ImageNet/val.lmdb"

ds_train, ds_base, ds_novel, ds_test = initialize_miniimagenet800(
    path_to_ds_train, path_to_ds_val
)

# Initialize the network backbone
model = models.resnext50_32x4d()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)
meta_baseline.init_pretraining(640, 64)

# Enable cuda
if torch.cuda.is_available():
    meta_baseline = meta_baseline.cuda()

optimizer = optim.SGD(
    meta_baseline.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001,
)


def lr_update(epoch):
    """Very simple learning rate multiplier"""
    if epoch >= 30 and epoch < 60:
        return 0.1
    elif epoch >= 60:
        return 0.01
    return 1.0


lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_update)
pt_args = PretrainArguments(
    modeldir="/mnt/ssd2/image-net/resnet50/modeldir",
    logdir="/mnt/ssd2/image-net/resnet50/logdir",
    logging_steps=100,
    batch_size=128,
    gradient_accumulation_steps=2,
    num_epochs=90,
    num_workers=20,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    distribted=True,
    scheduler_update_strategy=SchedulerUpdateStrategy.EPOCH,
    metric_for_best_model=None,
    modeldir_prefix="pretraining",
    disable_gradient_during_eval=True,
)

eval_taskgen = create_eval_taskgen(ds_base, ds_novel, n_samples=25, batch_size=4)
pre_trainer = PreTrainer(
    meta_baseline,
    pt_args,
    ds_train,
    eval_taskgen,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[WriterCallback()],
)
fs_trainer = FewshotTrainer(model,)
pre_trainer.train()

test_taskgen = create_test_taskgen(ds_test, n_samples=500, batch_size=5)
metrics = pre_trainer.evaluate(test_taskgen)
print("====================")
print("Classifier baseline:")
print(json.dumps(metrics, indent=4))
print("====================")

# Perform fewshot training using nway=5, kshot=1
fs_args = FewshotArguments(
    modeldir="/mnt/sdd2/fewshot-models/",
    logdir="/mnt/sdd2/fewshot-logs/",
    nways=5,
    ksupport=1,
    kquery=15,
    batch_size=1,
    num_epochs=30,
    epoch_steps=200,
    logging_steps=50,
    num_workers=10,
    gradient_accumulation_steps=4,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    modeldir_prefix="fewshot-training",
    disable_gradient_during_eval=True,
)

optimizer = optim.SGD(meta_baseline.parameters(), lr=0.001, momentum=0.9,)

# Chen et al doesn't use data augmentation in the meta-learning step
ds_train.transform = ds_test.transform
ds_base.transform = ds_test.transform

# Lets use more samples now to get a more
# precise measurement of the performance
eval_taskgen = create_eval_taskgen(ds_base, ds_novel, n_samples=200, batch_size=4)
test_taskgen = create_test_taskgen(ds_test, n_samples=500, batch_size=5)
fs_trainer = FewshotTrainer(
    meta_baseline,
    fs_args,
    ds_train,
    eval_taskgen,
    optimizers=(optimizer, None),
    callbacks=[WriterCallback()],
)

fs_trainer.train()
metrics = fs_trainer.evaluate(test_taskgen)
print("====================")
print("Meta-baseline:")
print(json.dumps(metrics, indent=4))
print("====================")

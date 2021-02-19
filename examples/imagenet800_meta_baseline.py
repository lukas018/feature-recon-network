import os
import sys
import random
from pathlib import Path
import json
from collections import defaultdict

import torch
from torch import optim
import torch.distributed as dist
from torchvision import models

from learn2learn.data import FilteredMetaDataset, SubsetMetaDataset, MetaDataset
from torchvision.transforms import (
    Compose,
    ToPILImage,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ColorJitter,
    CenterCrop,
    RandomResizedCrop,
    Normalize
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

    image_size = 224
    box_size = 256
    norm_params = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    normalize = Normalize(**norm_params)
    train_transform = Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    test_transform = Compose([
        Resize(box_size),
        CenterCrop(image_size),
        ToTensor(),
        normalize,
    ])

    ds_train.transform = train_transform
    ds_val.transform = test_transform

    def _prepare_metadataset(ds, bookkeeping_path):
        if Path(bookkeeping_path).exists():
            ds._bookkeeping_path = bookkeeping_path
            ds = MetaDataset(ds)
        else:
            # This will take some time
            ds = MetaDataset(ds)
            ds.serialize_bookkeeping(bookkeeping_path)
        return ds

    bk_train = Path(Path(train).parent, "train_bookkeeping.pkl")
    bk_val = Path(Path(train).parent, "validation_bookkeeping.pkl")
    ds_train = _prepare_metadataset(ds_train, bk_train)
    ds_val = _prepare_metadataset(ds_val, bk_val)

    # Randomly select indices for train and test
    _random = random.Random(10)
    idx = _random.sample(list(range(1000)), 1000)
    train_classes, val_classes = idx[:800], idx[800:]
    ds_train = FilteredMetaDataset(ds_train, train_classes)
    ds_val = FilteredMetaDataset(ds_val, val_classes)

    def remap_labels(dataset):
        labels = dataset.labels_to_indices.keys()
        remapper = {l: i for i, l in enumerate(labels)}
        labels_to_indices = {remapper[l]: v for l, v in dataset.labels_to_indices.items()}
        indices_to_labels = {i: l for l, ix in labels_to_indices.items() for i in ix}

        dataset.labels_to_indices = labels_to_indices
        dataset.indices_to_labels = indices_to_labels
        dataset.labels = list(dataset.labels_to_indices.keys())
        return dataset

    ds_train = remap_labels(ds_train)
    ds_val = remap_labels(ds_val)

    ds_train, ds_base = SubsetMetaDataset.create_intra_class_split(ds_train, 0.97)
    # We just use novel ds as a way of plotting validation error without having
    # to evaluate on the entire dataset every epoch.
    # Since we don't use it for model-selection we should not introduce any
    # bias into the finished model
    _, ds_novel = SubsetMetaDataset.create_intra_class_split(ds_val, 0.5)

    return ds_train, ds_base, ds_novel, ds_val


path_to_ds_train = "/mnt/ssd2/ImageNet/train.lmdb"
path_to_ds_val = "/mnt/ssd2/ImageNet/validation.lmdb"

ds_train, ds_base, ds_novel, ds_test = initialize_miniimagenet800(
    path_to_ds_train, path_to_ds_val
)
# Initialize the network backbone
model = models.resnext50_32x4d()
model.fc = torch.nn.Identity()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)
meta_baseline.init_pretraining(2048, 800)

# modeldir = "/mnt/ssd2/image-net/resnet50/modeldir/pretraining-151515"
# modeldir = "/mnt/ssd2/image-net/resnet50/modeldir/pretraining-163170/"
# meta_baseline.load_state_dict(torch.load(Path(modeldir, "model.pkl")))

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
    batch_size=256 // 4,
    gradient_accumulation_steps=4,
    num_epochs=90,
    num_workers=100,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    distributed=False,
    scheduler_update_strategy=SchedulerUpdateStrategy.EPOCH,
    metric_for_best_model=None,
    modeldir_prefix="pretraining",
    disable_gradient_during_eval=True,
)

eval_taskgen = create_eval_taskgen(
    ds_base,
    ds_novel,
    n_samples=25,
    batch_size=4
)
eval_taskgen.add("base-classification", ds_base, pt_args)
pre_trainer = PreTrainer(
    meta_baseline,
    pt_args,
    ds_train,
    eval_taskgen,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[WriterCallback()],
)

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

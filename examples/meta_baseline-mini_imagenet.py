#!/usr/bin/env python3


from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)

model = ResNet12()
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train)

train_arguments = TrainingArguments(
    modeldir="~/Downloads/models",
    logdir="~/Downloads/logdir",
    batch_size=8,
    max_epochs=100,
)
model = ResNet12()
meta_baseline = MetaBaseline(model)
meta_baseline.init_pretraining(640, 64)
pre_trainer = PreTrainer(model, ds_train, train_arguments, None, ds_base, ds_novel)
pre_trainer.train()

fs_arguments = FewshotTrainer(
    nways=5,
    ksupport=5,
    kquery=10,
   **train_arguments.asdict()
)

fs_trainer = FewshotTrainer(model, ds_train, fs_arguments, None, ds_base, ds_novel)
fs_trainer.train()
results = fs_trainer.evaluate(ds_test)
print(results)

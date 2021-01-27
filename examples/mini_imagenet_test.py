import json

import torch
from torch import optim

from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.data import mini_imagenet
from foi_fewshot.data import split_dataset
from foi_fewshot.models import ResNet12
from foi_fewshot.trainers import create_eval_taskgen
from foi_fewshot.trainers import create_test_taskgen
from foi_fewshot.trainers import EvaluationStrategy
from foi_fewshot.trainers import FewshotArguments
from foi_fewshot.trainers import FewshotTrainer
from foi_fewshot.trainers import PretrainArguments
from foi_fewshot.trainers import PreTrainer
from foi_fewshot.trainers import SchedulerUpdateStrategy
from foi_fewshot.trainers.callbacks import WriterCallback

# Initialize the network backcone
model = ResNet12()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)
# Initialize the datasets
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train, frac=0.95, even_class_dist=True)
# Init baseline
meta_baseline.init_pretraining(640, 64)
# Enable cuda
if torch.cuda.is_available():
    meta_baseline = meta_baseline.cuda()

optimizer = optim.SGD(
    meta_baseline.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
)


def lr_update(epoch):
    """Very simple learning rate multiplire"""
    return 0.1 if epoch >= 90 else 1.0


lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_update)
pt_args = PretrainArguments(
    modeldir="~/Downloads/models/",
    logdir="~/Downloads/logs/",
    logging_steps=100,
    batch_size=64,
    gradient_accumulation_steps=2,
    num_epochs=100,
    num_workers=10,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    # Use this to call the scheduler at the end of every epoch
    scheduler_update_strategy=SchedulerUpdateStrategy.EPOCH,
    # Chen et al. simply train for 100 epochs, so lets to the same
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

pre_trainer.train()
test_taskgen = create_test_taskgen(ds_test, n_samples=500, batch_size=5)
metrics = pre_trainer.evaluate(test_taskgen)
print("====================")
print("Classifier baseline:")
print(json.dumps(metrics, indent=4))
print("====================")

# Perform fewshot training using nway=5, kshot=1
fs_args = FewshotArguments(
    modeldir="~/Downloads/fewshot-models/",
    logdir="~/Downloads/fewshot-logs/",
    nways=5,
    ksupport=1,
    kquery=15,
    batch_size=1,
    num_epochs=30,
    epoch_steps=200,
    eval_steps=50,
    num_workers=10,
    gradient_accumulation_steps=4,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    modeldir_prefix="fewshot-training",
    # Use the performance on the novel dataset to select the best mode
    metric_for_best_model="eval-novel:n=5,k=1-loss",
    disable_gradient_during_eval=True,
)

optimizer = optim.SGD(
    meta_baseline.parameters(),
    lr=0.001,
    momentum=0.9,
)

# Chen et al doesn't use data augmentation in the meta-learning step
ds_train.transform = ds_test.transform
ds_base.transform = ds_test.transform

# Lets use more samples now to get a more precise measurement of the performance
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

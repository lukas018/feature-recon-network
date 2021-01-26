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
    scheduler_update_strategy=SchedulerUpdateStrategy.EPOCH,
    metric_for_best_model="eval-base-class-loss",
    modeldir_prefix="pretraining",
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

# Perform fewshot training
fs_args = FewshotArguments(
    modeldir="~/Downloads/fewshot-models/",
    logdir="~/Downloads/fewshot-logs/",
    nways=5,
    ksupport=1,
    kquery=15,
    batch_size=1,
    num_epochs=10,
    epoch_steps=250,
    eval_steps=50,
    num_workers=10,
    logging_steps=50,
    gradient_accumulation_steps=4,
    evaluation_strategy=EvaluationStrategy.STEPS,
    scheduler_update_strategy=SchedulerUpdateStrategy.STEPS,
    modeldir_prefix="fewshot-training",
)

fs_trainer = FewshotTrainer(
    meta_baseline,
    fs_args,
    ds_train,
    eval_taskgen,
    callbacks=[WriterCallback()],
)
fs_trainer.train()
test_taskgen = create_test_taskgen(ds_test, n_samples=500, batch_size=5)
metrics = fs_trainer.evaluate(test_taskgen)
print(metrics)

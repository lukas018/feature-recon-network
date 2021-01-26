from torch import optim

from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.data import mini_imagenet
from foi_fewshot.data import split_dataset
from foi_fewshot.models import ResNet12
from foi_fewshot.trainers import EvalTaskGenerator
from foi_fewshot.trainers import EvaluationStrategy
from foi_fewshot.trainers import FewshotArguments
from foi_fewshot.trainers import FewshotTrainer
from foi_fewshot.trainers import PretrainArguments
from foi_fewshot.trainers import PreTrainer
from foi_fewshot.trainers import SchedulerUpdateStrategy


# Initialize the network backcone
model = ResNet12()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)

# Initialize the datasets
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train, frac=0.95, even_class_dist=True)

pt_args = PretrainArguments(
    modeldir="~/Downloads/models/",
    logging_steps=250,
    batch_size=64,
    gradient_accumulation_steps=1,
    num_epochs=100,
    num_workers=10,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    scheduler_update_strategy=SchedulerUpdateStrategy.EPOCH,
    metric_for_best_model="eval-base-class-loss",
    modeldir_prefix="pretraining",
)

fs_args = FewshotArguments(
    modeldir="~/Downloads/models/",
    nways=5,
    ksupport=10,
    kquery=15,
    batch_size=5,
    num_epochs=100,
    epoch_steps=100,
    eval_steps=10,
    num_workers=10,
    logging_steps=250,
    gradient_accumulation_steps=1,
    metric_for_best_model="eval-novel-loss",
    evaluation_strategy=EvaluationStrategy.STEPS,
)

eval_taskgen = EvalTaskGenerator(fs_args)
eval_taskgen.add("base", ds_base)
eval_taskgen.add("novel", ds_novel)
eval_taskgen.add("base-class", ds_base, pt_args)


def lr_update(epoch):
    if 80 <= epoch:
        return 0.1
    elif 90 <= epoch:
        return 0.01
    else:
        return 1.0


# Init baseline
meta_baseline.init_pretraining(640, 64)
optimizer = optim.SGD(meta_baseline.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_update)
pre_trainer = PreTrainer(
    meta_baseline,
    pt_args,
    ds_train,
    eval_taskgen,
    optimizers=(optimizer, lr_scheduler),
)
pre_trainer.train()

# Perform fewshot training
fs_trainer = FewshotTrainer(
    meta_baseline,
    fs_args,
    ds_train,
    eval_taskgen,
)
fs_trainer.train()

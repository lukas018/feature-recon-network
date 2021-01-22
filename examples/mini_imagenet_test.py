from foi_fewshot.models import ResNet12
from tqdm import tqdm
from foi_fewshot.models import ResNet12
from foi_fewshot.data import mini_imagenet, split_dataset
from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.trainers import (
    PreTrainer,
    FewshotTrainer,
    TrainingArguments,
    FewshotArguments,
    EvalTaskGenerator,
    EvaluationStrategy
)
from foi_fewshot.trainers.trainer_utils import create_taskloader
from foi_fewshot.trainers.callbacks import WriterCallback

# Initialize the network backcone
model = ResNet12()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)

# Initialize the datasets
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train, frac=0.95, even_class_dist=True)

train_args = TrainingArguments(
    batch_size=128,
    gradient_accumulation_steps=1,
    num_epochs=100,
    evaluation_strategy=EvaluationStrategy.EPOCH
)

fs_args = FewshotArguments(
    modeldir="~/Downloads/models/",
    nways=2,
    ksupport=1,
    kquery=1,
    batch_size=2,
    num_epochs=100,
    epoch_steps=100,
    eval_steps=10,
    num_workers=10,
    gradient_accumulation_steps=1,
    metric_for_best_model='eval-novel-loss',
    evaluation_strategy=EvaluationStrategy.STEPS
)

eval_taskgen = EvalTaskGenerator(fs_args)
eval_taskgen.add("base", ds_base)
eval_taskgen.add("novel", ds_novel)
eval_taskgen.add("val-class", ds_base, train_args)

breakpoint()
# Pretrain the model
meta_baseline.init_pretraining(64)
pre_trainer = PreTrainer(
    meta_baseline,
    args=train_args,
    eval_taskgen=eval_taskgen
)
pre_trainer.train()

# Perform fewshot training
fs_trainer = FewshotTrainer(
    meta_baseline, train_args, ds_train, eval_taskgen
)
fs_trainer.train()

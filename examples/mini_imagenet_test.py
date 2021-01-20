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
    EvalTaskGenerator
)


# Initialize the network backcone
model = ResNet12()
# Wrap with Metabaseline Wrapper
meta_baseline = MetaBaseline(model)

# Initialize the datasets
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")
ds_train, ds_base = split_dataset(ds_train, frac=0.95)
train_args = FewshotArguments(
    modeldir="~/Downloads/models",
    logdir="~/Downloads/logdir",
    nways=5,
    ksupport=10,
    kquery=5,
    batch_size=2,
    num_epochs=100,
    epoch_steps=100,
    eval_steps=10,
    num_workers=10,
    gradient_accumulation_steps=10,
    metric_for_best_model='acc',
)
breakpoint()

eval_taskgen = EvalTaskGenerator(train_args)
breakpoint()
eval_taskgen.add("base", ds_base)
eval_taskgen.add("novel", ds_novel)

fs_trainer = FewshotTrainer(meta_baseline, train_args, eval_taskgen)
fs_trainer.train()

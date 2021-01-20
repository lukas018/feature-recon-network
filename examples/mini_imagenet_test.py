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
train_args = FewshotArguments(
    nways=2,
    ksupport=1,
    kquery=1,
    batch_size=2,
    num_epochs=100,
    epoch_steps=100,
    eval_steps=10,
    num_workers=10,
    gradient_accumulation_steps=2,
    metric_for_best_model="acc",
)

eval_taskgen = EvalTaskGenerator(train_args)
eval_taskgen.add("base", ds_base)
eval_taskgen.add("novel", ds_novel)

callbacks = WriterCallback(train_args.logdir)
fs_trainer = FewshotTrainer(
    meta_baseline, train_args, ds_train, eval_taskgen, callbacks=[callbacks]
)
fs_trainer.train()

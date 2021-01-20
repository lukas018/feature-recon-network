from foi_fewshot.models import resnet12
from foi_fewshot.trainer import TrainingArguments, PretrainArguments
from foi_fewshot.utils import split_dataset

import learn2learn as l2l

# Initialize the model
model = resnet12()

# Training Arguments
train_arguments = TrainingArguments()

# Prepare the dataset
ds = l2l.vision.datasets.MiniImagenet(mode="train", download=True)
train_ds, base_ds = split_dataset(ds, frac=0.9)
novel_ds = l2l.vision.datasets.MiniImagenet(mode="val", download=True)
test_ds = l2l.vision.datasets.MiniImagenet(mode="test", download=True)

# Pretrain the model
pretrainer = PreTrainer(model, train_ds, train_args, base_ds, novel_ds)
pretrainer.train()

# Fewshot Training
fs_args = FewshotArguments(**train_arguments, nway=10)
fs_val_args = FewshotArguments(**train_arguments, nway=5, kshot=10, kquery=10)

# Prepare fewshot training
fs_trainer = FewshotTrainer(model, train_ds, fs_args, base_ds, novel_ds)
fs_trainer.fewshot_trainer()
metrics = fs_trainer.eval(test_ds)

# Save the final results
metrics.write()

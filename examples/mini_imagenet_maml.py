# /urs/bin/python3
"""Example script of MAML

This script implements the nway=5, kshot=1 experiment in the
the miniImagenet dataset outlined in the original MAML paper
https://arxiv.org/pdf/1703.03400.pdf.
"""
import torch
from torch import optim

import learn2learn as l2l
from foi_fewshot.algorithms import MAML
from foi_fewshot.data import mini_imagenet
from foi_fewshot.data import split_dataset
from foi_fewshot.trainers import create_eval_taskgen
from foi_fewshot.trainers import create_test_taskgen
from foi_fewshot.trainers import EvaluationStrategy
from foi_fewshot.trainers import FewshotArguments
from foi_fewshot.trainers import FewshotTrainer
from foi_fewshot.trainers.callbacks import WriterCallback


# Initialize the datasets
ds_train, ds_novel, ds_test = mini_imagenet("~/Downloads")

# Perform fewshot training
fs_args = FewshotArguments(
    modeldir="~/Downloads/maml-models/",
    logdir="~/Downloads/maml-logs/",
    nways=5,
    ksupport=5,
    kquery=15,
    batch_size=1,
    num_epochs=100,
    epoch_steps=600,
    num_workers=10,
    logging_steps=300,
    gradient_accumulation_steps=32,
    evaluation_strategy=EvaluationStrategy.EPOCH,
    modeldir_prefix="maml-training",
)

eval_taskgen = create_eval_taskgen(None, ds_novel, n_samples=800, batch_size=1, nways=5, kshots=5)
test_taskgen = create_test_taskgen(ds_test, n_samples=800, batch_size=1, nways=5, kshots=5)

model = l2l.vision.models.MiniImagenetCNN(5)
maml = MAML(model, fast_lr=0.5, update_steps=1, eval_update_steps=1)

if torch.cuda.is_available():
    maml = maml.cuda()

optimizer = optim.Adam(maml.parameters(), lr=0.003)
fs_trainer = FewshotTrainer(
    maml,
    fs_args,
    ds_train,
    eval_taskgen,
    optimizers=(optimizer, None),
    callbacks=[WriterCallback()],
)

print(fs_trainer.evaluate())
fs_trainer.train()
metrics = fs_trainer.evaluate(test_taskgen)
print(metrics)

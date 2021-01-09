from torch import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
from .freshot_utils import fewshot_episode

from .data import initialize_taskdataset, classes_split, split_dataset, SummaryGroup

@dataclass_json
@dataclass
class TrainingState():
    epoch :int = 0
    global_step: int = 0
    current_step: int = 0


class FewshotTrainer():
    """FewshotTrainer

    General trainer wrapper for few-shot trainers
    """

    def __init__(self,
                 model,
                 train_ds,
                 args,
                 eval_args=None,
                 optimizer=None,
                 scheduler=None,
                 base_eval_dataset=None,
                 novel_eval_dataset=None,
                 on_epoch_end_callback = None,
                 on_update_callback = None
        ):

        self.model = model
        self.train_ds = train_ds
        self.args = args
        self.eval_args = eval_args

        if optimizer is None:
            self.optimzer = torch.optim.SGD(self.model.parameters(), momentum=0.9)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        self.base_eval_dataset = base_eval_dataset
        self.novel_eval_dataset = novel_eval_dataset

        self.on_epoch_end_callback = on_epoch_end_callback
        self.on_update_callback = on_update_callback
        self.validate = self.args.do_eval

        self.state = TrainingState()

    def save_checkpoint(self, modeldir=None):
        """Save the current state of the model
        """

        modeldir = modeldir if modeldir is not None else self.args.modeldir
        modeldir.mkdir(exist_ok=True)
        prefix = f"prefix-{self.state.epoch}"

        torch.save(self.model.state_dict(), Path(modeldir, f"model.pkl"))
        if self.optimizer is not None:
            torch.save(self.optimizer, Path(modeldir, "optimizer.pkl"))

        if self.scheduler is not None:
            torch.save(self.scheduler, Path(modeldir, "scheduler.pkl"))

        state_path = Path(modeldir, f"{prefix}-state.json")
        with open(state_path, 'w') as fh:
            fh.write(self.state.tojson())

    def load_checkpoint(self, modeldir):

        self.model.load_state_dict(torch.load(Path(modeldir, f"model.pkl")))

        optimizer_path = Path(modeldir, f"optimizer.pkl")
        if optimizer_path.isfile():
            self.optimizer.load(optimizer_path)

        scheduler_path = Path(modeldir, f"scheduler.pkl")
        if scheduler_path.isfile():
            self.scheduler.load(optimizer_path)

        state_path = Path(modeldir, f"state.json"), 'w'
        if state_path.isfile():
            with open(state_path, 'r') as fh:
                self.state = TrainingState.fromdict(json.load(fh))


    def fewshot_training(self, model_dir=None):
        """Trains the model using standard fewshot training
        """

        if model_dir:
            self.load_checkpoint(model_dir)

        self.model.train()

        dl_train = self.get_train_dataloader()
        loss = 0

        for epoch in range(self.state.epoch,
                           self.args.max_episodes * self.args.batch_size // self.args.epoch_length):

            for step in range(self.args.epoch_length * self.args.batch_size):
                batch = next(dl_train)
                _loss, res = fewshot_episode(self.model,
                                            batch,
                                            self.args.kquery,
                                            self.args.device)

                loss += _loss
                if step % self.args.batch_size == 0 and step == self.args.epoch_length - 1:
                    loss.backward()
                    self.optimzer.step()
                    self.optimizer.zero_grad()
                    loss = 0

                    self.state.global_step += 1
                    self.state.current_step += 1

                    if self.on_update_cb:
                        self.on_update_cb(self)


            if self.on_epoch_end_cb:
                self.on_epoch_end_cb(self)

            self.state.epoch += 1

            self.save_checkpoint()

            metrics = self.fewshot_eval()

    def _fewshot_eval(self, ds, num_episodes, args):
        self.learner.eval()

        dl  = initialize_taskdataset(ds, args.nways, args.kshot, args.num_workers)
        kquery = self.val_args.kquery if self.val_args is not None else self.args.kquery
        use_cuda = self.val_args.use_cuda if self.val_args is not None else self.args.use_cuda

        res = []
        for i in range(num_episodes):
            batch = next(dl)
            _, _res = fewshot_episode(self.learner, batch, kquery, use_cuda)
            res.append(_res)

        res = {key: [v[key] for v in res] for key in res[0].keys()}

        sg = SummaryGroup(self.args.writer)
        for key, vs in res.items():
            sg(key, *vs)

        return sg

    def fewshot_eval(self, dataset=None, num_epsiodes=None):

        datasets = []
        if dataset is not None:
            datasets.append(dataset)
        else:
            if self.base_eval_dataset is not None:
                datasets.append(self.base_eval_dataset)

            if self.novel_eval_dataset is not None:
                datasets.append(self.novel_eval_dataset)

        num_episodes = self.args.eval_episodes if num_episodes is None else num_episodes,
        res = [self.fewshot_eval(ds, num_episodes, self.args) for ds in datasets]

        return *res,


class PreTrainer(FewshotTrainer):

    def get_train_dataloader(self):
        args = self.args
        dl = DataLoader(self.eval_base_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)

    def get_eval_dataloader(self):
        args = self.eval_args if self.eval_args is not None else self.args
        dl = DataLoader(self.eval_base_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
        return dl


    def prediction_step(self, batch):
        images, labels = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        acc = (max(logits)[1] == labels) / len(labels)

        return loss, self.compute_metrics(logits, labels)


    def train(self, modeldir=None):

        if modeldir:
            self.load_checkpoint(modeldir)

        self.model.train()

        dl_train = self.get_train_dataloader()
        loss = 0

        for epoch in range(self.args.max_peoch):
            dl_train = self.get_train_dataloader()

            for i, batch in dl_train:

                self.state.global_step += 1
                self.state.curret_step += 1

                self.optimizer.zero_grad()
                loss, acc = prediction_step(batch)
                loss.backward()
                self.optimizer.step()

                self.on_update_callback(self)

            self.on_epoch_end_callback(self)

            if self.args.do_eval:
                netrics = self.fewshot_evaluate()
                {**metrics, **self.evaluate()}

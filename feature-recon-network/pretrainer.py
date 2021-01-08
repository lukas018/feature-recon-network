
class PreTrainer():

    def __init__(self, model, args, ds_train, ds_val, ds_novel):
        self.model = model
        self.model.init_pretraining(len(ds_train))

        self.stats = Statistics()
        self.args = args

        self.opt = self.arg.optimizer(model.parameters(),
                                      **self.args.optimizer_arguments)
        self.scheduler = self.args.scheduler(self.opt,
                                             **self.args.scheduler_arguments)


    def train_epoch(self, ds_train):
        dl_train = DataLoader(ds_train, batch_size=self.args.train_batch_size)

        for batch in dl_train:
            images, labels = batch
            self.opt.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy()
            loss.backward()
            self.opt.step()
            acc = (torch.max(logits)[1] == labels) / len(labels)
            self.stats["pretrain/loss/train"]

        if self.scheduler:
            self.scheduler.step()

    def eval(self, ds_val):
        self.model.eval()
        dl_val = DataLoader(ds_val, batch_size=self.args.val_batch_size)

        for batch in dl_val:
            images, labels = batch
            self.opt.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy()
            acc = (torch.max(logits)[1] == labels) / len(labels)

        base_dl = initialize_taskdataset(ds_val, self.args)

    def dump(self, epoch):
        modeldir = self.args.modeldir
        torch.save(self.model.state_dict(), Path(modeldir, f"pretrain-model-{epoch}.pkl"))
        torch.save(self.opt, Path(modeldir, f"pretrain-opt-{epoch}.pkl"))
        torch.save(self.scheduler, Path(modeldir, f"pretrain-scheduler-{epoch}.pkl"))

    def load(self, epoch):
        modeldir = self.args.modeldir
        self.model.load_state_dict(torch.load(Path(modeldir, f"pretrain-model-{epoch}.pkl")))
        self.opt.load(Path(modeldir, f"pretrain-opt-{epoch}.pkl"))
        self.scheduler.load(Path(modeldir, f"pretrain-scheduler-{epoch}.pkl"))


def pretrain_loop(trainer, fs_trainer):

    for epoch in range(trainer.args.max_epochs):
        trainer.train_epoch()
        res = fs_trainer.full_fewshot_eval()

        for key, v in res.items():
            self.stats["pretrain/" + key] = v

        trainer.dump(epoch)

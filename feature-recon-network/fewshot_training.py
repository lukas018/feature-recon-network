from torch import functional as F


def fewshot_episode(learner, batch, query_k, device="cuda"):
    """Perform fewshot epoch

    :param learner:
    :param batch:
    :param query_k:
    """

    data, labels = batch

    nway = len(set(labels))
    total_k = nway // len(labels)
    k = total_k - query_k

    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    evaluation_indices = np.zeros(data.size(0), dtype=bool)
    evaluation_indices[(np.arange(nway*k_total) % total_k) >= (total_k - query_k)] = True
    evaluation_indices = torch.from_numpy(evaluation_indices)
    adaptation_indices = torch.from_numpy(~evaluation_indices)

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    logits, aux_loss = learner(evaluation_data, adaptation_data)
    loss = F.cross_entropy(logits, labels)
    loss += aux_loss

    acc = ((torch.max(logits)[1]) == labels) / (nway*total_k)
    return loss, acc


class FewshotLearner():
    """FewshotLearner
    """

    def __init__(self, learner,
                 train_ds,
                 args,
                 base_val_ds=None,
                 novel_val_ds=None,
                 test_dsa=None,
                 val_args=None
                 ):

        self.train_ds = train_ds
        self.args = arguments
        self.base_val_ds = base_val_ds
        self.novel_val_ds = novel_val_ds
        self.test_ds = test_ds
        self.val_args = val_args

    def dump(self, epoch):
        modeldir = self.args.modeldir
        torch.save(self.model.state_dict(), Path(modeldir, f"fewshot-model-{epoch}.pkl"))
        torch.save(self.opt, Path(modeldir, f"fewshot-opt-{epoch}.pkl"))
        torch.save(self.scheduler, Path(modeldir, f"fewshot-scheduler-{epoch}.pkl"))

    def load(self, epoch):
        modeldir = self.args.modeldir
        self.model.load_state_dict(torch.load(Path(modeldir, f"fewshot-model-{epoch}.pkl")))
        self.opt.load(Path(modeldir, f"fewshot-opt-{epoch}.pkl"))
        self.scheduler.load(Path(modeldir, f"fewshot-scheduler-{epoch}.pkl"))

    def fewshot_training(self,  num_episodes=None):
        self.learner.train()

        kquery = self.args.kquery
        use_cuda = self.args.use_cuda
        num_episodes = num_episodes if num_episodes is not None else self.args.num_episodes
        losses, accs = [], []

        for i in range(num_eposides):
            batch = next(self.dl_train)
            opt.zero_grad()
            loss, acc = fewshot_episode(self.learner, batch, self.args.kquery, self.args.use_cuda)
            loss.backward()
            opt.update()

            losses.append(loss.detach().data.numpy())
            accs.append(acc)

        return np.mean(losses), np.mean(accs)

    def fewshot_eval(self, ds, num_episodes, args):
        self.learner.eval()
        dl  = initialize_taskdataset(ds, args.nways, args.kshot, args.num_workers)

        kquery = self.val_args.kquery if self.val_ags is not None else self.args.kquery
        use_cuda = self.val_args.use_cuda if self.val_args is not None else self.args.use_cuda

        losses, accs = [], []
        for i in range(num_episodes):
            batch = next(dl)
            loss, acc = fewshot_episode(self.learner, batch, kquery, use_cuda)
            losses.append(loss.detach().data.numpy())
            accs.append(acc)

        return np.mean(losses), np.mean(accs)

    def full_fewshot_eval(self):
        res = {}
        if self.base_val_ds is not None:
            res['base/loss'], res['base/acc'] = self.fewshot_eval(self.base_val_ds,
                                                                  self.args.eval_episodes,
                                                                  self.args)
        if self.novel_val_ds is not None:
            res['novel/loss'], res['novel/acc'] =self.fewshot_eval(self.novel_val_ds,
                                                                  self.args.eval_episodes,
                                                                  self.args)
        return res


def fewshot_loop(ds):
    writer = SummaryWriter(self.args.logdir)
    statistics = Statistics(alpha=0.0)

    for i in range(trainer.args.max_episodes // trainer.args.epoch_episodes):
        loss, acc = trainer.fewshot_training()

        statistics['fewshot/loss/train'] = loss
        statistics['fewshot/acc/train'] = acc

        res = self.full_fewshot_eval()
        for key, v in res.items():
            statistics["fewshot/" + key] = v

        statistics.write()


class MAMLTrainer(FewshotTrainer):

    def fewshot_episode(self, batch, *args, **kwargs):
        model = self.model.clone()
        return maml_fewshot(model, batch, *args, update=self.args.update_steps, **kwargs)

    def update(self, loss):
        loss.backward()
        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / self.args.batch_size).add_(p.data)
        self.optimizer.step()


class ReptileTrainer(FewshotTrainer):
    """Trainer for the Reptile algorithmk

    Reptile is a first-order variant of the MAML algorithm descrbied in the paper:
    https://arxiv.org/pdf/1803.02999.pdf

    The input model here can by any classification network, no wrapper class is required
    """

    def __init__(self):
        self.adapt_opt_state = self.optimizer.state_dict()
        self.optimizer_fn = None

    def fewshot_episode(self, batch, training=False):
        model = deepcopy(self.model)

        opt = self.optimizer_fn(self)
        opt.load_state_dict(self.optimizer.state_dict())
        reptile = reptile_fewshot(model,
                               batch,
                               optimizer=opt,
                               update=self.args.update_steps
                               *args,
                               **kwargs)

        # Construct the global gradient from the difference between the two models
        if training:
            self.adapt_opt_state = adapt_opt.state_dict()
            for p, l in zip(self.model.parameters(), model.parameters()):
                p.grad.data.add_(-1.0, l.data)


    def update(self, loss):
        # Here we don't care about the loss
        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / self.args.batch_size).add_(p.data)
        self.optimizer.step()

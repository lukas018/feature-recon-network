from torch import functional as F


def fewshot_epoch(learner, batch, query_k, device="cuda"):
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


def initialize_dataloader(ds, args):

    [
        RandomNWays(ds, args.nways) if isinstance(args.nways, tuple) else NWays(ds, args.nways),
        RandomKShot(ds, args.kshot) if isinstance(args.kshot, tuple) else KShot(ds, args.kshot),
        ]





def fewshot_training(
        frn,
        datasets,
        fewshot_args
):

    train_ds, val_ds, test_ds = datasets
    device = torch.device('cuda') if fewshot_args.use_cuda else torch.device('cpu')

    train_dl = initialize_dataloader(train_ds, fewshot_args)

    train_iter = iter(train_ds)
    val_iter = iter(val_ds)
    test_iter = iter(test_iter)

    frn.train()

    for i, batch in enumerate(train_ds):
        opt.zero_grad()
        loss, acc = fewshot_epoch(frn, batch, query_k, device)
        err.backward()
        opt.update()

        writer.add_scalar('fewshot/loss/train', loss, i)
        writer.add_scalar('fewshot/acc/train', acc, i)

        if i % validation_step:
            frn.validation()
            res = [fewshot_epoch(frn, next(val_iter), query_k, device) for _ in range(25)]
            loss, acc = *map(np.mean, map(np.array, zip(*res))),
            writer.add_scalar('fewshot/loss/train', loss, i)
            writer.add_scalar('fewshot/acc/train', acc, i)
            frn.train()


    frn.validation()
    for i, batch in enumerate(test_ds):

        for _ in range(eval_epochs):
            loss, acc = fewshot_epoch(frn,
                                      next(test_iter),
                                      query_k, device)
            tqdm.set_description(f"loss:{np.array(loss):.2f}, acc: {np.array(acc):.2f}")

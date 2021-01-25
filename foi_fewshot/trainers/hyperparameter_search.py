from multiprocessing import Queue
from typing import Optional


def run_hp_search(trainer, n_trials: int, direction: str, **kwargs):
    """Performs hp search by running he trainer ~n_trails~ times.

    :param trainer: Trainer object to perform hp search on
    :param n_trails: Number of trials to perform
    :param direction: Wherther to use `minimize` or `maximize` when finding
        optimal parameters

    :returns: BestTrail results from the study
    """

    import optuna

    timeout = kwargs.pop('timeout', None)
    n_jobs = kwargs.pop('n_jobs', 1)

    # Create a queue to use for distributing device ids
    gpu_queue: Optional[Queue] = None
    if trainer.args.device_ids is not None:
        gpu_queue = Queue()
        for device_id in trainer.args.device_ids:
            gpu_queue.put(device_id)

    def _objective(trial):
        # Gather avaiable gpu ids
        devices = None
        if gpu_queue is not None:
            devices = []
            for _ in range(gpu_queue.qsize() // n_jobs):
                devices.append(gpu_queue.get())
            trainer.args.device_ids = devices

        trainer.objective = None
        trainer.train(model_path=None, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, 'objective', None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)

        # Return the used device ids
        if devices is not None:
            for device_id in devices:
                gpu_queue.put(device_id)

        return trainer.objective

    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return best_trial

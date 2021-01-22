from multiprocessing import Manager

def run_hp_search(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)

    # Create a queue to use for distributing device ids
    gpu_queue = None
    if trainer.args.device_ids is not None:
        gpu_queue = Manager().queue()
        for device_id in trainer.args.device_ids:
            gpu_queue.put(device_id)


    def _objective(trial, checkpoint_dir=None):
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    model_path = os.path.join(checkpoint_dir, subdir)

        # Gather avaiable gpu ids
        devices = None
        if gpu_queue is not None:
            devices = []
            for _ in range(gpu_queue.qsize() // n_jobs):
                devices.append(gpu_queue.get())
            trainer.args.device_ids = devices

        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)

        # Return the used device ids
        if devices is not None:
            for device_id in devices:
                gpu_queue.push(device_id)

        return trainer.objective

    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial

    if gpu_queue is not None:
        gpu_queue.close()
    return best_trail

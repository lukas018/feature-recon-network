from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .trainer_arguments import TrainingArguments, EvaluationStrategy
from .trainer_utils import TrainerState, TrainerControl

class TrainerCallback:
    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of the initialization of the :class:`~transformers.Trainer`.
        """
        pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a prediction step.
        """
        pass


class CallbackHandler(TrainerCallback):
    """ Internal class that just calls the list of callbacks in order. """

    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warn(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warn(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
    ):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs,
    ):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == EvaluationStrategy.STEPS
            and state.global_step % args.eval_steps == 0
        ):
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True

        # Save
        if (
            not args.load_best_model_at_end
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.evaluation_strategy == EvaluationStrategy.EPOCH:
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True
        return control


class ProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


class WriterCallback(TrainerCallback):
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.writer = SummaryWriter(self.logdir)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.writer is not None:
            for key, values in kwargs["metrics"].items():
                self.writer(key, values, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.writer is not None:
            self.writer.close()

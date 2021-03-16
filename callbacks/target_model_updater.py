import copy

from .callback import Callback


class TargetModelUpdater(Callback):
    """Update the weights on the target model every `n` steps."""

    def __init__(self, target_freq: int) -> None:
        self.target_freq = target_freq
        self.num_steps = 0

    def on_train_begin(self):
        self.sync_target_model()

    def on_step_end(self, state, action, reward, next_state, done, was_random):
        self.num_steps += 1
        if self.num_steps % self.target_freq == 1:
            self.sync_target_model()

    def sync_target_model(self):
        self.runner.target_model.load_state_dict(copy.deepcopy(self.runner.model.state_dict()))
        self.runner.target_model.eval()
        for p in self.runner.target_model.parameters():
            p.requires_grad = False

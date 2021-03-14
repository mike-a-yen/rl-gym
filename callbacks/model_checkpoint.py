import os

import torch

from .callback import Callback


class ModelCheckpoint(Callback):
    def __init__(self, template: str = 'ep={episode:03d}-reward={reward}-steps={steps}.pth', best_only: bool = False) -> None:
        self.template = template
        self.best_only = best_only
        self.highest_reward = -float('inf')
        self.best_filename = None

    def on_episode_end(self):
        reward = sum([step['reward'] for step in self.runner.WandBLogger.history])
        run_id = self.runner.WandBLogger.run.id
        episode = self.runner.WandBLogger.num_episodes
        run_dir = self.runner.WandBLogger.run.dir
        steps = self.runner.WandBLogger.num_steps
        name = self.template.format(episode=episode, reward=reward, steps=steps, run=run_id)
        filename = os.path.join(run_dir, name)
        if self.highest_reward < reward:
            self.highest_reward = reward
            if self.best_only and self.best_filename is not None:  # remove the current best
                os.remove(self.best_filename)
            self.best_filename = filename
            self.save_model(filename)

    def save_model(self, filename: str) -> None:
        model = self.runner.model
        state = model.state_dict()
        torch.save(state, filename)

    def on_train_end(self) -> None:
        run_dir = self.runner.WandBLogger.run.dir
        name = 'terminal.pth'
        filename = os.path.join(run_dir, name)
        self.save_model(filename)

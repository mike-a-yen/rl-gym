import os

import torch

from .callback import Callback


class ModelCheckpoint(Callback):
    def __init__(self, template: str = 'ep={episode:3d}-reward={reward}-steps={steps}.pth') -> None:
        self.template = template
        self.highest_reward = -float('inf')
        self.best_model = None

    def on_episode_end(self, reward: int, randomness: float, frames: int, loss: float):
        run_id = self.runner.WandBLogger.run.id
        episode = self.runner.WandBLogger.num_episodes
        run_dir = self.runner.WandBLogger.run.dir
        steps = self.runner.WandBLogger.num_steps
        name = self.template.format(episode=episode, reward=reward, loss=loss, steps=steps, run=run_id)
        filename = os.path.join(run_dir, name)
        self.save_model(filename)
        if self.highest_reward < reward:
            self.highest_reward = reward
            self.best_model = filename

    def save_model(self, filename: str) -> None:
        model = self.runner.model
        state = model.state_dict()
        torch.save(state, filename)

    def on_train_end(self) -> None:
        run_dir = self.runner.WandBLogger.run.dir
        name = 'terminal.pth'
        filename = os.path.join(run_dir, name)
        self.save_model(filename)

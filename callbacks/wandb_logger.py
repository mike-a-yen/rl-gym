import wandb

from .callback import Callback


class WandBLogger(Callback):
    def __init__(self, project: str, **kwargs) -> None:
        self.run = wandb.init(project=project, **kwargs)
        self.num_episodes = 0
        self.num_steps = 0
        self.history = []
        self.highest_reward = -float('inf')

    def on_train_begin(self) -> None:
        self.run.watch(self.runner.model)

    def on_episode_begin(self):
        self.history = []

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        self.num_steps += 1
        self.history.append([action, reward, done, was_random])
        self.run.log(
            {
                'step_reward': reward,
                'done': done,
                'action_step': self.num_steps
            },
            step=self.num_steps
        )

    def on_episode_end(self, reward: int, randomness: float, frames: int, loss: float) -> None:
        self.num_episodes += 1
        action_hist = wandb.Histogram([step[0] for step in self.history])
        random_hist = wandb.Histogram([int(step[3]) for step in self.history])
        log = {
                'reward': reward,
                'randomness': randomness / frames,
                'frames': frames,
                'actions': action_hist,
                'was_random': random_hist,
                'episode': self.num_episodes
            }
        self.highest_reward = max(self.highest_reward, reward)
        if loss is not None:
            log.update({'loss': loss})
        self.run.log(log, step=self.num_steps)

    def on_train_end(self):
        self.run.summary.update({'highest_reward': self.highest_reward, 'train_episodes': self.num_episodes, 'train_steps': self.num_steps})
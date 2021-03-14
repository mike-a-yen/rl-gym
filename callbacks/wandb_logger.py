import numpy as np
import plotly.graph_objects as go
import wandb

from .callback import Callback


def get_Q_value_fig(Q_values, labels):
    xlim, num_nodes = Q_values.shape
    x = np.arange(xlim)
    data = [go.Scatter(x=x, y=Q_values[:, i], name=labels[i]) for i in range(num_nodes)]
    fig = go.Figure(data=data, layout=go.Layout(xaxis_title='Frame', yaxis_title='Activation'))
    return fig


class WandBLogger(Callback):
    def __init__(self, project: str, plot_every: int = 1, **kwargs) -> None:
        self.run = wandb.init(project=project, **kwargs)
        self.plot_every = plot_every
        self.num_episodes = 0
        self.num_steps = 0
        self.history = []
        self.highest_reward = -float('inf')

    def on_train_begin(self) -> None:
        self.run.watch(self.runner.model)

    def on_episode_begin(self):
        self.history = []

    def on_action_begin(self, state, Q):
        self.history.append({'Q': Q.detach().cpu().numpy()})

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        self.num_steps += 1
        self.history[-1].update({'action': action, 'reward': reward, 'done': done, 'was_random': was_random})
        self.run.log(
            {
                'step_reward': reward,
                'done': int(done),
                'action_step': self.num_steps
            },
            step=self.num_steps
        )

    def on_episode_end(self) -> None:
        self.num_episodes += 1
        action_hist = wandb.Histogram([step['action'] for step in self.history])
        random_hist = wandb.Histogram([int(step['was_random']) for step in self.history])
        randomness = sum([int(step['was_random']) for step in self.history])
        frames = len(self.history)
        episode_reward = sum([step['reward'] for step in self.history])
        log = {
                'buffer_size': len(self.runner.ReplayBuffer),
                'reward': episode_reward,
                'randomness': randomness / frames,
                'frames': frames,
                'actions': action_hist,
                'was_random': random_hist,
                'episode': self.num_episodes
            }
        self.highest_reward = max(self.highest_reward, episode_reward)
        if self.num_episodes % self.plot_every == 0:
            Q_values = np.concatenate([step['Q'] for step in self.history])
            log.update({'Q_activations': get_Q_value_fig(Q_values, self.runner.env.unwrapped.get_action_meanings())})
        self.run.log(log, step=self.num_steps)

    def on_train_end(self):
        self.run.summary.update({'highest_reward': self.highest_reward, 'train_episodes': self.num_episodes, 'train_steps': self.num_steps})
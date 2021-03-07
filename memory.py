from collections import deque, namedtuple
import random
from typing import Union

import numpy as np

from episode import Episode

Num = Union[int, float]


class Memory:
    def __init__(self, maxlen: int = 10_000, episode_size: int = None) -> None:
        self.episode_size = episode_size
        self.memory = deque(maxlen=maxlen)
        self.memory.append(Episode(0, maxsize=episode_size))

    def remember(self, state, action, reward, next_state, done):
        self.memory[-1].append(state, action, reward, next_state, done)
        if done:
            new_id = self.memory[-1].id + 1
            new_episode = Episode(new_id, self.episode_size)

    def sample(self, rate: Num):
        """Sample episodes from memory weighted by their length.
        """
        if isinstance(rate, float):
            n = int(rate * len(self))
            return self.sample(n)
        total_steps = sum([len(episode) for episode in self.memory])
        p = [len(episode) / total_steps for episode in self.memory]
        idxs = np.random.choice(len(self.memory), n, p=p)
        return [self.memory[i] for i in idxs]

    def sample_steps(self, n: Num):
        """Uniformly sample steps from every episode.

        Parameters
        ----------
        n : int
            the number of steps to return
        """
        if isinstance(n, float):
            n = int(n * self.num_steps)
            return self.sample_steps(n)
        total_steps = sum([episode.num_steps for episode in self.memory])
        p = [episode.num_steps / total_steps for episode in self.memory]
        idxs = np.random.choice(len(self.memory), n, p=p, replace=True)
        reservoir = []
        for i in idxs:
            episode = self.memory[i]
            reservoir.append(episode.sample(1)[0])
        return reservoir

    @property
    def latest_episode(self):
        return self.memory[-1]

    @property
    def best_episode(self):
        highest_reward = -float('inf')
        best_episode = None
        for episode in self.memory:
            if episode.total_reward > highest_reward:
                highest_reward = episode.total_reward
                best_episode = episode
        return episode

    @property
    def num_episodes(self) -> int:
        n = len(self.memory)
        if self.memory[-1].is_empty():
            n -= 1
        return n

    @property
    def num_steps(self) -> int:
        return sum([len(episode) for episode in self.memory])

    def is_empty(self):
        return len(self.memory[-1].steps) == 0

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.memory[i] for i in range(k.start, k.stop, k.step)]
        return self.memory[k]
    
    def __len__(self) -> int:
        return self.num_episodes
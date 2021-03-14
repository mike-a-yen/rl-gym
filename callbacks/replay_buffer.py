from collections import deque, namedtuple
import random
from typing import Union, Optional

import numpy as np
import torch
import torch.utils as utils

from .callback import Callback



ReplayStep = namedtuple('ReplayStep', 'state,action,reward,next_state,done,was_random')
Num = Union[int, float]


class ReplayBuffer(Callback, utils.data.Dataset):
    def __init__(self, maxsize: int = None) -> None:
        self.maxsize = maxsize
        self.buffer = []
        # self.states = []  # images
        # self.actions = []
        # self.rewards = []
        # self.next_states = []  # a single image
        # self.done = []
        # self.was_random = []

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        replay_step = ReplayStep(state, action, reward, next_state, done, was_random)
        self.buffer.append(replay_step)
        # self.states.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.next_states.append(next_state)
        # self.next_state = next_state
        # self.done.append(done)
        # self.was_random.append(was_random)
        if len(self) > self.maxsize:
            self.pop(0)

    def sample(self, n: int):
        return random.sample(self.buffer, n)

    def pop(self, idx):
        return self.buffer.pop(idx)
        # self.states.pop(idx)
        # self.actions.pop(idx)
        # self.rewards.pop(idx)
        # self.next_states.pop(idx)
        # self.done.pop(idx)
        # self.was_random.pop(idx)

    def __getitem__(self, idx: int):
        return self.buffer[idx]

    def __len__(self) -> int:
        return len(self.buffer)

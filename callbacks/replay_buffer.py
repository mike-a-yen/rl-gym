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
        self.write_idx = 0

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        replay_step = ReplayStep(state, action, reward, next_state, done, was_random)
        if len(self) <= self.write_idx:
            self.buffer.append(None)
        self.buffer[self.write_idx] = replay_step
        self.write_idx += 1
        if self.write_idx >= self.maxsize:
            self.write_idx = 0

    def sample(self, n: int):
        return random.sample(self.buffer, n)

    def pop(self, idx):
        """Pop an element from the buffer.

        This is O(n) operation.
        """
        return self.buffer.pop(idx)

    def __getitem__(self, idx: int):
        return self.buffer[idx]

    def __len__(self) -> int:
        return len(self.buffer)

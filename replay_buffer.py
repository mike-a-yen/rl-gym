from collections import deque, namedtuple
import random
from typing import Union, Optional

import numpy as np
import torch
import torch.utils as utils


ReplayStep = namedtuple('ReplayStep', 'state,action,reward,next_state,done,was_random')
Num = Union[int, float]


class ReplayDataset(utils.data.Dataset):
    def __init__(self, steps: deque) -> None:
        self.steps = steps

    def __getitem__(self, idx: int):
        return self.steps[idx]
    
    def __len__(self) -> int:
        return len(self.steps)


class BatchManager:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, batch):
        states, actions, rewards, next_states, is_done, was_random = zip(*batch)
        states = torch.tensor([sample.state for sample in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([sample.action for sample in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([sample.reward for sample in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([sample.next_state for sample in batch], dtype=torch.float32, device=self.device)
        is_done = torch.tensor([sample.done for sample in batch], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, is_done


class ReplayBuffer:
    def __init__(self, maxsize: int = None) -> None:
        self.maxsize = maxsize
        self.steps = deque(maxlen=self.maxsize)

    def to_dl(self, sampling: Optional[Num], batch_size, device) -> utils.data.DataLoader:
        if sampling is None:
            ds = ReplayDataset(self.steps)
        elif isinstance(sampling, float):
            n = int(sampling * len(self))
            return self.to_dl(n)
        else:
            ds = ReplayDataset(self.sample(sampling))
        batch_manager = BatchManager(device)
        dl = utils.data.DataLoader(
            ds,
            collate_fn=batch_manager,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        return dl

    def append(self, state, action, reward, next_state, done, was_random):
        self.steps.append(
            ReplayStep(state, action, reward, next_state, done, was_random)
        )

    def sample(self, n: int):
        return random.sample(self.steps, n)

    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.steps[i] for i in range(k.start, k.stop, k.step)]
        return self.steps[k]


# TODO: make a more efficient replay buffer
# allow states to have multiple successive images
class ImageLookbackReplayBuffer(ReplayBuffer):
    def __init__(self, lookback: int, maxsize: int = None) -> None:
        self.lookback = lookback
        self.maxsize = maxsize
        self.states = deque(maxlen=self.maxsize)  # images
        self.actions = deque(maxlen=self.maxsize)
        self.rewards = deque(maxlen=self.maxsize)
        self.next_state = None  # a single image
        self.done = deque(maxlen=self.maxsize)
        self.was_random = deque(maxlen=self.maxsize)

    def append(self, state, action, reward, next_state, done, was_random) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_state = next_state
        self.done.append(done)
        self.was_random.append(was_random)

    def to_dl(self, sampling: Optional[Num], batch_size, device) -> utils.data.DataLoader:
        if sampling is None:
            ds = ReplayDataset([self[i] for i in range(len(self))])
        elif isinstance(sampling, float):
            n = int(sampling * len(self))
            return self.to_dl(n)
        else:
            ds = ReplayDataset(self.sample(sampling))
        batch_manager = BatchManager(device)
        dl = utils.data.DataLoader(
            ds,
            collate_fn=batch_manager,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        return dl

    def sample(self, n: int):
        idxs = [random.randint(0, len(self) - 1) for _ in range(n)]
        return [self[i] for i in idxs]

    def get_lookback(self):
        upper = max(-len(self), -self.lookback)
        return [self.states[-i] for i in range(-1, upper, -1)]

    def __getitem__(self, idx: int):
        lower = max(-1, idx - self.lookback - 1)
        lookback_idxs = list(range(idx, lower, -1))
        image_states = [self.states[i] for i in lookback_idxs]
        assert len(image_states) <= self.lookback + 1
        state = self.stack_and_pad_states(image_states)
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_image = self.states[idx + 1] if idx + 1 < len(self) else self.next_state
        next_state = self.stack_and_pad_states([next_image] + image_states[:-1])
        done = self.done[idx]
        was_random = self.was_random[idx]
        return ReplayStep(state, action, reward, next_state, done, was_random)

    def stack_and_pad_states(self, states):
        C, W, H = states[0].shape
        stack_shape = (C * (self.lookback + 1), W, H)
        image_stack = np.zeros(stack_shape)
        for i, state in enumerate(states):
            image_stack[i * C: (i + 1) * C] = state
        return image_stack

    def __len__(self) -> int:
        return len(self.states)

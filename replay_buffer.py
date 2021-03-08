from collections import deque, namedtuple
import random
from typing import Union, Optional

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

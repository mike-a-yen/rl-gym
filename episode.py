from collections import deque, namedtuple
import random

EpisodeStep = namedtuple('EpisodeStep', 'state,action,reward,next_state,done')


class Episode:
    def __init__(self, id: int, maxsize: int = None) -> None:
        self.id = id
        self.maxsize = maxsize
        self.steps = deque(maxlen=self.maxsize)
        self.total_reward = 0
        self.num_steps = 0

    def append(self, state, action, reward, next_state, done):
        self.steps.append(
            EpisodeStep(state, action, reward, next_state, done)
        )
        self.total_reward += reward
        self.num_steps = min(self.num_steps + 1, len(self.steps))

    def sample(self, rate: float):
        n = int(len(self) * rate)
        return random.sample(self.steps, n)

    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.steps[i] for i in range(k.start, k.stop, k.step)]
        return self.steps[k]

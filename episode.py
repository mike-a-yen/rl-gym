from collections import deque, namedtuple


EpisodeStep = namedtuple('EpisodeStep', 'state,action,reward,next_state,done')


class Episode:
    def __init__(self, id: int, maxsize: int = 1024) -> None:
        self.id = id
        self.maxsize = maxsize
        self.steps = deque(maxlen=self.maxsize)

    def append(self, state, action, reward, next_state, done):
        self.steps.append(
            EpisodeStep(state, action, reward, next_state, done)
        )
    
    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.steps[i] for i in range(k.start, k.stop, k.step)]
        return self.steps[k]

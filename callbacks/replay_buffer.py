from collections import deque, namedtuple
import random

from .callback import Callback


EpisodeStep = namedtuple('EpisodeStep', 'state,action,reward,next_state,done,was_random')


class ReplayBuffer(Callback):
    def __init__(self, maxlen: int) -> None:
        self.buffer = deque(maxlen=maxlen)

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        self.buffer.append(
            EpisodeStep(state, action, reward, next_state, done, was_random)
        )

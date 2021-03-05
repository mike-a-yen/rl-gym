from collections import deque

from episode import Episode


class Memory:
    def __init__(self, maxlen: int = 10_000, episode_size: int = 1024) -> None:
        self.memory = deque(maxlen=maxlen)
        self.memory.append(Episode(0, maxsize=episode_size))

    def remember(self, state, action, reward, next_state, done):
        self.memory[-1].append(state, action, reward, next_state, done)
        if done:
            new_id = self.memory[-1].id + 1
            new_episode = Episode(new_id)

    @property
    def latest_episode(self):
        return self.memory[-1]

    @property
    def num_episodes(self) -> int:
        n = len(self.memory)
        if self.memory[-1].is_empty():
            n -= 1
        return n

    def is_empty(self):
        return len(self.memory[-1].steps) == 0

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.memory[i] for i in range(k.start, k.stop, k.step)]
        return self.memory[k]
    
    def __len__(self) -> int:
        return self.num_episodes
from typing import List

import gym
import numpy as np


class ActionLimiter(gym.ActionWrapper):
    def __init__(self, env, allowed_actions: List[str]) -> None:
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self._decoded_actions = []
        self.action_meanings = []
        for act in allowed_actions:
            assert act in action_meanings
            idx = action_meanings.index(act)
            self._decoded_actions.append(idx)
            self.action_meanings.append(action_meanings[idx])
        self.action_space = gym.spaces.Discrete(len(allowed_actions))

    def action(self, act: int) -> int:
        return self._decoded_actions[act]


class CropImage(gym.ObservationWrapper):
    def __init__(self, env, start: int, end: int, dim: int) -> None:
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.start = start if start is not None else 0
        self.end = end if end is not None else old_shape[dim]
        self.dim = dim
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        return self.crop(obs)

    def crop(self, obs):
        if self.dim == 0:
            img = obs[self.start: self.end]
        elif self.dim == 1:
            img = obs[:, self.start: self.end]
        elif self.dim == 2:
            img = obs[:, :, self.start: self.end]
        else:
            assert False, f"Invalid crop dimension {self.dim}"
        return img

    @property
    def shape(self):
        old_shape = self.env.observation_space.shape
        new_shape = []
        for dim in range(len(old_shape)):
            if dim == self.dim:
                size = self.end - self.start
            else:
                size = old_shape[dim]
            new_shape.append(size)
        return tuple(new_shape)


class FireOnResetWrapper(gym.Wrapper):
    def reset(self):
        self.env.reset()
        fire_action = self.unwrapped.get_action_meanings().index('FIRE')
        obs, _, done, _ = self.env.step(fire_action)
        if done:
            return self.reset()
        return obs


class UnitRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0
        return reward


class DeathPenaltyRewardWrapper(gym.Wrapper):
    def step(self, action):
        next_state, reward, done, meta = self.env.step(action)
        if done:
            reward = -1
        return next_state, reward, done, meta




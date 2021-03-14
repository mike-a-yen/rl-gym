import gym


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

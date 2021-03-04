import logging
import os

import gym
import hydra
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from gym_model import QGymModel
from agent import Agent


log = logging.getLogger(__file__)


class World:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.env = gym.make(self.cfg.env.name)

        self.logger = wandb.init(
            project=self.cfg.env.name,
            config=OmegaConf.to_container(self.cfg),
            monitor_gym=True
        )
        log.info(f'Logging to {self.logger.dir}')

        self.cfg.model.input_size = self.env.observation_space.shape[0]
        self.cfg.model.output_size = self.env.action_space.n
        self.agent = Agent(self.cfg, self.logger)

        self.num_episodes = 0

    def log_state(self, state):
        log = {'step': self.agent.num_steps}
        for desc, val in zip(self.cfg.env.states, state):
            log.update({desc: val})
        self.logger.log(log)

    def run_episode(self, fit: bool = True, render: bool = False) -> None:
        state, done = self.env.reset(), False
        total_reward = 0
        step = 0
        randomness = 0
        while not done:
            if render:
                self.env.render()
            self.log_state(state)
            action, was_random = self.agent.get_action(state)
            next_state, reward, done, meta = self.env.step(action) # take a random action
            reward = -1 if done else reward
            self.logger.log({'reward': reward, 'done': done})
            self.agent.remember(state, action, reward, next_state, done)
            step += 1
            total_reward += 1
            randomness += was_random
            state = next_state
        if fit:
            batch = [self.agent.memory[-i] for i in range(step, 0, -1)]
            episode_loss = self.agent.train_on_batch(batch)
        self.num_episodes += 1
        return total_reward, step, randomness


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    world = World(cfg)

    print('Before training')
    for _ in range(10):
        R, timesteps, randomness = world.run_episode(fit=False, render=True)
        print(f'R: {R} ({timesteps} @ {randomness})')

    for episode in tqdm(range(cfg.agent.num_episodes), unit='episode'):
        world.run_episode(fit=True, render=False)

    world.agent.epsilon = 0
    print(f'After training {cfg.agent.num_episodes} episodes.')
    for _ in range(10):
        R, timesteps, randomness = world.run_episode(fit=False, render=True)
        print(f'R: {R} ({timesteps} @ {randomness})')

    world.env.close()


if __name__ == '__main__':
    main()
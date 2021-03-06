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
            monitor_gym=False,
            tags=self.cfg.settings.tags
        )
        log.info(f'Logging to {self.logger.dir}')

        self.cfg.model.input_size = self.env.observation_space.shape
        self.cfg.model.output_size = self.env.action_space.n
        self.agent = Agent(self.cfg, self.logger)

        self.num_episodes = 0
        self.highest_reward = -1 * float('inf')

    def log_state(self, state):
        if self.cfg.env.get('states') is None:
            return
        log = {'step': self.agent.num_steps}
        for desc, val in zip(self.cfg.env.states, state):
            log.update({desc: val})
        self.logger.log(log)

    def run_episode(self, explore_steps: int = None, fit: bool = True, render: bool = False) -> None:
        """
        Run a single episode:
            1. Reset environment
            2. Loop over actions
            3. Once done, optionally fit on the episode memory
        """
        state, done = self.env.reset(), False
        total_reward = 0
        step = 0
        randomness = 0
        while not done and step < self.cfg.env.get('max_frames', float('inf')):
            if render:
                self.env.render()
            self.log_state(state)
            action, was_random = self.agent.get_action(state)
            next_state, reward, done, meta = self.env.step(action) # take a random action
            exploring = explore_steps is not None and step < explore_steps
            if exploring:
                done = False
            self.logger.log({'reward': reward, 'done': done, 'exploring': exploring})
            self.agent.remember(state, action, reward, next_state, done)
            step += 1
            total_reward += reward
            randomness += was_random
            state = next_state
        episode_loss = None
        if fit:
            batch = [self.agent.memory[-i] for i in range(step, 0, -1)]
            episode_loss = self.agent.train_on_batch(batch)
        self.num_episodes += 1
        self.highest_reward = max(self.highest_reward, total_reward)
        return total_reward, episode_loss, step, randomness


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    world = World(cfg)

    log.info('Before training')
    for _ in range(10):
        R, loss, timesteps, randomness = world.run_episode(
            explore_steps=cfg.agent.explore_steps, fit=False, render=cfg.settings.render
        )
        log.info(f'R: {R} ({timesteps} steps @ {randomness} random)')

    pbar = tqdm(range(cfg.agent.num_episodes), desc='Episode', unit='episode')
    for episode in pbar:
        R, loss, timesteps, randomness = world.run_episode(fit=True, render=False)
        pbar.set_description(f'Episode {episode} (loss: {loss:0.4f})')
        pbar.update(1)

    world.agent.epsilon = 0
    log.info(f'After training {cfg.agent.num_episodes} episodes.')
    for _ in range(10):
        R, loss, timesteps, randomness = world.run_episode(fit=False, render=cfg.settings.render)
        log.info(f'R: {R} ({timesteps} steps @ {randomness} random)')
    world.logger.summary.update({'highest_reward': world.highest_reward})
    world.env.close()


if __name__ == '__main__':
    main()
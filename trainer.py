from collections import namedtuple
import logging

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.utils as utils
from tqdm import tqdm

from callbacks import (
    CallbackRunner,
    ConvWeightVisualizer,
    ModelCheckpoint,
    WandBLogger,
    ReplayBuffer,
    TrainStepper
)


log = logging.getLogger(__file__)

ReplayBatch = namedtuple('ReplayBatch', 'states,actions,rewards,next_states,is_done')


class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.callback_runner = CallbackRunner(
            WandBLogger(self.env.spec.id, config=OmegaConf.to_container(self.agent.cfg), plot_every=100),
            ModelCheckpoint(best_only=True),
            # ConvWeightVisualizer(plot_every=100),
            ReplayBuffer(self.agent.cfg.replay_buffer),
            # TrainStepper(self.agent.cfg.batch_size, self.agent.cfg.baby_step, self.agent.cfg.target_step),
            model=self.agent.model,
            target_model=self.agent.target_model,
            env=self.env,
        )
        self.batch_manager = BatchManager(self.agent.device)

    def train(self, num_episodes: int, eval_every: int = 100, render_every: int = 1) -> None:
        # trial and error loop
        self.callback_runner('on_train_begin')
        with tqdm(total=num_episodes, desc='Episode', unit='episode') as pbar:
            for episode in range(num_episodes):
                render = episode % render_every == 0
                R = self.run_episode(render=render, baby_step=self.agent.cfg.baby_step)
                pbar.set_description(f'Episode {episode}')
                pbar.update(1)
                if episode % eval_every == 0:
                    self.callback_runner('on_eval_begin')
                    eval_reward = 0
                    for _ in range(self.agent.cfg.eval_episodes):
                        R = self.eval_episode()
                        eval_reward += R
                    log.info(f'Eval Reward: {eval_reward/self.agent.cfg.eval_episodes:0.2f}')
                    self.callback_runner('on_eval_end')
        self.callback_runner('on_train_end')

    def run_episode(self, render: bool = False, baby_step: int = 0, time_limit: int = 10000) -> None:
            """
            Run a single episode:
                1. Reset environment
                2. Loop over actions
                3. Once done, optionally fit on the episode memory
            """
            self.callback_runner('on_episode_begin')
            state, done = self.env.reset(), False
            total_reward, step = 0, 0
            while not done and step < time_limit:
                if render:
                    self.env.render()
                Q = self.agent.get_Q(state)
                self.callback_runner('on_action_begin', state, Q)
                action, was_random = self.agent.get_action(Q)
                self.callback_runner('on_step_begin', action, was_random)
                next_state, reward, done, meta = self.env.step(action) # take a random action
                self.callback_runner('on_step_end', state, action, reward, next_state, done, was_random)
                run_baby_step = (
                    baby_step != 0 and \
                    self.callback_runner.WandBLogger.num_steps % baby_step == 0
                    )
                if run_baby_step and len(self.callback_runner.ReplayBuffer) > self.agent.cfg.min_replay_buffer_size:
                    sync_target = self.callback_runner.WandBLogger.num_steps % self.agent.cfg.target_step == 1
                    replay_batch = self.get_replay_batch(self.agent.cfg.batch_size)
                    self.agent.train_on_step(
                        replay_batch.states, replay_batch.actions, replay_batch.rewards, replay_batch.next_states, replay_batch.is_done,
                        sync_target=sync_target
                    )
                step += 1
                total_reward += reward
                state = next_state
            self.callback_runner('on_episode_end')
            return total_reward

    def get_replay_batch(self, size: int):
        batch = self.callback_runner.ReplayBuffer.sample(size)
        states, actions, rewards, next_states, is_done, *_ = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=self.agent.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.agent.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.agent.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.agent.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.agent.device)
        return ReplayBatch(states, actions, rewards, next_states, is_done)

    def eval_episode(self, render: bool = True, time_limit: int = 10000) -> None:
            self.callback_runner('on_episode_begin')
            state, done = self.env.reset(), False
            total_reward, step = 0, 0
            while not done and step < time_limit:
                if render:
                    self.env.render()
                Q = self.agent.get_Q(state)
                self.callback_runner('on_action_begin', state, Q)
                action, was_random = self.agent.get_action(Q, epsilon=self.agent.cfg.epsilon.eval)
                self.callback_runner('on_step_begin', action, was_random)
                next_state, reward, done, meta = self.env.step(action) # take a random action
                self.callback_runner('on_step_end', state, action, reward, next_state, done, was_random)
                step += 1
                total_reward += reward
                state = next_state
            self.callback_runner('on_episode_end')
            return total_reward


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
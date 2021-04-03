from collections import namedtuple
import logging

import gym
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from tqdm import tqdm
import wandb

from callbacks import (
    CallbackRunner,
    ModelCheckpoint,
    WandBLogger,
    ReplayBuffer,
    TargetModelUpdater
)


log = logging.getLogger(__file__)

ReplayBatch = namedtuple('ReplayBatch', 'states,actions,rewards,next_states,is_done')


def configure_optimizer(cfg, model):
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam
    elif cfg.optimizer == 'sgd':
        optimizer = optim.SGD
    elif cfg.optimizer == 'rmsprop':
        optimizer = optim.RMSprop
    return optimizer(model.parameters(), cfg.lr)


def configure_loss(cfg):
    if cfg.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif cfg.loss == 'mse':
        criterion = nn.MSELoss()
    return criterion


class Trainer:
    def __init__(self, env, agent, trainer_cfg) -> None:
        self.cfg = trainer_cfg
        self.env = env
        self.agent = agent

        run_cfg = {'agent': OmegaConf.to_container(self.agent.cfg), 'trainer': OmegaConf.to_container(self.cfg)}
        self.callback_runner = CallbackRunner(
            WandBLogger(self.env.spec.id, config=run_cfg, plot_every=100),
            ModelCheckpoint(best_only=True),
            ReplayBuffer(self.cfg.replay_buffer),
            TargetModelUpdater(self.cfg.target_step),  # runs weight sync for target model
            model=self.agent.model,
            target_model=self.agent.target_model,
            env=self.env,
        )

        self.eval_env = gym.wrappers.Monitor(
            self.env,
            self.callback_runner.WandBLogger.run.dir,
            video_callable=lambda episode_id: True,
            mode='evaluation'
        )

        self.optimizer = configure_optimizer(self.cfg, self.agent.model)
        self.criterion = configure_loss(self.cfg)
        self.train_steps = 0

    def train(self, num_episodes: int, eval_every: int = 100, render_every: int = 1) -> None:
        # trial and error loop
        self.callback_runner('on_train_begin')
        with tqdm(total=num_episodes, desc='Episode', unit='episode') as pbar:
            for episode in range(num_episodes):
                render = render_every > 0 and episode % render_every == 0
                R = self.run_episode(render=render, baby_step=self.cfg.baby_step, time_limit=self.cfg.episode_time_limit)
                pbar.set_description(f'Episode {episode}')
                pbar.update(1)
                if episode % eval_every == 0:
                    self.callback_runner('on_eval_begin')
                    eval_reward = 0
                    for _ in range(self.cfg.eval_episodes):
                        R = self.eval_episode(render=render_every > 0, time_limit=self.cfg.episode_time_limit)
                        eval_reward += R
                    log.info(f'Eval Reward: {eval_reward/self.cfg.eval_episodes:0.2f}')
                    self.callback_runner('on_eval_end')
        self.callback_runner('on_train_end')

    def run_episode(self, render: bool = False, baby_step: int = 0, time_limit: int = 1000) -> None:
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
                eps = self.agent.epsilon.get(self.train_steps)
                action, was_random = self.agent.get_action(Q, epsilon=eps)
                self.callback_runner('on_step_begin', action, was_random)
                next_state, reward, done, meta = self.env.step(action)
                self.callback_runner('on_step_end', state, action, reward, next_state, done, was_random)
                run_baby_step = (baby_step != 0 and self.num_steps % baby_step == 0)
                if run_baby_step and self.replay_buffer_size > self.cfg.min_replay_buffer_size:
                    replay_batch = self.get_replay_batch(self.cfg.batch_size)
                    self.train_on_batch(replay_batch)
                step += 1
                total_reward += reward
                state = next_state
            self.callback_runner('on_episode_end')
            return total_reward

    def train_on_batch(self, batch):
        states, actions, rewards, next_states, is_done = batch
        action_Q = self.agent.model(states).gather(dim=1, index=actions.unsqueeze(dim=1))  # (B, 1)
        next_Q = self.agent.target_model(next_states).detach()

        Qnew = rewards + (1. - is_done) * self.agent.gamma * torch.max(next_Q, dim=1).values  # (B,)
        Qnew = Qnew.unsqueeze(1)

        self.optimizer.zero_grad()
        loss = self.criterion(Qnew, action_Q)
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1
        return loss

    def get_replay_batch(self, size: int):
        batch = self.callback_runner.ReplayBuffer.sample(size)
        states, actions, rewards, next_states, is_done, *_ = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=self.agent.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.agent.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.agent.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.agent.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.agent.device)
        return ReplayBatch(states, actions, rewards, next_states, is_done)

    def eval_episode(self, render: bool = True, time_limit: int = 100000) -> None:
        self.callback_runner('on_episode_begin')
        state, done = self.eval_env.reset(), False
        total_reward, step = 0, 0
        while not done and step < time_limit:
            if render:
                self.eval_env.render()
            Q = self.agent.get_Q(state)
            self.callback_runner('on_action_begin', state, Q)
            action, was_random = self.agent.get_action(Q, epsilon=self.agent.cfg.epsilon.eval)
            self.callback_runner('on_step_begin', action, was_random)
            next_state, reward, done, meta = self.eval_env.step(action) # take a random action
            self.callback_runner('on_step_end', state, action, reward, next_state, done, was_random)
            step += 1
            total_reward += reward
            state = next_state
        self.callback_runner('on_episode_end')
        video_filename = self.eval_env.video_recorder.path
        self.callback_runner.WandBLogger.run.log({'replay': wandb.Video(video_filename, fps=4, format="gif")})
        return total_reward

    @property
    def replay_buffer_size(self) -> int:
        return len(self.callback_runner.ReplayBuffer)

    @property
    def num_steps(self) -> int:
        return self.callback_runner.WandBLogger.num_steps

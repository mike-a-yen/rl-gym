from collections import deque, namedtuple
import copy
import logging
import random

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from gym_model import QGymModel


log = logging.getLogger(__file__)
Memory = namedtuple('Memory', 'state,action,reward,next_state,done')

MIN_REPLAY_SIZE = 2056
UPDATE_TARGET_DELAY = 128  # in batches


class Agent:
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        model_cfg = cfg.model
        self.model = QGymModel(model_cfg.input_size, model_cfg.hidden_size, model_cfg.output_size)
        self.target_model = QGymModel(model_cfg.input_size, model_cfg.hidden_size, model_cfg.output_size)
        self.sync_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), cfg.agent.lr)
        self.criterion = nn.MSELoss()

        self.gamma = cfg.agent.gamma
        self.epsilon = cfg.agent.epsilon
        self.epsilon_decay = cfg.agent.epsilon_decay
        self.batch_size = cfg.agent.batch_size

        self.memory = deque(maxlen=cfg.agent.memory_size)
        self.num_steps = 0  # number of training steps
        self.logger = logger

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            Memory(state, action, reward, next_state, done)
        )

    def get_Qs(self, state):
        X = torch.tensor(state, dtype=torch.float32)
        return self.model(X)

    def get_action(self, state, not_random: bool = False):
        was_random = False
        rand_action_thresh = self.epsilon * (self.epsilon_decay**self.num_steps)
        self.logger.log({'rand_action_threshold': rand_action_thresh, 'step': self.num_steps})
        if not not_random and random.random() < rand_action_thresh:  # select random
            action = random.randint(0, self.model.output_size - 1)
            was_random = True
        else:
            Q = self.get_Qs(state)
            action = int(Q.argmax().item())
        self.logger.log({'action': action, 'was_random': was_random, 'step': self.num_steps})
        return action, was_random        

    def tensorify_batch(self, batch):
        states = torch.tensor([sample.state for sample in batch], dtype=torch.float32)
        actions = torch.tensor([sample.action for sample in batch], dtype=torch.long)
        rewards = torch.tensor([sample.reward for sample in batch], dtype=torch.float32)
        next_states = torch.tensor([sample.next_state for sample in batch], dtype=torch.float32)
        is_done = torch.tensor([sample.done for sample in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, is_done

    def train_on_batch(self, batch):
        batch_size = len(batch)
        idx = torch.arange(batch_size)
        states, actions, rewards, next_states, is_done = self.tensorify_batch(batch)

        current_Q = self.model(states)
        next_Q = self.target_model(next_states)

        Qnew = rewards + (1. - is_done) * self.gamma * next_Q.max(dim=1).values

        self.optimizer.zero_grad()
        loss = self.criterion(Qnew, current_Q[idx, actions])
        loss.backward()
        self.optimizer.step()
        self.num_steps += 1
        if self.num_steps > 0 and self.num_steps % UPDATE_TARGET_DELAY == 0:
            self.sync_target_model()
        self.logger.log({'batch_size': batch_size, 'step_loss': loss.item(), 'step': self.num_steps})
        return loss

    def train_step(self) -> None:
        if len(self.memory) < self.cfg.agent.min_replay_size:
            return
        batch = self.sample_from_memory(self.batch_size, 'linear')
        return self.train_on_batch(batch)        

    def sync_target_model(self):
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def sample_from_memory(self, n_samples: int, method: str):
        N = len(self.memory)
        if method == 'linear':
            samples = np.random.triangular(0, N, N, n_samples)
            idxs = samples.astype(int)
            batch = [self.memory[idx] for idx in idxs]
        elif method == 'uniform':
            batch = random.choice(self.memory, n_samples)
        else:
            raise ValueError('Invalid choice for `agent.batch.method`')
        return batch

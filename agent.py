import copy
import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from tqdm import tqdm

from decay_schedule import LinearDecay
from hooks import visualize_Q_values


log = logging.getLogger(__file__)


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


class Agent:
    def __init__(self, model, target_model, agent_cfg) -> None:
        self.cfg = agent_cfg
        self.model = model
        self.target_model = target_model
        self.sync_target_model()
        self.device = list(self.model.parameters())[0].device

        self.optimizer = configure_optimizer(self.cfg, self.model)
        self.batch_size = self.cfg.batch_size
        self.criterion = configure_loss(self.cfg)

        self.gamma = self.cfg.gamma
        self.epsilon = LinearDecay(self.cfg.epsilon.start, self.cfg.epsilon.stop, self.cfg.epsilon.steps)

        self.num_steps = 0

    @torch.no_grad()
    def get_Q(self, state: np.ndarray) -> torch.Tensor:
        X = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.model(X)

    def get_action(self, Q, epsilon: Optional[float] = None):
        """Epsilon greedy policy."""
        was_random = False
        rand_action_thresh = self.epsilon.get(self.num_steps) if epsilon is None else epsilon
        if random.random() < rand_action_thresh:  # select random
            action = random.randint(0, self.model.output_size - 1)
            was_random = True
        else:
            action = int(Q.argmax().item())
        return action, was_random

    def train_on_step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, is_done: torch.Tensor, sync_target: bool = False):
        B = states.shape[0]
        idx = torch.arange(B, device=self.device)

        current_Q = self.model(states)
        next_Q = self.target_model(next_states).detach()

        Qnew = rewards + (1. - is_done) * self.gamma * next_Q.max(dim=1).values

        self.optimizer.zero_grad()
        loss = self.criterion(Qnew, current_Q[idx, actions])
        loss.backward()
        self.optimizer.step()
        self.num_steps += 1
        if sync_target:
            self.sync_target_model()
        return loss

    def train_on_dl(self, dl, sync_target: bool = False, silence: bool = False) -> float:
        episode_loss = 0
        num_samples = len(dl.dataset)
        num_batches = int(np.ceil(num_samples / dl.batch_size))
        with tqdm(total=num_batches, desc='Train steps', unit='sample', leave=False, disable=silence) as pbar:
            for i, step_data in enumerate(dl):
                step_loss = self.train_on_step(*step_data, sync_target=sync_target)
                episode_loss += float(step_loss.item())
                pbar.set_description(f'Train steps (loss: {episode_loss / (i + 1):0.4f}')
                pbar.update(1)
            if sync_target:
                self.sync_target_model()
        return episode_loss / num_batches

    def sync_target_model(self):
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False
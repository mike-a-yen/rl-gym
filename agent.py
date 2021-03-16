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


class Agent:
    def __init__(self, model, target_model, agent_cfg) -> None:
        self.cfg = agent_cfg
        self.model = model
        self.target_model = target_model
        self.device = list(self.model.parameters())[0].device

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

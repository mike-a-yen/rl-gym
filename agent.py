import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory import Memory

class Agent:
    def __init__(self, model, agent_cfg) -> None:
        self.cfg = agent_cfg
        self.model = model
        self.device = list(self.model.parameters())[0].device
        self.memory = Memory(
            maxlen=1000,
            episode_size=2048
        )

        self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr)
        self.criterion = nn.MSELoss()

        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon
        self.epsilon_decay = self.cfg.epsilon_decay
        
        self.num_steps = 0
    
    def get_Qs(self, state: np.ndarray) -> torch.Tensor:
        device = torch.device(self.device)
        X = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return self.model(X)

    def get_action(self, state, not_random: bool = False):
        was_random = False
        rand_action_thresh = self.epsilon * (self.epsilon_decay**self.num_steps)
        if not not_random and random.random() < rand_action_thresh:  # select random
            action = random.randint(0, self.model.output_size - 1)
            was_random = True
        else:
            Q = self.get_Qs(state)
            action = int(Q.argmax().item())
        return action, was_random        

    def tensorify_batch(self, batch):
        device = torch.device(self.device)
        states = torch.tensor([sample.state for sample in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([sample.action for sample in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([sample.reward for sample in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor([sample.next_state for sample in batch], dtype=torch.float32, device=device)
        is_done = torch.tensor([sample.done for sample in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, is_done

    def train_on_batch(self, batch):
        batch_size = len(batch)
        idx = torch.arange(batch_size, device=self.device)
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
        return loss  

    def sync_target_model(self):
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
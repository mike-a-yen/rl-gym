import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from tqdm import tqdm

from episode import Episode
from memory import Memory


log = logging.getLogger(__file__)


class EpisodeDataset(utils.data.Dataset):
    def __init__(self, episode: Episode) -> None:
        self.episode = episode
    
    def __getitem__(self, idx: int):
        return self.episode[idx]
    
    def __len__(self) -> int:
        return len(self.episode)


class EpisodeBatchManager:
    def __init__(self, device) -> None:
        self.device = device

    def __call__(self, batch):
        states, actions, rewards, next_states, is_done = zip(*batch)
        states = torch.tensor([sample.state for sample in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([sample.action for sample in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([sample.reward for sample in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([sample.next_state for sample in batch], dtype=torch.float32, device=self.device)
        is_done = torch.tensor([sample.done for sample in batch], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, is_done


class Agent:
    def __init__(self, model, target_model, agent_cfg) -> None:
        self.cfg = agent_cfg
        self.model = model
        self.target_model = target_model
        self.sync_target_model()
        self.device = list(self.model.parameters())[0].device
        self.memory = Memory(
            maxlen=128,  # max number of episodes to store
            episode_size=4096  # max number of frames per episode
        )

        self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr)
        self.batch_size = self.cfg.batch_size
        self.criterion = nn.MSELoss()

        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon
        self.epsilon_decay = self.cfg.epsilon_decay

        self.num_steps = 0
        self.num_episodes = 0

    @torch.no_grad()
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

    def train_on_step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, is_done: torch.Tensor, sync_target: bool = False):
        B = states.shape[0]
        idx = torch.arange(B, device=self.device)

        current_Q = self.model(states)
        next_Q = self.target_model(next_states)

        Qnew = rewards + (1. - is_done) * self.gamma * next_Q.max(dim=1).values

        self.optimizer.zero_grad()
        loss = self.criterion(Qnew, current_Q[idx, actions])
        loss.backward()
        self.optimizer.step()
        self.num_steps += 1
        if sync_target:
            self.sync_target_model()
        return loss

    def episode_to_dl(self, episode: Episode) -> utils.data.DataLoader:
        ds = EpisodeDataset(episode)
        batch_manager = EpisodeBatchManager(self.device)
        dl = utils.data.DataLoader(
            ds,
            collate_fn=batch_manager,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        return dl

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

    def train_on_episode(self, episode: Episode, sampling: float, **kwargs):
        if sampling is None:
            dl = self.episode_to_dl(episode)
        else:
            synthetic_episode = Episode(-1, None)
            for step in episode.sample(sampling):
                synthetic_episode.append(step.state, step.action, step.reward, step.next_state, step.done)
            dl = self.episode_to_dl(synthetic_episode)
        self.num_episodes += 1
        return self.train_on_dl(dl, sync_target=True, **kwargs)

    def train_on_memory(self, epochs: int = 1, sampling: float = None, sync_target: bool = False, **kwargs):
        synthetic_episode = Episode(-1)
        if sampling is None:  # get all steps in memory
            for episode in self.memory:
                for step in episode:
                    synthetic_episode.append(step.state, step.action, step.reward, step.next_state, step.done)
        else:
            for step in self.memory.sample_steps(sampling):
                synthetic_episode.append(step.state, step.action, step.reward, step.next_state, step.done)
        dl = self.episode_to_dl(synthetic_episode)
        for _ in range(epochs):
            loss = self.train_on_dl(dl, sync_target=sync_target, **kwargs)
        return loss

    def sync_target_model(self):
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.eval()
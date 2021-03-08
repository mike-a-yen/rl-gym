import logging

from omegaconf import OmegaConf
import torch
import torch.utils.data as data
from tqdm import tqdm

from callbacks import CallbackRunner, ModelCheckpoint, WandBLogger, ReplayBuffer


log = logging.getLogger(__file__)


class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.callback_runner = CallbackRunner(
            WandBLogger(self.env.spec.id, config=OmegaConf.to_container(self.agent.cfg)),
            ModelCheckpoint(),
            # ReplayBuffer(self.agent.cfg.replay_buffer),  TODO: have agent train off of this buffer rather than its own memory
            model=self.agent.model
        )

    def train(self, num_episodes: int, fit_episode: bool = False, render_every: int = 1) -> None:
        # trial and error loop
        self.callback_runner('on_train_begin')
        with tqdm(total=num_episodes, desc='Episode', unit='episode') as pbar:
            for episode in range(num_episodes):
                render = episode % render_every == 0
                total_reward, loss, step, randomness = self.run_episode(fit=fit_episode, render=render, baby_step=self.agent.cfg.baby_step)
                if loss is not None:
                    pbar.set_description(f'Episode {episode} (loss: {loss:0.4f})')
                else:
                    pbar.set_description(f'Episode {episode}')
                pbar.update(1)
        self.callback_runner('on_train_end')

    def state_transformation(self, state):
        """Normalize and transpose the image to (C, W, H)."""
        return (state / 255.).transpose((2, 0, 1))  # normalize image

    def run_episode(self, fit: bool = True, render: bool = False, baby_step: int = 0) -> None:
            """
            Run a single episode:
                1. Reset environment
                2. Loop over actions
                3. Once done, optionally fit on the episode memory
            """
            self.callback_runner('on_episode_begin')
            state, done = self.env.reset(), False
            state = self.state_transformation(state)
            total_reward = 0
            step = 0
            randomness = 0
            while not done:
                if render:
                    self.env.render()
                action, was_random = self.agent.get_action(state)
                next_state, reward, done, meta = self.env.step(action) # take a random action
                next_state = self.state_transformation(next_state)
                self.agent.memory.append(state, action, reward, next_state, done, was_random)
                self.callback_runner('on_step_end', state, action, reward, next_state, done, was_random)
                run_baby_step = baby_step != 0 and step % baby_step == 0
                if run_baby_step and self.callback_runner.WandBLogger.num_steps > self.agent.batch_size:
                    sync_target = self.callback_runner.WandBLogger.num_steps % self.agent.cfg.target_step == 0
                    self.agent.train_on_memory(1, sampling=self.agent.batch_size, sync_target=sync_target, silence=True)
                # TODO: train on replay buffer callback
                # if run_baby_step and self.callback_runner.ReplayBuffer.num_steps > self.agent.batch_size:
                #     sync_target = self.agent.num_steps % self.agent.cfg.target_step == 0
                #     self.agent.train_on_memory
                step += 1
                total_reward += reward
                randomness += was_random
                state = next_state
            episode_loss = None
            if fit:  # fit on this episode
                episode_loss = self.agent.train_on_memory(1, sampling=self.agent.cfg.memory_sampling)
            self.callback_runner('on_episode_end', total_reward, randomness, step, episode_loss)
            return total_reward, episode_loss, step, randomness

import logging
from pathlib import Path

import gym
import hydra
import torch
from tqdm import tqdm

from agent import Agent
from gym_model import GymConvModel
from trainer import Trainer

log = logging.getLogger(__file__)


def get_env_shapes(env):
    input_shape = env.observation_space.shape
    output_shape = env.action_space.shape
    if len(output_shape) == 0:
        output_shape = env.action_space.n
    return input_shape, output_shape


def create_model(input_shape, output_shape, cfg):
    device = torch.device(cfg.model.device)
    num_channels = cfg.agent.get('lookback', 0) + 1
    h, w, c = input_shape[0], input_shape[1], num_channels
    return GymConvModel((h, w, c), cfg.model.convs, cfg.model.hidden_size, output_shape).to(device)


def configure_env(cfg):
    env = gym.make(cfg.env.name)
    playback_dir = Path().cwd() / 'recording'
    playback_dir.mkdir(exist_ok=True, parents=True)
    env = gym.wrappers.Monitor(env, Path.cwd() / 'recording')
    return env


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    env = configure_env(cfg)    
    input_shape, output_shape = get_env_shapes(env)

    model = create_model(input_shape, output_shape, cfg)
    target_model = create_model(input_shape, output_shape, cfg)
    agent = Agent(model, target_model, cfg.agent)
    trainer = Trainer(env, agent)

    trainer.train(agent.cfg.num_episodes, fit_episode=False, render_every=cfg.settings.render_every)
    # # trial and error loop
    # with tqdm(total=cfg.agent.num_episodes, desc='Episode', unit='episode') as pbar:
    #     for episode in range(cfg.agent.num_episodes):
    #         render = episode % cfg.settings.render_every == 0
    #         total_reward, loss, step, randomness = trainer.run_episode(fit=True, render=render, baby_step=cfg.agent.baby_step)
    #         pbar.set_description(f'Episode {episode} (loss: {loss:0.4f})')
    #         pbar.update(1)

    # check results loop
    for _ in range(5):
        trainer.run_episode(fit=False, render=True)
    log.info(f'Highest Reward: {trainer.callback_runner.WandBLogger.highest_reward}')
    env.close()
    log.info('Done.')


if __name__ == '__main__':
    main()

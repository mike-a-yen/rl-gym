import logging
from pathlib import Path

import gym
import hydra
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from agent import Agent
from gym_model import GymConvModel
from trainer import Trainer

import wrappers as wrappers


log = logging.getLogger(__file__)


def get_env_shapes(env):
    input_shape = env.observation_space.shape
    output_shape = env.action_space.shape
    if len(output_shape) == 0:
        output_shape = env.action_space.n
    return input_shape, output_shape


def create_model(input_shape, output_shape, cfg):
    device = torch.device(cfg.model.device)
    c, h, w = input_shape
    return GymConvModel((h, w, c), cfg.model.convs, cfg.model.hidden_size, output_shape).to(device)


def configure_env(cfg):
    env = gym.make(cfg.env.name)
    for wrapper_config in cfg.env.wrappers:
        env = hydra.utils.instantiate(wrapper_config, env)
    return env


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    log.info(f'\n{OmegaConf.to_yaml(cfg)}')
    env = configure_env(cfg)
    input_shape, output_shape = get_env_shapes(env)
    log.info(f'In <- {input_shape}')
    log.info(f'Out -> {output_shape}')

    model = create_model(input_shape, output_shape, cfg)
    target_model = create_model(input_shape, output_shape, cfg)
    agent = Agent(model, target_model, cfg.agent)
    trainer = Trainer(env, agent, cfg.trainer)
    trainer.callback_runner.WandBLogger.run.config.update({
        'env': OmegaConf.to_container(cfg.env),
        'model': OmegaConf.to_container(cfg.model)
    })
    trainer.train(cfg.trainer.num_episodes, cfg.trainer.eval_every, render_every=cfg.settings.render_every)

    for _ in range(5):
        trainer.eval_episode(render=True)
    log.info(f'Highest Reward: {trainer.callback_runner.WandBLogger.highest_reward}')
    trainer.env.close()
    trainer.eval_env.close()
    log.info('Done.')


if __name__ == '__main__':
    main()

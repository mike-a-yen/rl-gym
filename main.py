import logging

import gym
import hydra
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


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    env = gym.make(cfg.env.name)
    input_shape, output_shape = get_env_shapes(env)
    model = GymConvModel(input_shape, cfg.model.hidden_size, output_shape)
    target_model = GymConvModel(input_shape, cfg.model.hidden_size, output_shape)
    agent = Agent(model, target_model, cfg.agent)
    trainer = Trainer(env, agent)

    with tqdm(total=cfg.agent.num_episodes, unit='episode') as pbar:
        for episode in range(cfg.agent.num_episodes):
            trainer.run_episode(fit=True, render=True)
            pbar.update(1)

    for _ in range(5):
        trainer.run_episode(fit=False, render=True)
    #trainer.logger.summary.update({'highest_reward': world.highest_reward})
    env.close()
    log.info('Done.')


if __name__ == '__main__':
    main()

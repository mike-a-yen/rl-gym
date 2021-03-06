class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.highest_reward = 0

    def run_episode(self, fit: bool = True, render: bool = False) -> None:
            """
            Run a single episode:
                1. Reset environment
                2. Loop over actions
                3. Once done, optionally fit on the episode memory
            """
            state, done = self.env.reset(), False
            total_reward = 0
            step = 0
            randomness = 0
            while not done:
                state = state / 255.  # normalize image
                if render:
                    self.env.render()
                action, was_random = self.agent.get_action(state)
                next_state, reward, done, meta = self.env.step(action) # take a random action
                self.agent.memory.remember(state, action, reward, next_state, done)
                # self.logger.log({'reward': reward, 'done': done})
                step += 1
                total_reward += reward
                randomness += was_random
                state = next_state
            episode_loss = None
            if fit:  # fit on this episode
                batch = [self.agent.memory.latest_episode[i] for i in range(0, len(self.agent.memory.latest_episode))]
                episode_loss = self.agent.train_on_batch(batch)
                self.agent.sync_target_model()
            self.highest_reward = max(self.highest_reward, total_reward)
            return total_reward, episode_loss, step, randomness

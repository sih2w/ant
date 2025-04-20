from collections import defaultdict
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from tqdm import tqdm
import scavenging_ant.envs.scavenging_ant as scavenging_ant
import numpy as np
import matplotlib.pyplot as plt

class QAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, env, observation) -> int:
        observation = tuple(observation)
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation,
        action: int,
        reward: float,
        terminated: bool,
        next_observation,
    ):
        observation = tuple(observation)
        next_observation = tuple(next_observation)

        future_q_value = (not terminated) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[observation][action]
        )

        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    learning_rate = 0.001
    n_episodes = 50
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = scavenging_ant.ScavengingAntEnv(
        render_mode=None,
        render_fps=60,
        persistent_obstacles=True,
        persistent_food=True,
        persistent_nests=True,
        grid_height=10,
        grid_width=15,
        food_count=10,
        percent_obstacles=0.10
    )
    
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env.reset()

    agent = QAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        observation, info = env.reset(seed=0)
        done = False

        while not done:
            action = agent.get_action(env, observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.update(observation, action, reward, terminated, next_observation)

            done = terminated or truncated
            observation = next_observation

        agent.decay_epsilon()

    env.close()

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    reward_moving_average = (
            np.convolve(
                np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
            np.convolve(
                np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
            np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()
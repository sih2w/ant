import numpy as np
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from collections import defaultdict
from tqdm import tqdm
from gymnasium.wrappers import FlattenObservation, TimeLimit

def create_env(render_mode: str = None):
    return scavenging_ant.ScavengingAntEnv(
        render_mode=render_mode,
        render_fps=5,
        grid_height=10,
        grid_width=18,
        food_count=2,
        nest_count=3,
        percent_obstacles=0.10,
        seed=0
    )

if __name__ == "__main__":
    episodes = 1000
    seed = 100

    env = create_env()
    env = TimeLimit(env, max_episode_steps=500)
    env = FlattenObservation(env)

    q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards_per_episode = np.zeros(episodes)

    learning_rate_alpha = 0.10
    discount_factor_gamma = 0.90
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)
    rng = np.random.default_rng()

    for episode in tqdm(range(episodes)):
        state = env.reset(seed=seed)[0]
        state = state.tolist()
        state = tuple(state)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = tuple(new_state)

            q[state][action] = q[state][action] + learning_rate_alpha * (
                    reward + discount_factor_gamma * np.max(q[new_state]) - q[state][action]
            )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_alpha = 0.0001

    env.close()

    env = create_env(render_mode="human")
    env = TimeLimit(env, max_episode_steps=500)
    env = FlattenObservation(env)

    for episode in range(episodes):
        state = env.reset(seed=seed)[0]
        state = state.tolist()
        state = tuple(state)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q[state])
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = tuple(new_state)
            state = new_state

    env.close()
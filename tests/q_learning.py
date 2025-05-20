import numpy as np
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from collections import defaultdict
from tqdm import tqdm

def create_env(render_mode=None):
    return scavenging_ant.ScavengingAntEnv(
        render_mode=render_mode,
        render_fps=10,
        grid_height=10,
        grid_width=20,
        food_count=10,
        nest_count=1,
        seed=0,
        agent_count=2,
        percent_obstacles=0.10
    )

if __name__ == "__main__":
    episodes = 2500
    seed = 0
    env = create_env()

    q = {name: defaultdict(lambda: np.zeros(env.action_space(name).n)) for name in env.agents}
    rng = np.random.default_rng()

    learning_rate_alpha = 0.1
    discount_factor_gamma = 0.95
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)

    for episode in tqdm(range(episodes)):
        observations, _ = env.reset(seed=seed)
        observations = env.flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for name, observation in observations.items():
                if rng.random() > epsilon:
                    actions[name] = np.argmax(q[name][observations[name]])
                else:
                    actions[name] = env.action_space(name).sample()

            new_observations, rewards, terminations, truncations, _ = env.step(actions)
            new_observations = env.flatten_observations(new_observations)

            # if env.get_step_count() >= 1000:
            #     rewards = {name: -1000 for name in env.agents}

            for _, termination in terminations.items():
                terminated = termination
                if terminated:
                    break
            else:
                for _, truncation in truncations.items():
                    truncated = truncation
                    if truncated:
                        break

            for name, new_observation in new_observations.items():
                q[name][observations[name]][actions[name]] = q[name][observations[name]][actions[name]] + learning_rate_alpha * (
                        rewards[name] + discount_factor_gamma * np.max(q[name][new_observation]) - q[name][observations[name]][actions[name]]
                )

            observations = new_observations

        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

    env.close()
    env = create_env(render_mode="human")

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed)
        observations = env.flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {name: np.argmax(q[name][observations[name]]) for name in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(actions)
            observations = env.flatten_observations(observations)

            for _, termination in terminations.items():
                terminated = termination
                if terminated:
                    break
            else:
                for _, truncation in truncations.items():
                    truncated = truncation
                    if truncated:
                        break

    env.close()
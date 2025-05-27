import numpy as np
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

def create_env(render_mode: str = None, seed: int = 0, grid_width: int = 10, grid_height: int = 5):
    return scavenging_ant.ScavengingAntEnv(
        render_mode=render_mode,
        render_fps=5,
        grid_height=grid_height, # 5
        grid_width=grid_width, # 7
        seed=seed,
        food_count=7, # 10
        nest_count=1, # 1
        agent_count=3, # 2
        percent_obstacles=0.30 # 0.10
    )

def flatten_observations(observations):
    for name, observation in observations.items():
        observations[name] = (
        observation["position"][0],
        observation["position"][1],
        observation["carrying_food"]
    )

    return observations

if __name__ == "__main__":
    episodes = 5_000
    seed = 0
    grid_width = 7
    grid_height = 5

    env = create_env(
        render_mode=None,
        seed=seed,
        grid_width=grid_width,
        grid_height=grid_height
    )

    q = {name: defaultdict(lambda: np.zeros(env.action_space(name).n)) for name in env.agents}
    rng = np.random.default_rng()

    learning_rate_alpha = 0.01
    discount_factor_gamma = 0.70
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)

    episode = 0
    pbar = tqdm(total=episodes)

    while episode < episodes:
        temp_q = deepcopy(q)
        observations, _ = env.reset(seed=seed)
        observations = flatten_observations(observations)
        terminated, truncated, use_episode = False, False, True

        while not terminated and not truncated:
            actions = {}
            for name, observation in observations.items():
                if rng.random() > epsilon:
                    actions[name] = np.argmax(temp_q[name][observations[name]])
                else:
                    actions[name] = env.action_space(name).sample()

            new_observations, rewards, terminations, truncations, _ = env.step(actions)
            new_observations = flatten_observations(new_observations)

            use_episode = env.get_step_count() <= 1000
            if not use_episode:
                break

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
                temp_q[name][observations[name]][actions[name]] = temp_q[name][observations[name]][actions[name]] + learning_rate_alpha * (rewards[name] + discount_factor_gamma * np.max(q[name][new_observation]) - temp_q[name][observations[name]][actions[name]])

            observations = new_observations

        if use_episode:
            epsilon = max(epsilon - epsilon_decay_rate, 0.01)
            q = temp_q
            pbar.update(1)
            episode += 1

    pbar.close()

    for name, agent_q in q.items():
        print(f"\nAgent: ", name)
        for observation, action_values in agent_q.items():
            print(f"State: {observation} | Down: {action_values[0]: .5f} | Up: {action_values[1]: .5f} | Left: {action_values[2]: .5f} | Right: {action_values[3]: .5f} |")

    env.close()
    env = create_env(
        render_mode="human",
        seed=seed,
        grid_width=grid_width,
        grid_height=grid_height
    )

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed)
        observations = flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {name: np.argmax(q[name][observations[name]]) for name in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(actions)
            observations = flatten_observations(observations)

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
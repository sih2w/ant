import numpy as np
import pickle
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

def create_env(
        render_mode: str = None,
        seed: int = 0,
        grid_width: int = 10,
        grid_height: int = 5,
        agent_count: int = 1,
        food_count: int = 1,
        obstacle_count: int = 1,
        render_fps: int = 5,
):
    return scavenging_ant.ScavengingAntEnv(
        render_mode=render_mode,
        render_fps=render_fps,
        grid_height=grid_height,
        grid_width=grid_width,
        seed=seed,
        food_count=food_count,
        nest_count=1,
        agent_count=agent_count,
        obstacle_count=obstacle_count,
    )

def flatten_observations(observations):
    for name, observation in observations.items():
        positions = [*observation["agent_position"]]
        for position in observation["dropped_food_positions"]:
            positions.append(position[0])
            positions.append(position[1])

        observations[name] = (
            observation["carrying_food"],
            observation["dropped_food_count"],
            *positions
        )

    return observations

if __name__ == "__main__":
    # Learning parameters
    episodes = 2_000
    seed = 0
    learning_rate_alpha = 0.10
    discount_factor_gamma = 0.95
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)
    max_steps_per_episode = 1000

    # Environment parameters
    grid_width = 7
    grid_height = 5
    agent_count = 2
    food_count = 5
    obstacle_count = 1

    loaded_from_file = False
    file_name = (
        "q_learning_models/"
        f"learning_rate_{learning_rate_alpha}_"
        f"discount_factor_gamma_{discount_factor_gamma}_"
        f"episodes_{episodes}_"
        f"seed_{seed}_"
        f"width_{grid_width}_"
        f"height_{grid_height}_"
        f"agent_{agent_count}_"
        f"food_{food_count}_"
        f"obstacle_{obstacle_count}"
    )

    try:
        with open(file_name, "rb") as file:
            # Attempt to load a file that uses the given parameters
            q = pickle.load(file)
            loaded_from_file = True
            print("Loaded from file")

    except FileNotFoundError:
        env = create_env(
            render_mode=None,
            seed=seed,
            grid_width=grid_width,
            grid_height=grid_height,
            agent_count=agent_count,
            food_count=food_count,
            obstacle_count=obstacle_count,
            render_fps=1000
        )

        q = {agent_name: defaultdict(lambda: np.zeros(env.action_space(agent_name).n)) for agent_name in env.agents}
        rng = np.random.default_rng()

        episode = 0
        pbar = tqdm(total=episodes)

        while episode < episodes:
            episode_q = deepcopy(q) # Clone the current Q table so that this episode has all prior information.
            observations, _ = env.reset(seed=seed)
            observations = flatten_observations(observations)
            terminated, truncated, use_episode = False, False, True

            while not terminated and not truncated:
                actions = {}
                for agent_name, observation in observations.items():
                    if rng.random() > epsilon:
                        actions[agent_name] = np.argmax(episode_q[agent_name][observations[agent_name]])
                    else:
                        actions[agent_name] = env.action_space(agent_name).sample()

                new_observations, rewards, terminations, truncations, _ = env.step(actions)
                new_observations = flatten_observations(new_observations)

                use_episode = env.get_step_count() <= max_steps_per_episode
                if not use_episode:
                    # If the number of steps in this episode exceeds a maximum number of steps,
                    # terminate this episode and discard the changes made to the Q table. This
                    # should ensure that episodes that take too long are not influencing the learning.
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

                # Update the Q table for each agent.
                for agent_name, new_observation in new_observations.items():
                    episode_q[agent_name][observations[agent_name]][actions[agent_name]] = episode_q[agent_name][observations[agent_name]][actions[agent_name]] + learning_rate_alpha * (rewards[agent_name] + discount_factor_gamma * np.max(q[agent_name][new_observation]) - episode_q[agent_name][observations[agent_name]][actions[agent_name]])

                observations = new_observations

            if use_episode:
                q = episode_q  # Overwrite the current Q table with the updates made during the episode.
                epsilon = max(epsilon - epsilon_decay_rate, 0.01)
                pbar.update(1)
                episode = episode + 1

        pbar.close()
        env.close()

    if not loaded_from_file:
        # If the Q-learning model was not loaded from a file, save this Q-learning model to a file.
        with open(file_name, "wb") as file:
            # Convert the default dict object to a normal dictionary. Pickle cannot save instances.
            saved_q = {}
            for agent_name, agent_q in q.items():
                saved_q[agent_name] = {}
                for observation, action_values in agent_q.items():
                    saved_q[agent_name][observation] = action_values

            pickle.dump(saved_q, file)
            print("Saved to file")

    # Create a new environment with human rendering.
    env = create_env(
        render_mode="human",
        seed=seed,
        grid_width=grid_width,
        grid_height=grid_height,
        agent_count=agent_count,
        food_count=food_count,
        obstacle_count=obstacle_count,
    )

    # Visualize trained model.
    for episode in range(episodes):
        observations, _ = env.reset(seed=seed)
        observations = flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {agent_name: np.argmax(q[agent_name][observations[agent_name]]) for agent_name in env.agents}
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
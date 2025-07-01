import ast
import os
from typing import Callable
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
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
        post_render_callback: Callable = None,
        square_pixel_width: int = 60,
        nest_count: int = 1,
        agent_vision_radius: int = 5
):
    return scavenging_ant.ScavengingAntEnv(
        render_mode=render_mode,
        render_fps=render_fps,
        grid_height=grid_height,
        grid_width=grid_width,
        seed=seed,
        food_count=food_count,
        nest_count=nest_count,
        agent_count=agent_count,
        obstacle_count=obstacle_count,
        square_pixel_width=square_pixel_width,
        post_render_callback=post_render_callback,
        agent_vision_radius=agent_vision_radius,
    )

def flatten_observation(observation):
    positions = [*observation["agent_position"]]
    for position in observation["dropped_food_positions"]:
        positions.append(position[0])
        positions.append(position[1])

    return (
        observation["carrying_food"],
        observation["dropped_food_count"],
        observation["agent_detected"],
        *positions
    )

def flatten_observations(observations):
    for agent_name, observation in observations.items():
        observations[agent_name] = flatten_observation(observation)

    return observations

if __name__ == "__main__":
    # Learning parameters
    episodes = 10_000
    seed = 1
    learning_rate_alpha = 0.10
    discount_factor_gamma = 0.95
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)
    max_steps_per_episode = 2000
    agents_exchange_info = True

    # Environment parameters
    grid_width = 15
    grid_height = 9
    agent_count = 2
    food_count = 5
    obstacle_count = 10
    nest_count = 1
    square_pixel_width = 80
    agent_vision_radius = 1

    file_name = (
        f"{learning_rate_alpha}_"
        f"{discount_factor_gamma}_"
        f"{episodes}_"
        f"{seed}_"
        f"{grid_width}_"
        f"{grid_height}_"
        f"{agent_count}_"
        f"{food_count}_"
        f"{nest_count}_"
        f"{obstacle_count}_"
        f"{agent_vision_radius}_"
        f"{agents_exchange_info}"
    )

    os.makedirs(name="./q_learning_models", exist_ok=True)
    os.makedirs(name="./q_learning_graphs", exist_ok=True)
    os.makedirs(name="./q_learning_exchanges", exist_ok=True)

    try:
        # The Q table is a dictionary of dictionaries. {[string]: {[string]: {float}}}
        with open(f"./q_learning_models/{file_name}.json", "r") as file:
            q = {}
            loaded_q = json.load(file)

            for agent_name, agent_q in loaded_q.items():
                q[agent_name] = {}
                for observation, action_values in agent_q.items():
                    q[agent_name][ast.literal_eval(observation)] = action_values

        # The episode steps are saved as a list. This allows the graph to be changed at runtime.
        with open(f"./q_learning_graphs/{file_name}.json", "r") as file:
            episode_steps = json.load(file)

        with open(f"./q_learning_exchanges/{file_name}.json", "r") as file:
            episode_exchanges = json.load(file)

    except FileNotFoundError:
        env = create_env(
            render_mode=None,
            seed=seed,
            grid_width=grid_width,
            grid_height=grid_height,
            agent_count=agent_count,
            food_count=food_count,
            obstacle_count=obstacle_count,
            nest_count=nest_count,
            render_fps=1000,
            agent_vision_radius=agent_vision_radius,
        )

        q = {agent_name: defaultdict(lambda: np.zeros(env.action_space(agent_name).n)) for agent_name in env.agents}
        episode_steps = []
        episode_exchanges = []

        rng = np.random.default_rng()
        episode = 0
        pbar = tqdm(total=episodes)

        while episode < episodes:
            episode_q = deepcopy(q) # Clone the current Q table so that this episode has all prior information.
            observations, _ = env.reset(seed=seed)
            observations = flatten_observations(observations)
            terminated, truncated, use_episode = False, False, True

            episode_step_count = 0
            episode_exchange_count = 0

            while not terminated and not truncated:
                actions = {}
                for agent_name, observation in observations.items():
                    if rng.random() > epsilon:
                        actions[agent_name] = np.argmax(episode_q[agent_name][observations[agent_name]])
                    else:
                        actions[agent_name] = env.action_space(agent_name).sample()

                new_observations, rewards, terminations, truncations, infos = env.step(actions)
                new_observations = flatten_observations(new_observations)

                episode_step_count = episode_step_count + 1
                use_episode = episode_step_count <= max_steps_per_episode

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

                if agents_exchange_info:
                    # For each agent who is within proximity of another agent, fill in missing Q-values.
                    for agent_name, info in infos.items():
                        for other_agent_name in info["nearby_agents"]:
                            episode_exchange_count = episode_exchange_count + 1
                            for observation, actions in episode_q[other_agent_name].items():
                                if episode_q[agent_name].get(observation) is None:
                                    episode_q[agent_name][observation] = deepcopy(actions)

                observations = new_observations

            if use_episode:
                q = episode_q  # Overwrite the current Q table with the updates made during the episode.
                epsilon = max(epsilon - epsilon_decay_rate, 0.01)
                pbar.update(1)
                episode = episode + 1
                episode_steps.append(episode_step_count)
                episode_exchanges.append(episode_exchange_count)

        pbar.close()
        env.close()

        with open(f"./q_learning_models/{file_name}.json", "w") as file:
            saved_q = {}
            for agent_name, agent_q in q.items():
                saved_q[agent_name] = {}
                for observation, action_values in agent_q.items():
                    saved_q[agent_name][str(observation)] = action_values.tolist()
            json.dump(saved_q, file)

        with open(f"./q_learning_graphs/{file_name}.json", "w") as file:
            json.dump(episode_steps, file)

        with open(f"./q_learning_exchanges/{file_name}.json", "w") as file:
            json.dump(episode_exchanges, file)

    plt.plot([x for x in range(episodes)], episode_exchanges)
    plt.title("Agent Information Exchanges")
    plt.xlabel("Episode")
    plt.ylabel("Exchange Count")
    plt.show()

    plt.plot([x for x in range(episodes)], episode_steps)
    plt.title("Exchange Enabled" if agents_exchange_info else "Exchange Disabled")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()

    # Settings for visualizing agent policy
    selected_agent_index = 0
    switching_agent = False
    selected_agent_observation = None
    selected_agent_color = None
    padding = 0.40

    def post_render_callback(canvas, window):
        global selected_agent_index
        global switching_agent

        if selected_agent_observation is None:
            return

        for row in range(grid_height):
            for column in range(grid_width):
                # Copy the current observation of the agent and then update the
                # agent's position to be the grid position of the current cell. This
                # will return the policy for a specific position.
                grid_observation = deepcopy(selected_agent_observation)
                grid_observation["agent_position"] = (column, row)
                grid_observation = flatten_observation(grid_observation)

                actions = q[f"agent_{selected_agent_index}"].get(grid_observation)
                if actions is not None:
                    top_point = (
                        (column * square_pixel_width) + (square_pixel_width / 2),
                        (row * square_pixel_width) + (square_pixel_width * padding)
                    )
                    left_point = (
                        (column * square_pixel_width) + (square_pixel_width * padding),
                        (row * square_pixel_width) + (square_pixel_width / 2)
                    )
                    right_point = (
                        (column * square_pixel_width) + square_pixel_width - (square_pixel_width * padding),
                        (row * square_pixel_width) + (square_pixel_width / 2)
                    )
                    bottom_point = (
                        (column * square_pixel_width) + (square_pixel_width / 2),
                        (row * square_pixel_width) + square_pixel_width - (square_pixel_width * padding)
                    )

                    action = np.argmax(actions)
                    points = []

                    if action == 0:
                        points = [left_point, bottom_point, right_point]
                    elif action == 1:
                        points = [left_point, top_point, right_point]
                    elif action == 2:
                        points = [left_point, top_point, bottom_point]
                    elif action == 3:
                        points = [right_point, top_point, bottom_point]

                    if len(points) > 0:
                        # Draw a triangle to represent the favored move direction.
                        pygame.draw.polygon(
                            surface=canvas,
                            color=selected_agent_color,
                            points=points
                        )

        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            if not switching_agent:
                switching_agent = True
                selected_agent_index += 1
                if selected_agent_index >= agent_count:
                    selected_agent_index = 0
        else:
            switching_agent = False

    # Create a new environment with human rendering.
    env = create_env(
        render_mode="human",
        seed=seed,
        grid_width=grid_width,
        grid_height=grid_height,
        agent_count=agent_count,
        food_count=food_count,
        obstacle_count=obstacle_count,
        nest_count=nest_count,
        post_render_callback=post_render_callback,
        square_pixel_width=square_pixel_width,
        render_fps=5,
        agent_vision_radius=agent_vision_radius,
    )

    # Visualize trained model.
    for episode in range(episodes):
        observations, info = env.reset(seed=seed)
        observations = flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {agent_name: np.argmax(q[agent_name][observations[agent_name]]) for agent_name in env.agents}
            observations, rewards, terminations, truncations, info = env.step(actions)
            selected_agent_color = info[f"agent_{selected_agent_index}"]["agent_color"]
            selected_agent_observation = observations[f"agent_{selected_agent_index}"]

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
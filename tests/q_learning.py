import ast
import math
import os
import random
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

def flatten_observation(observation):
    positions = [*observation["agent_position"]]
    for position in observation["dropped_food_positions"]:
        positions.append(position[0])
        positions.append(position[1])

    return (
        int(observation["carrying_food"]),
        int(observation["dropped_food_count"]),
        int(observation["agent_detected"]),
        *positions
    )

def flatten_observations(observations):
    for agent_name, observation in observations.items():
        observations[agent_name] = flatten_observation(observation)
    return observations

def get_triangle_points(radius: float, position: (int, int), radians: float):
    points = [[-1, -1], [0, 1], [1, -1]]
    for index, point in enumerate(points):
        points[index] = [
            position[0] + radius * (point[0] * math.cos(radians) - point[1] * math.sin(radians)),
            position[1] + radius * (point[0] * math.sin(radians) + point[1] * math.cos(radians))
        ]
    return points

def get_points_from_action(action: int, radius: float, position: (int, int)):
    if action == 0:
        return get_triangle_points(radius, position, math.radians(0))
    elif action == 1:
        return get_triangle_points(radius, position, math.radians(180))
    elif action == 2:
        return get_triangle_points(radius, position, math.radians(90))
    elif action == 3:
        return get_triangle_points(radius, position, math.radians(-90))
    return [[0, 0], [0, 1], [1, 0], [1, 1]]

if __name__ == "__main__":
    # Learning parameters
    episodes = 5000
    seed = 0 # random.randint(1, 10000)
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
    exchange_delay = 1

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
        f"{agents_exchange_info}_"
        f"{exchange_delay}"
    )

    os.makedirs(name="./q_learning_models", exist_ok=True)
    os.makedirs(name="./q_learning_graphs", exist_ok=True)
    os.makedirs(name="./q_learning_exchanges", exist_ok=True)

    try:
        with open(f"./q_learning_models/{file_name}.json", "r") as file:
            q = {}
            loaded_q = json.load(file)

            for agent_name, agent_q in loaded_q.items():
                q[agent_name] = {}
                for observation, action_values in agent_q.items():
                    q[agent_name][ast.literal_eval(observation)] = action_values

        with open(f"./q_learning_graphs/{file_name}.json", "r") as file:
            episode_steps = json.load(file)

        with open(f"./q_learning_exchanges/{file_name}.json", "r") as file:
            episode_exchanges = json.load(file)

    except FileNotFoundError:
        env = scavenging_ant.ScavengingAntEnv(
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
            proximity_count = 0
            exchange_count = 0

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

                in_proximity = False
                information_exchanged = False

                for agent_name, info in infos.items():
                    for other_agent_name in info["nearby_agents"]:
                        in_proximity = True
                        if agents_exchange_info and exchange_count % exchange_delay == 0:
                            for observation, actions in episode_q[other_agent_name].items():
                                if observation not in episode_q[agent_name]:
                                    # Fill in missing Q-values if information sharing is available.
                                    information_exchanged = True
                                    episode_q[agent_name][observation] = deepcopy(actions)

                if information_exchanged:
                    exchange_count = exchange_count + 1

                if in_proximity:
                    proximity_count = proximity_count + 1

                observations = new_observations

            if use_episode:
                q = episode_q  # Overwrite the current Q table with the updates made during the episode.
                epsilon = max(epsilon - epsilon_decay_rate, 0.01)
                pbar.update(1)
                episode = episode + 1

                episode_steps.append(episode_step_count)
                episode_exchanges.append({
                    "exchange_count": exchange_count,
                    "proximity_count": proximity_count,
                })

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

    episodes = [x for x in range(episodes)]
    exchange_count = []
    proximity_count = []

    for episode_exchange in episode_exchanges:
        exchange_count.append(episode_exchange["exchange_count"])
        proximity_count.append(episode_exchange["proximity_count"])

    if agents_exchange_info:
        plt.plot(episodes, exchange_count, color="blue")
        plt.title("Exchange Count")
        plt.xlabel("Episodes")
        plt.show()

    plt.plot(episodes, proximity_count, color="orange")
    plt.title("Proximity Count")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps per Episode")
    plt.xlabel("Episodes")
    plt.show()

    # Create a new environment with human rendering.
    env = scavenging_ant.ScavengingAntEnv(
        render_mode="human",
        seed=seed,
        grid_width=grid_width,
        grid_height=grid_height,
        agent_count=agent_count,
        food_count=food_count,
        obstacle_count=obstacle_count,
        nest_count=nest_count,
        square_pixel_width=square_pixel_width,
        render_fps=60,
        agent_vision_radius=agent_vision_radius,
    )

    pygame.init()
    pygame.display.set_caption("Scavenging Ant")

    window_size = env.get_window_size()
    window = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()

    selected_agent_index = 0
    switching_agent = False
    running = True
    stepping_enabled = True
    stepping = False

    while running:
        # Visualize trained model.
        observations, info = env.reset(seed=seed)
        observations = flatten_observations(observations)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if stepping_enabled and not stepping:
                stepping = True

                actions = {agent_name: np.argmax(q[agent_name][observations[agent_name]]) for agent_name in env.agents}
                observations, rewards, terminations, truncations, info = env.step(actions)
                selected_agent_color = info[f"agent_{selected_agent_index}"]["agent_color"]
                selected_agent_observation = observations[f"agent_{selected_agent_index}"]

                canvas = pygame.Surface(window_size)
                env.draw(canvas)

                for row in range(grid_height):
                    for column in range(grid_width):
                        grid_observation = deepcopy(selected_agent_observation)
                        grid_observation["agent_position"] = (column, row)
                        grid_observation = flatten_observation(grid_observation)

                        actions = q[f"agent_{selected_agent_index}"].get(grid_observation)
                        if actions is not None:
                            action = int(np.argmax(actions))
                            position = (
                                column * square_pixel_width + square_pixel_width / 2,
                                row * square_pixel_width + square_pixel_width / 2
                            )

                            radius = square_pixel_width / 10
                            foreground_points = get_points_from_action(action, radius, position)
                            background_points = get_points_from_action(action, radius + 5, position)

                            pygame.draw.polygon(
                                surface=canvas,
                                color=(46, 48, 51),
                                points=background_points,
                            )

                            pygame.draw.polygon(
                                surface=canvas,
                                color=selected_agent_color,
                                points=foreground_points,
                            )

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

                window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    running = False
                    break
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_COMMA]:
                    if not switching_agent:
                        switching_agent = True
                        selected_agent_index += 1
                        if selected_agent_index >= agent_count:
                            selected_agent_index = 0
                else:
                    switching_agent = False
                    if keys[pygame.K_SPACE]:
                        stepping_enabled = True
                    else:
                        stepping_enabled = False
                        stepping = False

            clock.tick(env.render_fps)

    pygame.display.quit()
    pygame.quit()
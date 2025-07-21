import ast
import os
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
from scripts.scavenging_ant import ScavengingAntEnv
from scripts.layered_sprite import LayeredSprite
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

def flatten_observation(observation):
    food_positions = []
    for position in observation["food_positions"]:
        food_positions.append(position[0])
        food_positions.append(position[1])

    return (
        int(observation["carrying_food"]),
        int(observation["agent_detected"]),
        *observation["agent_position"],
        *observation["carried_food"],
        *food_positions
    )

def flatten_observations(observations):
    for agent_name, observation in observations.items():
        observations[agent_name] = flatten_observation(observation)
    return observations

def get_rotation_from_action(action: int):
    if action == 0:
        return 180
    elif action == 1:
        return 0
    elif action == 2:
        return 90
    elif action == 3:
        return -90

if __name__ == "__main__":
    # Learning parameters
    episodes = 100_000
    seed = 5 # random.randint(1, 10000)
    learning_rate_alpha = 0.10
    discount_factor_gamma = 0.70
    epsilon = 1
    epsilon_decay_rate = epsilon / (episodes / 2)
    agents_exchange_info = False

    # Environment parameters
    grid_width = 10
    grid_height = 5
    agent_count = 2
    food_count = 3
    obstacle_count = 10
    nest_count = 1
    square_pixel_width = 50
    agent_vision_radius = 1
    exchange_delay = 1

    file_name = (
        f"{learning_rate_alpha}_"
        f"{discount_factor_gamma}_"
        f"{epsilon_decay_rate}_"
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

    os.makedirs(name="../tests/q_learning_models", exist_ok=True)
    os.makedirs(name="../tests/q_learning_episodes", exist_ok=True)

    try:
        with open(f"../tests/q_learning_models/{file_name}.json", "r") as file:
            q = {}
            loaded_q = json.load(file)

            for agent_name, agent_q in loaded_q.items():
                q[agent_name] = {}
                convert_progress_bar = tqdm(total=len(agent_q.items()), desc=f"Importing {agent_name} Policy")
                for observation, action_values in agent_q.items():
                    q[agent_name][ast.literal_eval(observation)] = action_values
                    convert_progress_bar.update(1)
                convert_progress_bar.close()

        with open(f"../tests/q_learning_episodes/{file_name}.json", "r") as file:
            episode_data = json.load(file)

    except FileNotFoundError:
        env = ScavengingAntEnv(
            render_mode=None,
            seed=seed,
            grid_width=grid_width,
            grid_height=grid_height,
            agent_count=agent_count,
            food_count=food_count,
            obstacle_count=obstacle_count,
            nest_count=nest_count,
            agent_vision_radius=agent_vision_radius,
        )

        q = {agent_name: defaultdict(lambda: np.zeros(env.action_space(agent_name).n)) for agent_name in env.agents}
        episode_data = []

        rng = np.random.default_rng()
        episode_progress_bar = tqdm(total=episodes, desc="Training")

        for episode in range(episodes):
            observations, _ = env.reset(seed=seed)
            observations = flatten_observations(observations)
            terminated, truncated = False, False

            step_count = 0
            proximity_count = 0
            exchange_count = 0
            total_rewards = {agent_name: 0 for agent_name in env.agents}

            while not terminated and not truncated:
                actions = {}
                for agent_name, observation in observations.items():
                    if rng.random() > epsilon:
                        actions[agent_name] = np.argmax(q[agent_name][observations[agent_name]])
                    else:
                        actions[agent_name] = env.action_space(agent_name).sample()

                new_observations, rewards, terminations, truncations, infos = env.step(actions)
                new_observations = flatten_observations(new_observations)

                for agent_name, reward in rewards.items():
                    total_rewards[agent_name] += reward

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
                    observation = observations[agent_name]
                    action = actions[agent_name]
                    q[agent_name][observation][action] = q[agent_name][observation][action] + learning_rate_alpha * (rewards[agent_name] + discount_factor_gamma * np.max(q[agent_name][new_observation]) - q[agent_name][observation][action])

                in_proximity = False
                exchanged = False

                for agent_name, info in infos.items():
                    nearby_agents = info["nearby_agents"]
                    if not in_proximity and len(nearby_agents) > 0:
                        in_proximity = True

                    if agents_exchange_info and exchange_count % exchange_delay == 0:
                        for nearby_agent_name in nearby_agents:
                            # Give the current agent observations from the nearby agent that it doesn't already have.
                            for observation, actions in q[nearby_agent_name].items():
                                if observation not in q[agent_name]:
                                    q[agent_name][observation] = deepcopy(actions)
                                    exchanged = True

                if exchanged:
                    exchange_count = exchange_count + 1

                if in_proximity:
                    proximity_count = proximity_count + 1

                step_count = step_count + 1
                observations = new_observations

            epsilon = max(epsilon - epsilon_decay_rate, 0.01)
            episode_data.append({
                "step_count": step_count,
                "exchange_count": exchange_count,
                "proximity_count": proximity_count,
                "total_rewards": total_rewards
            })

            episode_progress_bar.update(1)

        episode_progress_bar.close()
        env.close()

        with open(f"../tests/q_learning_models/{file_name}.json", "w") as file:
            saved_q = {}
            for agent_name, agent_q in q.items():
                saved_q[agent_name] = {}
                for observation, action_values in agent_q.items():
                    saved_q[agent_name][str(observation)] = action_values.tolist()
            json.dump(saved_q, file)

        with open(f"../tests/q_learning_episodes/{file_name}.json", "w") as file:
            json.dump(episode_data, file)

    episodes = []
    episode_steps = []
    exchange_count = []
    proximity_count = []
    total_rewards = []

    for episode, data in enumerate(episode_data):
        if episode % 100 == 0:
            episodes.append(episode)
            if data.get("step_count") is not None:
                episode_steps.append(data["step_count"])
            if data.get("exchange_count") is not None:
                exchange_count.append(data["exchange_count"])
            if data.get("proximity_count") is not None:
                proximity_count.append(data["proximity_count"])
            if data.get("total_rewards") is not None:
                total_rewards.append(data["total_rewards"])

    if agents_exchange_info and len(exchange_count) > 0:
        plt.plot(episodes, exchange_count, color="blue")
        plt.title("Exchange Count")
        plt.xlabel("Episodes")
        plt.show()

    if len(proximity_count) > 0:
        plt.plot(episodes, proximity_count, color="orange")
        plt.title("Proximity Count")
        plt.xlabel("Episodes")
        plt.show()

    if len(episode_steps) > 0:
        plt.plot(episodes, episode_steps, color="green")
        plt.title("Steps per Episode")
        plt.xlabel("Episodes")
        plt.show()

    if len(total_rewards) > 0:
        episode_agent_rewards = {}
        for _, agent_rewards in enumerate(total_rewards):
            for agent_name, episode_reward in agent_rewards.items():
                if episode_agent_rewards.get(agent_name) is None:
                    episode_agent_rewards[agent_name] = []
                episode_agent_rewards[agent_name].append(episode_reward)

        for agent_name, episode_rewards in episode_agent_rewards.items():
            plt.plot(episodes, episode_rewards, label=agent_name)

        plt.title("Rewards per Episode")
        plt.xlabel("Episodes")
        plt.legend()
        plt.show()

    # Create a new environment with human rendering.
    env = ScavengingAntEnv(
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
    run_interval_time = 0

    selected_agent_index = 0
    switching_agent = False
    auto_run_enabled = False
    switching_auto_run = False
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
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or stepping_enabled and not stepping

            if draw_next_step:
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
                            radius = square_pixel_width / 2
                            LayeredSprite(
                                foreground_image="../images/icons8-slide-up-200.png",
                                background_image="../images/icons8-slide-up-outline-200.png",
                                dimensions=(radius, radius),
                                rotation=get_rotation_from_action(int(np.argmax(actions))),
                                color=selected_agent_color
                            ).draw(
                                canvas=canvas,
                                position=(
                                    column * square_pixel_width + square_pixel_width / 2 - radius / 2,
                                    row * square_pixel_width + square_pixel_width / 2 - radius / 2,
                                ),
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
                if keys[pygame.K_a]:
                    if not switching_auto_run:
                        switching_auto_run = True
                        auto_run_enabled = not auto_run_enabled
                        run_interval_time = 0
                else:
                    switching_auto_run = False

                if keys[pygame.K_s]:
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

            delta_time = clock.tick(env.render_fps) / 1000
            if auto_run_enabled:
                run_interval_time += delta_time
                if run_interval_time >= 0.10:
                    run_interval_time = 0

    pygame.display.quit()
    pygame.quit()
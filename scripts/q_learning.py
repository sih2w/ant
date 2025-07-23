import ast
import os
from typing import Any
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
from scripts.scavenging_ant import ScavengingAntEnv
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

type LoadedQ = dict[str, dict[tuple[int, ...], np.ndarray[float]]]
type SavedQ = dict[str, dict[str, list[float]]]
type EpisodeData = list[dict]
type RawObservation = dict[str, Any]
type FlattenedObservation = tuple[int, ...]

EPISODES = 10_000
SEED = 10
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.70
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = False
GRID_WIDTH = 10
GRID_HEIGHT = 10
AGENT_COUNT = 2
FOOD_COUNT = 3
OBSTACLE_COUNT = 10
NEST_COUNT = 1
SQUARE_PIXEL_WIDTH = 45
AGENT_VISION_RADIUS = 1
EXCHANGE_DELAY = 1
RENDER_FPS = 30
SECONDS_BETWEEN_AUTO_STEP = 0.50
SPARSE_INTERVAL = int(EPISODES / 100)

def flatten_observation(observation: RawObservation) -> FlattenedObservation:
    """
    Converts a RawObservation into a FlattenedObservation.
    """
    flattened_observation = [
        int(observation["carrying_food"]),
        int(observation["agent_detected"]),
        observation["agent_position"][0],
        observation["agent_position"][1],
    ]

    for carried_food in observation["carried_food"]:
        flattened_observation.append(carried_food)

    for food_position in observation["food_positions"]:
        flattened_observation.append(food_position[0])
        flattened_observation.append(food_position[1])
    return tuple(flattened_observation)

def flatten_observations(raw_observations: dict[str, RawObservation]) -> dict[str, FlattenedObservation]:
    """
    Converts a dict of RawObservations into a dict of FlattenedObservations.
    """
    flattened_observations: dict[str, FlattenedObservation] = {}
    for agent_name, observation in raw_observations.items():
        flattened_observations[agent_name] = flatten_observation(observation)
    return flattened_observations

def get_rotation_from_action(action: int) -> int or None:
    """
    Returns the rotation from a given action.
    """
    if action == 0:
        return 180
    elif action == 1:
        return 0
    elif action == 2:
        return 90
    elif action == 3:
        return -90

def load_q(file_name: str) -> LoadedQ:
    """
    Loads a Q-learning model from a given file name.
    """
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
        return q

def load_episode_data(file_name: str) -> EpisodeData:
    """
    Loads EpisodeData from a given file name.
    """
    with open(f"../tests/q_learning_episodes/{file_name}.json", "r") as file:
        return json.load(file)

def save_q(file_name: str, q: LoadedQ) -> None:
    """
    Saves a Q-learning model from a given file name.
    """
    with open(f"../tests/q_learning_models/{file_name}.json", "w") as file:
        saved_q = {}
        for agent_name, agent_q in q.items():
            saved_q[agent_name] = {}
            for observation, action_values in agent_q.items():
                saved_q[agent_name][str(observation)] = action_values.tolist()
        json.dump(saved_q, file)

def save_episode_data(file_name: str, episode_data: EpisodeData):
    """
    Saves EpisodeData from a given file name.
    """
    with open(f"../tests/q_learning_episodes/{file_name}.json", "w") as file:
        json.dump(episode_data, file)

def sparse_episode_data(
        episode_data: EpisodeData
) -> (list[int], list[int], list[int], list[int], list[int]):
    """
    Breaks EpisodeData down into separate lists that can be used for plotting.
    """
    episodes = []
    episode_steps = []
    exchange_count = []
    proximity_count = []
    total_rewards = []

    for episode in range(0, len(episode_data), SPARSE_INTERVAL):
        data = episode_data[episode]
        if data is not None:
            if data.get("step_count") is not None:
                episode_steps.append(data["step_count"])
            if data.get("exchange_count") is not None:
                exchange_count.append(data["exchange_count"])
            if data.get("proximity_count") is not None:
                proximity_count.append(data["proximity_count"])
            if data.get("total_rewards") is not None:
                total_rewards.append(data["total_rewards"])
            episodes.append(episode)

    return (
        episodes,
        episode_steps,
        exchange_count,
        proximity_count,
        total_rewards
    )

def train() -> (LoadedQ, EpisodeData):
    """
    Trains Q-learning agents in an environment defined by the passed parameters.
    """
    env = ScavengingAntEnv(
        render_mode=None,
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        agent_vision_radius=AGENT_VISION_RADIUS,
    )

    q = {agent_name: defaultdict(lambda: np.zeros(env.action_space(agent_name).n)) for agent_name in env.agents}
    episode_data = []

    epsilon = EPSILON_START
    rng = np.random.default_rng()
    episode_progress_bar = tqdm(total=EPISODES, desc="Training")

    for episode in range(EPISODES):
        observations, _ = env.reset(seed=SEED)
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
                q[agent_name][observation][action] = q[agent_name][observation][action] + LEARNING_RATE_ALPHA * (
                            rewards[agent_name] + DISCOUNT_FACTOR_GAMMA * np.max(q[agent_name][new_observation]) -
                            q[agent_name][observation][action])

            in_proximity = False
            exchanged = False

            for agent_name, info in infos.items():
                nearby_agents = info["nearby_agents"]
                if not in_proximity and len(nearby_agents) > 0:
                    in_proximity = True

                if AGENTS_EXCHANGE_INFO and exchange_count % EXCHANGE_DELAY == 0:
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

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0.01)
        episode_data.append({
            "step_count": step_count,
            "exchange_count": exchange_count,
            "proximity_count": proximity_count,
            "total_rewards": total_rewards
        })

        episode_progress_bar.update(1)

    episode_progress_bar.close()
    env.close()

    return q, episode_data

def draw_exchange_chart(episodes: list[int], exchange_count: list[int]):
    plt.plot(episodes, exchange_count, color="blue")
    plt.title("Exchange Count")
    plt.xlabel("Episodes")
    plt.show()

def draw_proximity_chart(episodes: list[int], proximity_count: list[int]):
    plt.plot(episodes, proximity_count, color="orange")
    plt.title("Proximity Count")
    plt.xlabel("Episodes")
    plt.show()

def draw_episode_steps_chart(episodes: list[int], episode_steps: list[int]):
    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps per Episode")
    plt.xlabel("Episodes")
    plt.show()

def draw_rewards_chart(episodes: list[int], total_rewards: list[dict[str, int]]):
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

def validate():
    """
    Creates a visual representation of the Q-learning agents' learned policies.
    Agents will take the action they found to be optimal at each step.
    Users can toggle auto running with the "A" key.
    Users can alternate between agent policy visualizations with the "S" key.
    Users can progress a single step at a time with the "SPACE" key.
    """
    env = ScavengingAntEnv(
        render_mode="human",
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        square_pixel_width=SQUARE_PIXEL_WIDTH,
        render_fps=RENDER_FPS,
        agent_vision_radius=AGENT_VISION_RADIUS,
    )

    pygame.init()
    pygame.display.set_caption("Q-Learning Ants")
    pygame.display.set_icon(pygame.image.load("../images/icons8-ant-30.png"))

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
        observations, info = env.reset(seed=SEED)
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
                selected_agent_name = f"agent_{selected_agent_index}"
                selected_agent_observation = observations[selected_agent_name]

                canvas = pygame.Surface(window_size)
                env.draw(canvas)

                for row in range(GRID_HEIGHT):
                    for column in range(GRID_WIDTH):
                        grid_observation = deepcopy(selected_agent_observation)
                        grid_observation["agent_position"] = (column, row)
                        grid_observation = flatten_observation(grid_observation)
                        actions = q[selected_agent_name].get(grid_observation)

                        if actions is not None:
                            image = pygame.image.load(f"../images/arrows/{selected_agent_name}.png")
                            rotation = get_rotation_from_action(int(np.argmax(actions)))
                            position = (
                                column * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_width() / 2,
                                row * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_height() / 2,
                            )
                            image = pygame.transform.rotate(image, rotation)
                            canvas.blit(image, position)

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
                        if selected_agent_index >= AGENT_COUNT:
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
                if run_interval_time >= SECONDS_BETWEEN_AUTO_STEP:
                    run_interval_time = 0

    pygame.display.quit()
    pygame.quit()

if __name__ == "__main__":
    file_name = (
        f"{LEARNING_RATE_ALPHA}_"
        f"{DISCOUNT_FACTOR_GAMMA}_"
        f"{EPSILON_DECAY_RATE}_"
        f"{EPISODES}_"
        f"{SEED}_"
        f"{GRID_WIDTH}_"
        f"{GRID_HEIGHT}_"
        f"{AGENT_COUNT}_"
        f"{FOOD_COUNT}_"
        f"{NEST_COUNT}_"
        f"{OBSTACLE_COUNT}_"
        f"{AGENT_VISION_RADIUS}_"
        f"{AGENTS_EXCHANGE_INFO}_"
        f"{EXCHANGE_DELAY}"
    )

    os.makedirs(name="../tests/q_learning_models", exist_ok=True)
    os.makedirs(name="../tests/q_learning_episodes", exist_ok=True)

    try:
        q = load_q(file_name)
        episode_data = load_episode_data(file_name)

    except FileNotFoundError:
        q, episode_data = train()
        save_q(file_name, q)
        save_episode_data(file_name, episode_data)

    episodes, episode_steps, exchange_count, proximity_count, total_rewards = sparse_episode_data(episode_data)

    if AGENTS_EXCHANGE_INFO:
        if len(exchange_count) > 0:
            draw_exchange_chart(episodes, exchange_count)

    if len(proximity_count) > 0:
        draw_proximity_chart(episodes, proximity_count)

    if len(episode_steps) > 0:
        draw_episode_steps_chart(episodes, episode_steps)

    if len(total_rewards) > 0:
        draw_rewards_chart(episodes, total_rewards)

    validate()
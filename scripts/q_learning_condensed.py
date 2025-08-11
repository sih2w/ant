import os
from typing import Optional
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
from scripts.scavenging_ant import ScavengingAntEnv
from tqdm import tqdm
from scripts.types import *

EPISODES = 1_000
SEED = 0
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.95
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = True
GRID_WIDTH = 10
GRID_HEIGHT = 10
AGENT_COUNT = 2
FOOD_COUNT = 10
OBSTACLE_COUNT = 10
NEST_COUNT = 1
SQUARE_PIXEL_WIDTH = 45
AGENT_VISION_RADIUS = 1
EXCHANGE_DELAY = 1

RENDER_FPS = 30
SECONDS_BETWEEN_AUTO_STEP = 0.20
ACTION_COUNT = 4
SPARSE_INTERVAL = int(EPISODES / 100)
SAVE_AFTER_TRAINING = True
SHOW_AFTER_TRAINING = True
ACTION_ROTATIONS = (180, 0, 90, -90)
SAVE_DIRECTORY = "../runs/q_learning"
FILE_NAME = (
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


def convert_dict_to_json_compatible(data):
    if isinstance(data, dict):
        return {str(key): convert_dict_to_json_compatible(value) for key, value in data.items()}
    return data


def convert_json_compatible_to_dict(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key.startswith("(") and key.endswith(")"):
                key = eval(key)
            new_dict[key] = convert_json_compatible_to_dict(value)
        return new_dict
    return data


def load_data() -> (StateActions, EpisodeData):
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.json", "r") as file:
        data = json.load(file)
        return convert_json_compatible_to_dict(data["state_actions"]), data["episode_data"]


def save_data(state_actions: StateActions, episode_data: EpisodeData) -> None:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.json", "w") as file:
        json.dump({
            "state_actions": convert_dict_to_json_compatible(state_actions),
            "episode_data": episode_data
        }, file)


def sparsify(data: list) -> list:
    sparse_data = []
    for index in range(0, len(data), SPARSE_INTERVAL):
        sparse_data.append(data[index])
    return sparse_data


def get_return_actions(
        state_actions: StateActions,
        agent_name: AgentName,
        location: AgentLocation
) -> Optional[Actions]:
    if "return_policy" not in state_actions:
        state_actions["return_policy"] = {}

    if agent_name not in state_actions["return_policy"]:
        state_actions["return_policy"][agent_name] = {}

    if location not in state_actions["return_policy"][agent_name]:
        state_actions["return_policy"][agent_name][location] = np.zeros(ACTION_COUNT).tolist()

    return state_actions["return_policy"][agent_name][location]


def get_search_actions(
        state_actions: StateActions,
        agent_name: AgentName,
        location: AgentLocation,
        food_positions: FoodPositions
) -> Optional[Actions]:
    if "search_policy" not in state_actions:
        state_actions["search_policy"] = {}

    if agent_name not in state_actions["search_policy"]:
        state_actions["search_policy"][agent_name] = {}

    if location not in state_actions["search_policy"][agent_name]:
        state_actions["search_policy"][agent_name][location] = {}

    if food_positions not in state_actions["search_policy"][agent_name][location]:
        state_actions["search_policy"][agent_name][location][food_positions] = np.zeros(ACTION_COUNT).tolist()

    return state_actions["search_policy"][agent_name][location][food_positions]


def get_actions(
        state_actions: StateActions,
        agent_name: str,
        agent_location: AgentLocation,
        food_positions: FoodPositions,
        carrying_food: bool,
) -> Actions:
    if carrying_food:
        return get_return_actions(state_actions, agent_name, agent_location)
    return get_search_actions(state_actions, agent_name, agent_location, food_positions)


def update_actions(
        state_actions: StateActions,
        agent_name: str,
        was_carrying_food: bool,
        is_carrying_food: bool,
        old_agent_location: AgentLocation,
        new_agent_location: AgentLocation,
        old_food_positions: FoodPositions,
        new_food_positions: FoodPositions,
        selected_action: int,
        reward: float
):
    old_actions = get_actions(state_actions, agent_name, old_agent_location, old_food_positions, was_carrying_food)
    new_actions = get_actions(state_actions, agent_name, new_agent_location, new_food_positions, is_carrying_food)
    old_actions[selected_action] = old_actions[selected_action] + LEARNING_RATE_ALPHA * (
            reward + DISCOUNT_FACTOR_GAMMA * np.max(new_actions) - old_actions[selected_action])


def train() -> (StateActions, EpisodeData):
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

    state_actions: StateActions = {}
    episode_data: EpisodeData = {"steps": [], "rewards": []}

    epsilon = EPSILON_START
    rng = np.random.default_rng()
    episode_progress_bar = tqdm(total=EPISODES, desc="Training")

    for episode in range(EPISODES):
        observations, _ = env.reset(seed=SEED)
        terminated, truncated = False, False
        step_count = 0
        total_rewards = {agent_name: 0 for agent_name in env.agents}

        while not terminated and not truncated:
            actions = {}
            for agent_name, observation in observations.items():
                if rng.random() > epsilon:
                    actions[agent_name] = int(
                        np.argmax(
                            get_actions(
                                state_actions,
                                agent_name,
                                observation["agent_position"],
                                observation["food_positions"],
                                observation["carrying_food"],
                            )
                        )
                    )
                else:
                    actions[agent_name] = env.action_space(agent_name).sample()

            new_observations, rewards, terminations, truncations, infos = env.step(actions)
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

            for agent_name, new_observation in new_observations.items():
                old_observation = observations[agent_name]
                update_actions(
                    state_actions,
                    agent_name,
                    old_observation["carrying_food"],
                    new_observation["carrying_food"],
                    old_observation["agent_position"],
                    new_observation["agent_position"],
                    old_observation["food_positions"],
                    new_observation["food_positions"],
                    actions[agent_name],
                    rewards[agent_name]
                )

            step_count = step_count + 1
            observations = new_observations

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0.01)
        episode_data["steps"].append(step_count)
        episode_data["rewards"].append(total_rewards)
        episode_progress_bar.update(1)

        if AGENTS_EXCHANGE_INFO:
            shared_return_policy = {}
            for agent_name in env.agents:
                state_actions["return_policy"][agent_name] = state_actions["return_policy"].get(agent_name, {})

                for agent_location, actions in state_actions["return_policy"][agent_name].items():
                    shared_return_policy[agent_location] = actions

            for agent_name in env.agents:
                state_actions["return_policy"][agent_name] = shared_return_policy.copy()

    episode_progress_bar.close()
    env.close()

    return state_actions, episode_data


def visualize(state_actions: StateActions):
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
        observations, info = env.reset(seed=SEED)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or stepping_enabled and not stepping

            if draw_next_step:
                stepping = True
                actions = {}
                for agent_name in env.agents:
                    actions[agent_name] = int(
                        np.argmax(
                            get_actions(
                                state_actions,
                                agent_name,
                                observations[agent_name]["agent_position"],
                                observations[agent_name]["food_positions"],
                                observations[agent_name]["carrying_food"],
                            )
                        )
                    )

                observations, rewards, terminations, truncations, info = env.step(actions)
                selected_agent_name = f"agent_{selected_agent_index}"
                selected_agent_observation = observations[selected_agent_name]

                canvas = pygame.Surface(window_size)
                env.draw(canvas)

                food_positions = selected_agent_observation["food_positions"]
                carrying_food = selected_agent_observation["carrying_food"]

                for row in range(GRID_HEIGHT):
                    for column in range(GRID_WIDTH):
                        agent_position = (column, row)
                        actions = get_actions(
                            state_actions,
                            selected_agent_name,
                            agent_position,
                            food_positions,
                            carrying_food
                        )

                        image = pygame.image.load(f"../images/arrows/{selected_agent_name}.png")
                        rotation = ACTION_ROTATIONS[int(np.argmax(actions))]
                        position = (
                            column * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_width() / 2,
                            row * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_height() / 2,
                        )
                        image = pygame.transform.rotate(image, rotation)
                        canvas.blit(image, position)

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
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    try:
        state_actions, episode_data = load_data()
    except FileNotFoundError:
        state_actions, episode_data = train()
        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episode_data)

    episode_steps = sparsify(episode_data["steps"])
    episodes = [episode for episode in range(len(episode_steps))]

    plt.plot(episodes, episode_steps, color="green")
    plt.title(f"Steps {"With" if AGENTS_EXCHANGE_INFO else "Without"} Exchange")
    plt.xlabel("Episodes")
    plt.show()

    rewards = sparsify(episode_data["rewards"])
    episode_agent_rewards = {}
    for _, agent_rewards in enumerate(rewards):
        for agent_name, episode_reward in agent_rewards.items():
            if episode_agent_rewards.get(agent_name) is None:
                episode_agent_rewards[agent_name] = []
            episode_agent_rewards[agent_name].append(episode_reward)
    for agent_name, episode_rewards in episode_agent_rewards.items():
        plt.plot(episodes, episode_rewards, label=agent_name)

    plt.title(f"Rewards {"With" if AGENTS_EXCHANGE_INFO else "Without"} Exchange")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

    if SHOW_AFTER_TRAINING:
        visualize(state_actions)

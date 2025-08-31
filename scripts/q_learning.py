from typing import Any
import dill
import os
from collections import defaultdict
import numpy as np
import pygame
import matplotlib.pyplot as plt
from numpy.random import Generator
from scripts.scavenging_ant import ScavengingAntEnv
from tqdm import tqdm
from scripts.types import *

EPISODES = 1000
SEED = 2
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.70
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = True
GRID_WIDTH = 20
GRID_HEIGHT = 10
AGENT_COUNT = 2
FOOD_COUNT = 10
OBSTACLE_COUNT = 10
NEST_COUNT = 1
AGENT_VISION_RADIUS = 1

SQUARE_PIXEL_WIDTH = 40
RENDER_FPS = 30
SECONDS_BETWEEN_AUTO_STEP = 0.10
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
    f"{AGENTS_EXCHANGE_INFO}"
)


def state_actions_factory() -> StateActions:
    return {
        "return_policy": defaultdict(lambda: defaultdict(lambda: np.zeros(ACTION_COUNT).tolist())),
        "search_policy": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(ACTION_COUNT).tolist()))),
    }


def exchanged_actions_factory() -> ExchangedActions:
    return {
        "return_policy": defaultdict(lambda: defaultdict(lambda: False)),
        "search_policy": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: False))),
    }


def load_data() -> Tuple[StateActions, ExchangedActions, List[Episode]]:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "rb") as file:
        data = dill.load(file)
        return data["state_actions"], data["exchanged_actions"], data["episode_data"]


def save_data(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        episode_data: List[Episode]
) -> None:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "wb") as file:
        dill.dump({
            "state_actions": state_actions,
            "exchanged_actions": exchanged_actions,
            "episode_data": episode_data
        }, file)


def sparsify(data: Any) -> Any:
    sparse_data = []
    for index in range(0, len(data), SPARSE_INTERVAL):
        sparse_data.append(data[index])
    return sparse_data


def get_action_values(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        agent_name: AgentName,
        agent_location: Location,
        food_locations: FoodLocations,
        carrying_food: bool,
) -> Actions:
    if carrying_food:
        if exchanged_actions:
            used = exchanged_actions["return_policy"][agent_name][agent_location]
            if used is False:
                exchanged_actions["return_policy"][agent_name][agent_location] = True
        return state_actions["return_policy"][agent_name][agent_location]
    else:
        if exchanged_actions:
            used = exchanged_actions["search_policy"][agent_name][agent_location][food_locations]
            if used is False:
                exchanged_actions["search_policy"][agent_name][agent_location][food_locations] = True
        return state_actions["search_policy"][agent_name][agent_location][food_locations]


def update_action_values(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        agent_name: AgentName,
        was_carrying_food: bool,
        is_carrying_food: bool,
        old_agent_location: Location,
        new_agent_location: Location,
        old_food_positions: FoodLocations,
        new_food_positions: FoodLocations,
        selected_action: int,
        reward: float
):
    old_actions = get_action_values(state_actions, exchanged_actions, agent_name, old_agent_location,
                                    old_food_positions, was_carrying_food)
    new_actions = get_action_values(state_actions, exchanged_actions, agent_name, new_agent_location,
                                    new_food_positions, is_carrying_food)
    old_actions[selected_action] = old_actions[selected_action] + LEARNING_RATE_ALPHA * (
            reward + DISCOUNT_FACTOR_GAMMA * np.max(new_actions) - old_actions[selected_action])


def are_close_enough(
        agent_1_location: Location,
        agent_2_location: Location
) -> bool:
    return np.array_equal(agent_1_location, agent_2_location)


def fill_missing_search_policy(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName,
) -> None:
    source = state_actions["search_policy"][from_agent_name]
    destination = state_actions["search_policy"][to_agent_name]
    exchange_destination = exchanged_actions["search_policy"][to_agent_name]

    for agent_location, food_location_actions in source.items():
        for food_locations, action_values in food_location_actions.items():
            if not food_locations in destination[agent_location]:
                destination[agent_location][food_locations] = action_values.copy()
                exchange_destination[agent_location][food_locations] = False


def fill_missing_return_policy(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName,
) -> None:
    source = state_actions["return_policy"][from_agent_name]
    destination = state_actions["return_policy"][to_agent_name]
    exchange_destination = exchanged_actions["return_policy"][to_agent_name]

    for agent_location, action_values in source.items():
        if agent_location not in destination:
            destination[agent_location] = action_values.copy()
            exchange_destination[agent_location] = False


def exchange(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        observations: Dict[AgentName, Observation]
) -> None:
    for agent_1_name, agent_1_observation in observations.items():
        for agent_2_name, agent_2_observation in observations.items():
            if agent_1_name == agent_2_name or not are_close_enough(
                    agent_1_observation["agent_location"],
                    agent_2_observation["agent_location"],
            ):
                continue

            if agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
                # Agent 1 returning to nest. Agent 2 returning to nest.
                fill_missing_return_policy(state_actions, exchanged_actions, agent_1_name, agent_2_name)
                fill_missing_return_policy(state_actions, exchanged_actions, agent_2_name, agent_1_name)
            elif not agent_1_observation["carrying_food"] and not agent_2_observation["carrying_food"]:
                # Agent 1 searching for food. Agent 2 searching for food.
                fill_missing_search_policy(state_actions, exchanged_actions, agent_1_name, agent_2_name)
                fill_missing_search_policy(state_actions, exchanged_actions, agent_2_name, agent_1_name)
            elif not agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
                # Agent 1 searching for food. Agent 2 returning to nest.
                fill_missing_return_policy(state_actions, exchanged_actions, agent_1_name, agent_2_name)
                fill_missing_search_policy(state_actions, exchanged_actions, agent_2_name, agent_1_name)
            else:
                # Agent 1 returning to nest. Agent 2 searching for food.
                fill_missing_return_policy(state_actions, exchanged_actions, agent_2_name, agent_1_name)
                fill_missing_search_policy(state_actions, exchanged_actions, agent_1_name, agent_2_name)


def has_episode_ended(
        terminations: Dict[AgentName, bool],
        truncations: Dict[AgentName, bool]
) -> bool:
    for _, termination in terminations.items():
        terminated = termination
        if terminated:
            return True
    else:
        for _, truncation in truncations.items():
            truncated = truncation
            if truncated:
                return True
    return False


def get_greedy_actions(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        observations: Dict[AgentName, Observation]
) -> Dict[AgentName, int]:
    greedy_actions = {}
    for agent_name, observation in observations.items():
        greedy_actions[agent_name] = int(
            np.argmax(
                get_action_values(
                    state_actions,
                    exchanged_actions,
                    agent_name,
                    observation["agent_location"],
                    observation["food_locations"],
                    observation["carrying_food"],
                )
            )
        )

    return greedy_actions


def get_epsilon_greedy_actions(
        state_actions: StateActions,
        exchanged_actions: ExchangedActions,
        observations: Dict[AgentName, Observation],
        epsilon: float,
        rng: Generator
) -> Dict[AgentName, int]:
    epsilon_greedy_actions = {}
    for agent_name, observation in observations.items():
        if rng.random() > epsilon:
            epsilon_greedy_actions[agent_name] = int(
                np.argmax(
                    get_action_values(
                        state_actions,
                        exchanged_actions,
                        agent_name,
                        observation["agent_location"],
                        observation["food_locations"],
                        observation["carrying_food"],
                    )
                )
            )
        else:
            epsilon_greedy_actions[agent_name] = rng.integers(0, ACTION_COUNT)

    return epsilon_greedy_actions


def train() -> Tuple[StateActions, ExchangedActions, List[Episode]]:
    env = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT
    )

    state_actions: StateActions = state_actions_factory()
    exchanged_actions: ExchangedActions = exchanged_actions_factory()
    episode_data: List[Episode] = []

    epsilon = EPSILON_START
    rng = np.random.default_rng(seed=SEED)
    episode_progress_bar = tqdm(total=EPISODES, desc="Training")

    for episode in range(EPISODES):
        observations, _ = env.reset(seed=SEED)
        terminations, truncations = {}, {}
        current_episode: Episode = {
            "steps": 0,
            "rewards": {agent_name: 0 for agent_name in env.agent_names}
        }

        while not has_episode_ended(terminations, truncations):
            selected_actions = get_epsilon_greedy_actions(state_actions, exchanged_actions, observations, epsilon, rng)
            new_observations, rewards, terminations, truncations, infos = env.step(selected_actions)

            for agent_name, reward in rewards.items():
                current_episode["rewards"][agent_name] += reward

            for agent_name, new_observation in new_observations.items():
                old_observation = observations[agent_name]
                update_action_values(
                    state_actions,
                    exchanged_actions,
                    agent_name,
                    old_observation["carrying_food"],
                    new_observation["carrying_food"],
                    old_observation["agent_location"],
                    new_observation["agent_location"],
                    old_observation["food_locations"],
                    new_observation["food_locations"],
                    selected_actions[agent_name],
                    rewards[agent_name]
                )

            if AGENTS_EXCHANGE_INFO:
                exchange(state_actions, exchanged_actions, new_observations)

            current_episode["steps"] += 1
            observations = new_observations

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0.01)
        episode_progress_bar.update(1)
        episode_data.append(current_episode)

    episode_progress_bar.close()

    return state_actions, exchanged_actions, episode_data


def draw_current_step(
        env: ScavengingAntEnv,
        exchanged_actions: ExchangedActions,
        observations: Dict[AgentName, Observation],
        selected_agent_index: int,
        window: pygame.Surface,
        window_size: (int, int),
) -> None:
    selected_agent_name = f"agent_{selected_agent_index}"
    selected_agent_observation = observations[selected_agent_name]

    canvas = pygame.Surface(window_size)
    env.draw(canvas)

    food_locations = selected_agent_observation["food_locations"]
    carrying_food = selected_agent_observation["carrying_food"]

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            agent_position = (column, row)
            action_values = get_action_values(
                state_actions,
                exchanged_actions,
                selected_agent_name,
                agent_position,
                food_locations,
                carrying_food
            )

            image = pygame.image.load(f"../images/arrows/{selected_agent_name}.png")
            rotation = ACTION_ROTATIONS[int(np.argmax(action_values))]
            position = (
                column * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_width() / 2,
                row * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_height() / 2,
            )
            image = pygame.transform.rotate(image, rotation)
            canvas.blit(image, position)

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.flip()


def visualize(state_actions: StateActions, exchanged_actions: ExchangedActions) -> None:
    env = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        square_pixel_width=SQUARE_PIXEL_WIDTH
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
    stepping_enabled = False
    stepping = False

    while running:
        observations, _ = env.reset(seed=SEED)
        terminations, truncations = {}, {}

        draw_current_step(
            env,
            exchanged_actions,
            observations,
            selected_agent_index,
            window,
            window_size
        )

        while running and not has_episode_ended(terminations, truncations):
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or (stepping_enabled and not stepping)

            if draw_next_step:
                stepping = True
                selected_actions = get_greedy_actions(state_actions, exchanged_actions, observations)
                observations, rewards, terminations, truncations, info = env.step(selected_actions)

                draw_current_step(
                    env,
                    exchanged_actions,
                    observations,
                    selected_agent_index,
                    window,
                    window_size
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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

            delta_time = clock.tick(60) / 1000
            if auto_run_enabled:
                run_interval_time += delta_time
                if run_interval_time >= SECONDS_BETWEEN_AUTO_STEP:
                    run_interval_time = 0

    pygame.display.quit()
    pygame.quit()


def plot_episode_data(episode_data: List[Episode]) -> None:
    episode_data = sparsify(episode_data)
    episodes = [episode * SPARSE_INTERVAL for episode in range(len(episode_data))]

    episode_steps = []
    episode_rewards = []

    for episode in episode_data:
        episode_steps.append(episode["steps"])
        episode_rewards.append(episode["rewards"])

    plt.plot(episodes, episode_steps, color="green")
    plt.title(f"Steps {"With" if AGENTS_EXCHANGE_INFO else "Without"} Exchange")
    plt.xlabel("Episodes")
    plt.show()

    episode_agent_rewards = {}
    for _, agent_rewards in enumerate(episode_rewards):
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


def get_state_use_count(
        dictionary: DefaultDict,
        total_exchanges: int = 0,
        used_count: int = 0
) -> Tuple[int, int]:
    for value in dictionary.values():
        if isinstance(value, bool):
            total_exchanges += 1
            if value is True:
                used_count += 1
        elif isinstance(value, defaultdict):
            total_exchanges, used_count = get_state_use_count(value, total_exchanges, used_count)

    return total_exchanges, used_count


def plot_exchanges(total_exchanges: int = 0, states_used: int = 0) -> None:
    plt.bar(["Total", "Used"], [total_exchanges, states_used])
    plt.title(f"Exchange Use Rate {100 * states_used / max(1, total_exchanges)}%")
    plt.ylabel("Exchange Count")
    plt.show()


if __name__ == "__main__":
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    try:
        state_actions, exchanged_actions, episode_data = load_data()
    except FileNotFoundError:
        state_actions, exchanged_actions, episode_data = train()
        if SAVE_AFTER_TRAINING:
            save_data(state_actions, exchanged_actions, episode_data)

    plot_episode_data(episode_data)
    exchanged_actions: DefaultDict = exchanged_actions
    total_exchanges, states_used = get_state_use_count(exchanged_actions)
    plot_exchanges(total_exchanges, states_used)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions, exchanged_actions)

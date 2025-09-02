import math
from typing import Any
import dill
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator
from scripts.scavenging_ant import ScavengingAntEnv, ACTION_ROTATIONS
from tqdm import tqdm
from scripts.types import *
from scripts.utils import *

EPISODES = 1000
SEED = 1
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.95
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = False
GRID_WIDTH = 10
GRID_HEIGHT = 10
AGENT_COUNT = 1
FOOD_COUNT = 10
OBSTACLE_COUNT = 10
NEST_COUNT = 1
AGENT_VISION_RADIUS = 10
CARRY_CAPACITY = 5

SQUARE_PIXEL_WIDTH = 40
RENDER_FPS = 30
SECONDS_BETWEEN_AUTO_STEP = 0.10
ACTION_COUNT = len(ACTION_ROTATIONS)
SPARSE_INTERVAL = 1
DRAW_ARROWS = False
SAVE_AFTER_TRAINING = True
SHOW_AFTER_TRAINING = True
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
    f"f{CARRY_CAPACITY}"
)


def policy_factory() -> Policy:
    return {
        "actions": np.zeros(ACTION_COUNT).tolist(),
        "used": False,
        "given": False
    }


def state_actions_factory() -> StateActions:
    return {
        "returning": defaultdict(lambda: defaultdict(lambda: policy_factory())),
        "searching": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: policy_factory()))),
    }


def load_data() -> Tuple[StateActions, List[Episode]]:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "rb") as file:
        data = dill.load(file)
        return data["state_actions"], data["episode_data"]


def save_data(
        state_actions: StateActions,
        episode_data: List[Episode]
) -> None:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "wb") as file:
        dill.dump({
            "state_actions": state_actions,
            "episode_data": episode_data
        }, file)


def sparsify(data: Any) -> Any:
    sparse_data = []
    for index in range(0, len(data), SPARSE_INTERVAL):
        sparse_data.append(data[index])
    return sparse_data


def get_action_values(
        state_actions: StateActions,
        agent_name: AgentName,
        agent_location: Location,
        food_locations: FoodLocations,
        carrying_food: bool,
) -> Actions:
    policy: Policy
    if carrying_food:
        policy = state_actions["returning"][agent_name][agent_location]
    else:
        policy = state_actions["searching"][agent_name][agent_location][food_locations]

    if AGENTS_EXCHANGE_INFO:
        if policy["given"]:
            policy["used"] = True
    return policy["actions"]


def update_action_values(
        state_actions: StateActions,
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
    old_actions = get_action_values(state_actions, agent_name, old_agent_location, old_food_positions,
                                    was_carrying_food)
    new_actions = get_action_values(state_actions, agent_name, new_agent_location, new_food_positions, is_carrying_food)
    old_actions[selected_action] = old_actions[selected_action] + LEARNING_RATE_ALPHA * (
            reward + DISCOUNT_FACTOR_GAMMA * np.max(new_actions) - old_actions[selected_action])


def are_close_enough(
        agent_1_location: Location,
        agent_2_location: Location
) -> bool:
    dx = agent_1_location[0] - agent_2_location[0]
    dy = agent_1_location[1] - agent_2_location[1]
    distance = math.floor(math.hypot(dx, dy))
    return distance <= AGENT_VISION_RADIUS


def fill_missing_search_policy(
        state_actions: StateActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName,
) -> None:
    source = state_actions["searching"][from_agent_name]
    destination = state_actions["searching"][to_agent_name]

    for agent_location, food_locations_to_policy in source.items():
        for food_locations, policy in food_locations_to_policy.items():
            if not food_locations in destination[agent_location]:
                destination[agent_location][food_locations]["actions"] = policy["actions"].copy()
                destination[agent_location][food_locations]["given"] = True


def fill_missing_return_policy(
        state_actions: StateActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName,
) -> None:
    source = state_actions["returning"][from_agent_name]
    destination = state_actions["returning"][to_agent_name]

    for agent_location, policy in source.items():
        if agent_location not in destination:
            destination[agent_location]["actions"] = policy["actions"].copy()
            destination[agent_location]["given"] = True


def fill_policy_gaps(
        state_actions: StateActions,
        agent_1_name: AgentName,
        agent_2_name: AgentName,
        agent_1_observation: Observation,
        agent_2_observation: Observation,
) -> None:
    if agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        fill_missing_return_policy(state_actions, agent_1_name, agent_2_name)
        fill_missing_return_policy(state_actions, agent_2_name, agent_1_name)
    elif not agent_1_observation["carrying_food"] and not agent_2_observation["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        fill_missing_search_policy(state_actions, agent_1_name, agent_2_name)
        fill_missing_search_policy(state_actions, agent_2_name, agent_1_name)
    elif not agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        fill_missing_return_policy(state_actions, agent_1_name, agent_2_name)
        fill_missing_search_policy(state_actions, agent_2_name, agent_1_name)
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        fill_missing_return_policy(state_actions, agent_1_name, agent_2_name)
        fill_missing_search_policy(state_actions, agent_2_name, agent_1_name)


def exchange(
        state_actions: StateActions,
        observations: Dict[AgentName, Observation]
) -> None:
    length = len(observations)
    for index_1 in range(length):
        for index_2 in range(index_1 + 1, length):
            agent_1_name = f"agent_{index_1}"
            agent_2_name = f"agent_{index_2}"
            agent_1_observation: Observation = observations[agent_1_name]
            agent_2_observation: Observation = observations[agent_2_name]

            close_enough = are_close_enough(
                agent_1_observation["agent_location"],
                agent_2_observation["agent_location"]
            )

            if close_enough:
                fill_policy_gaps(
                    state_actions,
                    agent_1_name,
                    agent_2_name,
                    agent_1_observation,
                    agent_2_observation
                )


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
        observations: Dict[AgentName, Observation]
) -> Dict[AgentName, int]:
    greedy_actions = {}
    for agent_name, observation in observations.items():
        greedy_actions[agent_name] = int(
            np.argmax(
                get_action_values(
                    state_actions,
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


def train() -> Tuple[StateActions, List[Episode]]:
    env = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        carry_capacity=CARRY_CAPACITY,
    )

    state_actions: StateActions = state_actions_factory()
    episode_data: List[Episode] = []

    epsilon = EPSILON_START
    rng = np.random.default_rng(seed=SEED)
    episode_progress_bar = tqdm(total=EPISODES, desc="Training")

    for episode in range(EPISODES):
        observations, _ = env.reset()
        terminations, truncations = {}, {}
        current_episode: Episode = {
            "steps": 0,
            "rewards": {agent_name: 0 for agent_name in env.agent_names}
        }

        while not has_episode_ended(terminations, truncations):
            selected_actions = get_epsilon_greedy_actions(state_actions, observations, epsilon, rng)
            new_observations, rewards, terminations, truncations, infos = env.step(selected_actions)

            for agent_name, reward in rewards.items():
                current_episode["rewards"][agent_name] += reward

            for agent_name, new_observation in new_observations.items():
                old_observation = observations[agent_name]
                update_action_values(
                    state_actions,
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
                exchange(state_actions, new_observations)

            current_episode["steps"] += 1
            observations = new_observations

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0.01)
        episode_progress_bar.update(1)
        episode_data.append(current_episode)

    episode_progress_bar.close()

    return state_actions, episode_data


def draw(
        env: ScavengingAntEnv,
        observations: Dict[AgentName, Observation],
        selected_agent_index: int,
        window: pygame.Surface,
        window_size: (int, int),
) -> None:
    canvas = pygame.Surface(window_size)
    env.draw(canvas)

    if DRAW_ARROWS:
        selected_agent_name = f"agent_{selected_agent_index}"
        selected_agent_observation = observations[selected_agent_name]

        food_locations = selected_agent_observation["food_locations"]
        carrying_food = selected_agent_observation["carrying_food"]

        for row in range(GRID_HEIGHT):
            for column in range(GRID_WIDTH):
                agent_position = (column, row)
                action_values = get_action_values(
                    state_actions,
                    selected_agent_name,
                    agent_position,
                    food_locations,
                    carrying_food
                )

                image = pygame.image.load(f"../images/icons8-triangle-48.png")
                image = change_image_color(image, env.get_agent_color(selected_agent_name))
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


def visualize(state_actions: StateActions) -> None:
    env = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        square_pixel_width=SQUARE_PIXEL_WIDTH,
        carry_capacity=CARRY_CAPACITY,
    )

    pygame.init()
    pygame.display.set_caption("Q-Learning Ants")
    pygame.display.set_icon(pygame.image.load("../images/icons8-ant-48.png"))

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
        observations, _ = env.reset()
        terminations, truncations = {}, {}

        draw(
            env,
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
                selected_actions = get_greedy_actions(state_actions, observations)
                observations, rewards, terminations, truncations, info = env.step(selected_actions)

                draw(
                    env,
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

            delta_time = clock.tick(RENDER_FPS) / 1000
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
    plt.title("Steps")
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

    plt.title("Rewards")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()


def get_search_exchanges(state_actions: StateActions) -> Tuple[int, int]:
    given_count, used_count = 0, 0
    for agent_name, agent_location_to_food_locations in state_actions["searching"].items():
        for agent_location, food_locations_to_policy in agent_location_to_food_locations.items():
            for food_locations, policy in food_locations_to_policy.items():
                if policy["given"]:
                    given_count += 1
                    if policy["used"]:
                        used_count += 1

    return given_count, used_count


def get_return_exchanges(state_actions: StateActions) -> Tuple[int, int]:
    given_count, used_count = 0, 0
    for agent_name, agent_location_to_policy in state_actions["returning"].items():
        for agent_location, policy in agent_location_to_policy.items():
            if policy["given"]:
                given_count += 1
                if policy["used"]:
                    used_count += 1

    return given_count, used_count


def plot_exchanges(state_actions: StateActions) -> None:
    given_count, used_count = get_search_exchanges(state_actions)
    plt.bar(["Given", "Used"], [given_count, used_count], color=["orange", "red"])
    plt.title("Search Exchanges")
    plt.ylabel("Amount")
    plt.show()

    given_count, used_count = get_return_exchanges(state_actions)
    plt.bar(["Given", "Used"], [given_count, used_count], color=["orange", "red"])
    plt.title("Return Exchanges")
    plt.ylabel("Amount")
    plt.show()


if __name__ == "__main__":
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    try:
        state_actions, episode_data = load_data()
    except FileNotFoundError:
        state_actions, episode_data = train()
        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episode_data)

    plot_exchanges(state_actions)
    plot_episode_data(episode_data)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions)

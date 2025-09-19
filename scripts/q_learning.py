import copy
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

EPISODES = 10000
SEED = 4
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.90
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = True
GRID_WIDTH = 15
GRID_HEIGHT = 10
AGENT_COUNT = 2
FOOD_COUNT = 10
OBSTACLE_COUNT = 20
NEST_COUNT = 3
AGENT_VISION_RADIUS = 0
CARRY_CAPACITY = 1

SQUARE_PIXEL_WIDTH = 40
RENDER_FPS = 30
SECONDS_BETWEEN_AUTO_STEP = 0.10
ACTION_COUNT = len(ACTION_ROTATIONS)
SPARSE_INTERVAL = 1000
DRAW_ARROWS = False
SAVE_AFTER_TRAINING = True
SHOW_AFTER_TRAINING = True
SAVE_DIRECTORY = "../runs/q_learning"
FILE_NAME = (
    f"LRA={LEARNING_RATE_ALPHA} "
    f"DFG={DISCOUNT_FACTOR_GAMMA} "
    f"EDR={EPSILON_DECAY_RATE} "
    f"E={EPISODES} "
    f"S={SEED} "
    f"GW={GRID_WIDTH} "
    f"GH={GRID_HEIGHT} "
    f"AC={AGENT_COUNT} "
    f"FC={FOOD_COUNT} "
    f"NC={NEST_COUNT} "
    f"OC={OBSTACLE_COUNT} "
    f"AVR={AGENT_VISION_RADIUS} "
    f"AEI={AGENTS_EXCHANGE_INFO} "
    f"CC={CARRY_CAPACITY}"
)


def policy_factory() -> Policy:
    return {
        "actions": np.zeros(ACTION_COUNT).tolist()
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


def _is_shared(key, shared_map, agent_map) -> bool:
    return key in shared_map and key in agent_map and np.array_equal(shared_map[key], agent_map[key])


def _maybe_clone_shared(shared_map, agent_map, key):
    if key in shared_map and key not in agent_map:
        agent_map[key] = copy.deepcopy(shared_map[key])
        return agent_map[key]
    return None


def get_returning_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    agent_location: Location
) -> Tuple[Policy, bool]:
    shared_map = state_actions["returning"]["shared"]
    agent_map = state_actions["returning"][agent_name]

    shared = _is_shared(agent_location, shared_map, agent_map)
    policy = _maybe_clone_shared(shared_map, agent_map, agent_location)

    if policy is None:
        policy = agent_map[agent_location]

    return policy, shared


def get_searching_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    agent_location: Location,
    food_locations: FoodLocations,
) -> Tuple[Policy, bool]:
    shared_map = state_actions["searching"]["shared"][agent_location]
    agent_map = state_actions["searching"][agent_name][agent_location]

    shared = _is_shared(food_locations, shared_map, agent_map)
    policy = _maybe_clone_shared(shared_map, agent_map, food_locations)

    if policy is None:
        policy = agent_map[food_locations]

    return policy, shared


def get_action_values(
        state_actions: StateActions,
        agent_name: AgentName,
        agent_location: Location,
        food_locations: FoodLocations,
        carrying_food: bool,
) -> (Actions, bool):
    if carrying_food:
        policy, shared = get_returning_policy(state_actions, agent_name, agent_location)
    else:
        policy, shared = get_searching_policy(state_actions, agent_name, agent_location, food_locations)

    return policy["actions"], shared


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
    old_actions, _ = get_action_values(state_actions, agent_name, old_agent_location, old_food_positions, was_carrying_food)
    new_actions, _ = get_action_values(state_actions, agent_name, new_agent_location, new_food_positions, is_carrying_food)
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


def share_search_policy(
        state_actions: StateActions,
        from_agent_name: AgentName
) -> int:
    source = state_actions["searching"][from_agent_name]
    destination = state_actions["searching"]["shared"]
    exchange_count = 0

    for agent_location, food_locations_to_policy in source.items():
        for food_location, policy in food_locations_to_policy.items():
            if not food_location in destination[agent_location]:
                destination[agent_location][food_location] = copy.deepcopy(policy)
                exchange_count += 1
    return exchange_count


def share_return_policy(
        state_actions: StateActions,
        from_agent_name: AgentName
) -> int:
    source = state_actions["returning"][from_agent_name]
    destination = state_actions["returning"]["shared"]
    exchange_count = 0

    for agent_location, policy in source.items():
        if not agent_location in destination:
            destination[agent_location] = copy.deepcopy(policy)
            exchange_count += 1
    return exchange_count


def fill_policy_gaps(
        state_actions: StateActions,
        agent_1_name: AgentName,
        agent_2_name: AgentName,
        agent_1_observation: Observation,
        agent_2_observation: Observation,
        episode: Episode,
) -> None:
    if agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_2_name)
    elif not agent_1_observation["carrying_food"] and not agent_2_observation["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)
    elif not agent_1_observation["carrying_food"] and agent_2_observation["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)


def exchange(
        state_actions: StateActions,
        observations: Dict[AgentName, Observation],
        episode: Episode
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
                    agent_2_observation,
                    episode
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
        action_values, _ = get_action_values(
            state_actions,
            agent_name,
            observation["agent_location"],
            observation["food_locations"],
            observation["carrying_food"],
        )

        greedy_actions[agent_name] = int(np.argmax(action_values))

    return greedy_actions


def get_epsilon_greedy_actions(
        state_actions: StateActions,
        observations: Dict[AgentName, Observation],
        epsilon: float,
        rng: Generator,
        episode: Episode
) -> (Dict[AgentName, int]):
    epsilon_greedy_actions = {}
    for agent_name, observation in observations.items():
        if rng.random() > epsilon:
            action_values, shared = get_action_values(
                state_actions,
                agent_name,
                observation["agent_location"],
                observation["food_locations"],
                observation["carrying_food"],
            )

            epsilon_greedy_actions[agent_name] = int(np.argmax(action_values))
            if shared:
                if observation["carrying_food"]:
                    episode["return_exchange_use_count"] += 1
                else:
                    episode["search_exchange_use_count"] += 1
        else:
            epsilon_greedy_actions[agent_name] = rng.integers(0, ACTION_COUNT)

    return epsilon_greedy_actions


def train(env: ScavengingAntEnv) -> Tuple[StateActions, List[Episode]]:
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
            "rewards": {agent_name: 0 for agent_name in env.agent_names},
            "search_exchange_count": 0,
            "return_exchange_count": 0,
            "return_exchange_use_count": 0,
            "search_exchange_use_count": 0
        }

        while not has_episode_ended(terminations, truncations):
            selected_actions = get_epsilon_greedy_actions(state_actions, observations, epsilon, rng, current_episode)
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
                exchange(state_actions, new_observations, current_episode)

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
                action_values, _ = get_action_values(
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


def visualize(state_actions: StateActions, env: ScavengingAntEnv) -> None:
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
    episode_search_exchanges = []
    episode_search_exchange_uses = []
    episode_return_exchanges = []
    episode_return_exchange_uses = []

    for episode in episode_data:
        episode_steps.append(episode["steps"])
        episode_rewards.append(episode["rewards"])
        episode_search_exchanges.append(episode["search_exchange_count"])
        episode_search_exchange_uses.append(episode["search_exchange_use_count"])
        episode_return_exchanges.append(episode["return_exchange_count"])
        episode_return_exchange_uses.append(episode["return_exchange_use_count"])

    plt.plot(episodes, episode_search_exchange_uses, color="red", label="Used Search Exchanges")
    plt.plot(episodes, episode_search_exchanges, color="blue", label="Search Exchanges")
    plt.plot(episodes, episode_return_exchanges, color="green", label="Return Exchanges")
    plt.plot(episodes, episode_return_exchange_uses, color="purple", label="Used Return Exchanges")
    plt.title("Exchanges")
    plt.xlabel(f"Episode (Step = {SPARSE_INTERVAL})")
    plt.legend()
    plt.show()

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episode (Step = {SPARSE_INTERVAL})")
    plt.show()

    episode_agent_rewards = defaultdict(list)
    for agent_rewards in episode_rewards:
        for name, reward in agent_rewards.items():
            episode_agent_rewards[name].append(reward)

    for name, rewards in episode_agent_rewards.items():
        plt.plot(episodes, rewards, label=name)

    plt.title("Rewards")
    plt.xlabel(f"Episode (Step = {SPARSE_INTERVAL})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

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

    try:
        state_actions, episode_data = load_data()
    except FileNotFoundError:
        state_actions, episode_data = train(env)
        if SAVE_AFTER_TRAINING:
            print("Saving")
            save_data(state_actions, episode_data)

    plot_episode_data(episode_data)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions, env)

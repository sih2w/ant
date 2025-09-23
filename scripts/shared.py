import copy
import math
import numpy as np
from collections import defaultdict
from scripts.utils import change_image_color
from scripts.scavenging_ant import ScavengingAntEnv, ACTION_ROTATIONS
from scripts.types import *
from scripts.config import *


def policy_factory() -> Policy:
    return {
        "actions": np.zeros(ACTION_COUNT).tolist()
    }


def state_actions_factory() -> StateActions:
    return {
        "returning": defaultdict(lambda: defaultdict(lambda: policy_factory())),
        "searching": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: policy_factory()))),
    }


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


def draw(
        env: ScavengingAntEnv,
        observations: Dict[AgentName, Observation],
        selected_agent_index: int,
        window: pygame.Surface,
        window_size: (int, int),
        state_actions: StateActions,
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
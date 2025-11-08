import copy
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from workspace.classes.environment import Environment
from workspace.shared.types import *
from workspace.shared.enums import *
from workspace.shared.run_settings import *
from workspace.functions.policy_functions import get_training_actions, exchange
from workspace.functions.episode_functions import has_episode_ended,episode_factory
from workspace.functions.policy_functions import update_policy, gridded_policy_factory, state_actions_factory
from multiprocessing import Process


def average_gridded_policies(gridded_policies: List[GriddedPolicy]) -> GriddedPolicy:
    averaged_gridded_policy = gridded_policy_factory()

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            averaged_policy = averaged_gridded_policy[row][column]
            averaged_actions = averaged_policy[PolicyAttr.ACTIONS.value]

            for gridded_policy in gridded_policies:
                policy = gridded_policy[row][column]
                actions = policy[PolicyAttr.ACTIONS.value]
                for index, value in enumerate(actions):
                    averaged_actions[index] += value

            for index, value in enumerate(averaged_actions):
                averaged_actions[index] /= len(gridded_policies)

    return averaged_gridded_policy


def average_returning_policies(returning_policies: List[ReturningPolicies]) -> ReturningPolicies:
    average_returning_policy: ReturningPolicies = returning_policies[0]

    progress_bar = tqdm(
        total=AGENT_COUNT,
        desc="Average Return Policies"
    )

    for agent_index in range(AGENT_COUNT):
        gridded_policies: List[GriddedPolicy] = []
        for worker_gridded_policies in returning_policies:
            gridded_policies.append(worker_gridded_policies[agent_index])
        average_returning_policy[agent_index] = average_gridded_policies(gridded_policies)
        progress_bar.update(1)
    progress_bar.close()

    return average_returning_policy


def average_searching_policies(searching_policies: List[SearchingPolicies]) -> SearchingPolicies:
    average_searching_policy: SearchingPolicies = searching_policies[0]

    progress_bar = tqdm(
        total=AGENT_COUNT,
        desc="Average Search Policies"
    )

    for agent_index in range(AGENT_COUNT):
        gridded_policies: Dict[FoodLocations, List[GriddedPolicy]] = {}
        for worker_searching_policies in searching_policies:
            food_locations_to_gridded_policies = worker_searching_policies[agent_index]
            for food_locations, gridded_policy in food_locations_to_gridded_policies.items():
                gridded_policies.setdefault(food_locations, [])
                gridded_policies[food_locations].append(gridded_policy)
        for food_locations, gridded_policy in gridded_policies.items():
            average_searching_policy[agent_index][food_locations] = average_gridded_policies(gridded_policy)
        progress_bar.update(1)
    progress_bar.close()

    return average_searching_policy


def average_state_actions(worker_state_actions: List[StateActions]) -> StateActions:
    returning_policies = []
    for state_actions in worker_state_actions:
        returning_policies.append(state_actions["returning"])

    searching_policies = []
    for state_actions in worker_state_actions:
        searching_policies.append(state_actions["searching"])

    return {
        "returning": average_returning_policies(returning_policies),
        "searching": average_searching_policies(searching_policies),
    }


def train_episode(
        environment: Environment,
        epsilon: float,
        state_actions: StateActions,
        rng: np.random.Generator,
) -> Episode:
    states, _ = environment.reset()
    terminations, truncations = [], []

    episode: Episode = episode_factory()
    episode[EpisodeAttr.EPSILON.value] = epsilon

    while not has_episode_ended(terminations, truncations):
        selected_actions: List[int] = get_training_actions(
            state_actions=state_actions,
            states=states,
            epsilon=epsilon,
            rng=rng
        )

        new_states, rewards, terminations, truncations, infos = environment.step(selected_actions)

        for index, reward in enumerate(rewards):
            episode[EpisodeAttr.REWARDS.value][index] += reward

        for index, new_state in enumerate(new_states):
            update_policy(
                state_actions=state_actions,
                agent_index=index,
                old_state=states[index],
                new_state=new_state,
                selected_action_index=selected_actions[index],
                reward=rewards[index]
            )

        if EXCHANGE_INFO:
            exchange(
                environment=environment,
                state_actions=state_actions,
                episode=episode
            )

        episode[EpisodeAttr.STEPS.value] += 1
        states = new_states

    return episode


def train_parallel(
        environment: Environment,
        worker_index: int,
        worker_state_actions: List[StateActions],
        worker_episodes: List[List[Episode]],
):
    rng = np.random.default_rng(seed=(SEED + worker_index))
    environment = copy.deepcopy(environment)
    epsilon = 1.00

    episodes: List[Episode] = []
    state_actions = state_actions_factory()

    progress_bar = tqdm(
        total=WORKER_EPISODE_COUNT,
        desc=f"Training [Worker {worker_index}]"
    )

    for _ in range(WORKER_EPISODE_COUNT):
        episode = train_episode(
            environment=environment,
            epsilon=epsilon,
            state_actions=state_actions,
            rng=rng,
        )

        epsilon -= EPSILON_DECAY_RATE
        progress_bar.update(1)
        episodes.append(episode)

    progress_bar.close()

    worker_episodes[worker_index] = episodes
    worker_state_actions[worker_index] = state_actions


def train_from_existing(
        environment: Environment,
        state_actions: StateActions,
        episodes: List[Episode],
) -> None:
    rng = np.random.default_rng(seed=(SEED + WORKER_COUNT + 1))
    epsilon = 1.00 - (WORKER_EPISODE_COUNT * EPSILON_DECAY_RATE)

    progress_bar = tqdm(
        total=MERGED_EPISODE_COUNT,
        desc=f"Training [Merged]"
    )

    for current_episode in range(MERGED_EPISODE_COUNT):
        episode = train_episode(
            environment=environment,
            epsilon=epsilon,
            state_actions=state_actions,
            rng=rng,
        )

        epsilon = max(epsilon - EPSILON_DECAY_RATE, EPSILON_MIN)
        progress_bar.update(1)
        episodes.append(episode)

    progress_bar.close()


def train(environment: Environment) -> Tuple[StateActions, List[Episode]]:
    manager = mp.Manager()
    worker_state_actions = manager.list(range(WORKER_COUNT))
    worker_episodes = manager.list(range(WORKER_COUNT))

    workers: List[Process] = []

    for worker_index in range(WORKER_COUNT):
        workers.append(
            Process(
                target=train_parallel,
                args=(
                    environment,
                    worker_index,
                    worker_state_actions,
                    worker_episodes
                )
            )
        )

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
        worker.close()

    state_actions = average_state_actions(worker_state_actions)
    episodes = worker_episodes[0]

    train_from_existing(
        environment=environment,
        state_actions=state_actions,
        episodes=episodes
    )

    return state_actions, episodes
import copy
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from workspace.classes.environment import ScavengingAntEnv
from workspace.types import *
from workspace.functions.policy_functions import get_training_actions, update_policy_use, exchange
from workspace.functions.episode_functions import has_episode_ended,episode_factory
from workspace.functions.policy_functions import update_policy, gridded_policy_factory, state_actions_factory
from workspace.enums import EpisodeAttribute, PolicyAttribute
from multiprocessing import Process


# def average_gridded_policies(
#         gridded_policies: List[GriddedPolicy],
#         grid_width: int,
#         grid_height: int
# ) -> GriddedPolicy:
#     averaged_gridded_policy = gridded_policy_factory(grid_width, grid_height)
#
#     for row in range(grid_height):
#         for column in range(grid_width):
#             averaged_policy = averaged_gridded_policy[row][column]
#             averaged_actions = averaged_policy[PolicyAttribute.ACTIONS.value]
#
#             for gridded_policy in gridded_policies:
#                 policy = gridded_policy[row][column]
#                 actions = policy[PolicyAttribute.ACTIONS.value]
#                 for index, value in enumerate(actions):
#                     averaged_actions[index] += value
#
#             for index, value in enumerate(averaged_actions):
#                 averaged_actions[index] /= len(gridded_policies)
#
#     return averaged_gridded_policy
#
#
# def average_returning_policies(
#         returning_policies: List[ReturningPolicies],
#         grid_width: int,
#         grid_height: int
# ) -> ReturningPolicies:
#     pass
#
#
# def average_searching_policies(
#         searching_policies: List[SearchingPolicies],
#         grid_width: int,
#         grid_height: int
# ) -> SearchingPolicies:
#     pass
#
#
# def average_state_actions(
#         worker_state_actions: List[StateActions],
#         grid_width: int,
#         grid_height: int
# ) -> StateActions:
#     returning_policies = []
#     for state_actions in worker_state_actions:
#         returning_policies.append(state_actions["returning"])
#
#     searching_policies = []
#     for state_actions in worker_state_actions:
#         searching_policies.append(state_actions["searching"])
#
#     return {
#         "returning": average_returning_policies(returning_policies, grid_width, grid_height),
#         "searching": average_searching_policies(searching_policies, grid_width, grid_height),
#     }
#
#
# def average_episodes(episodes: List[List[Episode]]) -> List[Episode]:
#     return episodes[0]


def get_food_pickup_callbacks(agent_count: int) -> List[FoodPickupCallback]:
    def food_pickup_callback(agent_index: int, environment_state: EnvironmentState) -> bool:
        return True

    return [food_pickup_callback] * agent_count


def get_action_verification_callbacks(agent_count: int) -> List[ActionVerificationCallback]:
    def action_verification_callback(agent_index: int, action_index: int, environment_state: EnvironmentState) -> Tuple[bool, int]:
        return True, -1

    return [action_verification_callback] * agent_count


def train_episode(
        environment: ScavengingAntEnv,
        discount_factor_gamma: float,
        learning_rate_alpha: float,
        epsilon: float,
        exchange_info: bool,
        agent_vision_radius: float,
        state_actions: StateActions,
        rng: np.random.Generator,
) -> Episode:
    states, _ = environment.reset()
    terminations, truncations = [], []

    agent_count = environment.get_agent_count()
    grid_width = environment.get_grid_width()
    grid_height = environment.get_grid_height()

    food_pickup_callbacks = get_food_pickup_callbacks(agent_count)
    action_verification_callbacks = get_action_verification_callbacks(agent_count)

    episode: Episode = episode_factory(agent_count)

    while not has_episode_ended(terminations, truncations):
        selected_actions: List[int] = get_training_actions(
            state_actions=state_actions,
            states=states,
            epsilon=epsilon,
            rng=rng,
            grid_height=grid_height,
            grid_width=grid_width,
        )

        for agent_index, action_index in enumerate(selected_actions):
            callback = action_verification_callbacks[agent_index]
            success, new_action_index = callback(agent_index, action_index, environment.get_environment_state())
            if not success:
                selected_actions[agent_index] = new_action_index

        update_policy_use(
            episode=episode,
            states=states,
            state_actions=state_actions,
            grid_width=grid_width,
            grid_height=grid_height,
        )

        new_states, rewards, terminations, truncations, infos = environment.step(selected_actions, food_pickup_callbacks)

        for index, reward in enumerate(rewards):
            episode[EpisodeAttribute.REWARDS.value][index] += reward

        for index, new_state in enumerate(new_states):
            update_policy(
                state_actions=state_actions,
                agent_index=index,
                old_state=states[index],
                new_state=new_state,
                selected_action_index=selected_actions[index],
                reward=rewards[index],
                discount_factor_gamma=discount_factor_gamma,
                learning_rate_alpha=learning_rate_alpha,
                grid_width=grid_width,
                grid_height=grid_height,
            )

        if exchange_info:
            exchange(
                state_actions=state_actions,
                states=new_states,
                episode=episode,
                grid_width=environment.get_grid_width(),
                grid_height=environment.get_grid_height(),
                agent_vision_radius=agent_vision_radius
            )

        episode[EpisodeAttribute.STEPS.value] += 1
        states = new_states

    return episode


def train_parallel(
        environment: ScavengingAntEnv,
        episode_count: int,
        worker_index: int,
        discount_factor_gamma: float,
        learning_rate_alpha: float,
        epsilon_decay_rate: float,
        epsilon_min: float,
        exchange_info: bool,
        agent_vision_radius: float,
        worker_state_actions: List[StateActions],
        worker_episodes: List[List[Episode]],
):
    rng = np.random.default_rng(seed=environment.get_seed() + worker_index)
    environment = copy.deepcopy(environment)
    epsilon = 1.00

    episodes: List[Episode] = []
    state_actions = state_actions_factory(
        grid_width=environment.get_grid_width(),
        grid_height=environment.get_grid_height(),
        agent_count=environment.get_agent_count()
    )

    progress_bar = tqdm(
        total=episode_count,
        desc=f"Training [Worker {worker_index}]"
    )

    for current_episode in range(episode_count):
        episode = train_episode(
            environment=environment,
            discount_factor_gamma=discount_factor_gamma,
            learning_rate_alpha=learning_rate_alpha,
            epsilon=epsilon,
            exchange_info=exchange_info,
            agent_vision_radius=agent_vision_radius,
            state_actions=state_actions,
            rng=rng,
        )

        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)
        progress_bar.update(1)
        episodes.append(episode)

    progress_bar.close()

    worker_episodes[worker_index] = episodes
    worker_state_actions[worker_index] = state_actions


def train(
        environment: ScavengingAntEnv,
        episode_count: int,
        exchange_info: bool,
        epsilon_decay_rate: float,
        epsilon_min: float,
        discount_factor_gamma: float,
        learning_rate_alpha: float,
        agent_vision_radius: float,
        worker_count: int,
) -> Tuple[StateActions, List[Episode]]:
    manager = mp.Manager()
    worker_state_actions = manager.list(range(worker_count))
    worker_episodes = manager.list(range(worker_count))

    workers: List[Process] = []

    for worker_index in range(worker_count):
        workers.append(
            Process(
                target=train_parallel,
                args=(
                    environment,
                    episode_count,
                    worker_index,
                    discount_factor_gamma,
                    learning_rate_alpha,
                    epsilon_decay_rate,
                    epsilon_min,
                    exchange_info,
                    agent_vision_radius,
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

    return worker_state_actions[0], worker_episodes[0]
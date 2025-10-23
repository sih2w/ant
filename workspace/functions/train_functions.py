import numpy as np
from tqdm import tqdm
from workspace.classes.environment import ScavengingAntEnv
from workspace.shared_types import *
from workspace.functions.policy_functions import get_training_actions, update_policy_use, exchange
from workspace.functions.episode_functions import has_episode_ended
from workspace.functions.policy_functions import update_policy, state_actions_factory


def train(
        environment: ScavengingAntEnv,
        episode_count: int = 100,
        exchange_info: bool = True,
        epsilon_decay_rate: float = 0.00001,
        epsilon_min: float = 0.01,
        discount_factor_gamma: float = 0.99,
        learning_rate_alpha: float = 0.1,
        agent_vision_radius: float = 1.0,
) -> Tuple[StateActions, List[Episode]]:
    state_actions: StateActions = state_actions_factory(
        environment.get_grid_width(),
        environment.get_grid_height()
    )

    episode_data: List[Episode] = []
    epsilon = 1
    rng = np.random.default_rng(seed=environment.get_seed())
    progress_bar = tqdm(total=episode_count, desc="Training")

    for _ in range(episode_count):
        states, _ = environment.reset()
        episode: Episode = {
            "steps": 0,
            "rewards": [0 for _ in range(environment.get_agent_count())],
            "given_search_policies": 0,
            "given_return_policies": 0,
            "averaged_search_policies": 0,
            "averaged_return_policies": 0,
            "used_search_policies": 0,
            "used_return_policies": 0
        }

        terminations: List[bool] = []
        truncations: List[bool] = []

        while not has_episode_ended(terminations, truncations):
            selected_actions: List[int] = get_training_actions(
                state_actions=state_actions,
                states=states,
                epsilon=epsilon,
                rng=rng
            )

            update_policy_use(
                episode=episode,
                states=states,
                state_actions=state_actions
            )

            new_states, rewards, terminations, truncations, infos = environment.step(selected_actions)
            for index, reward in enumerate(rewards):
                episode["rewards"][index] += reward

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

            episode["steps"] += 1
            states = new_states

        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)
        progress_bar.update(1)
        episode_data.append(episode)

    progress_bar.close()

    return state_actions, episode_data
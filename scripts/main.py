import copy
from collections import defaultdict
from typing import Callable, Optional, TypeVar
import dill
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scripts.temporal_difference_learning import q_learning, sarsa_learning
from scripts.scavenging_ant import ScavengingAntEnv, ACTION_ROTATIONS
from scripts.utils import *
from scripts.constants import *
from scripts.types import *
from scripts.state_action_functions import get_decided_actions, get_training_actions
from scripts.exchange_functions import exchange


T = TypeVar("T")
PolicyUpdator: TypeAlias = Callable[[StateActions, AgentName, State, State, int, int, int, float], None]


def policy_factory() -> Policy:
    return {
        "actions": np.zeros(ACTION_COUNT).tolist(),
        "given": False,
        "used": False
    }


def create_grid(callback: Callable[[], T]) -> List[List[T]]:
    grid = []
    for row in range(GRID_HEIGHT):
        new_row = []
        for column in range(GRID_WIDTH):
            new_row.append(callback())
        grid.append(new_row)
    return grid


def state_actions_factory() -> StateActions:
    return {
        "returning": defaultdict(lambda: create_grid(lambda: policy_factory())),
        "searching": defaultdict(lambda: defaultdict(lambda: create_grid(lambda: policy_factory()))),
    }


def has_episode_ended(
        terminations: Dict[AgentName, bool],
        truncations: Dict[AgentName, bool],
        episode: Optional[Episode]
) -> bool:
    def has_max_steps_reached(episode: Optional[Episode]) -> bool:
        return episode is not None and episode["steps"] >= MAX_STEPS

    def contains_true_values(data: Dict[AgentName, bool]) -> bool:
        return any(data.values())

    if has_max_steps_reached(episode):
        return True
    if contains_true_values(terminations):
        return True
    if contains_true_values(truncations):
        return True
    return False


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


def average_episode_data(episode_data: List[Episode]) -> List[Episode]:
    new_episode_data = []

    for current in range(0, len(episode_data), GRAPH_STEP):
        episode_sum: Episode = {
            "steps": 0,
            "search_exchange_count": 0,
            "search_exchange_use_count": 0,
            "return_exchange_count": 0,
            "return_exchange_use_count": 0,
            "rewards": {}
        }
        count = 0

        start = max(0, current - GRAPH_STEP)
        end = min(current, len(episode_data))

        for previous in range(start, end):
            episode: Episode = episode_data[previous]
            episode_sum["steps"] += episode["steps"]
            episode_sum["search_exchange_count"] += episode["search_exchange_count"]
            episode_sum["search_exchange_use_count"] += episode["search_exchange_use_count"]
            episode_sum["return_exchange_count"] += episode["return_exchange_count"]
            episode_sum["return_exchange_use_count"] += episode["return_exchange_use_count"]
            count += 1

            for agent_name, reward in episode["rewards"].items():
                episode_sum["rewards"].setdefault(agent_name, 0)
                episode_sum["rewards"][agent_name] += reward

        if count > 0:
            episode_average: Episode = {
                "steps": round(episode_sum["steps"] / count),
                "search_exchange_count": round(episode_sum["search_exchange_count"] / count),
                "search_exchange_use_count": round(episode_sum["search_exchange_use_count"] / count),
                "return_exchange_count": round(episode_sum["return_exchange_count"] / count),
                "return_exchange_use_count": round(episode_sum["return_exchange_use_count"] / count),
                "rewards": {agent_name: round(total_reward / count) for agent_name, total_reward in episode_sum["rewards"].items()}
            }
            new_episode_data.append(episode_average)

    return new_episode_data


def plot_episode_data(episode_data: List[Episode]) -> None:
    episode_data = average_episode_data(episode_data)
    episodes = [episode * GRAPH_STEP for episode in range(len(episode_data))]

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
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.legend()
    plt.show()

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.show()

    episode_agent_rewards = defaultdict(list)
    for agent_rewards in episode_rewards:
        for name, reward in agent_rewards.items():
            episode_agent_rewards[name].append(reward)

    for name, rewards in episode_agent_rewards.items():
        plt.plot(episodes, rewards, label=name)

    plt.title("Rewards")
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.legend()
    plt.show()


def draw_arrows(
    env: ScavengingAntEnv,
    states: Dict[AgentName, State],
    selected_agent_index: int,
    state_actions: StateActions,
    canvas: pygame.Surface
) -> None:
    agent_name = f"agent_{selected_agent_index}"
    states = copy.deepcopy(states)

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            states[agent_name]["agent_location"] = (column, row)
            agent_actions = get_decided_actions(state_actions, states)

            image = pygame.image.load(f"../images/icons8-triangle-48.png")
            image = change_image_color(image, env.get_agent_color(agent_name))
            rotation = ACTION_ROTATIONS[agent_actions[agent_name]]
            position = (
                column * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_width() / 2,
                row * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - image.get_height() / 2,
            )
            image = pygame.transform.rotate(image, rotation)
            canvas.blit(image, position)


def draw(
        env: ScavengingAntEnv,
        window_size: Tuple[int, int],
        window: pygame.Surface,
        states: Dict[AgentName, State],
        selected_agent_index: int,
        state_actions: StateActions
) -> None:
    canvas = pygame.Surface(window_size)
    env.draw(canvas)

    if DRAW_ARROWS:
        draw_arrows(
            env=env,
            states=states,
            selected_agent_index=selected_agent_index,
            state_actions=state_actions,
            canvas=canvas
        )

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.flip()


def visualize(
        state_actions: StateActions,
        env: ScavengingAntEnv
) -> None:
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
        states, _ = env.reset()
        terminations, truncations = {}, {}

        draw(
            env=env,
            window_size=window_size,
            window=window,
            states=states,
            selected_agent_index=selected_agent_index,
            state_actions=state_actions
        )

        while running and not has_episode_ended(terminations, truncations, None):
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or (stepping_enabled and not stepping)

            if draw_next_step:
                stepping = True
                agent_actions = get_decided_actions(state_actions, states)
                states, rewards, terminations, truncations, info = env.step(agent_actions)

                draw(
                    env=env,
                    window_size=window_size,
                    window=window,
                    states=states,
                    selected_agent_index=selected_agent_index,
                    state_actions=state_actions
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


def train(
        env: ScavengingAntEnv,
        policy_updator: PolicyUpdator
) -> Tuple[StateActions, List[Episode]]:
    state_actions: StateActions = state_actions_factory()
    episode_data: List[Episode] = []

    epsilon = EPSILON_START
    rng = np.random.default_rng(seed=SEED)
    progress_bar = tqdm(total=EPISODES, desc="Training")

    for _ in range(EPISODES):
        states, _ = env.reset()
        episode: Episode = {
            "steps": 0,
            "rewards": {agent_name: 0 for agent_name in env.agent_names},
            "search_exchange_count": 0,
            "return_exchange_count": 0,
            "return_exchange_use_count": 0,
            "search_exchange_use_count": 0
        }

        terminations: Dict[AgentName, bool] = {}
        truncations: Dict[AgentName, bool] = {}
        previous_actions: Dict[AgentName, int] = {}

        while not has_episode_ended(terminations, truncations, episode):
            selected_actions: Dict[AgentName, int] = get_training_actions(
                state_actions=state_actions,
                states=states,
                epsilon=epsilon,
                rng=rng,
                episode=episode
            )

            new_states, rewards, terminations, truncations, infos = env.step(selected_actions)
            for agent_name, reward in rewards.items():
                episode["rewards"][agent_name] += reward

            if len(previous_actions) == 0:
                previous_actions = selected_actions

            for agent_name, new_state in new_states.items():
                policy_updator(
                    state_actions,
                    agent_name,
                    states[agent_name],
                    new_state,
                    selected_actions[agent_name],
                    previous_actions[agent_name],
                    rewards[agent_name],
                    epsilon
                )

            if AGENTS_EXCHANGE_INFO:
                exchange(state_actions, new_states, episode)

            episode["steps"] += 1
            states = new_states
            previous_actions = selected_actions

        epsilon = max(epsilon - EPSILON_DECAY_RATE, EPSILON_MIN)
        progress_bar.update(1)
        episode_data.append(episode)

    progress_bar.close()

    return state_actions, episode_data


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
        policy_updator = sarsa_learning if LEARNING_METHOD == "Sarsa" else q_learning
        state_actions, episode_data = train(env, policy_updator)

        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episode_data)
    plot_episode_data(episode_data)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions, env)
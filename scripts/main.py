from typing import Any, Callable
import dill
import os
import matplotlib.pyplot as plt
from scripts.q_learning import train, get_greedy_actions
from scripts.shared import *


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


def sparse_episode_data(episode_data: List[Episode]) -> Any:
    new_episode_data = []
    for index in range(0, len(episode_data), GRAPH_STEP):
        new_episode_data.append(episode_data[index])
    return new_episode_data


def plot_episode_data(episode_data: List[Episode]) -> None:
    episode_data = sparse_episode_data(episode_data)
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


def visualize(
        state_actions: StateActions,
        env: ScavengingAntEnv,
        action_selector: Callable[[StateActions, Dict[AgentName, Observation]], Dict[AgentName, int]],
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
        observations, _ = env.reset()
        terminations, truncations = {}, {}

        draw(
            env,
            observations,
            selected_agent_index,
            window,
            window_size,
            state_actions,
        )

        while running and not has_episode_ended(terminations, truncations):
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or (stepping_enabled and not stepping)

            if draw_next_step:
                stepping = True
                selected_actions = action_selector(state_actions, observations)
                observations, rewards, terminations, truncations, info = env.step(selected_actions)

                draw(
                    env,
                    observations,
                    selected_agent_index,
                    window,
                    window_size,
                    state_actions,
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
        state_actions, episode_data = train(env) # Can swap train function
        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episode_data)
    plot_episode_data(episode_data)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions, env, get_greedy_actions) # Can swap action selector
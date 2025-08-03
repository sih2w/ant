import ast
import os
import numpy as np
import json
import pygame
import matplotlib.pyplot as plt
from scripts.scavenging_ant import ScavengingAntEnv
from copy import deepcopy
from tqdm import tqdm

type EpisodeData = {
    "steps": [int],
    "rewards": [{[str]: float}]
}

type Q = {
    [tuple, ...]: {
        [tuple[int, int]]: {
            [tuple[str, bool]]: tuple[float, ...]
        }
    }
}

type Observation = {
    "agent_position": tuple[int, int],
    "carrying_food": bool,
    "carried_food": tuple[int, ...],
    "food_positions": tuple[int, ...],
    "agent_detected": bool
}

ACTION_COUNT = 4
EPISODES = 1_000
SEED = 10
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.70
EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
AGENTS_EXCHANGE_INFO = False
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
SECONDS_BETWEEN_AUTO_STEP = 0.50
SPARSE_INTERVAL = int(EPISODES / 100)

def get_rotation_from_action(action: int) -> int or None:
    """
    Returns the rotation from a given action.
    """
    if action == 0:
        return 180
    elif action == 1:
        return 0
    elif action == 2:
        return 90
    elif action == 3:
        return -90

def convert_keys_to_default(dictionary: dict):
    if isinstance(dictionary, dict):
        return {ast.literal_eval(key): convert_keys_to_default(value) for key, value in dictionary.items()}
    return dictionary

def load_data(directory: str, file_name: str) -> (Q, EpisodeData):
    with open(f"{directory}/{file_name}.json", "r") as file:
        data = json.load(file)
        data["q"] = convert_keys_to_default(data["q"])
        return data["q"], data["episode_data"]

def convert_keys_to_strings(dictionary: dict):
    if isinstance(dictionary, dict):
        return {str(key): convert_keys_to_strings(value) for key, value in dictionary.items()}
    return dictionary

def save_data(directory: str, file_name: str, q: Q, episode_data: EpisodeData) -> None:
    with open(f"{directory}/{file_name}.json", "w") as file:
        saved_q = deepcopy(q)
        saved_q = convert_keys_to_strings(saved_q)
        json.dump({"q": saved_q, "episode_data": episode_data}, file)

def sparsify(data: list) -> list:
    sparse_data = []
    for index in range(0, len(data), SPARSE_INTERVAL):
        sparse_data.append(data[index])
    return sparse_data

def reconcile(
        q: Q,
        observation: Observation,
        agent_name: str
):
    if q.get(observation["food_positions"]) is None:
        q[observation["food_positions"]] = {}
    if q[observation["food_positions"]].get(observation["agent_position"]) is None:
        q[observation["food_positions"]][observation["agent_position"]] = {}
    if q[observation["food_positions"]][observation["agent_position"]].get((agent_name, observation["carrying_food"])) is None:
        q[observation["food_positions"]][observation["agent_position"]][(agent_name, observation["carrying_food"])] = np.zeros(ACTION_COUNT).tolist()

def get_actions(
        q: Q,
        observation: Observation,
        agent_name: str
):
    return q[observation["food_positions"]][observation["agent_position"]][(agent_name, observation["carrying_food"])]

def get_actions_at_agent_position(
        q: Q,
        observation: Observation,
        agent_name: str,
        agent_position: (int, int)
):
    actions = q.get(observation["food_positions"])
    if actions is not None:
        actions = actions.get(agent_position)
        if actions is not None:
            actions = actions.get((agent_name, observation["carrying_food"]))
            return actions

def update_actions(
        q: Q,
        observation: Observation,
        new_observation: Observation,
        agent_name: str,
        selected_action: int,
        reward: float
):
    old_actions = get_actions(q, observation, agent_name)
    new_actions = get_actions(q, new_observation, agent_name)
    old_actions[selected_action] = old_actions[selected_action] + LEARNING_RATE_ALPHA * (
                reward + DISCOUNT_FACTOR_GAMMA * np.max(new_actions) - old_actions[selected_action])

def train() -> (Q, EpisodeData):
    """
    Trains Q-learning agents in an environment defined by the passed parameters.
    """
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

    q: Q = {}
    episode_data: EpisodeData = {"steps": [], "rewards": []}

    epsilon = EPSILON_START
    rng = np.random.default_rng()
    episode_progress_bar = tqdm(total=EPISODES, desc="Training")

    for episode in range(EPISODES):
        observations, _ = env.reset(seed=SEED)
        for agent_name, observation in observations.items():
            # Add the observation to the Q table if it doesn't already exist.
            # This initializes all action values for this observation to zero.
            reconcile(q, observation, agent_name)

        terminated, truncated = False, False
        step_count = 0
        total_rewards = {agent_name: 0 for agent_name in env.agents}

        while not terminated and not truncated:
            actions = {}
            for agent_name, observation in observations.items():
                if rng.random() > epsilon:
                    # Get the best action for the current observation.
                    actions[agent_name] = np.argmax(get_actions(q, observation, agent_name))
                else:
                    # Get a random action for the current observation.
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

            # Update the Q table.
            for agent_name, new_observation in new_observations.items():
                reconcile(q, new_observation, agent_name)
                update_actions(
                    q,
                    observations[agent_name],
                    new_observation,
                    agent_name,
                    actions[agent_name],
                    rewards[agent_name]
                )

            step_count = step_count + 1
            observations = new_observations

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0.01)
        episode_data["steps"].append(step_count)
        episode_data["rewards"].append(total_rewards)
        episode_progress_bar.update(1)

    episode_progress_bar.close()
    env.close()

    return q, episode_data

def validate():
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
        # Visualize trained model.
        observations, info = env.reset(seed=SEED)
        for agent_name, observation in observations.items():
            reconcile(q, observation, agent_name)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or stepping_enabled and not stepping

            if draw_next_step:
                stepping = True

                actions = {agent_name: np.argmax(get_actions(q, observations[agent_name], agent_name)) for agent_name in env.agents}
                observations, rewards, terminations, truncations, info = env.step(actions)

                for agent_name, observation in observations.items():
                    reconcile(q, observation, agent_name)

                selected_agent_name = f"agent_{selected_agent_index}"
                selected_agent_observation = observations[selected_agent_name]

                canvas = pygame.Surface(window_size)
                env.draw(canvas)

                for row in range(GRID_HEIGHT):
                    for column in range(GRID_WIDTH):
                        actions = get_actions_at_agent_position(
                            q,
                            selected_agent_observation,
                            selected_agent_name,
                            (column, row),
                        )

                        if actions is not None:
                            image = pygame.image.load(f"../images/arrows/{selected_agent_name}.png")
                            rotation = get_rotation_from_action(int(np.argmax(actions)))
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
    file_name = (
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

    directory = "../tests/q_learning_condensed"
    os.makedirs(name=directory, exist_ok=True)

    try:
        q, episode_data = load_data(directory, file_name)
    except FileNotFoundError:
        q, episode_data = train()
        save_data(directory, file_name, q, episode_data)

    episode_steps = sparsify(episode_data["steps"])
    episodes = [episode for episode in range(len(episode_steps))]

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps per Episode")
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

    plt.title("Rewards per Episode")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

    validate()
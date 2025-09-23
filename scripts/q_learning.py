from numpy.random import Generator
from tqdm import tqdm
from scripts.shared import *
from scripts.types import *
from scripts.scavenging_ant import ScavengingAntEnv


EPSILON_START = 1
EPSILON_DECAY_RATE = EPSILON_START / (EPISODES / 2)
LEARNING_RATE_ALPHA = 0.10
DISCOUNT_FACTOR_GAMMA = 0.90


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
            window_size,
            state_actions,
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
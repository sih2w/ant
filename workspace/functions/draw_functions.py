import copy
from workspace.classes.environment import Environment, ACTION_ROTATIONS
from workspace.functions.pygame_functions import change_image_color
from workspace.shared.types import *
from workspace.functions.policy_functions import get_decided_actions
from workspace.functions.episode_functions import has_episode_ended
from workspace.shared.run_settings import *


def draw_arrow(
        environment: Environment,
        action_index: int,
        agent_index: int,
        agent_location: Location,
        canvas: pygame.Surface,
) -> None:
    image = pygame.image.load("images/icons8-triangle-48.png")
    image = change_image_color(image, environment.get_agent_color(agent_index))
    rotation = ACTION_ROTATIONS[action_index]
    position = environment.get_position_on_grid(agent_location, image.get_width())
    image = pygame.transform.rotate(image, rotation)
    canvas.blit(image, position)

    return None


def draw_arrows(
        environment: Environment,
        states: List[State],
        agent_index: int,
        state_actions: StateActions,
        canvas: pygame.Surface,
) -> None:
    states = copy.deepcopy(states)

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            agent_location = (column, row)
            states[agent_index]["agent_location"] = agent_location
            agent_actions = get_decided_actions(state_actions, states)
            draw_arrow(
                environment=environment,
                action_index=agent_actions[agent_index],
                agent_index=agent_index,
                agent_location=agent_location,
                canvas=canvas,
            )

    return None


def draw(
        environment: Environment,
        window_size: Tuple[int, int],
        window: pygame.Surface,
        states: List[State],
        agent_index: int,
        state_actions: StateActions
) -> None:
    canvas = pygame.Surface(window_size)
    environment.draw(canvas)

    if DRAW_ARROWS:
        draw_arrows(
            environment=environment,
            states=states,
            agent_index=agent_index,
            state_actions=state_actions,
            canvas=canvas
        )

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.flip()

    return None


def init_pygame() -> None:
    pygame.init()
    pygame.display.set_caption("Q-Learning Ants")
    pygame.display.set_icon(pygame.image.load("images/icons8-ant-48.png"))

    return None


def test(
        state_actions: StateActions,
        environment: Environment
) -> None:
    init_pygame()

    window_size = environment.get_window_size()
    window = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()

    run_interval_time = 0
    agent_index = 0

    switching_agent = False
    auto_run_enabled = False
    switching_auto_run = False
    running = True
    stepping_enabled = False
    stepping = False

    while running:
        states, _ = environment.reset()
        terminations, truncations = {}, {}

        draw(
            environment=environment,
            window_size=window_size,
            window=window,
            states=states,
            agent_index=agent_index,
            state_actions=state_actions
        )

        while running and not has_episode_ended(terminations, truncations):
            draw_next_step = auto_run_enabled and run_interval_time == 0
            draw_next_step = draw_next_step or (stepping_enabled and not stepping)

            if draw_next_step:
                stepping = True
                agent_actions = get_decided_actions(state_actions, states)
                states, rewards, terminations, truncations, info = environment.step(agent_actions)

            if draw_next_step or switching_agent:
                draw(
                    environment=environment,
                    window_size=window_size,
                    window=window,
                    states=states,
                    agent_index=agent_index,
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
                        agent_index += 1
                        if agent_index >= AGENT_COUNT:
                            agent_index = 0
                else:
                    switching_agent = False
                    if keys[pygame.K_SPACE]:
                        stepping_enabled = True
                    else:
                        stepping_enabled = False
                        stepping = False

            delta_time = clock.tick(60) / 1000
            if auto_run_enabled:
                run_interval_time += delta_time
                if run_interval_time >= 0.10:
                    run_interval_time = 0

    pygame.display.quit()
    pygame.quit()

    return None
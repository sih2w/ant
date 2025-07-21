from scripts.scavenging_ant import ScavengingAntEnv
import pygame

if __name__ == "__main__":
    env = ScavengingAntEnv(
        render_mode="human",
        render_fps=1,
        grid_height=3,
        grid_width=3,
        food_count=1,
        seed=0,
        agent_count=1
    )

    pygame.init()
    pygame.display.set_caption("Scavenging Ant")

    window_size = env.get_window_size()
    window = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    running = True

    while running:
        observations, info = env.reset(seed=0)
        terminated = False

        while not terminated:
            actions = {}
            for name in env.agents:
                actions[name] = env.action_space(name).sample()

            observations, rewards, terminations, truncations, infos = env.step(actions)
            canvas = pygame.Surface(window_size)
            env.draw(canvas)

            print(observations)
            # for agent_name, info in infos.items():
            #     print(agent_name, info["nearby_agents"])
            #     for other_agent_name in info["nearby_agents"]:
            #        print(agent_name, other_agent_name)

            window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    running = False
                    break

            clock.tick(env.render_fps)

            for _, termination in terminations.items():
                if termination:
                    terminated = True
                    break
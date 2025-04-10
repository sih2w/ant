from __future__ import annotations
import numpy as np
import pygame
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import Env

AGENT_ACTIONS = [
    [0, 1], # Move down
    [0, -1], # Move up
    [-1, 0], # Move left
    [1, 0], # Move right
]

class Positionable:
    def __init__(self, position: [int, int]):
        self.__position = position

    def set_position(self, position: [int, int]):
        self.__position = position

    def get_position(self):
        return self.__position

class Food(Positionable):
    def __init__(
            self,
            weight: float = 0.10,
            position: [int, int] = None,
            aroma_radius: float = 1,
            square_pixel_width: float = 0,
            random: np.random.default_rng = None,
    ):
        super().__init__(position)
        self.__aroma_radius = aroma_radius
        self.__carried = False
        self.__weight = weight
        self.__pixel_offset = random.integers(
            low=-square_pixel_width // 2,
            high=square_pixel_width // 2,
            size=(2, ),
            dtype=np.int16
        )

    def get_pixel_offset(self):
        return self.__pixel_offset

    def get_weight(self):
        return self.__weight

    def set_carried(self, is_carried: bool):
        self.__carried = is_carried

    def is_carried(self):
        return self.__carried

    def get_aroma_radius(self):
        return self.__aroma_radius

class Nest(Positionable):
    def __init__(
            self,
            capacity: float = 2,
            position: [int, int] = None,
    ):
        super().__init__(position)
        self.__capacity = capacity
        self.__agents_occupied = []
        self.__position = position

    def add_agent(self, agent: Agent) -> bool:
        if len(self.__agents_occupied) < self.__capacity:
            self.__agents_occupied.append(agent)
            return True
        return False

    def remove_agent(self, agent: Agent):
        self.__agents_occupied.remove(agent)

class Agent(Positionable):
    def __init__(
            self,
            carry_capacity: float = 0.10,
            position: [int, int] = None,
            scent_radius: float = 1,
            vision_radius: float = 1
    ):
        super().__init__(position)
        self.__carry_capacity = carry_capacity
        self.__scent_radius = scent_radius
        self.__vision_radius = vision_radius
        self.__carried_food = None

    def get_carry_capacity(self):
        return self.__carry_capacity

    def get_scent_radius(self):
        return self.__scent_radius

    def get_vision_radius(self):
        return self.__vision_radius

    def set_carried_food(self, food: Food or None):
        self.__carried_food = food

    def get_carried_food(self):
        return self.__carried_food

class Obstacle(Positionable):
    def __init__(
            self,
            position: [int, int] = None
    ):
        super().__init__(position)
        self.__position = position

class ScavengingAntEnv(Env):
    metadata = {
        "render_fps": 30,
        "name": "scavenging_ants_environment_v0",
        "render_modes": [
            "human",
        ],
    }

    def __init__(
            self,
            grid_width: int = 20,
            grid_height: int = 10,
            square_pixel_width: int = 80,
            percent_obstacles: float = 0.10,
            render_mode: str = None,
            render_fps: int = 10,
            nest_count: int = 1,
            food_count: int = 1,
            seed: int = 0,
            persistent_obstacles: bool = False,
            persistent_nests: bool = False,
            persistent_food: bool = False,
    ):
        self.__grid_width = grid_width
        self.__grid_height = grid_height
        self.__square_pixel_width = square_pixel_width
        self.__obstacle_count = self.__percent_to_count(percent_obstacles)
        self.__food_count = food_count
        self.__nest_count = nest_count
        self.__food = []
        self.__obstacles = []
        self.__nests = []
        self.__window = None
        self.__clock = None
        self.__random = np.random.default_rng(seed)
        self.__persistent_obstacles = persistent_obstacles
        self.__persistent_nests = persistent_nests
        self.__persistent_food = persistent_food

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.action_space = Discrete(len(AGENT_ACTIONS))

        """
        1. Agent position plane
        2. Obstacle position plane
        3. Nest position plane
        4. Food position plane
        5. Is carrying food plane
        """
        self.observation_space = Box(
            low=0,
            high=np.iinfo(np.int16).max,
            shape=(5, grid_height, grid_width),
            dtype=np.int16
        )

        max_agent_radius = max(grid_width, grid_height)
        self.__agent = Agent(
            carry_capacity=min(self.__random.random(), 0.50) * max_agent_radius,
            scent_radius=min(self.__random.random(), 0.50) * max_agent_radius,
            vision_radius=min(self.__random.random(), 0.50) * max_agent_radius,
            position=[-1, -1]
        )

    def __get_random_position(self):
        return self.__random.integers(low=0, high=[self.__grid_width - 1, self.__grid_height - 1], dtype=np.int16)

    def __get_new_random_position(self, excluded_positions: list):
        attempt = 0
        while True and attempt < 100:
            attempt += 1
            new_position = self.__get_random_position()
            is_position_new = True

            for excluded_position in excluded_positions:
                if np.array_equal(new_position, excluded_position):
                    is_position_new = False
                    break

            if is_position_new:
                return new_position

        return self.__get_random_position()

    def __get_excluded_positions(self):
        excluded_positions = [self.__agent.get_position()]
        for obstacle in self.__obstacles:
            excluded_positions.append(obstacle.get_position())

        for nest in self.__nests:
            excluded_positions.append(nest.get_position())

        for food in self.__food:
            excluded_positions.append(food.get_position())
        return excluded_positions

    def __percent_to_count(self, percent: float):
        cell_count = self.__grid_width * self.__grid_height
        return int(cell_count * percent)

    def __spawn_food(self):
        self.__food = []
        excluded_positions = self.__get_excluded_positions()
        max_radius = max(self.__grid_width, self.__grid_height)

        for _ in range(self.__food_count):
            self.__food.append(
                Food(
                    weight=self.__random.random(),
                    position=self.__get_new_random_position(excluded_positions),
                    aroma_radius=min(self.__random.random(), 0.50) * max_radius,
                    square_pixel_width=self.__square_pixel_width,
                    random=self.__random
                )
            )

    def __spawn_obstacles(self):
        self.__obstacles = []
        for index in range(self.__obstacle_count):
            excluded_positions = self.__get_excluded_positions()
            self.__obstacles.append(
                Obstacle(
                    position=self.__get_new_random_position(excluded_positions)
                )
            )

    def __spawn_nests(self):
        self.__nests = []
        for index in range(self.__nest_count):
            excluded_positions = self.__get_excluded_positions()
            self.__nests.append(
                Nest(
                    position=self.__get_new_random_position(excluded_positions),
                    capacity=self.__random.integers(low=1, high=5),
                )
            )

    def __spawn_agent(self):
        excluded_positions = self.__get_excluded_positions()
        self.__agent.set_position(self.__get_new_random_position(excluded_positions))
        self.__agent.set_carried_food(None)

    def __coordinates_to_plane(self, coordinates: [[int, int]]):
        plane = np.zeros(shape=(self.__grid_height, self.__grid_width), dtype=np.int16)
        for index, coordinate in enumerate(coordinates):
            plane[coordinate[1], coordinate[0]] += 1
        return plane

    @staticmethod
    def __get_in_vision(agent: Agent, items: [Positionable], persistent: bool):
        coordinates = []
        agent_position = np.array(agent.get_position())
        agent_vision_radius = agent.get_vision_radius()

        for index, item in enumerate(items):
            item_position = np.array(item.get_position())
            direction = item_position - agent_position

            if persistent or np.linalg.norm(direction) <= agent_vision_radius:
                coordinates.append(item_position)

        return coordinates

    def __get_observation(self):
        return np.stack((
            self.__coordinates_to_plane([self.__agent.get_position()]),
            self.__coordinates_to_plane(self.__get_in_vision(self.__agent, self.__nests, self.__persistent_nests)),
            self.__coordinates_to_plane(self.__get_in_vision(self.__agent, self.__food, self.__persistent_food)),
            self.__coordinates_to_plane(self.__get_in_vision(self.__agent, self.__obstacles, self.__persistent_obstacles)),
            np.zeros(shape=(self.__grid_height, self.__grid_width)) if self.__agent.get_carried_food() is None else np.ones(shape=(self.__grid_height, self.__grid_width))
        ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed or np.random.randint(low=0, high=1000))

        self.__spawn_nests()
        self.__spawn_food()
        self.__spawn_agent()
        self.__spawn_obstacles()

        return self.__get_observation(), {}

    def __update_agent(self, action: int):
        reward = 0
        old_position = np.array(self.__agent.get_position())
        direction = np.array(AGENT_ACTIONS[action])
        new_position = old_position + direction

        for obstacle in self.__obstacles:
            if np.array_equal(new_position, obstacle.get_position()):
                # Penalize the agent if it attempted to move into an obstacle.
                reward -= 1
                break
        else:
            clipped_position = np.clip(new_position, [0, 0], [self.__grid_width - 1, self.__grid_height - 1])
            if np.array_equal(clipped_position, old_position):
                # Penalize the agent if it attempted to move out of bounds.
                reward -= 1
            else:
                carried_food = self.__agent.get_carried_food()
                if carried_food is None:
                    for food in self.__food:
                        if not food.is_carried() and np.array_equal(food.get_position(), new_position):
                            self.__agent.set_carried_food(food)
                            food.set_carried(True)
                            reward += 1
                            break
                    else:
                        # Penalize the agent for taking a step without picking up food.
                        reward -= 0.25
                else:
                    carried_food.set_position(new_position)
                    for nest in self.__nests:
                        if np.array_equal(nest.get_position(), new_position):
                            self.__agent.set_carried_food(None)
                            self.__food.remove(carried_food)
                            reward += 1
                            break
                    else:
                        # Penalize the agent for taking a step without depositing food in a nest.
                        reward -= 0.25

                # Move the agent to the new position if the position was valid.
                self.__agent.set_position(new_position)

        return reward

    def step(self, action: int):
        reward = self.__update_agent(action)
        terminated = len(self.__food) == 0

        if self.render_mode == "human":
            self.render()

        return (
            self.__get_observation(),
            reward,
            terminated,
            False,
            {}
        )

    def __draw_nests(self, canvas):
        for nest in self.__nests:
            position = nest.get_position()
            pygame.draw.rect(
                surface=canvas,
                color=(255, 255, 255),
                rect=(
                    position[0] * self.__square_pixel_width,
                    position[1] * self.__square_pixel_width,
                    self.__square_pixel_width, self.__square_pixel_width
                ),
            )

    def __draw_agent(self, canvas):
        pygame.draw.circle(
            surface=canvas,
            color=(229, 69, 23),
            center=(self.__agent.get_position() + 0.50) * self.__square_pixel_width,
            radius=self.__square_pixel_width / 4,
        )

    def __draw_food(self, canvas):
        for food in self.__food:
            position = np.array((food.get_position() + 0.50) * self.__square_pixel_width)
            if not food.is_carried():
                position += food.get_pixel_offset()

            pygame.draw.circle(
                surface=canvas,
                color=(229, 170, 22),
                center=position,
                radius=self.__square_pixel_width / 10,
            )

    def __draw_obstacles(self, canvas):
        for obstacle in self.__obstacles:
            position = obstacle.get_position()
            pygame.draw.rect(
                surface=canvas,
                color=(46, 48, 51),
                rect=(
                    position[0] * self.__square_pixel_width,
                    position[1] * self.__square_pixel_width,
                    self.__square_pixel_width,
                    self.__square_pixel_width
                ),
            )

    def __draw_grass(self, canvas):
        for row in range(self.__grid_height):
            for column in range(self.__grid_width):
                pygame.draw.rect(
                    surface=canvas,
                    color=(0, 128, 19) if (row % 2 == 0 and column % 2 == 1) or (row % 2 == 1 and column % 2 == 0) else (5, 198, 34),
                    rect=(
                        column * self.__square_pixel_width,
                        row * self.__square_pixel_width,
                        self.__square_pixel_width,
                        self.__square_pixel_width
                    ),
                )

    def render(self):
        window_size = [
            self.__grid_width * self.__square_pixel_width,
            self.__grid_height * self.__square_pixel_width
        ]

        if self.render_mode == "human":
            if self.__window is None:
                pygame.init()
                self.__window = pygame.display.set_mode(window_size)

            if self.__clock is None:
                self.__clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    exit()

        canvas = pygame.Surface(window_size)

        self.__draw_grass(canvas)
        self.__draw_obstacles(canvas)
        self.__draw_nests(canvas)
        self.__draw_agent(canvas)
        self.__draw_food(canvas)

        if self.render_mode == "human":
            self.__window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.__clock.tick(self.render_fps)

    def close(self):
        if self.__window is not None:
            pygame.display.quit()
            pygame.quit()
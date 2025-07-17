from __future__ import annotations
import functools
import math
import random
import colorsys
import numpy as np
import pygame
from scripts.layered_sprite import LayeredSprite
from gymnasium.spaces import Discrete, Box, Dict, Tuple
from pettingzoo import ParallelEnv

AGENT_ACTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))

class Positionable:
    def __init__(self, position: [int, int]):
        self.__position = position

    @staticmethod
    def get_positions(positionables: [Positionable]):
        positions = []
        for positionable in positionables:
            positions.append(positionable.get_position())
        return positions

    def set_position(self, position: [int, int]):
        if isinstance(position, np.ndarray):
            self.__position = position.tolist()
        else:
            self.__position = position

    def get_position(self):
        return self.__position

class Food(Positionable):
    def __init__(
            self,
            weight: float = 0.10,
            position: [int, int] = None,
            square_pixel_width: float = 0
    ):
        super().__init__(position)
        self.__hidden = False
        self.__carried = False
        self.__weight = weight
        self.__pixel_offset = np.random.randint(
            low=-square_pixel_width // 2,
            high=square_pixel_width // 2,
            size=(2, ),
            dtype=np.int16
        )

    def get_pixel_offset(self):
        return self.__pixel_offset

    def get_weight(self):
        return self.__weight

    def set_carried(self, carried: bool):
        self.__carried = carried

    def is_carried(self):
        return self.__carried

    def is_hidden(self):
        return self.__hidden

    def set_hidden(self, hidden: bool):
        self.__hidden = hidden

class Nest(Positionable):
    def __init__(
            self,
            capacity: float = 2,
            position: [int, int] = None,
    ):
        super().__init__(position)
        self.__capacity = capacity
        self.__position = position

class Agent(Positionable):
    def __init__(
            self,
            carry_capacity: float = 0.10,
            position: [int, int] = None,
            color: [int, int, int, int] = None,
    ):
        super().__init__(position)
        self.__carry_capacity = carry_capacity
        self.__carried_food = None
        self.__color = color

        if self.__color is None:
            self.__color = colorsys.hsv_to_rgb(random.random(), 1, 1)
            self.__color = (
                int(self.__color[0] * 255),
                int(self.__color[1] * 255),
                int(self.__color[2] * 255),
            )

    def get_color(self):
        return self.__color

    def get_carry_capacity(self):
        return self.__carry_capacity

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

class ScavengingAntEnv(ParallelEnv):
    metadata = {
        "render_fps": 30,
        "name": "scavenging_ant_environment_v0",
        "render_modes": [
            "human",
        ],
    }

    def __init__(
            self,
            grid_width: int = 20,
            grid_height: int = 10,
            square_pixel_width: int = 80,
            render_mode: str = None,
            render_fps: int = 10,
            nest_count: int = 1,
            food_count: int = 1,
            agent_count: int = 1,
            agent_vision_radius: int = 1,
            obstacle_count: int = 1,
            seed: int = np.random.randint(1, 1000),
            agent_colors: [[int, int, int]] = None
    ):
        self.__grid_width = grid_width
        self.__grid_height = grid_height
        self.__square_pixel_width = square_pixel_width
        self.__obstacle_count = math.floor(min(grid_width * grid_height * 0.10, obstacle_count))
        self.__food_count = food_count
        self.__nest_count = nest_count
        self.__food = []
        self.__obstacles = []
        self.__nests = []

        self.__random = np.random.default_rng(seed)
        self.__step_count = 0
        self.__agent_vision_radius = agent_vision_radius

        self.possible_agents = ["agent_" + str(index) for index in range(agent_count)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        self.render_fps = render_fps

        agent_colors = agent_colors or [(255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]

        self.__agents = {
            name: Agent(
                carry_capacity=np.random.uniform(low=0, high=0.50),
                position=[-1, -1],
                color=agent_colors.pop() if len(agent_colors) > 0 else None
            ) for name in self.possible_agents
        }

        for _ in range(self.__nest_count):
            self.__nests.append(
                Nest(
                    position=[-1, -1],
                    capacity=np.random.randint(low=1, high=5),
                )
            )

        for _ in range(self.__food_count):
            self.__food.append(
                Food(
                    weight=np.random.randint(low=1, high=10),
                    position=[-1, -1],
                    square_pixel_width=self.__square_pixel_width
                )
            )

        for index in range(self.__obstacle_count):
            self.__obstacles.append(
                Obstacle(
                    position=[-1, -1]
                )
            )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return Discrete(len(AGENT_ACTIONS))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return Dict({
            "agent_position": Box(
                low=np.array([0, 0]),
                high=np.array([self.__grid_width, self.__grid_height]),
                shape=(2, ),
                dtype=np.int16
            ),
            "carrying_food": Discrete(2),
            "dropped_food_count": Discrete(self.__food_count),
            "food_positions": Tuple(
                *{
                    Box(
                        low=np.array([0, 0]),
                        high=np.array([self.__grid_width, self.__grid_height]),
                        shape=(2, ),
                        dtype=np.int16
                    ) for _ in range(len(self.__food))
                }),
            "carried_food": Tuple(
               *{
                   Discrete(2) for _ in range(len(self.__food))
               }),
            "agent_detected": Discrete(2)
        })

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

    def __get_nearby_agents(self, agent: Agent, agent_name: str):
        nearby_agents = []
        agent_position = agent.get_position()

        for other_agent_name, other_agent in self.__agents.items():
            if agent_name != other_agent_name:
                other_agent_position = other_agent.get_position()
                direction = np.array(other_agent_position) - np.array(agent_position)

                if np.linalg.norm(direction) <= self.__agent_vision_radius:
                    nearby_agents.append(other_agent_name)

        return nearby_agents

    def __get_observation(self, agent_name: str):
        agent = self.__agents[agent_name]
        carried_food = []
        food_positions = []

        for food in self.__food:
            food_positions.append(food.get_position())
            carried_food.append(int(food.is_carried()))

        return {
            "agent_position": agent.get_position(),
            "carrying_food": agent.get_carried_food() is not None,
            "carried_food": tuple(carried_food),
            "food_positions": tuple(food_positions),
            "agent_detected": len(self.__get_nearby_agents(agent, agent_name)) > 0,
        }

    def __get_observations(self):
        return {
            agent_name: self.__get_observation(agent_name) for agent_name in self.possible_agents
        }

    def __get_info(self):
        info = {}

        for agent_name in self.agents:
            agent = self.__agents[agent_name]
            info[agent_name] = {
                "agent_color": agent.get_color(),
                "nearby_agents": self.__get_nearby_agents(agent, agent_name),
            }

        return info

    def reset(self, seed: int = None, options = None):
        for obstacle in self.__obstacles:
            obstacle.set_position([-1, -1])

        for nest in self.__nests:
            nest.set_position([-1, -1])

        for food in self.__food:
            food.set_hidden(False)
            food.set_carried(False)
            food.set_position([-1, -1])

        for _, agent in self.__agents.items():
            agent.set_position([-1, -1])
            agent.set_carried_food(None)

        self.__step_count = 0
        self.__random = np.random.default_rng(seed=seed)

        for obstacle in self.__obstacles:
            excluded_positions = Positionable.get_positions(self.__obstacles)
            obstacle.set_position(self.__get_new_random_position(excluded_positions))

        for nest in self.__nests:
            excluded_positions = Positionable.get_positions(self.__nests) + Positionable.get_positions(self.__obstacles)
            nest.set_position(self.__get_new_random_position(excluded_positions))

        for food in self.__food:
            excluded_positions = Positionable.get_positions(self.__nests) + Positionable.get_positions(self.__obstacles)
            food.set_position(self.__get_new_random_position(excluded_positions))

        excluded_positions = Positionable.get_positions(self.__obstacles) + Positionable.get_positions(self.__food)
        for _, agent in self.__agents.items():
            agent.set_position(self.__get_new_random_position(excluded_positions))

        return self.__get_observations(), self.__get_info()

    def __update_agent(self, agent_name: str, action: int):
        agent = self.__agents[agent_name]
        reward = 0

        old_position = np.array(agent.get_position())
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
                carried_food = agent.get_carried_food()
                if carried_food is None:
                    for food in self.__food:
                        if not food.is_hidden() and not food.is_carried() and np.array_equal(food.get_position(), new_position):
                            agent.set_carried_food(food)
                            food.set_carried(True)
                            # Reward the agent for picking up food.
                            reward += 1
                            break
                    else:
                        # Penalize the agent for taking a step without picking up food.
                        reward -= 1
                else:
                    carried_food.set_position(new_position)
                    for nest in self.__nests:
                        if np.array_equal(nest.get_position(), new_position):
                            agent.set_carried_food(None)
                            carried_food.set_hidden(True)
                            # Reward the agent for depositing food.
                            reward += 1
                            break
                    else:
                        # Penalize the agent for taking a step without depositing food in a nest.
                        reward -= 1

                # Move the agent to the new position if the position was valid.
                agent.set_position(new_position)

        return reward

    def step(self, actions):
        rewards = {agent_name: self.__update_agent(agent_name, actions[agent_name]) for agent_name in self.agents}
        self.__step_count += 1

        terminated = True
        for food in self.__food:
            terminated = food.is_hidden()
            if not terminated:
                break

        terminations = {agent_name: terminated for agent_name in self.agents}
        truncations = {agent_name: False for agent_name in self.agents}

        if terminated:
            # Reward each agent for depositing all food.
            rewards = {agent_name: 100 for agent_name in self.agents}

        return (
            self.__get_observations(),
            rewards,
            terminations,
            truncations,
            self.__get_info()
        )

    def get_step_count(self):
        return self.__step_count

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

    def __draw_agents(self, canvas):
        for _, agent in self.__agents.items():
            radius = self.__square_pixel_width
            position = agent.get_position()
            position = (
                position[0] * self.__square_pixel_width,
                position[1] * self.__square_pixel_width
            )

            LayeredSprite(
                foreground_image="../images/icons8-animal-200.png",
                background_image="../images/icons8-animal-outline-200.png",
                dimensions=(radius, radius),
                rotation=0,
                color=agent.get_color()
            ).draw(
                canvas=canvas,
                position=position,
            )

    def __draw_food(self, canvas):
        for food in self.__food:
            if not food.is_hidden():
                radius = self.__square_pixel_width
                if food.is_carried():
                    radius = radius / 2

                position = food.get_position()
                position = (
                    position[0] * self.__square_pixel_width + self.__square_pixel_width / 2 - radius / 2,
                    position[1] * self.__square_pixel_width
                )

                if not food.is_carried():
                    position += food.get_pixel_offset()

                image = pygame.image.load("../images/icons8-apple-200.png")
                image = pygame.transform.scale(image, (radius, radius))
                canvas.blit(image, position)

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

    def get_window_size(self):
        return (
            self.__grid_width * self.__square_pixel_width,
            self.__grid_height * self.__square_pixel_width
        )

    def draw(self, canvas):
        if self.render_mode == "human":
            self.__draw_grass(canvas)
            self.__draw_obstacles(canvas)
            self.__draw_nests(canvas)
            self.__draw_agents(canvas)
            self.__draw_food(canvas)

    def close(self):
        pass
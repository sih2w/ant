from __future__ import annotations
import math
from scripts.types import *
import numpy as np
import pygame

AGENT_ACTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))

class ScavengingAntEnv:
    def __init__(
            self,
            grid_width: int = 20,
            grid_height: int = 10,
            square_pixel_width: int = 80,
            nest_count: int = 1,
            food_count: int = 1,
            agent_count: int = 1,
            obstacle_count: int = 1,
            seed: int = np.random.randint(1, 1000)
    ):
        self.__grid_width = grid_width
        self.__grid_height = grid_height
        self.__square_pixel_width = square_pixel_width
        self.__obstacle_count = math.floor(min(grid_width * grid_height * 0.10, obstacle_count))
        self.__food_count = food_count
        self.__nest_count = nest_count

        self.__food: List[Food] = []
        self.__obstacles: List[Obstacle] = []
        self.__nests: List[Nest] = []
        self.__agents: Dict[str, Agent] = {}

        self.__random = np.random.default_rng(seed)
        self.__step_count = 0

        agent_count = max(1, min(agent_count, 3))
        self.agent_names = [f"agent_{index}" for index in range(agent_count)]

        for agent_name in self.agent_names:
            self.__agents[agent_name] = {
                "location": (-1, -1),
                "carried_food": None
            }

        for _ in range(self.__nest_count):
            self.__nests.append({
                "location": (-1, -1)
            })

        for _ in range(self.__food_count):
            self.__food.append({
                "location": (-1, -1),
                "carried": False,
                "hidden": False
            })

        for index in range(self.__obstacle_count):
            self.__obstacles.append({
                "location": (-1, -1),
            })

    def __get_random_location(self) -> Location:
        return (
            int(self.__random.integers(0, self.__grid_width)),
            int(self.__random.integers(0, self.__grid_height))
        )

    def __get_new_random_location(self, excluded_locations: List[Location]) -> Location:
        excluded = np.asarray(excluded_locations, dtype=np.int16)
        if excluded.size == 0:
            return self.__get_random_location()

        attempts = 100
        for _ in range(attempts):
            position = self.__get_random_location()
            if not np.any(np.all(excluded == position, axis=1)):
                return position

        return self.__get_random_location()

    @staticmethod
    def __get_observation(agent: Agent, food_locations: FoodLocations) -> Observation:
        return {
            "agent_location": agent["location"],
            "carrying_food": agent["carried_food"] is not None,
            "food_locations": food_locations,
        }

    def __get_observations(self) -> Dict[str, Observation]:
        observations = {}
        food_locations = []

        for food in self.__food:
            food_locations.append(food["location"])
        food_locations = tuple(food_locations)

        for agent_name, agent in self.__agents.items():
            observations[agent_name] = self.__get_observation(agent, food_locations)

        return observations

    def reset(self, seed: int = None):
        for obstacle in self.__obstacles:
            obstacle["location"] = (-1, -1)

        for nest in self.__nests:
            nest["location"] = (-1, -1)

        for food in self.__food:
            food["hidden"] = False
            food["carried"] = False
            food["location"] = (-1, -1)

        for _, agent in self.__agents.items():
            agent["location"] = (-1, -1)
            agent["carried_food"] = None

        self.__step_count = 0
        self.__random = np.random.default_rng(seed=seed)

        excluded_locations = []
        for obstacle in self.__obstacles:
            location = self.__get_new_random_location(excluded_locations)
            excluded_locations.append(location)
            obstacle["location"] = location

        for nest in self.__nests:
            location = self.__get_new_random_location(excluded_locations)
            excluded_locations.append(location)
            nest["location"] = location

        for food in self.__food:
            location = self.__get_new_random_location(excluded_locations)
            excluded_locations.append(location)
            food["location"] = location

        for _, agent in self.__agents.items():
            location = self.__get_new_random_location(excluded_locations)
            excluded_locations.append(location)
            agent["location"] = location

        return self.__get_observations(), {}

    def __inside_grid(self, location: Location) -> bool:
        return 0 <= location[0] < self.__grid_width and 0 <= location[1] < self.__grid_height

    def __update_agent(self, agent: Agent, action: int):
        old_location = agent["location"]
        direction = AGENT_ACTIONS[action]
        new_location = (old_location[0] + direction[0], old_location[1] + direction[1])
        reward = 0

        for obstacle in self.__obstacles:
            if new_location == obstacle["location"]:
                reward -= 1000 # Penalize the agent if it attempted to move into an obstacle.
                break
        else:
            if not self.__inside_grid(new_location):
                reward -= 1000 # Penalize the agent if it attempted to move out of bounds.
            else:
                carried_food = agent["carried_food"]
                if carried_food is None:
                    for food in self.__food:
                        can_pick_up = not food["hidden"]
                        can_pick_up = can_pick_up and not food["carried"]
                        can_pick_up = can_pick_up and food["location"] == new_location

                        if can_pick_up:
                            agent["carried_food"] = food
                            food["carried"] = True
                            reward += 100 # Reward the agent for picking up food.
                            break
                    else:
                        reward -= 1 # Penalize the agent for taking a step without picking up food.
                else:
                    carried_food["location"] = new_location

                    for nest in self.__nests:
                        if nest["location"] == new_location:
                            agent["carried_food"] = None
                            carried_food["hidden"] = True
                            reward += 100 # Reward the agent for depositing food.
                            break
                    else:
                        reward -= 1 # Penalize the agent for taking a step without depositing food in a nest.

                # Move the agent to the new position if the position was valid.
                agent["location"] = new_location

        return reward

    def step(self, selected_actions: Dict[AgentName, int]):
        rewards: Dict[AgentName, int] = {}
        for agent_name, agent in self.__agents.items():
            rewards[agent_name] = self.__update_agent(agent, selected_actions[agent_name])

        all_food_hidden = True
        for food in self.__food:
            all_food_hidden = food["hidden"]
            if not all_food_hidden:
                break

        terminations = {}
        truncations = {}

        for agent_name, agent in self.__agents.items():
            terminations[agent_name] = all_food_hidden
            truncations[agent_name] = False

        return (
            self.__get_observations(),
            rewards,
            terminations,
            truncations,
            {}
        )

    def get_step_count(self):
        return self.__step_count

    def __draw_nests(self, canvas):
        for nest in self.__nests:
            image = pygame.image.load("../images/icons8-log-cabin-48.png")
            position = nest["location"]
            position = (
                position[0] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_width() / 2,
                position[1] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_height() / 2
            )
            canvas.blit(image, position)

    def __draw_agents(self, canvas):
        for agent_name, agent in self.__agents.items():
            image = pygame.image.load(f"../images/ants/{agent_name}.png")
            position = agent["location"]
            position = (
                position[0] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_width() / 2,
                position[1] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_height() / 2
            )
            canvas.blit(image, position)

    def __draw_food(self, canvas):
        for food in self.__food:
            if not food["carried"]:
                image = pygame.image.load("../images/icons8-whole-apple-48.png")
                position = food["location"]
                position = (
                    position[0] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_width() / 2,
                    position[1] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_height() / 2
                )
                canvas.blit(image, position)

    def __draw_obstacles(self, canvas):
        for obstacle in self.__obstacles:
            image = pygame.image.load("../images/icons8-obstacle-48.png")
            position = obstacle["location"]
            position = (
                position[0] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_width() / 2,
                position[1] * self.__square_pixel_width + self.__square_pixel_width / 2 - image.get_height() / 2
            )
            canvas.blit(image, position)

    def __draw_grass(self, canvas):
        for row in range(self.__grid_height):
            for column in range(self.__grid_width):
                pygame.draw.rect(
                    surface=canvas,
                    color=(46, 48, 51) if (row % 2 == 0 and column % 2 == 1) or (row % 2 == 1 and column % 2 == 0) else (29, 31, 33),
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
        self.__draw_grass(canvas)
        self.__draw_obstacles(canvas)
        self.__draw_nests(canvas)
        self.__draw_agents(canvas)
        self.__draw_food(canvas)

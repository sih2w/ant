from workspace.shared.types import *
from workspace.functions.pygame_functions import change_image_color
from workspace.shared.run_settings import *
import numpy as np
import pygame


AGENT_ACTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))
ACTION_ROTATIONS = (180, 0, 90, -90)


def default_food_pickup_callback(environment_state: EnvironmentState, food_index: int):
    return True


def default_action_override_callback(environment_state: EnvironmentState, action_index: int):
    return False, -1


def default_exchange_callback(environment_state: EnvironmentState):
    return True


class Environment:
    def __init__(self):
        self.__food: List[Food] = []
        self.__obstacles: List[Obstacle] = []
        self.__nests: List[Nest] = []
        self.__agents: List[Agent] = []

        random = np.random.default_rng(SEED)
        excluded_locations = []

        for index in range(AGENT_COUNT):
            location = self.__get_new_random_location(excluded_locations, random)
            excluded_locations.append(location)
            color = pygame.Color(0)
            color.hsla = (360.00 / (len(self.__agents) + 1.00), 100.00, 50.00, 100.00)

            self.__agents.append({
                "location": location,
                "carried_food": [],
                "last_action": 0,
                "spawn_location": location,
                "carry_capacity": CARRY_CAPACITY,
                "color": color,
                "food_pickup_callback": default_food_pickup_callback,
                "action_override_callback": default_action_override_callback,
                "exchange_callback": default_exchange_callback
            })

        for _ in range(NEST_COUNT):
            location = self.__get_new_random_location(excluded_locations, random)
            excluded_locations.append(location)
            self.__nests.append({
                "location": location,
                "spawn_location": location,
            })

        for _ in range(FOOD_COUNT):
            location = self.__get_new_random_location(excluded_locations, random)
            excluded_locations.append(location)
            self.__food.append({
                "location": location,
                "carried": False,
                "deposited": False,
                "spawn_location": location,
            })

        for index in range(OBSTACLE_COUNT):
            location = self.__get_new_random_location(excluded_locations, random)
            excluded_locations.append(location)
            self.__obstacles.append({
                "location": location,
                "spawn_location": location,
            })

    def register_food_pickup_callbacks(self, callbacks: List[FoodPickupCallback]):
        for index, callback in enumerate(callbacks):
            self.__agents[index]["food_pickup_callback"] = callback

    def register_action_override_callbacks(self, callbacks: List[ActionOverrideCallback]):
        for index, callback in enumerate(callbacks):
            self.__agents[index]["action_override_callback"] = callback

    def register_exchange_callbacks(self, callbacks: List[ExchangeCallback]):
        for index, callback in enumerate(callbacks):
            self.__agents[index]["exchange_callback"] = callback

    def get_environment_state(self) -> EnvironmentState:
        return {
            "agent_locations": [agent["location"] for agent in self.__agents],
            "carried_food": [self.__get_carried_food_indexes(agent) for agent in self.__agents],
            "nest_locations": [nest["location"] for nest in self.__nests],
            "obstacle_locations": [obstacle["location"] for obstacle in self.__obstacles],
            "food_locations": [food["location"] for food in self.__food],
            "deposited_food": [food["deposited"] for food in self.__food],
        }

    def __get_carried_food_indexes(self, agent: Agent) -> List[int]:
        food_indexes = []
        for index, food in enumerate(agent["carried_food"]):
            food_indexes.append(self.__food.index(food))
        return food_indexes

    @staticmethod
    def __get_random_location(random: np.random.Generator) -> Location:
        return (
            int(random.integers(0, GRID_WIDTH)),
            int(random.integers(0, GRID_HEIGHT))
        )

    def __get_new_random_location(self, excluded_locations: List[Location], random: np.random.Generator) -> Location:
        excluded = np.asarray(excluded_locations, dtype=np.int16)
        if excluded.size == 0:
            return self.__get_random_location(random)

        attempts = 100
        for _ in range(attempts):
            position = self.__get_random_location(random)
            if not np.any(np.all(excluded == position, axis=1)):
                return position

        return self.__get_random_location(random)

    @staticmethod
    def __get_state(agent: Agent, food_locations: FoodLocations) -> State:
        return {
            "agent_location": agent["location"],
            "carrying_food": len(agent["carried_food"]) > 0,
            "food_locations": food_locations,
        }

    def __get_active_food_locations(self) -> FoodLocations:
        food_locations = []
        for food in self.__food:
            if not food["deposited"] and not food["carried"]:
                food_locations.append(food["location"])
        food_locations = tuple(food_locations)
        return food_locations

    def __get_states(self) -> List[State]:
        food_locations = self.__get_active_food_locations()
        states = []

        for agent in self.__agents:
            states.append(self.__get_state(agent, food_locations))

        return states

    def reset(self):
        for obstacle in self.__obstacles:
            obstacle["location"] = obstacle["spawn_location"]

        for nest in self.__nests:
            nest["location"] = nest["spawn_location"]

        for food in self.__food:
            food["deposited"] = False
            food["carried"] = False
            food["location"] = food["spawn_location"]

        for agent in self.__agents:
            agent["location"] = agent["spawn_location"]
            agent["carried_food"].clear()
            agent["last_action"] = 0

        return self.__get_states(), {}

    @staticmethod
    def __outside_grid(location: Location) -> bool:
        return location[0] < 0 or location[0] >= GRID_WIDTH or location[1] < 0 or location[1] >= GRID_HEIGHT

    def __inside_obstacle(self, location: Location) -> bool:
        for obstacle in self.__obstacles:
            if location == obstacle["location"]:
                return True
        return False

    def __pickup_food(self, agent: Agent, location: Location) -> bool:
        if len(agent["carried_food"]) < agent["carry_capacity"]:
            environment_state = self.get_environment_state()

            for index, food in enumerate(self.__food):
                can_pick_up = not food["deposited"]
                can_pick_up = can_pick_up and not food["carried"]
                can_pick_up = can_pick_up and food["location"] == location
                can_pick_up = can_pick_up and agent["food_pickup_callback"](environment_state, index)

                if can_pick_up:
                    agent["carried_food"].append(food)
                    food["carried"] = True
                    return True
        return False

    def __deposit_food(self, agent: Agent, location: Location) -> bool:
        if len(agent["carried_food"]) > 0:
            for nest in self.__nests:
                if nest["location"] == location:
                    for food in agent["carried_food"]:
                        food["location"] = location
                        food["carried"] = False
                        food["deposited"] = True
                    agent["carried_food"].clear()
                    return True
        return False

    def __get_food_deposited(self) -> int:
        deposited = 0
        for food in self.__food:
            if food["deposited"]:
                deposited += 1
        return deposited

    def __update_agent(self, agent: Agent, action_index: int):
        direction = AGENT_ACTIONS[action_index]
        old_location = agent["location"]
        new_location = (old_location[0] + direction[0], old_location[1] + direction[1])

        outside_grid = self.__outside_grid(new_location)
        inside_obstacle = self.__inside_obstacle(new_location)
        invalid_location = outside_grid or inside_obstacle

        if invalid_location:
            return -1000

        reward = 0
        picked_up_food = self.__pickup_food(agent, new_location)
        if picked_up_food:
            reward += 4

        deposited_food = self.__deposit_food(agent, new_location)
        if deposited_food:
            reward += 1

        agent["location"] = new_location
        for food in agent["carried_food"]:
            food["location"] = new_location

        return reward if reward > 0 else -1

    def step(self, selected_actions: List[int]):
        rewards: List[int] = []
        environment_state = self.get_environment_state()

        for index, agent in enumerate(self.__agents):
            action_index = selected_actions[index]
            override_selection, new_action_index = agent["action_override_callback"](environment_state, action_index)
            if override_selection:
                action_index = new_action_index
            rewards.append(self.__update_agent(agent, action_index))
            agent["last_action"] = action_index

        terminated = self.__get_food_deposited() == len(self.__food)
        terminations = [terminated for _ in range(len(self.__agents))]
        truncations = [False for _ in range(len(self.__agents))]

        return (
            self.__get_states(),
            rewards,
            terminations,
            truncations,
            {}
        )

    @staticmethod
    def get_position_on_grid(location: Location, width: float) -> Tuple[float, float]:
        return (
            location[0] * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - width / 2,
            location[1] * SQUARE_PIXEL_WIDTH + SQUARE_PIXEL_WIDTH / 2 - width / 2
        )

    def get_agent_color(self, index: int):
        return self.__agents[index]["color"]

    def __draw_nests(self, canvas):
        for nest in self.__nests:
            image = pygame.image.load("images/icons8-log-cabin-48.png")
            position = self.get_position_on_grid(nest["location"], image.get_width())
            canvas.blit(image, position)

    def __draw_agents(self, canvas):
        for agent in self.__agents:
            image = pygame.image.load("images/icons8-ant-48.png")
            image = change_image_color(image, agent["color"])
            rotation = ACTION_ROTATIONS[agent["last_action"]]
            image = pygame.transform.rotate(image, rotation)
            position = self.get_position_on_grid(agent["location"], image.get_width())
            canvas.blit(image, position)

    def __draw_food(self, canvas):
        font = pygame.font.SysFont("arialblack", 30)

        for food in self.__food:
            if not food["deposited"]:
                image = pygame.image.load("images/icons8-whole-apple-48.png")
                position = self.get_position_on_grid(food["location"], image.get_width())
                canvas.blit(image, position)

                text = font.render(f"{self.__food.index(food)}", True, (255, 255, 255))
                canvas.blit(text, position)


    def __draw_obstacles(self, canvas):
        for obstacle in self.__obstacles:
            image = pygame.image.load("images/icons8-obstacle-48.png")
            position = self.get_position_on_grid(obstacle["location"], image.get_width())
            canvas.blit(image, position)

    @staticmethod
    def __draw_grass(canvas):
        for row in range(GRID_HEIGHT):
            for column in range(GRID_WIDTH):
                pygame.draw.rect(
                    surface=canvas,
                    color=(46, 48, 51) if (row % 2 == 0 and column % 2 == 1) or (
                                row % 2 == 1 and column % 2 == 0) else (29, 31, 33),
                    rect=(
                        column * SQUARE_PIXEL_WIDTH,
                        row * SQUARE_PIXEL_WIDTH,
                        SQUARE_PIXEL_WIDTH,
                        SQUARE_PIXEL_WIDTH
                    ),
                )

    @staticmethod
    def get_window_size():
        return (
            GRID_WIDTH * SQUARE_PIXEL_WIDTH,
            GRID_HEIGHT * SQUARE_PIXEL_WIDTH
        )

    def draw(self, canvas):
        self.__draw_grass(canvas)
        self.__draw_obstacles(canvas)
        self.__draw_nests(canvas)
        self.__draw_agents(canvas)
        self.__draw_food(canvas)

from typing import List, Tuple, TypeAlias, TypedDict, Dict, Callable
import pygame


Used: TypeAlias = bool or None


Location: TypeAlias = Tuple[int, int]


FoodLocations: TypeAlias = Tuple[Location, ...]


Actions: TypeAlias = List[float]


Episode: TypeAlias = List


Policy: TypeAlias = List


GriddedPolicy: TypeAlias = List[List[Policy]]


ReturningPolicies: TypeAlias = List[GriddedPolicy]


SearchingPolicies: TypeAlias = List[Dict[FoodLocations, GriddedPolicy]]


class EnvironmentState(TypedDict):
    agent_locations: List[Location]
    carried_food: List[List[int]]
    nest_locations: List[Location]
    obstacle_locations: List[Location]
    food_locations: List[Location]
    deposited_food: List[bool]


class StateActions(TypedDict):
    returning: ReturningPolicies
    searching: SearchingPolicies


class State(TypedDict):
    agent_location: Location
    carrying_food: bool
    food_locations: FoodLocations


class Food(TypedDict):
    location: Location
    carried: bool
    deposited: bool
    spawn_location: Location


class Nest(TypedDict):
    location: Location
    spawn_location: Location


class Agent(TypedDict):
    location: Location
    carried_food: List[Food]
    last_action: int
    spawn_location: Location
    carry_capacity: int
    color: pygame.Color


class Obstacle(TypedDict):
    location: Location
    spawn_location: Location


FoodPickupCallback: TypeAlias = Callable[[int, EnvironmentState], bool]

ActionVerificationCallback: TypeAlias = Callable[[int, int, EnvironmentState], Tuple[bool, int]]
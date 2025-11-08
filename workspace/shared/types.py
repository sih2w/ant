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


FoodPickupCallback: TypeAlias = Callable[[EnvironmentState, int], bool]


ActionOverrideCallback: TypeAlias = Callable[[EnvironmentState, int], Tuple[bool, int]]


ExchangeCallback: TypeAlias = Callable[[Policy, Policy], bool]


class Agent(TypedDict):
    location: Location
    carried_food: List[Food]
    last_action: int
    spawn_location: Location
    carry_capacity: int
    color: pygame.Color
    food_pickup_callback: FoodPickupCallback
    action_override_callback: ActionOverrideCallback
    exchange_callback: ExchangeCallback


class Obstacle(TypedDict):
    location: Location
    spawn_location: Location
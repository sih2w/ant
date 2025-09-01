from typing import List, Tuple, TypeAlias, Dict, TypedDict, DefaultDict

Used: TypeAlias = bool or None
AgentName: TypeAlias = str
Location: TypeAlias = Tuple[int, int]
FoodLocations: TypeAlias = Tuple[Location, ...]
Actions: TypeAlias = List[float]


class Episode(TypedDict):
    steps: int
    rewards: Dict[AgentName, int]


class Policy(TypedDict):
    actions: Actions
    used: bool
    given: bool


class StateActions(TypedDict):
    returning: DefaultDict[AgentName, DefaultDict[Location, Policy]]
    searching: DefaultDict[AgentName, DefaultDict[Location, DefaultDict[FoodLocations, Policy]]]


class Observation(TypedDict):
    agent_location: Location
    carrying_food: bool
    food_locations: FoodLocations


class Food(TypedDict):
    location: Location
    carried: bool
    hidden: bool
    spawn_location: Location


class Nest(TypedDict):
    location: Location
    spawn_location: Location


class Agent(TypedDict):
    location: Location
    carried_food: Food | None
    last_action: int
    spawn_location: Location


class Obstacle(TypedDict):
    location: Location
    spawn_location: Location

from collections import defaultdict


class Agent:
    def __init__(self, agent_id: int):
        self._id = agent_id
        self._visited_locations = defaultdict(int)

    @property
    def id(self) -> int:
        return self._id

    @property
    def visited_locations(self) -> defaultdict:
        return self._visited_locations

    @visited_locations.setter
    def visited_locations(self, new_visited: dict | defaultdict):
        if not isinstance(new_visited, (dict, defaultdict)):
            raise ValueError("visited_locations must be a dict or defaultdict")
        self._visited_locations = defaultdict(int, new_visited)


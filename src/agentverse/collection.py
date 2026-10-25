"""Fixed resource collection task with position and inventory observations."""

import gymnasium as gym
from gymnasium import spaces

MAP = ("B.....", ".##.#.", "..R.#.", ".#....", "...#R.", "......")
CONTRACT = {"map": list(MAP), "actions": ["up", "right", "down", "left"],
            "completion_reward": 1.0, "resource_reward": 0.2,
            "step_reward": -0.01, "wall_penalty": -0.05,
            "max_steps": 100, "version": 1,
            "observation": "inventory_mask * traversable_cells + position",
            "start": "base", "collection": "first entry, two resource bits"}


class ResourceCollectionWorld(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        if render_mode not in (None, "ansi"):
            raise ValueError("Only ansi rendering is supported")
        self.render_mode = render_mode
        self.cells = tuple((r, c) for r, row in enumerate(MAP)
                           for c, value in enumerate(row) if value != "#")
        self.indices = {cell: i for i, cell in enumerate(self.cells)}
        self.base = self.indices[(0, 0)]
        self.resources = tuple(self.indices[(r, c)] for r, row in enumerate(MAP)
                               for c, value in enumerate(row) if value == "R")
        self.starts = (self.base,)
        self.observation_space = spaces.Discrete(len(self.cells) * 4)
        self.action_space = spaces.Discrete(4)
        self.position = self.base
        self.collected = 0
        self.steps = 0
        self.done = True

    @property
    def state(self):
        return self.collected * len(self.cells) + self.position

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if (options or {}).get("start", self.base) != self.base:
            raise ValueError("Collection starts at base")
        self.position, self.collected, self.steps, self.done = self.base, 0, 0, False
        return self.state, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished environment")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(action)]
        r, c = self.cells[self.position]
        target = (r + dr, c + dc)
        collision = target not in self.indices
        reward = CONTRACT["step_reward"] + (CONTRACT["wall_penalty"] if collision else 0)
        if not collision:
            self.position = self.indices[target]
            if self.position in self.resources:
                bit = 1 << self.resources.index(self.position)
                if not self.collected & bit:
                    self.collected |= bit
                    reward += CONTRACT["resource_reward"]
        self.steps += 1
        terminated = self.position == self.base and self.collected == 3
        truncated = self.steps >= CONTRACT["max_steps"] and not terminated
        if terminated:
            reward += CONTRACT["completion_reward"]
        self.done = terminated or truncated
        return self.state, reward, terminated, truncated, {"collision": collision}

    def render(self):
        rows = [list(row) for row in MAP]
        for bit, position in enumerate(self.resources):
            if self.collected & (1 << bit):
                r, c = self.cells[position]
                rows[r][c] = "."
        r, c = self.cells[self.position]
        rows[r][c] = "A"
        return "\n".join("".join(row) for row in rows)

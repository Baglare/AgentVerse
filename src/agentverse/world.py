"""Fixed, fully observable navigation task. Actions: up, right, down, left."""

import gymnasium as gym
from gymnasium import spaces

MAP = (".....", ".##..", "...#.", ".#...", "....G")
CONTRACT = {"map": list(MAP), "actions": ["up", "right", "down", "left"],
            "goal_reward": 1.0, "step_reward": -0.01,
            "wall_penalty": -0.05, "max_steps": 100, "version": 1}


class NavigationWorld(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        if render_mode not in (None, "ansi"):
            raise ValueError("Only ansi rendering is supported")
        self.render_mode = render_mode
        self.cells = tuple((r, c) for r, row in enumerate(MAP)
                           for c, value in enumerate(row) if value != "#")
        self.indices = {cell: i for i, cell in enumerate(self.cells)}
        self.goal = self.indices[(4, 4)]
        self.starts = tuple(i for i in range(len(self.cells)) if i != self.goal)
        self.observation_space = spaces.Discrete(len(self.cells))
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.steps = 0
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = (options or {}).get("start")
        if start is not None and start not in self.starts:
            raise ValueError("Start must be a non-goal traversable state")
        self.state = int(self.np_random.choice(self.starts)) if start is None else int(start)
        self.steps = 0
        self.done = False
        return self.state, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished environment")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(action)]
        r, c = self.cells[self.state]
        target = (r + dr, c + dc)
        collision = target not in self.indices
        if not collision:
            self.state = self.indices[target]
        self.steps += 1
        terminated = self.state == self.goal
        truncated = self.steps >= 100 and not terminated
        reward = (1.0 if terminated else 0.0) - 0.01 - (0.05 if collision else 0.0)
        self.done = terminated or truncated
        return self.state, reward, terminated, truncated, {"collision": collision}

    def render(self):
        rows = [list(row) for row in MAP]
        if self.state is not None:
            r, c = self.cells[self.state]
            rows[r][c] = "A"
        return "\n".join("".join(row) for row in rows)

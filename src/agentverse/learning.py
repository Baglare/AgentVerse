"""Seeded training and paired evaluation with independent random streams."""

import numpy as np

from .world import NavigationWorld


def train(episodes, seed):
    if episodes <= 0 or seed < 0:
        raise ValueError("Episodes must be positive and seed nonnegative")
    env = NavigationWorld()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    history = []
    for episode in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**32)))
        epsilon = 1.0 - 0.95 * min(episode / max(1, episodes * 0.8), 1.0)
        total = 0.0
        while True:
            action = int(rng.integers(4)) if rng.random() < epsilon else int(q[state].argmax())
            following, reward, terminated, truncated, _ = env.step(action)
            # Time limits do not make the underlying state terminal.
            target = reward + (0.0 if terminated else 0.95 * q[following].max())
            q[state, action] += 0.1 * (target - q[state, action])
            state = following
            total += reward
            if terminated or truncated:
                history.append({"steps": env.steps, "reward": total,
                                "success": terminated, "epsilon": epsilon})
                break
    return q, history


def evaluate(episodes, seed, q=None):
    if episodes <= 0 or seed < 0:
        raise ValueError("Episodes must be positive and seed nonnegative")
    env = NavigationWorld()
    # Domain separation from training, even when a caller reuses its seed.
    starts_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    actions_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    starts = [int(x) for x in starts_rng.choice(env.starts, size=episodes)]
    records = []
    for start in starts:
        state, _ = env.reset(options={"start": start})
        total = 0.0
        while True:
            action = int(actions_rng.integers(4)) if q is None else int(q[state].argmax())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                records.append({"steps": env.steps, "reward": total, "success": terminated})
                break
    return {"seed": seed, "episodes": episodes, "starts": starts,
            "success_rate": float(np.mean([r["success"] for r in records])),
            "mean_steps": float(np.mean([r["steps"] for r in records])),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "records": records}

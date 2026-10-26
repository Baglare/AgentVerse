"""M3-A: fixed-protocol, paired goal relocation experiment (CPU only)."""

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

import gymnasium
import numpy as np

from .learning import evaluate
from .storage import AgentStore, publish, write_json
from .world import CONTRACT, MAP, NavigationWorld

CHECKPOINTS = (0, 25, 50, 100, 200, 500)
TRAIN_SEEDS = tuple(range(10))
EVAL_SEEDS = tuple(range(10000, 10010))
ARMS = ("scratch", "transfer", "shuffled_prior")
THRESHOLD = 0.9


class TransferWorld(NavigationWorld):
    """Reuse M1 transitions, changing only the terminal goal and valid starts."""

    def __init__(self, goal):
        super().__init__()
        if tuple(goal) not in self.indices:
            raise ValueError("Goal must be a traversable (row, column)")
        self.goal = self.indices[tuple(goal)]
        self.starts = tuple(i for i in range(len(self.cells)) if i != self.goal)
        reached, pending = {tuple(goal)}, [tuple(goal)]
        while pending:
            r, c = pending.pop()
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                cell = (r + dr, c + dc)
                if cell in self.indices and cell not in reached:
                    reached.add(cell)
                    pending.append(cell)
        if reached != set(self.cells):
            raise ValueError("Not all starts can reach goal")

    def render(self):
        rows = [list(row.replace("G", ".")) for row in MAP]
        r, c = self.cells[self.goal]
        rows[r][c] = "G"
        if self.state is not None:
            r, c = self.cells[self.state]
            rows[r][c] = "A"
        return "\n".join("".join(row) for row in rows)


def compatibility(metadata, q, env):
    """Exact legacy source contract plus explicit state/action correspondence."""
    if metadata.get("skill_id") != "navigation" or metadata.get("contract") != CONTRACT:
        raise ValueError("Incompatible source Navigation contract")
    AgentStore.validate_q(q)
    source_cells = tuple((r, c) for r, row in enumerate(metadata["contract"]["map"])
                         for c, value in enumerate(row) if value != "#")
    if (env.cells != source_cells or env.indices != {cell: i for i, cell in enumerate(source_cells)}
            or env.action_space.n != 4
            or metadata["contract"]["actions"] != ["up", "right", "down", "left"]):
        raise ValueError("Incompatible state indices or action meanings")
    return {"source_contract": deepcopy(metadata["contract"]),
            "state_index_to_cell": [list(cell) for cell in source_cells],
            "action_deltas": [[-1, 0], [0, 1], [1, 0], [0, -1]],
            "target_goal": list(env.cells[env.goal]), "original_goal": [4, 4],
            "goal_delta": [env.cells[env.goal][0] - 4, env.cells[env.goal][1] - 4],
            "all_starts_reachable": True,
            "changed_fields": [] if env.cells[env.goal] == (4, 4) else ["goal", "non_goal_start_set"]}


def initial_q(source, arm, seed):
    if arm == "scratch":
        return np.zeros_like(source), None
    if arm == "transfer":
        return source.copy(), None
    if arm != "shuffled_prior":
        raise ValueError("Unknown experiment arm")
    permutation = np.random.default_rng(np.random.SeedSequence([seed, 303])).permutation(len(source))
    return source[permutation].copy(), permutation.tolist()


def evaluate_target(q, goal, starts):
    """Greedy first-argmax ties, no RNG and no writes to the supplied table."""
    env = TransferWorld(goal)
    records = []
    for start in starts:
        state, _ = env.reset(options={"start": start})
        total = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(int(q[state].argmax()))
            total += reward
            if terminated or truncated:
                records.append({"success": bool(terminated), "steps": env.steps, "reward": total})
                break
    return {"success_rate": float(np.mean([r["success"] for r in records])),
            "mean_steps": float(np.mean([r["steps"] for r in records])),
            "mean_reward": float(np.mean([r["reward"] for r in records])), "records": records}


def curve_metrics(rows):
    x = np.array([row["episode"] for row in rows])
    # Explicit trapezoids also support NumPy 2.0 (without np.trapezoid).
    def auc(key):
        y = np.array([row[key] for row in rows])
        return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
    return {"first_observed_threshold_episode": next(
                (r["episode"] for r in rows if r["success_rate"] >= THRESHOLD), None),
            "success_auc": auc("success_rate"), "reward_auc": auc("mean_reward")}


def train_arm(source, goal, arm, seed, eval_seed, checkpoints=CHECKPOINTS):
    env = TransferWorld(goal)
    q, permutation = initial_q(source, arm, seed)
    start_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    train_starts = [int(x) for x in start_rng.choice(env.starts, size=checkpoints[-1])]
    eval_rng = np.random.default_rng(np.random.SeedSequence([eval_seed, 404]))
    eval_starts = [int(x) for x in eval_rng.choice(env.starts, size=100)]
    rows, history, models = [], [], {}
    for episode in range(checkpoints[-1] + 1):
        if episode in checkpoints:
            result = evaluate_target(q, goal, eval_starts)
            rows.append({"episode": episode, **result})
            models[episode] = q.copy()
        if episode == checkpoints[-1]:
            break
        # Key by episode and step: divergence in trajectory cannot shift later draws.
        action_rng = np.random.default_rng(np.random.SeedSequence([seed, 202, episode]))
        explore = action_rng.random(100)
        random_actions = action_rng.integers(4, size=100)
        epsilon = 1.0 - 0.95 * min(episode / max(1, checkpoints[-1] * 0.8), 1.0)
        state, _ = env.reset(options={"start": train_starts[episode]})
        total = 0.0
        while True:
            step = env.steps
            action = int(random_actions[step]) if explore[step] < epsilon else int(q[state].argmax())
            following, reward, terminated, truncated, _ = env.step(action)
            target = reward + (0.0 if terminated else 0.95 * q[following].max())
            q[state, action] += 0.1 * (target - q[state, action])
            state = following
            total += reward
            if terminated or truncated:
                history.append({"episode": episode + 1, "steps": env.steps,
                                "reward": total, "success": bool(terminated), "epsilon": epsilon})
                break
    return {"arm": arm, "training_seed": seed, "evaluation_seed": eval_seed,
            "train_starts": train_starts, "eval_starts": eval_starts,
            "state_permutation": permutation, "checkpoints": rows, "training": history,
            **curve_metrics(rows)}, models


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def aggregate(runs):
    rows = []
    for arm in ARMS:
        selected = [run for run in runs if run["arm"] == arm]
        for index, checkpoint in enumerate(CHECKPOINTS):
            row = {"arm": arm, "episode": checkpoint}
            for metric in ("success_rate", "mean_steps", "mean_reward"):
                values = [run["checkpoints"][index][metric] for run in selected]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_sd"] = float(np.std(values, ddof=1))
            rows.append(row)
    differences = []
    for seed in TRAIN_SEEDS:
        paired = {run["arm"]: run for run in runs if run["training_seed"] == seed}
        for control in ("scratch", "shuffled_prior"):
            for key in ("success_auc", "reward_auc"):
                differences.append({"seed": seed, "comparison": "transfer_minus_" + control,
                                    "metric": key, "difference": paired["transfer"][key] - paired[control][key]})
            for index, checkpoint in enumerate(CHECKPOINTS):
                for key in ("success_rate", "mean_steps", "mean_reward"):
                    differences.append({"seed": seed, "comparison": "transfer_minus_" + control,
                                        "metric": key, "episode": checkpoint,
                                        "difference": paired["transfer"]["checkpoints"][index][key]
                                        - paired[control]["checkpoints"][index][key]})
    return rows, differences


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_target(path, runs, goal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("success_rate", "mean_reward")):
        for arm in ARMS:
            values = np.array([[row[metric] for row in run["checkpoints"]]
                               for run in runs if run["arm"] == arm])
            line, = ax.plot(CHECKPOINTS, values.mean(axis=0), label=arm)
            for value in values:
                ax.plot(CHECKPOINTS, value, color=line.get_color(), alpha=0.12)
            ax.fill_between(CHECKPOINTS, values.mean(axis=0) - values.std(axis=0, ddof=1),
                            values.mean(axis=0) + values.std(axis=0, ddof=1), alpha=0.12,
                            color=line.get_color())
        ax.set(xlabel="Training episodes", ylabel=metric, title=f"Goal {goal}: mean, SD, all seeds")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)


def run_experiment(root, agent="Alph", version=None, destination=Path("outputs/m3-transfer")):
    store = AgentStore(root, agent)
    before = hashes(store.path)
    source, metadata, _ = store.load("navigation", version)
    old = {skill: evaluate(100, 90000, store.load(skill)[0], skill)
           for skill in ("navigation", "resource_collection") if store.versions(skill)}
    protocol = {"checkpoints": list(CHECKPOINTS), "training_seeds": list(TRAIN_SEEDS),
                "evaluation_seeds": list(EVAL_SEEDS), "evaluation_episodes": 100,
                "alpha": 0.1, "gamma": 0.95, "epsilon": "1 - .95 * min(episode / 400, 1)",
                "threshold": THRESHOLD, "tie_break": "first argmax (up, right, down, left)",
                "rng": {"library": "NumPy default_rng / PCG64 / SeedSequence",
                        "training_starts": "[training_seed,101]",
                        "actions": "[training_seed,202,episode]; 100 uniforms then 100 integer actions",
                        "shuffle": "[training_seed,303]; whole rows permuted, actions retained",
                        "evaluation_starts": "[evaluation_seed,404]; same at every checkpoint and arm"},
                "budget": "500 episodes per arm; at most 100 steps each; actual transitions recorded",
                "auc": "trapezoidal checkpoint curve, unnormalized, episode range 0..500",
                "threshold_rule": "first observed checkpoint >= .90; null means not reached; no interpolation",
                "metrics": "steps/rewards include failures and 100-step timeouts; SD across 10 paired seeds",
                "scope": "One fixed source model, same map, changed goal only; no cross-skill/general intelligence evidence",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "gymnasium": gymnasium.__version__}}
    result = {"status": "M3-A evaluated; M3 remains experimental; M3-B not implemented",
              "source": {"identity": store.identity, "metadata": metadata}, "protocol": protocol, "targets": []}
    for goal in ((4, 3), (0, 4), (4, 4)):
        compatibility(metadata, source, TransferWorld(goal))

    def writer(path):
        write_json(path / "protocol.json", protocol)  # Written before any experiment result.
        for goal in ((4, 3), (0, 4), (4, 4)):
            name = f"goal-{goal[0]}-{goal[1]}"
            directory = path / name
            directory.mkdir()
            target = {"goal": list(goal), "positive_control": goal == (4, 4),
                      "contract": compatibility(metadata, source, TransferWorld(goal)), "runs": []}
            for seed, eval_seed in zip(TRAIN_SEEDS, EVAL_SEEDS):
                for arm in ARMS:
                    run, models = train_arm(source, goal, arm, seed, eval_seed)
                    model_name = f"{arm}-{seed}.npz"
                    np.savez_compressed(directory / model_name,
                                        **{f"episode_{ep}": q for ep, q in models.items()})
                    run["model_file"] = f"{name}/{model_name}"
                    target["runs"].append(run)
            target["summary"], target["paired_differences"] = aggregate(target["runs"])
            flat = [{"seed": run["training_seed"], "arm": run["arm"],
                     **{k: v for k, v in row.items() if k != "records"}}
                    for run in target["runs"] for row in run["checkpoints"]]
            write_csv(directory / "checkpoints.csv", flat)
            write_csv(directory / "summary.csv", target["summary"])
            write_csv(directory / "paired-differences.csv", target["paired_differences"])
            write_csv(directory / "seed-metrics.csv", [{k: run[k] for k in
                      ("arm", "training_seed", "first_observed_threshold_episode", "success_auc", "reward_auc")}
                      for run in target["runs"]])
            plot_target(directory / "curves.png", target["runs"], goal)
            result["targets"].append(target)
        after_performance = {skill: evaluate(100, 90000, store.load(skill)[0], skill) for skill in old}
        after = hashes(store.path)
        if before != after or old != after_performance:
            raise ValueError("Source integrity or old skill performance changed")
        result["preservation"] = {"all_agent_files_sha256_before": before,
                                  "all_agent_files_sha256_after": after,
                                  "unchanged": True, "evaluation_before": old,
                                  "evaluation_after": after_performance}
        write_json(path / "results.json", result)
        write_json(path / "sha256.json", hashes(path))

    output = publish(Path(destination), str(uuid4()), writer)
    return {"directory": str(output.resolve()), "source_version": metadata["version"],
            "status": result["status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="Alph")
    parser.add_argument("--source-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--version", help="Navigation source version; default latest, pinned in report")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m3-transfer"))
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.source_dir, args.agent, args.version, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"m3-transfer: {error}\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()

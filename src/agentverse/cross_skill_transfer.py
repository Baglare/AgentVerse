"""M3-B: conservative Navigation advantages as Collection Q initialization."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import gymnasium
import numpy as np

from .collection import CONTRACT as TARGET_CONTRACT, ResourceCollectionWorld
from .learning import evaluate
from .storage import AgentStore, publish, write_json
from .transfer import hashes, write_csv
from .world import CONTRACT as SOURCE_CONTRACT, NavigationWorld

ADAPTER_VERSION = "relative-vector-local-walls-v1"
ARMS = ("scratch", "transfer", "shuffled_prior", "untrained_source")
# Equal actual transitions, rather than unequal episode-length budgets.
CHECKPOINTS = (0, 250, 500, 1000, 2000, 5000, 10000, 30000)
SEEDS = tuple(range(10))
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def blocked(env, cell):
    return tuple((cell[0] + dr, cell[1] + dc) not in env.indices for dr, dc in DELTAS)


class CrossSkillAdapter:
    """Exact displacement + immediate wall signature; no search or target rollouts.

    Subgoals: first uncollected resource in row-major order, then base.
    A target row receives Q_source(s,a) - mean_a Q_source(s,a), or all zeros.
    Only initialization is transferred; target updates are independent.
    """

    def __init__(self, q, metadata, target_contract=None):
        target_contract = TARGET_CONTRACT if target_contract is None else target_contract
        if metadata.get("skill_id") != "navigation" or metadata.get("contract") != SOURCE_CONTRACT:
            raise ValueError("Incompatible source contract")
        if target_contract != TARGET_CONTRACT:
            raise ValueError("Incompatible target contract")
        if SOURCE_CONTRACT["actions"] != target_contract["actions"]:
            raise ValueError("Incompatible action meanings")
        AgentStore.validate_q(q)
        self.source = q.copy()
        self.source.setflags(write=False)
        self.source_env = NavigationWorld()
        self.target_env = ResourceCollectionWorld()

    def mapping(self, state):
        env, src = self.target_env, self.source_env
        if type(state) is not int or not 0 <= state < env.observation_space.n:
            raise ValueError("Invalid target state")
        mask, position = divmod(state, len(env.cells))
        # A resource is collected on entry: these encoded states cannot be visited.
        if position in env.resources and not mask & (1 << env.resources.index(position)):
            return None, "inconsistent_inventory", None
        if mask == 3 and position == env.base:
            return None, "terminal", None
        subgoal = next((p for bit, p in enumerate(env.resources) if not mask & (1 << bit)), env.base)
        r, c = env.cells[position]
        gr, gc = env.cells[subgoal]
        sr, sc = src.cells[src.goal]
        source_cell = (sr - (gr - r), sc - (gc - c))
        if source_cell not in src.indices or source_cell == src.cells[src.goal]:
            return None, "relative_vector_unavailable", subgoal
        if blocked(src, source_cell) != blocked(env, (r, c)):
            return None, "local_walls_mismatch", subgoal
        return src.indices[source_cell], "mapped", subgoal

    def initialize(self):
        env = self.target_env
        q = np.zeros((env.observation_space.n, 4), dtype=np.float64)
        mapping = []
        for state in range(len(q)):
            source, reason, goal = self.mapping(state)
            if source is not None:
                row = self.source[source]
                q[state] = row - row.mean()
            mapping.append({"target_state": state, "source_state": source,
                            "reason": reason, "subgoal_position": goal})
        counts = dict(Counter(row["reason"] for row in mapping))
        eligible = len(q) - counts.get("terminal", 0) - counts.get("inconsistent_inventory", 0)
        mapped = counts.get("mapped", 0)
        return q, {"adapter_version": ADAPTER_VERSION, "mapping": mapping, "counts": counts,
                   "encoded_states": len(q), "eligible_states": eligible,
                   "mapped_states": mapped, "coverage_all": mapped / len(q),
                   "coverage_eligible": mapped / eligible,
                   "nonzero_prior_states": int(np.count_nonzero(np.any(q != 0, axis=1))),
                   "source_shape": list(self.source.shape), "target_shape": list(q.shape),
                   "source_contract": deepcopy(SOURCE_CONTRACT),
                   "target_contract": deepcopy(TARGET_CONTRACT),
                   "actions": list(SOURCE_CONTRACT["actions"]),
                   "remote_topology_equivalence": "UNVERIFIED; local signature only"}


def source_variant(source, arm, seed):
    """Shuffle nonterminal source entries, preserving their exact value multiset."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    result = source.copy()
    permutation = None
    if arm == "untrained_source":
        result.fill(0)
    elif arm == "shuffled_prior":
        indices = list(NavigationWorld().starts)
        flat = source[indices].ravel()
        permutation = np.random.default_rng(np.random.SeedSequence([seed, 303])).permutation(len(flat))
        result[indices] = flat[permutation].reshape(-1, 4)
        permutation = permutation.tolist()
    return result, permutation


def evaluate_target(q):
    """One deterministic fixed-base rollout. It is NOT 100 independent samples."""
    AgentStore.validate_q(q, "resource_collection")
    env = ResourceCollectionWorld()
    state, _ = env.reset()
    total = 0.0
    while True:
        state, reward, terminal, timeout, _ = env.step(int(q[state].argmax()))
        total += reward
        if terminal or timeout:
            return {"success_rate": float(terminal), "mean_reward": total,
                    "successful_steps": env.steps if terminal else None,
                    "steps": env.steps, "failures": int(not terminal), "timeouts": int(timeout),
                    "evaluation_rollouts": 1}


def curve_metrics(rows):
    x = np.array([row["training_steps"] for row in rows])
    if len(x) < 2 or x[0] != 0 or not np.all(np.diff(x) > 0):
        raise ValueError("Checkpoints must increase from zero")
    result = {}
    for name, key in (("success_auc", "success_rate"), ("reward_auc", "mean_reward")):
        y = np.array([row[key] for row in rows])
        result[name] = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
        result[name + "_normalized"] = result[name] / int(x[-1])
    return result


def train_arm(source, metadata, arm, seed, checkpoints=CHECKPOINTS):
    if (len(checkpoints) < 2 or checkpoints[0] != 0
            or any(type(x) is not int or x < 0 for x in checkpoints)
            or any(b <= a for a, b in zip(checkpoints, checkpoints[1:]))):
        raise ValueError("Invalid checkpoint budget")
    variant, permutation = source_variant(source, arm, seed)
    prior, coverage = CrossSkillAdapter(variant, metadata).initialize()
    q = np.zeros_like(prior) if arm == "scratch" else prior.copy()
    initial = q.copy()
    # Shared per-transition draws: trajectory divergence never shifts RNG alignment.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    explore = rng.random(checkpoints[-1])
    actions = rng.integers(4, size=checkpoints[-1])
    env = ResourceCollectionWorld()
    state, _ = env.reset()
    history, rows, models = [], [], {}
    episode_reward = 0.0
    visited, mapped_visits = Counter(), 0
    for step in range(checkpoints[-1] + 1):
        if step in checkpoints:
            rows.append({"training_steps": step, "completed_episodes": len(history), **evaluate_target(q)})
            models[step] = q.copy()
        if step == checkpoints[-1]:
            break
        visited[state] += 1
        if coverage["mapping"][state]["source_state"] is not None:
            mapped_visits += 1
        epsilon = 1.0 - .95 * min(step / (checkpoints[-1] * .8), 1.0)
        action = int(actions[step]) if explore[step] < epsilon else int(q[state].argmax())
        following, reward, terminal, timeout, _ = env.step(action)
        target = reward + (0.0 if terminal else .95 * q[following].max())
        q[state, action] += .1 * (target - q[state, action])
        state = following
        episode_reward += reward
        if terminal or timeout:
            history.append({"episode": len(history) + 1, "training_steps": step + 1,
                            "success": bool(terminal), "timeout": bool(timeout),
                            "steps": env.steps, "reward": episode_reward})
            state, _ = env.reset()
            episode_reward = 0.0
    return {"arm": arm, "training_seed": seed, "checkpoints": rows, "training": history,
            "actual_training_steps": checkpoints[-1], "completed_episodes": len(history),
            "partial_episode": {"steps": env.steps, "reward": episode_reward},
            "source_permutation": permutation,
            "variant_q_sha256": hashlib.sha256(variant.tobytes()).hexdigest(),
            "initial_q_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "checkpoint_q_sha256": {str(step): hashlib.sha256(model.tobytes()).hexdigest()
                                    for step, model in models.items()},
            "coverage": {k: v for k, v in coverage.items() if k not in ("mapping", "source_contract", "target_contract")},
            "mapped_training_visit_fraction": mapped_visits / checkpoints[-1],
            "training_state_visits": dict(visited), **curve_metrics(rows)}, models


def statistics(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1)) if n > 1 else 0.0
    # Fixed main protocol has n=10: Student t(9) .975. Smaller test runs get no CI.
    interval = [float(mean - 2.262157 * sd / np.sqrt(n)), float(mean + 2.262157 * sd / np.sqrt(n))] if n == 10 else None
    return {"n": n, "mean": mean, "sample_sd": sd, "descriptive_95ci": interval}


def aggregate(runs):
    summary, differences, comparisons = [], [], []
    for arm in ARMS:
        chosen = [r for r in runs if r["arm"] == arm]
        for metric in ("success_auc", "reward_auc", "success_auc_normalized", "reward_auc_normalized"):
            summary.append({"arm": arm, "metric": metric, **statistics([r[metric] for r in chosen])})
    for control in ARMS:
        if control == "transfer":
            continue
        for seed in sorted({r["training_seed"] for r in runs}):
            pair = {r["arm"]: r for r in runs if r["training_seed"] == seed}
            for metric in ("success_auc", "reward_auc"):
                differences.append({"training_seed": seed, "comparison": "transfer_minus_" + control,
                                    "metric": metric, "difference": pair["transfer"][metric] - pair[control][metric]})
        for metric in ("success_auc", "reward_auc"):
            stats = statistics([d["difference"] for d in differences
                                if d["comparison"] == "transfer_minus_" + control and d["metric"] == metric])
            comparisons.append({"comparison": "transfer_minus_" + control, "metric": metric, **stats,
                                "direction_of_mean": "positive" if stats["mean"] > 0 else "negative" if stats["mean"] < 0 else "neutral",
                                "interpretation": "descriptive fixed-source, fixed-map result; no generalization claim"})
    return summary, differences, comparisons


def checkpoint_summary(runs):
    rows = []
    for arm in ARMS:
        selected = [r for r in runs if r["arm"] == arm]
        for index, checkpoint in enumerate(selected[0]["checkpoints"]):
            measured = [r["checkpoints"][index] for r in selected]
            successful = [r["successful_steps"] for r in measured if r["successful_steps"] is not None]
            row = {"arm": arm, "training_steps": checkpoint["training_steps"], "n": len(selected),
                   "successes": len(successful), "failures": sum(r["failures"] for r in measured),
                   "timeouts": sum(r["timeouts"] for r in measured),
                   "successful_steps_mean": float(np.mean(successful)) if successful else None}
            for key in ("success_rate", "mean_reward"):
                stats = statistics([r[key] for r in measured])
                row[key + "_mean"], row[key + "_sd"] = stats["mean"], stats["sample_sd"]
            rows.append(row)
    return rows


def source_effect_check(source, metadata):
    """In-memory intervention on the real loaded source; never save it as a skill."""
    prior, report = CrossSkillAdapter(source, metadata).initialize()
    chosen = next((r for r in report["mapping"] if r["source_state"] is not None), None)
    if chosen is None:
        return {"status": "NO_MAPPING", "source_intervention_verified": False}
    modified = source.copy()
    modified[chosen["source_state"], 2] += 1.0
    altered, _ = CrossSkillAdapter(modified, metadata).initialize()
    observed = altered[chosen["target_state"]] - prior[chosen["target_state"]]
    verified = bool(np.allclose(observed, [-.25, -.25, .75, -.25]))
    if not verified:
        raise ValueError("Source intervention did not affect prior as specified")
    return {"status": "PASS", "source_intervention_verified": verified,
            "source_state": chosen["source_state"], "target_state": chosen["target_state"],
            "in_memory_intervention": "add 1 to source action down; no file write",
            "expected_prior_delta": [-.25, -.25, .75, -.25], "observed_prior_delta": observed.tolist(),
            "changed_target_rows": np.flatnonzero(np.any(altered != prior, axis=1)).tolist()}


def plot_results(path, runs, differences):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ("success_rate", "mean_reward")):
        for arm in ARMS:
            selected = [r for r in runs if r["arm"] == arm]
            x = [r["training_steps"] for r in selected[0]["checkpoints"]]
            y = np.array([[r[key] for r in run["checkpoints"]] for run in selected])
            line, = ax.plot(x, y.mean(axis=0), label=arm)
            for row in y:
                ax.plot(x, row, alpha=.12, color=line.get_color())
            sd = y.std(axis=0, ddof=1) if len(y) > 1 else np.zeros(len(x))
            ax.fill_between(x, y.mean(axis=0) - sd, y.mean(axis=0) + sd, alpha=.12, color=line.get_color())
        ax.set(xlabel="Target training transitions", ylabel=key, title="Seeds: mean, sample SD, individual curves")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(path / "learning_curves.png")
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    controls = [a for a in ARMS if a != "transfer"]
    for ax, metric in zip(axes, ("success_auc", "reward_auc")):
        for i, arm in enumerate(controls):
            vals = [d["difference"] for d in differences if d["comparison"] == "transfer_minus_" + arm and d["metric"] == metric]
            ax.scatter(np.full(len(vals), i), vals, alpha=.5)
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, fmt="ks", capsize=5)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set(xticks=range(len(controls)), xticklabels=controls, ylabel="Transfer minus control: " + metric,
               title="Paired seeds; black = mean +/- sample SD")
    fig.tight_layout()
    try:
        fig.savefig(path / "transfer_comparison.png")
    finally:
        plt.close(fig)


def legacy_hashes(root):
    """All saved agents, baseline evidence and prior M2/M3-A experiment evidence."""
    root = Path(root)
    names = {"agents", "baselines", "m3-transfer"}
    names.update(p.name for p in root.glob("m2-*") if p.is_dir())
    return {name: hashes(root / name) for name in sorted(names)}


def run_experiment(root, agent="Alph", version=None, destination=Path("outputs/m3-cross-skill"),
                   *, checkpoints=CHECKPOINTS, seeds=SEEDS):
    started = time.perf_counter()
    root, destination = Path(root), Path(destination)
    store = AgentStore(root, agent)
    # Output must never be inside a retained skill/evidence directory.
    before = legacy_hashes(root)
    for name in before:
        if destination.resolve().is_relative_to((root / name).resolve()):
            raise ValueError("Output overlaps retained evidence")
    source, metadata, _ = store.load("navigation", version)
    source.setflags(write=False)
    _, compatibility = CrossSkillAdapter(source, metadata).initialize()
    mechanism = source_effect_check(source, metadata)
    old = {skill: evaluate(20, 90000, store.load(skill)[0], skill)
           for skill in ("navigation", "resource_collection") if store.versions(skill)}
    protocol = {"version": "m3b-fixed-steps-v1", "adapter_version": ADAPTER_VERSION,
                "arms": list(ARMS), "training_seeds": list(seeds), "checkpoints_transitions": list(checkpoints),
                "alpha": .1, "gamma": .95, "prior_scale": 1.0,
                "initialization": "Q_source(s,a) - mean_a Q_source(s,a); fallback zero; no ongoing shaping",
                "subgoals": "first uncollected resource in row-major order, then base; used only for initialization",
                "epsilon": "1 - .95 * min(global_transition / (.8 * budget), 1)",
                "budget": "equal actual target transitions; final partial episode retained; episodes can differ",
                "training_rng": "PCG64 SeedSequence([seed,202]); budget uniforms then budget integers, per transition",
                "shuffle_rng": "PCG64 SeedSequence([seed,303]); permutation of flattened nonterminal source entries",
                "evaluation_rng": "none: exactly one deterministic base-start greedy rollout per checkpoint",
                "tie_break": "first argmax: up,right,down,left", "start": "fixed M2 base in every arm",
                "auc": "sum((x[i+1]-x[i])*(y[i]+y[i+1])/2), x=actual target transitions; normalized=raw/budget",
                "ci": "descriptive paired Student-t interval, df=9, multiplier 2.262157; fixed source/map only",
                "truncation": "bootstrap at time limit, no bootstrap at task completion; same across arms",
                "scope": "one frozen M1 source; unchanged M2 target; no held-out maps or source-seed replication",
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "gymnasium": gymnasium.__version__}}
    manifest = {"identity": store.identity, "metadata": metadata,
                "q_file": str((store.path / "navigation" / metadata["version"] / "q.npz").resolve()),
                "q_file_sha256": metadata["q_sha256"], "adapter_version": ADAPTER_VERSION,
                "source_training_target": "navigation only, per retained source metadata/history",
                "source_selection": "explicit version or latest resolved before target runs; no outcome-based selection"}
    result = {"stage": "M3-B", "status": "implemented and evaluated; effect reported descriptively",
              "source": manifest, "protocol": protocol, "runs": []}

    def writer(path):
        write_json(path / "protocol.json", protocol)
        write_json(path / "source_manifest.json", manifest)
        write_json(path / "compatibility_report.json", compatibility)
        write_json(path / "integrity_before.json", before)
        for seed in seeds:
            for arm in ARMS:
                run, models = train_arm(source, metadata, arm, seed, checkpoints)
                name = f"{arm}-{seed}.npz"
                np.savez_compressed(path / name, **{f"step_{step}": q for step, q in models.items()})
                run["model_file"] = name
                result["runs"].append(run)
        result["summary"], result["paired_differences"], result["comparisons"] = aggregate(result["runs"])
        result["checkpoint_summary"] = checkpoint_summary(result["runs"])
        mechanism["scratch_equals_untrained_all_seeds"] = all(
            paired["scratch"]["checkpoint_q_sha256"] == paired["untrained_source"]["checkpoint_q_sha256"]
            and paired["scratch"]["training"] == paired["untrained_source"]["training"]
            for seed in seeds
            for paired in [{r["arm"]: r for r in result["runs"] if r["training_seed"] == seed}])
        mechanism["trained_prior_differs_from_untrained"] = all(
            paired["transfer"]["initial_q_sha256"] != paired["untrained_source"]["initial_q_sha256"]
            for seed in seeds
            for paired in [{r["arm"]: r for r in result["runs"] if r["training_seed"] == seed}])
        result["mechanism_checks"] = mechanism
        write_json(path / "mechanism_checks.json", mechanism)
        write_csv(path / "summary.csv", result["summary"])
        write_csv(path / "checkpoint_summary.csv", result["checkpoint_summary"])
        write_csv(path / "paired_differences.csv", result["paired_differences"])
        write_csv(path / "comparisons.csv", result["comparisons"])
        write_csv(path / "checkpoints.csv", [{"arm": r["arm"], "seed": r["training_seed"], **row}
                                             for r in result["runs"] for row in r["checkpoints"]])
        write_csv(path / "seed_metrics.csv", [{k: r[k] for k in ("arm", "training_seed", "success_auc", "reward_auc",
                                                               "actual_training_steps", "completed_episodes", "mapped_training_visit_fraction")}
                                               for r in result["runs"]])
        plot_results(path, result["runs"], result["paired_differences"])
        after_performance = {skill: evaluate(20, 90000, store.load(skill)[0], skill) for skill in old}
        after = legacy_hashes(root)
        integrity = {"before": before, "after": after, "unchanged": before == after,
                     "evaluation_before": old, "evaluation_after": after_performance,
                     "evaluations_unchanged": old == after_performance, "identity": store.identity}
        write_json(path / "integrity_report.json", integrity)
        if not integrity["unchanged"] or not integrity["evaluations_unchanged"]:
            raise ValueError("Retained evidence or performance changed")
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(path / "results.json", result)
        write_json(path / "sha256.json", hashes(path))

    output = publish(destination, str(uuid4()), writer)
    return {"directory": str(output.resolve()), "source_version": metadata["version"],
            "elapsed_seconds": result["elapsed_seconds"], "comparisons": result["comparisons"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="Alph")
    parser.add_argument("--source-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--version", help="Navigation source version; default latest, pinned in manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m3-cross-skill"))
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.source_dir, args.agent, args.version, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"m3-cross-skill: {error}\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()

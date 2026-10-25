"""M1 command line interface; all generated artifacts live under outputs/."""

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

from .learning import evaluate, train
from .storage import AgentStore, publish, read_json, write_json


def summary(result):
    return {k: v for k, v in result.items() if k not in ("records", "starts")}


def report(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _, skill, history = store.load()
    evaluations = [read_json(p) for p in (store.path / "evaluations").glob("*/result.json")
                   if not p.parent.name.startswith(".pending-")]
    evaluations = [e for e in evaluations if e["version"] == skill["version"]]
    latest = max(evaluations, key=lambda e: e["created_ns"], default=None)
    result = {"agent": store.identity, "skill": skill, "version_count": len(store.versions()),
              "scope": "Fixed M1 map only; no evidence of generalization to unseen maps.",
              "step_metric": "Mean across all episodes, including timeouts (100 steps).",
              "evaluation": None if latest is None else
                  {"trained": summary(latest["trained"]), "random": summary(latest["random"])}}

    def writer(path):
        write_json(path / "report.json", result)
        window = min(100, len(history))
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for ax, key in zip(axes, ("success", "steps", "reward")):
            values = np.array([r[key] for r in history], dtype=float)
            ax.plot(np.arange(window, len(values) + 1),
                    np.convolve(values, np.ones(window) / window, mode="valid"))
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"{store.identity['name']} — training ({window}-episode mean)")
        axes[-1].set_xlabel("Training episode")
        fig.tight_layout()
        try:
            fig.savefig(path / "training.png")
        finally:
            plt.close(fig)

    output = publish(store.path / "reports", str(uuid4()), writer)
    return {**result, "report_directory": str(output.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agentverse")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "baseline", "report"):
        child = commands.add_parser(command)
        child.add_argument("--output-dir", type=Path, default=Path("outputs"))
        if command != "baseline":
            child.add_argument("--agent", required=True)
        if command != "report":
            child.add_argument("--episodes", type=int, default=10000 if command == "train" else 100)
            child.add_argument("--seed", type=int, default=42 if command == "train" else 123)
    playback = commands.add_parser("replay", help="Watch a saved policy or a random agent")
    playback.add_argument("--agent", default="Alph")
    playback.add_argument("--mode", choices=("trained", "random"), default="trained")
    playback.add_argument("--seed", type=int, default=123)
    playback.add_argument("--interval-ms", type=int, default=300)
    playback.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.command == "replay":
        if args.seed < 0 or args.interval_ms <= 0:
            parser.error("seed must be nonnegative and interval-ms positive")
        from .replay import replay
        try:
            result = replay(args.output_dir, args.agent, args.mode, args.seed, args.interval_ms)
        except (OSError, ValueError, KeyError, ImportError) as error:
            parser.exit(1, f"agentverse: {error}\n")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return result
    if args.command != "report" and (args.episodes <= 0 or args.seed < 0):
        parser.error("episodes must be positive and seed nonnegative")
    try:
        if args.command == "baseline":
            metrics = evaluate(args.episodes, args.seed)
            output = publish(args.output_dir / "baselines", str(uuid4()),
                             lambda p: write_json(p / "result.json", metrics))
            result = {**summary(metrics), "directory": str(output.resolve())}
        else:
            store = AgentStore(args.output_dir, args.agent, create=args.command == "train")
            if args.command == "train":
                q, history = train(args.episodes, args.seed)
                skill = store.save(q, history, args.seed)
                result = {"agent": store.identity, "skill": skill}
            elif args.command == "evaluate":
                q, skill, _ = store.load()
                trained = evaluate(args.episodes, args.seed, q)
                random = evaluate(args.episodes, args.seed)
                assert trained["starts"] == random["starts"]
                evaluation = {"created_ns": time.time_ns(), "version": skill["version"],
                              "trained": trained, "random": random}
                output = store.save_evaluation(evaluation)
                result = {"trained": summary(trained), "random": summary(random),
                          "directory": str(output.resolve())}
            else:
                result = report(store)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"agentverse: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result

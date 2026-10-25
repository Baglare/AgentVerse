"""Immutable version directories published by rename after complete writes."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import UUID, uuid4
from zipfile import BadZipFile

import numpy as np

from .skills import skill_spec


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(parent, name, writer):
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        writer(temporary)
        destination = parent / name
        if destination.exists():
            raise FileExistsError(destination)
        # Every published directory is nonempty; rename cannot replace it.
        for attempt in range(3):
            try:
                temporary.rename(destination)
                break
            except PermissionError:
                # Windows may briefly hold a newly written file open.
                # Keep the same unpublished directory and fail closed if persistent.
                if os.name != "nt" or attempt == 2:
                    raise
                time.sleep(0.05)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class AgentStore:
    def __init__(self, root, name, *, create=False):
        if not name or not name.strip():
            raise ValueError("Agent name cannot be empty")
        key = hashlib.sha256(name.encode("utf-8")).hexdigest()
        self.path = Path(root) / "agents" / key
        if create and not self.path.exists():
            identity = {"agent_id": str(uuid4()), "name": name, "schema": 1}
            try:
                publish(self.path.parent, key, lambda p: write_json(p / "agent.json", identity))
            except FileExistsError:
                pass  # A concurrent creator published the identity first.
        self.identity = read_json(self.path / "agent.json")
        if self.identity["name"] != name or self.identity["schema"] != 1:
            raise ValueError("Agent identity mismatch")
        UUID(self.identity["agent_id"])

    def save(self, q, history, seed, skill="navigation"):
        _, contract = skill_spec(skill)
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.validate_q(q, skill)
        self.validate_history(history)
        version = str(uuid4())
        metadata = {"schema": 1, "agent_id": self.identity["agent_id"],
                    "skill_id": skill, "version": version,
                    "created_ns": time.time_ns(), "training_seed": seed,
                    "episodes": len(history), "contract": contract,
                    "algorithm": {"name": "tabular-q-learning", "alpha": 0.1,
                                  "gamma": 0.95, "epsilon_start": 1.0,
                                  "epsilon_end": 0.05, "decay_fraction": 0.8}}

        def writer(path):
            with (path / "q.npz").open("xb") as stream:
                np.savez_compressed(stream, q=q)
                stream.flush()
                os.fsync(stream.fileno())
            metadata["q_sha256"] = hashlib.sha256((path / "q.npz").read_bytes()).hexdigest()
            write_json(path / "training.json", history)
            metadata["training_sha256"] = hashlib.sha256((path / "training.json").read_bytes()).hexdigest()
            write_json(path / "metadata.json", metadata)

        publish(self.path / skill, version, writer)
        return metadata

    @staticmethod
    def validate_q(q, skill="navigation"):
        env = skill_spec(skill)[0]()
        if (not isinstance(q, np.ndarray)
                or q.shape != (env.observation_space.n, env.action_space.n) or q.dtype != np.dtype("float64")
                or not np.isfinite(q).all()):
            raise ValueError("Invalid Q-table shape, dtype or values")

    @staticmethod
    def validate_history(history):
        if not isinstance(history, list) or not history:
            raise ValueError("Training history must contain episodes")
        for record in history:
            if (not isinstance(record, dict)
                    or type(record.get("steps")) is not int or not 1 <= record["steps"] <= 100
                    or type(record.get("success")) is not bool
                    or type(record.get("reward")) not in (int, float)
                    or not np.isfinite(record["reward"])):
                raise ValueError("Invalid training history")

    def versions(self, skill="navigation"):
        _, contract = skill_spec(skill)
        parent = self.path / skill
        versions = []
        if parent.exists():
            for path in parent.iterdir():
                if path.name.startswith(".pending-") or path.name in ("evaluations", "reports"):
                    continue
                UUID(path.name)
                metadata = read_json(path / "metadata.json")
                if (not isinstance(metadata, dict)
                        or metadata.get("version") != path.name or metadata.get("schema") != 1
                        or metadata.get("agent_id") != self.identity["agent_id"]
                        or metadata.get("skill_id") != skill or metadata.get("contract") != contract
                        or type(metadata.get("created_ns")) is not int
                        or type(metadata.get("episodes")) is not int or metadata["episodes"] <= 0
                        or type(metadata.get("training_seed")) is not int or metadata["training_seed"] < 0
                        or not isinstance(metadata.get("q_sha256"), str)):
                    raise ValueError("Skill metadata is incompatible")
                versions.append(metadata)
        return sorted(versions, key=lambda m: (m["created_ns"], m["version"]))

    def load(self, skill="navigation", version=None):
        versions = self.versions(skill)
        if not versions:
            raise ValueError(f"No {skill} skill saved; train this agent first")
        metadata = versions[-1] if version is None else next(
            (m for m in versions if m["version"] == version), None)
        if metadata is None:
            raise ValueError("Unknown skill version")
        path = self.path / skill / metadata["version"]
        payload = (path / "q.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["q_sha256"]:
            raise ValueError("Q-table checksum mismatch")
        try:
            with np.load(BytesIO(payload), allow_pickle=False) as data:
                if data.files != ["q"]:
                    raise ValueError("Unexpected NPZ contents")
                q = data["q"]
        except (BadZipFile, EOFError, TypeError) as error:
            raise ValueError("Invalid Q-table archive") from error
        self.validate_q(q, skill)
        payload = (path / "training.json").read_bytes()
        # M1 versions predate history checksums; keep them readable without migration.
        if "training_sha256" in metadata:
            if hashlib.sha256(payload).hexdigest() != metadata["training_sha256"]:
                raise ValueError("Training history checksum mismatch")
        elif skill != "navigation":
            raise ValueError("Missing training history checksum")
        history = json.loads(payload)
        self.validate_history(history)
        if len(history) != metadata["episodes"]:
            raise ValueError("Training history length mismatch")
        return q, metadata, history

    def save_evaluation(self, result, skill="navigation"):
        skill_spec(skill)
        parent = self.path if skill == "navigation" else self.path / skill
        return publish(parent / "evaluations", str(uuid4()),
                       lambda p: write_json(p / "result.json", result))

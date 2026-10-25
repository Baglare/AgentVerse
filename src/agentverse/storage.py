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

import numpy as np

from .world import CONTRACT, NavigationWorld


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

    def save(self, q, history, seed):
        self.validate_q(q)
        version = str(uuid4())
        metadata = {"schema": 1, "agent_id": self.identity["agent_id"],
                    "skill_id": "navigation", "version": version,
                    "created_ns": time.time_ns(), "training_seed": seed,
                    "episodes": len(history), "contract": CONTRACT,
                    "algorithm": {"name": "tabular-q-learning", "alpha": 0.1,
                                  "gamma": 0.95, "epsilon_start": 1.0,
                                  "epsilon_end": 0.05, "decay_fraction": 0.8}}

        def writer(path):
            with (path / "q.npz").open("xb") as stream:
                np.savez_compressed(stream, q=q)
                stream.flush()
                os.fsync(stream.fileno())
            metadata["q_sha256"] = hashlib.sha256((path / "q.npz").read_bytes()).hexdigest()
            write_json(path / "metadata.json", metadata)
            write_json(path / "training.json", history)

        publish(self.path / "navigation", version, writer)
        return metadata

    @staticmethod
    def validate_q(q):
        env = NavigationWorld()
        if (q.shape != (env.observation_space.n, 4) or q.dtype != np.dtype("float64")
                or not np.isfinite(q).all()):
            raise ValueError("Invalid Q-table shape, dtype or values")

    def versions(self):
        parent = self.path / "navigation"
        versions = []
        if parent.exists():
            for path in parent.iterdir():
                if path.name.startswith(".pending-"):
                    continue
                UUID(path.name)
                metadata = read_json(path / "metadata.json")
                if (metadata["version"] != path.name or metadata["schema"] != 1
                        or metadata["agent_id"] != self.identity["agent_id"]
                        or metadata["skill_id"] != "navigation" or metadata["contract"] != CONTRACT):
                    raise ValueError("Skill metadata is incompatible")
                versions.append(metadata)
        return sorted(versions, key=lambda m: (m["created_ns"], m["version"]))

    def load(self):
        versions = self.versions()
        if not versions:
            raise ValueError("No navigation skill saved; train this agent first")
        metadata = versions[-1]
        path = self.path / "navigation" / metadata["version"]
        payload = (path / "q.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["q_sha256"]:
            raise ValueError("Q-table checksum mismatch")
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            if data.files != ["q"]:
                raise ValueError("Unexpected NPZ contents")
            q = data["q"]
        self.validate_q(q)
        return q, metadata, read_json(path / "training.json")

    def save_evaluation(self, result):
        return publish(self.path / "evaluations", str(uuid4()),
                       lambda p: write_json(p / "result.json", result))

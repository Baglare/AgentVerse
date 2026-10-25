import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

from gymnasium.utils.env_checker import check_env
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.registry import backend_registry
import numpy as np
import pytest

from agentverse.cli import main
from agentverse.collection import ResourceCollectionWorld
from agentverse.learning import evaluate, train
from agentverse.replay import ReplaySession, ReplayViewer, replay
from agentverse.storage import AgentStore, publish

SKILL = "resource_collection"


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def test_collection_contract_rewards_and_success():
    env = ResourceCollectionWorld(render_mode="ansi")
    check_env(env, skip_render_check=True)
    state, _ = env.reset(seed=42)
    assert state == env.base and env.collected == 0
    assert len(env.render().splitlines()) == 6
    assert env.step(0)[1] == pytest.approx(-0.06)
    with pytest.raises(ValueError):
        env.step(4)
    # First resource: down, down, right, right.
    for action in (2, 2, 1, 1):
        state, reward, terminated, _, _ = env.step(action)
    assert reward == pytest.approx(0.19) and not terminated
    assert state == len(env.cells) + env.position and env.collected == 1
    env.step(3)
    assert env.step(1)[1] == pytest.approx(-0.01)
    # Returning with only one resource is not success.
    for action in (3, 3, 0, 0):
        _, _, terminated, _, _ = env.step(action)
    assert env.position == env.base and not terminated
    # Collect second resource and return through the bottom of the map.
    for action in (2, 2, 1, 1, 2, 1, 1, 2):
        _, reward, terminated, _, _ = env.step(action)
    assert env.collected == 3 and not terminated
    assert reward == pytest.approx(0.19)
    for action in (2, 3, 3, 3, 3, 0, 0, 0, 0, 0):
        _, reward, terminated, truncated, _ = env.step(action)
    assert terminated and not truncated and reward == pytest.approx(0.99)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    assert env.collected == 0
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    with pytest.raises(ValueError):
        env.reset(options={"start": 1})


@pytest.fixture(scope="module")
def collection_trained():
    return train(3000, 42, SKILL)


def test_collection_reproducibility_and_random_comparison(collection_trained):
    q, history = collection_trained
    repeat, repeated_history = train(3000, 42, SKILL)
    np.testing.assert_array_equal(q, repeat)
    assert history == repeated_history
    before = q.copy()
    trained = evaluate(100, 123, q, SKILL)
    random = evaluate(100, 123, skill=SKILL)
    assert trained == evaluate(100, 123, q, SKILL)
    assert random == evaluate(100, 123, skill=SKILL)
    assert trained["starts"] == random["starts"]
    assert trained["success_rate"] == 1.0 > random["success_rate"]
    assert trained["mean_reward"] > random["mean_reward"]
    assert trained["mean_steps"] < random["mean_steps"]
    np.testing.assert_array_equal(q, before)


def test_identity_legacy_hashes_and_independent_versions(tmp_path, collection_trained):
    store = AgentStore(tmp_path, "Alph", create=True)
    q, history = train(50, 42)
    first = store.save(q, history, 42)
    legacy = store.path / "navigation" / first["version"] / "metadata.json"
    metadata = json.loads(legacy.read_text())
    metadata.pop("training_sha256")  # Exact pre-M2 schema, no migration.
    legacy.write_text(json.dumps(metadata))
    second = store.save(q + 1, history, 43)
    identity = store.identity.copy()
    snapshot = hashes(store.path / "navigation")
    identity_hash = hashlib.sha256((store.path / "agent.json").read_bytes()).hexdigest()
    expected = evaluate(20, 123, store.load()[0])
    cq, ch = collection_trained
    c1 = store.save(cq, ch, 42, SKILL)
    c2 = store.save(cq + 1, ch, 43, SKILL)
    reopened = AgentStore(tmp_path, "Alph", create=True)
    assert reopened.identity == identity
    assert hashlib.sha256((store.path / "agent.json").read_bytes()).hexdigest() == identity_hash
    assert hashes(store.path / "navigation") == snapshot
    assert evaluate(20, 123, reopened.load()[0]) == expected
    np.testing.assert_array_equal(reopened.load(version=first["version"])[0], q)
    np.testing.assert_array_equal(reopened.load(SKILL, c1["version"])[0], cq)
    assert len(reopened.versions()) == len(reopened.versions(SKILL)) == 2
    assert reopened.load()[1]["version"] == second["version"]
    assert reopened.load(SKILL)[1]["version"] == c2["version"]
    for skill, version in (("navigation", first["version"]), (SKILL, c1["version"])):
        before = hashes(store.path)
        with pytest.raises(FileExistsError):
            publish(store.path / skill, version, lambda p: (p / "x").write_text("x"))
        assert hashes(store.path) == before
    with pytest.raises(ValueError):
        reopened.load(SKILL, "../../navigation")


@pytest.mark.parametrize("corruption", ["checksum", "shape", "dtype", "nan", "object",
                                      "contract", "identity", "skill", "history", "history_length",
                                      "metadata", "archive", "history_record", "missing_history_hash"])
def test_collection_corruption_rejected(tmp_path, collection_trained, corruption):
    q, history = collection_trained
    store = AgentStore(tmp_path, "Alph", create=True)
    metadata = store.save(q, history, 42, SKILL)
    path = store.path / SKILL / metadata["version"]
    if corruption == "checksum":
        (path / "q.npz").write_bytes(b"broken")
    elif corruption in ("contract", "identity", "skill"):
        key = {"contract": "contract", "identity": "agent_id", "skill": "skill_id"}[corruption]
        metadata[key] = "incompatible"
    elif corruption == "history":
        (path / "training.json").write_text("[]")
    elif corruption == "history_length":
        metadata["episodes"] += 1
    elif corruption == "metadata":
        metadata = []
    elif corruption == "missing_history_hash":
        metadata.pop("training_sha256")
    elif corruption == "history_record":
        (path / "training.json").write_text('[{"steps": 0, "success": true, "reward": 1}]')
        metadata["training_sha256"] = hashlib.sha256((path / "training.json").read_bytes()).hexdigest()
    elif corruption == "archive":
        (path / "q.npz").write_bytes(b"PK\x03\x04broken")
        metadata["q_sha256"] = hashlib.sha256((path / "q.npz").read_bytes()).hexdigest()
    else:
        bad = {"shape": np.zeros((1, 4)), "dtype": q.astype(np.float32),
               "nan": q * np.nan, "object": np.array([[object()]], dtype=object)}[corruption]
        np.savez(path / "q.npz", q=bad)
        metadata["q_sha256"] = hashlib.sha256((path / "q.npz").read_bytes()).hexdigest()
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        store.load(SKILL)


@pytest.mark.parametrize("skill", ["navigation", SKILL])
@pytest.mark.parametrize("random", [False, True])
def test_both_replays_read_only_and_match_evaluation(tmp_path, monkeypatch, collection_trained, skill, random):
    q, history = collection_trained if skill == SKILL else train(50, 42)
    store = AgentStore(tmp_path, "Alph", create=True)
    metadata = store.save(q, history, 42, skill)
    snapshot = hashes(tmp_path)
    monkeypatch.setattr(backend_registry, "resolve_backend", lambda _: ("tkagg", "tk"))

    def show(viewer):
        viewer.timer = Mock()
        while viewer.session.step():
            pass
        viewer.draw()
        reference = evaluate(1, 123, None if random else q, skill)
        assert viewer.session.env.steps == reference["mean_steps"]
        assert viewer.session.total_reward == reference["mean_reward"]
        assert viewer.session.terminated == bool(reference["success_rate"])
        if skill == SKILL:
            assert "Kaynaklar:" in viewer.axes.get_title()
            for bit, marker in enumerate(viewer.resource_markers):
                assert marker.get_visible() == (not bool(viewer.session.env.collected & (1 << bit)))
            r, c = viewer.session.env.cells[viewer.session.env.position]
            np.testing.assert_array_equal(viewer.marker.get_offsets(), [[c, r]])
        viewer.reset()
        assert viewer.session.env.steps == 0
        if skill == SKILL:
            assert viewer.session.env.collected == 0
            assert all(m.get_visible() for m in viewer.resource_markers)
        viewer.close()
        plt.close(viewer.figure)

    monkeypatch.setattr(ReplayViewer, "show", show)
    result = replay(tmp_path, "Alph", "random" if random else "trained", 123, 300, skill)
    assert result["version"] == (None if random else metadata["version"])
    assert hashes(tmp_path) == snapshot


def test_collection_cli_all_commands(tmp_path, monkeypatch):
    shared = ["--skill", SKILL, "--output-dir", str(tmp_path)]
    training = main(["train", "--agent", "Alph", "--episodes", "10", *shared])
    comparison = main(["evaluate", "--agent", "Alph", "--episodes", "10", *shared])
    baseline = main(["baseline", "--episodes", "10", *shared])
    assert comparison["random"] == {k: v for k, v in baseline.items() if k != "directory"}
    report = main(["report", "--agent", "Alph", *shared])
    assert report["skill"] == training["skill"]
    assert report["evaluation"]["trained"] == comparison["trained"]
    runner = Mock(return_value={"skill_id": SKILL})
    monkeypatch.setattr("agentverse.replay.replay", runner)
    main(["replay", "--agent", "Alph", *shared])
    runner.assert_called_once_with(tmp_path, "Alph", "trained", 123, 300, skill=SKILL)
    for command in ("train", "evaluate", "baseline", "report", "replay"):
        with pytest.raises(SystemExit):
            main([command, "--skill", "../navigation"])


def test_cli_historical_version_selection(tmp_path, monkeypatch, collection_trained):
    store = AgentStore(tmp_path, "Alph", create=True)
    q, history = collection_trained
    first = store.save(q, history, 42, SKILL)
    store.save(q + 1, history, 43, SKILL)
    shared = ["--agent", "Alph", "--skill", SKILL, "--version", first["version"],
              "--output-dir", str(tmp_path)]
    result = main(["evaluate", "--episodes", "1", *shared])
    assert result["trained"]["success_rate"] == 1.0
    assert main(["report", *shared])["skill"]["version"] == first["version"]
    runner = Mock(return_value={})
    monkeypatch.setattr("agentverse.replay.replay", runner)
    main(["replay", *shared])
    runner.assert_called_once_with(tmp_path, "Alph", "trained", 123, 300,
                                   skill=SKILL, version=first["version"])

import hashlib
import json
import os
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from agentverse.cli import main
from agentverse.learning import evaluate, train
from agentverse.storage import AgentStore, publish
from agentverse.world import NavigationWorld


def test_gym_contract_and_all_starts_reachable():
    env = NavigationWorld()
    check_env(env, skip_render_check=True)
    reached = {env.cells[env.goal]}
    while True:
        expanded = reached | {p for p in env.cells if any(
            abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in reached)}
        if expanded == reached:
            break
        reached = expanded
    assert reached == set(env.cells)
    assert env.goal not in env.starts


def test_rewards_limits_and_lifecycle():
    env = NavigationWorld()
    env.reset(options={"start": env.indices[(0, 0)]})
    assert env.step(0)[1] == pytest.approx(-0.06)
    assert env.step(1)[1] == pytest.approx(-0.01)
    env.reset(options={"start": env.indices[(4, 3)]})
    _, reward, terminated, truncated, _ = env.step(1)
    assert reward == pytest.approx(0.99)  # +1 goal and -0.01 movement
    assert terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(1)
    env.reset(options={"start": 0})
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated and env.steps == 100
    with pytest.raises(ValueError):
        env.reset(options={"start": env.goal})
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_seeded_resets():
    a, b = NavigationWorld(), NavigationWorld()
    assert a.reset(seed=42) == b.reset(seed=42)
    assert [a.reset()[0] for _ in range(20)] == [b.reset()[0] for _ in range(20)]


@pytest.fixture(scope="module")
def trained():
    return train(1500, 42)


def test_training_reproducible_and_beats_random(trained):
    q, history = trained
    q2, history2 = train(1500, 42)
    np.testing.assert_array_equal(q, q2)
    assert history == history2
    before = q.copy()
    greedy = evaluate(100, 123, q)
    random = evaluate(100, 123)
    assert greedy["starts"] == random["starts"]
    assert greedy == evaluate(100, 123, q)
    assert random == evaluate(100, 123)
    np.testing.assert_array_equal(q, before)
    assert greedy["success_rate"] == 1.0
    assert greedy["success_rate"] > random["success_rate"]
    assert greedy["mean_steps"] < random["mean_steps"]
    assert greedy["mean_reward"] > random["mean_reward"]
    assert len(history) == 1500 and history[-1]["epsilon"] == pytest.approx(0.05)


def test_roundtrip_identity_versions_and_no_overwrite(tmp_path, trained):
    q, history = trained
    store = AgentStore(tmp_path, "Alph", create=True)
    first = store.save(q, history, 42)
    original = {p.name: p.read_bytes() for p in
                (store.path / "navigation" / first["version"]).iterdir()}
    second = store.save(q + 1, history, 43)
    reopened = AgentStore(tmp_path, "Alph", create=True)
    assert reopened.identity == store.identity
    assert first["version"] != second["version"]
    assert len(reopened.versions()) == 2
    loaded, metadata, loaded_history = reopened.load()
    np.testing.assert_array_equal(loaded, q + 1)
    assert metadata == second and loaded_history == history
    assert original == {p.name: p.read_bytes() for p in
                        (store.path / "navigation" / first["version"]).iterdir()}
    assert AgentStore(tmp_path, "Beta", create=True).identity != store.identity
    with pytest.raises(FileExistsError):
        publish(store.path / "navigation", first["version"],
                lambda p: (p / "other").write_text("must not replace"))


def test_failed_publication_not_visible(tmp_path):
    def fail(path):
        (path / "partial").write_text("partial")
        raise OSError("simulated interrupted save")
    with pytest.raises(OSError):
        publish(tmp_path, "version", fail)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows rename retry")
@pytest.mark.parametrize("failures", [1, 3])
def test_windows_publication_permission_errors(tmp_path, monkeypatch, failures):
    original = Path.rename
    calls = 0

    def rename(path, target):
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise PermissionError("simulated file lock")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    writer = lambda p: (p / "complete").write_text("complete")
    if failures == 3:
        with pytest.raises(PermissionError):
            publish(tmp_path, "version", writer)
        assert list(tmp_path.iterdir()) == []
    else:
        result = publish(tmp_path, "version", writer)
        assert (result / "complete").read_text() == "complete"
    assert calls == min(failures + 1, 3)


@pytest.mark.parametrize("corruption", ["checksum", "shape", "nan", "object", "contract"])
def test_corrupt_skills_rejected(tmp_path, trained, corruption):
    q, history = trained
    store = AgentStore(tmp_path, "Alph", create=True)
    metadata = store.save(q, history, 42)
    path = store.path / "navigation" / metadata["version"]
    if corruption == "contract":
        metadata["contract"] = {}
    elif corruption == "checksum":
        (path / "q.npz").write_bytes(b"broken")
    else:
        bad = {"shape": np.zeros((1, 4)), "nan": q * np.nan,
               "object": np.array([[object()]], dtype=object)}[corruption]
        np.savez(path / "q.npz", q=bad)
        metadata["q_sha256"] = hashlib.sha256((path / "q.npz").read_bytes()).hexdigest()
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_cli_train_evaluate_baseline_report(tmp_path, capsys):
    shared = ["--output-dir", str(tmp_path)]
    training = main(["train", "--agent", "Alph", "--episodes", "20", "--seed", "42", *shared])
    comparison = main(["evaluate", "--agent", "Alph", "--episodes", "10", "--seed", "123", *shared])
    baseline = main(["baseline", "--episodes", "10", "--seed", "123", *shared])
    assert comparison["random"]["mean_reward"] == baseline["mean_reward"]
    report = main(["report", "--agent", "Alph", *shared])
    assert report["skill"]["version"] == training["skill"]["version"]
    assert (Path(report["report_directory"]) / "training.png").stat().st_size > 0
    assert report["evaluation"]["trained"] == comparison["trained"]


def test_cli_invalid_and_missing_agent(tmp_path):
    with pytest.raises(SystemExit) as invalid:
        main(["train", "--agent", "Alph", "--episodes", "0"])
    assert invalid.value.code == 2
    with pytest.raises(SystemExit) as missing:
        main(["evaluate", "--agent", "Missing", "--output-dir", str(tmp_path)])
    assert missing.value.code == 1

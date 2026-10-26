"""M3 tests use temporary sources; never depend on the user's trained agent."""

from copy import deepcopy
import json

import numpy as np
import pytest

from agentverse.learning import evaluate, train
from agentverse.storage import AgentStore
from agentverse.transfer import (ARMS, CHECKPOINTS, EVAL_SEEDS, TRAIN_SEEDS, TransferWorld,
                                compatibility, curve_metrics, evaluate_target, hashes,
                                initial_q, run_experiment, train_arm)
from agentverse.world import CONTRACT, NavigationWorld


@pytest.fixture
def source(tmp_path):
    store = AgentStore(tmp_path / "source", "Temporary", create=True)
    q, history = train(100, 42)
    metadata = store.save(q, history, 42)
    cq, ch = train(100, 43, "resource_collection")
    store.save(cq, ch, 43, "resource_collection")
    return store, q, metadata


@pytest.mark.parametrize("goal", [(4, 3), (0, 4), (4, 4)])
def test_world_same_indices_transitions_and_reachability(goal):
    env, original = TransferWorld(goal), NavigationWorld()
    assert env.cells == original.cells and env.indices == original.indices
    # Exhaustive transition comparison except goal-dependent termination/reward.
    for start in env.starts:
        for action in range(4):
            env.reset(options={"start": start})
            r, c = env.cells[start]
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
            following = env.indices.get((r + dr, c + dc), start)
            state, reward, terminal, truncated, _ = env.step(action)
            assert state == following and terminal == (state == env.goal)
            assert reward == pytest.approx((1 if terminal else 0) - .01 - (.05 if state == start else 0))
            assert not truncated
        # BFS yields an actual route from each permitted start, executed via step.
        queue = [(start, [])]
        seen = {start}
        for state, route in queue:
            if state == env.goal:
                break
            r, c = env.cells[state]
            for action, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                nxt = env.indices.get((r + dr, c + dc))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, route + [action]))
        assert state == env.goal
        env.reset(options={"start": start})
        for action in route:
            _, _, terminal, _, _ = env.step(action)
        assert terminal
    assert CONTRACT["map"][-1] == "....G"
    assert env.render().count("G") <= 1


@pytest.mark.parametrize("change", ["map", "actions", "step_reward", "max_steps", "version", "skill_id", "indices"])
def test_reject_incompatible_contract_even_same_shape(source, change):
    _, q, metadata = source
    metadata = deepcopy(metadata)
    env = TransferWorld((4, 3))
    if change == "indices":
        env.indices = {cell: len(env.cells) - 1 - i for cell, i in env.indices.items()}
    elif change == "skill_id":
        metadata[change] = "resource_collection"
    else:
        metadata["contract"][change] = "incompatible"
    with pytest.raises(ValueError):
        compatibility(metadata, q, env)


def test_independent_copies_shuffle_distribution_and_budget(source):
    _, q, metadata = source
    before = q.copy()
    compatibility(metadata, q, TransferWorld((4, 3)))
    for arm in ARMS:
        initial, permutation = initial_q(q, arm, 0)
        assert not np.shares_memory(initial, q)
        if arm == "scratch":
            assert not initial.any()
        elif arm == "transfer":
            np.testing.assert_array_equal(initial, q)
        else:
            assert sorted(permutation) == list(range(len(q)))
            np.testing.assert_array_equal(initial, q[permutation])
            np.testing.assert_array_equal(np.sort(initial, axis=0), np.sort(q, axis=0))
        initial[:] = 999
    runs = [train_arm(q, (4, 3), arm, 5, 10005, (0, 25, 50)) for arm in ARMS]
    for run, models in runs:
        assert len(run["training"]) == 50
        assert [row["episode"] for row in run["checkpoints"]] == [0, 25, 50]
        assert all(len(row["records"]) == 100 for row in run["checkpoints"])
        assert list(models) == [0, 25, 50]
        assert run["train_starts"] == runs[0][0]["train_starts"]
        assert run["eval_starts"] == runs[0][0]["eval_starts"]
        assert [r["epsilon"] for r in run["training"]] == [r["epsilon"] for r in runs[0][0]["training"]]
    np.testing.assert_array_equal(q, before)
    control, control_models = train_arm(q, (4, 4), "transfer", 5, 10005, (0, 1))
    np.testing.assert_array_equal(control_models[0], q)
    assert control["checkpoints"][0]["records"] == evaluate_target(
        q, (4, 4), control["eval_starts"])["records"]
    run, models = train_arm(q, (4, 3), "transfer", 5, 10005, (0, 25, 50))
    assert run == runs[1][0]
    for checkpoint in models:
        np.testing.assert_array_equal(models[checkpoint], runs[1][1][checkpoint])
    assert set(TRAIN_SEEDS).isdisjoint(EVAL_SEEDS) and len(TRAIN_SEEDS) >= 10


def test_evaluation_read_only_and_no_fabricated_success():
    q = np.zeros((len(NavigationWorld().cells), 4))
    q.setflags(write=False)
    env = TransferWorld((4, 3))
    starts = list(env.starts)
    result = evaluate_target(q, (4, 3), starts)
    assert result == evaluate_target(q, (4, 3), starts)
    assert result["success_rate"] == 0 and result["mean_steps"] == 100
    metrics = curve_metrics([{"episode": ep, **result} for ep in CHECKPOINTS])
    assert metrics["first_observed_threshold_episode"] is None
    assert metrics["success_auc"] == 0
    assert not q.any()


def test_full_protocol_temporary_agent_preserves_old_skills(source, tmp_path):
    store, _, metadata = source
    before = hashes(store.path)
    performance = {skill: evaluate(20, 70000, store.load(skill)[0], skill)
                   for skill in ("navigation", "resource_collection")}
    output = run_experiment(tmp_path / "source", "Temporary", metadata["version"], tmp_path / "experiment")
    from pathlib import Path
    path = Path(output["directory"])
    results = json.loads((path / "results.json").read_text())
    assert hashes(store.path) == before
    assert results["preservation"]["unchanged"]
    for skill in performance:
        assert evaluate(20, 70000, store.load(skill)[0], skill) == performance[skill]
    assert [target["positive_control"] for target in results["targets"]] == [False, False, True]
    for target in results["targets"]:
        assert len(target["runs"]) == 30
        for run in target["runs"]:
            assert len(run["training"]) == 500
            assert [r["episode"] for r in run["checkpoints"]] == list(CHECKPOINTS)
            with np.load(path / run["model_file"], allow_pickle=False) as models:
                assert len(models.files) == len(CHECKPOINTS)
    manifest = json.loads((path / "sha256.json").read_text())
    assert manifest == {k: v for k, v in hashes(path).items() if k != "sha256.json"}
    assert len(list(path.rglob("*.png"))) == 3
    assert len(list(path.rglob("*.csv"))) == 12


def test_missing_or_corrupt_source_does_not_publish_success(source, tmp_path):
    destination = tmp_path / "experiment"
    with pytest.raises(FileNotFoundError):
        run_experiment(tmp_path / "missing", "Temporary", destination=destination)
    assert not destination.exists()
    store, _, metadata = source
    (store.path / "navigation" / metadata["version"] / "q.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(tmp_path / "source", "Temporary", destination=destination)
    assert not destination.exists()

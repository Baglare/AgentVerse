"""Synthetic mechanism checks, independent of Alph's experimental outcomes."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from agentverse.cross_skill_transfer import (
    ARMS, CHECKPOINTS, CrossSkillAdapter, aggregate, curve_metrics, evaluate_target,
    checkpoint_summary, legacy_hashes, main, run_experiment, source_effect_check, source_variant, train_arm,
)
from agentverse.collection import CONTRACT as TARGET, ResourceCollectionWorld
from agentverse.storage import AgentStore
from agentverse.transfer import hashes
from agentverse.world import CONTRACT, NavigationWorld


@pytest.fixture
def synthetic():
    q = np.arange(84, dtype=float).reshape(21, 4) / 100
    q[NavigationWorld().goal] = 0
    return q, {"skill_id": "navigation", "contract": deepcopy(CONTRACT)}


@pytest.mark.parametrize("side,key", [("source", "map"), ("source", "actions"),
                                       ("source", "max_steps"), ("source", "version"),
                                       ("target", "map"), ("target", "actions"),
                                       ("target", "observation"), ("target", "collection")])
def test_contracts_rejected(synthetic, side, key):
    q, meta = synthetic
    target = deepcopy(TARGET)
    (meta["contract"] if side == "source" else target)[key] = "invalid"
    with pytest.raises(ValueError, match="contract"):
        CrossSkillAdapter(q, meta, target)


@pytest.mark.parametrize("bad", [np.zeros((120, 4)), np.zeros((21, 4), dtype=np.float32),
                                 np.full((21, 4), np.nan)])
def test_bad_source_q(synthetic, bad):
    with pytest.raises(ValueError):
        CrossSkillAdapter(bad, synthetic[1])


def test_exact_mapping_centering_and_source_intervention(synthetic):
    source, meta = synthetic
    adapter = CrossSkillAdapter(source, meta)
    prior, report = adapter.initialize()
    assert prior.shape == (120, 4) and source.shape == (21, 4)
    assert 0 < report["coverage_all"] < 1
    env, nav = ResourceCollectionWorld(), NavigationWorld()
    mapped = [r for r in report["mapping"] if r["reason"] == "mapped"]
    assert mapped
    for row in report["mapping"]:
        state, s = row["target_state"], row["source_state"]
        if s is None:
            assert not prior[state].any()
        else:
            pos = state % len(env.cells)
            goal = env.cells[row["subgoal_position"]]
            cell = env.cells[pos]
            assert (goal[0] - cell[0], goal[1] - cell[1]) == (4 - nav.cells[s][0], 4 - nav.cells[s][1])
            np.testing.assert_allclose(prior[state], source[s] - source[s].mean())
    chosen = mapped[0]
    intervention = source.copy()
    intervention[chosen["source_state"], 2] += 2
    changed, _ = CrossSkillAdapter(intervention, meta).initialize()
    expected = np.array([-.5, -.5, 1.5, -.5])
    np.testing.assert_allclose(changed[chosen["target_state"]] - prior[chosen["target_state"]], expected)
    # Adapter snapshot cannot be changed through the caller's source array.
    source[:] = 999
    np.testing.assert_array_equal(adapter.initialize()[0], prior)
    assert not adapter.source.flags.writeable
    reasons = report["counts"]
    assert all(reasons[r] > 0 for r in ("relative_vector_unavailable", "local_walls_mismatch", "terminal", "inconsistent_inventory"))
    with pytest.raises(ValueError):
        adapter.mapping(120)


def test_real_source_intervention_report_and_neutral_signal(synthetic):
    source, meta = synthetic
    before = source.copy()
    result = source_effect_check(source, meta)
    assert result["status"] == "PASS" and result["source_intervention_verified"]
    np.testing.assert_allclose(result["observed_prior_delta"], result["expected_prior_delta"])
    np.testing.assert_array_equal(source, before)
    zero, report = CrossSkillAdapter(np.zeros_like(source), meta).initialize()
    assert not zero.any() and report["nonzero_prior_states"] == 0


def test_adapter_never_steps_target_or_accesses_evaluation(synthetic, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Adapter accessed dynamics/evaluation")
    monkeypatch.setattr(ResourceCollectionWorld, "step", forbidden)
    monkeypatch.setattr(ResourceCollectionWorld, "reset", forbidden)
    monkeypatch.setattr("agentverse.cross_skill_transfer.evaluate_target", forbidden)
    prior, _ = CrossSkillAdapter(*synthetic).initialize()
    assert prior.any()


def test_controls_distribution_isolation(synthetic):
    source, meta = synthetic
    before = source.copy()
    shuffled, permutation = source_variant(source, "shuffled_prior", 7)
    indices = list(NavigationWorld().starts)
    np.testing.assert_array_equal(np.sort(shuffled[indices].ravel()), np.sort(source[indices].ravel()))
    assert permutation != list(range(80))
    assert not np.array_equal(shuffled, source)
    zeros, _ = source_variant(source, "untrained_source", 7)
    assert not CrossSkillAdapter(zeros, meta).initialize()[0].any()
    for arm in ARMS:
        variant, _ = source_variant(source, arm, 7)
        variant[:] = 666
    np.testing.assert_array_equal(source, before)
    with pytest.raises(ValueError):
        source_variant(source, "invalid", 0)


def test_budget_reproducible_zero_checkpoint_and_readonly_eval(synthetic):
    source, meta = synthetic
    before = source.copy()
    checkpoints = (0, 25, 50, 200)
    outputs = [train_arm(source, meta, arm, 3, checkpoints) for arm in ARMS]
    for (run, models), arm in zip(outputs, ARMS):
        assert run["actual_training_steps"] == 200
        assert [r["training_steps"] for r in run["checkpoints"]] == list(checkpoints)
        assert sum(r["steps"] for r in run["training"]) + run["partial_episode"]["steps"] == 200
        assert run["checkpoints"][0]["completed_episodes"] == 0
        frozen = models[0].copy()
        frozen.setflags(write=False)
        assert run["checkpoints"][0] == {"training_steps": 0, "completed_episodes": 0, **evaluate_target(frozen)}
        again, again_models = train_arm(source, meta, arm, 3, checkpoints)
        assert again == run
        for step in models:
            np.testing.assert_array_equal(models[step], again_models[step])
        assert not np.shares_memory(models[0], source)
    np.testing.assert_array_equal(outputs[0][1][0], outputs[3][1][0])
    assert outputs[0][0]["checkpoints"] == outputs[3][0]["checkpoints"]
    np.testing.assert_array_equal(source, before)
    assert CHECKPOINTS[0] == 0 and CHECKPOINTS[-1] == 30000
    summary = checkpoint_summary([run for run, _ in outputs])
    assert len(summary) == 16
    assert summary[0]["successful_steps_mean"] is None
    assert summary[0]["successes"] == 0 and summary[0]["failures"] == 1


def test_weighted_auc_and_honest_negative_result():
    rows = [{"training_steps": x, "success_rate": y, "mean_reward": -2*y}
            for x, y in [(0, 0), (25, 1), (100, 0)]]
    metrics = curve_metrics(rows)
    assert metrics["success_auc"] == 50 and metrics["reward_auc"] == -100
    assert metrics["success_auc_normalized"] == .5
    runs = [{"arm": arm, "training_seed": seed,
             "success_auc": -1 if arm == "transfer" else 5,
             "reward_auc": -10 if arm == "transfer" else 2,
             "success_auc_normalized": 0, "reward_auc_normalized": 0}
            for seed in range(10) for arm in ARMS]
    _, differences, comparisons = aggregate(runs)
    assert len(differences) == 60
    assert all(c["direction_of_mean"] == "negative" for c in comparisons)
    assert comparisons[0]["mean"] == -6
    assert comparisons[0]["descriptive_95ci"] == [-6, -6]
    q = np.zeros((120, 4))
    q.setflags(write=False)
    result = evaluate_target(q)
    assert result["success_rate"] == 0 and result["successful_steps"] is None
    assert result["failures"] == result["timeouts"] == result["evaluation_rollouts"] == 1
    with pytest.raises(ValueError):
        curve_metrics(rows[::-1])


@pytest.fixture
def saved(synthetic, tmp_path):
    store = AgentStore(tmp_path / "source", "Temporary", create=True)
    q, _ = synthetic
    history = [{"steps": 1, "success": False, "reward": -.01}]
    metadata = store.save(q, history, 7)
    store.save(np.zeros((120, 4)), history, 8, "resource_collection")
    return store, metadata


def test_experiment_artifacts_preservation_and_cli(saved, tmp_path, monkeypatch):
    store, metadata = saved
    root = tmp_path / "source"
    (root / "m3-transfer").mkdir()
    (root / "m3-transfer" / "old-results.json").write_text('{"historical": true}')
    before = legacy_hashes(root)
    output = run_experiment(root, "Temporary", metadata["version"], tmp_path / "experiment",
                            checkpoints=(0, 25, 100), seeds=(0, 1))
    path = Path(output["directory"])
    result = json.loads((path / "results.json").read_text())
    assert legacy_hashes(root) == before
    integrity = json.loads((path / "integrity_report.json").read_text())
    assert integrity["unchanged"] and integrity["evaluations_unchanged"]
    assert integrity["identity"] == store.identity
    assert len(result["runs"]) == 8
    assert result["mechanism_checks"]["scratch_equals_untrained_all_seeds"]
    assert result["mechanism_checks"]["trained_prior_differs_from_untrained"]
    assert result["source"]["metadata"]["version"] == metadata["version"]
    for name in ("results.json", "summary.csv", "learning_curves.png", "transfer_comparison.png",
                 "source_manifest.json", "compatibility_report.json", "integrity_report.json"):
        assert (path / name).stat().st_size > 0
    for run in result["runs"]:
        with np.load(path / run["model_file"], allow_pickle=False) as data:
            assert len(data.files) == 3 and data["step_0"].shape == (120, 4)
    manifest = json.loads((path / "sha256.json").read_text())
    assert manifest == {k: v for k, v in hashes(path).items() if k != "sha256.json"}
    captured = []
    monkeypatch.setattr("agentverse.cross_skill_transfer.run_experiment", lambda *args: captured.append(args) or {})
    main(["--source-dir", str(root), "--agent", "Temporary", "--version", metadata["version"],
          "--output-dir", str(tmp_path / "cli")])
    assert captured[0] == (root, "Temporary", metadata["version"], tmp_path / "cli")


@pytest.mark.parametrize("fault", ["missing", "corrupt", "unknown", "contract", "history", "overlap"])
def test_bad_sources_no_success_publication(saved, tmp_path, fault):
    store, metadata = saved
    root, dest, version = tmp_path / "source", tmp_path / "experiment", metadata["version"]
    source_path = store.path / "navigation" / version
    if fault == "missing":
        root = tmp_path / "missing"
    elif fault == "corrupt":
        (source_path / "q.npz").write_bytes(b"bad")
    elif fault == "unknown":
        version = "unknown"
    elif fault == "contract":
        metadata = deepcopy(metadata)
        metadata["contract"]["actions"] = ["left", "right", "up", "down"]
        (source_path / "metadata.json").write_text(json.dumps(metadata))
    elif fault == "history":
        (source_path / "training.json").write_text("[]")
    else:
        dest = store.path / "experiment"
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(root, "Temporary", version, dest, checkpoints=(0, 1), seeds=(0,))
    assert not dest.exists()

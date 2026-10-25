from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.registry import backend_registry
import numpy as np
import pytest

from agentverse.cli import main
from agentverse.learning import evaluate
from agentverse.replay import ReplaySession, ReplayViewer, replay
from agentverse.storage import AgentStore
from agentverse.world import NavigationWorld


def trajectory(session):
    records = [(session.state, None, 0.0)]
    while session.step():
        records.append((session.state, session.last_action, session.total_reward))
    return records


@pytest.mark.parametrize("random", [False, True])
def test_seeded_replay_matches_environment_and_evaluation(random):
    env = NavigationWorld()
    q = None if random else np.zeros((env.observation_space.n, 4))
    session = ReplaySession(123, q)
    env.reset(options={"start": session.state})
    rng = np.random.default_rng(np.random.SeedSequence([123, 2]))
    before = None if q is None else q.copy()
    while not env.done:
        action = int(rng.integers(4)) if random else int(q[env.state].argmax())
        state, reward, terminated, truncated, info = env.step(action)
        previous_reward = session.total_reward
        assert session.step()
        assert session.last_action == action
        assert session.state == state
        assert session.total_reward == pytest.approx(previous_reward + reward)
        assert session.terminated == terminated and session.truncated == truncated
        assert session.collision == info["collision"]
    steps = session.env.steps
    assert not session.step() and session.env.steps == steps
    reference = evaluate(1, 123, q)
    assert reference["starts"][0] == ReplaySession(123, q).state
    assert reference["mean_steps"] == steps
    assert reference["mean_reward"] == session.total_reward
    assert reference["success_rate"] == float(session.terminated)
    if q is not None:
        np.testing.assert_array_equal(q, before)


def test_random_reset_repeats_full_trajectory():
    session = ReplaySession(123)
    first = trajectory(session)
    session.reset()
    assert session.env.steps == 0 and not session.env.done
    assert session.last_action is None and session.total_reward == 0
    assert trajectory(session) == first == trajectory(ReplaySession(123))


def test_viewer_controls_and_actual_marker_position():
    session = ReplaySession(123, np.zeros((NavigationWorld().observation_space.n, 4)))
    viewer = ReplayViewer(session, "Alph")
    viewer.timer = Mock()
    try:
        viewer.tick()
        assert session.env.steps == 0
        viewer.toggle()
        assert viewer.playing
        viewer.timer.start.assert_called_once()
        viewer.tick()
        assert session.env.steps == 1
        viewer.toggle()
        viewer.tick()
        assert not viewer.playing and session.env.steps == 1
        viewer.advance()
        assert session.env.steps == 2
        r, c = session.env.cells[session.state]
        np.testing.assert_array_equal(viewer.marker.get_offsets(), [[c, r]])
        viewer.reset()
        assert not viewer.playing and session.env.steps == 0
        viewer.toggle()
        for _ in range(100):
            viewer.tick()
        assert session.truncated and not viewer.playing
        viewer.toggle()
        assert not viewer.playing
        viewer.close()
        viewer.timer.stop.assert_called()
    finally:
        plt.close(viewer.figure)


def test_saved_q_loaded_without_training_or_writes(tmp_path, monkeypatch):
    store = AgentStore(tmp_path, "Alph", create=True)
    q = np.arange(NavigationWorld().observation_space.n * 4, dtype=np.float64).reshape(-1, 4)
    metadata = store.save(q, [{"steps": 1, "reward": 0.99, "success": True}], 42)
    snapshot = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(backend_registry, "resolve_backend", lambda _: ("tkagg", "tk"))
    monkeypatch.setattr("agentverse.learning.train", Mock(side_effect=AssertionError("No training")))

    def show(viewer):
        np.testing.assert_array_equal(viewer.session.q, q)
        expected = int(q[viewer.session.state].argmax())
        viewer.advance()
        assert viewer.session.last_action == expected
        viewer.close()
        plt.close(viewer.figure)

    monkeypatch.setattr(ReplayViewer, "show", show)
    result = replay(tmp_path, "Alph", "trained", 123, 300)
    assert result["version"] == metadata["version"] and result["steps"] == 1
    assert snapshot == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_random_needs_no_saved_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_registry, "resolve_backend", lambda _: ("tkagg", "tk"))
    def show(viewer):
        assert viewer.session.q is None
        viewer.close()
        plt.close(viewer.figure)
    monkeypatch.setattr(ReplayViewer, "show", show)
    assert replay(tmp_path, "Missing", "random", 123, 300)["version"] is None
    assert list(tmp_path.iterdir()) == []


def test_headless_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match="interactive"):
        replay(tmp_path, "Alph", "trained", 123, 300)


def test_cli_dispatch_and_validation(tmp_path, monkeypatch):
    runner = Mock(return_value={"mode": "trained"})
    monkeypatch.setattr("agentverse.replay.replay", runner)
    assert main(["replay", "--agent", "Alph", "--seed", "7", "--output-dir", str(tmp_path)]) == {"mode": "trained"}
    runner.assert_called_once_with(tmp_path, "Alph", "trained", 7, 300)
    for args in (["--seed", "-1"], ["--interval-ms", "0"], ["--mode", "unknown"]):
        with pytest.raises(SystemExit) as error:
            main(["replay", *args])
        assert error.value.code == 2


def test_missing_model_cli_error(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_registry, "resolve_backend", lambda _: ("tkagg", "tk"))
    with pytest.raises(SystemExit) as error:
        main(["replay", "--output-dir", str(tmp_path)])
    assert error.value.code == 1

# AgentVerse

**An experimental, local-first toolkit for training persistent agents in simulated environments.**

> **Working title · Early prototype (M1 / v0.1.0)**
>
> The long-term idea is simple: create an agent, train it, keep its identity and learned skills, and eventually test it in different community-built environments. **Today, this repository implements only the first, deliberately small step.**

AgentVerse explores a question: *Can independently trained agents preserve skills and eventually reuse what they learn across tasks—without requiring an LLM, a paid API, or a central inference server?*

The first prototype uses tabular Q-learning in a fixed 2D navigation world. It is a reproducible learning experiment, **not** a general-purpose intelligent agent or an existing multi-world platform.

## Current status

| Capability | M1 status |
| --- | --- |
| Named agent with a persistent identity | Implemented |
| Fixed 5×5 Gymnasium navigation environment | Implemented |
| Seeded tabular Q-learning | Implemented |
| Random-policy baseline and paired evaluation | Implemented |
| Versioned, checksummed navigation skill snapshots | Implemented |
| Training metrics and PNG learning curves | Implemented |
| Interactive replay of a saved policy or random agent | Implemented |
| Multiple independent skills per agent | **Planned (M2)** |
| Continuing training from an existing skill | **Not implemented** |
| Skill transfer to a new environment | **Research goal (M3)** |
| Multiple competing agents and community worlds | **Future work** |

**An important distinction:** M1 can *save, load, and retain multiple versions* of a navigation skill. Calling `train` again currently starts a **new Q-table from scratch** and saves a new version; it does not continue training the previous one. Cross-task transfer and continual learning have **not** been demonstrated.

## Why build this?

Most game AI either follows hand-authored behavior or uses a model trained for one specific environment. AgentVerse aims to experiment with a third direction: **named, inspectable, locally trained agents** whose progress is stored independently of a particular run.

The intended future is a small open platform with:

- **AgentCore:** identity, learned policies, skill storage, and evaluation.
- **Training Grounds:** reproducible environments in which agents can learn.
- **Arena:** controlled comparisons between differently trained agents.
- **Community environments:** additional tasks connected through a defined interface.
- **Adaptive Boss Lab:** a possible application of the same learning infrastructure.

These are **goals**, not features of the current release. In particular, sharing an environment API does not magically make a skill transferable to a new task.

## Quick start

**Requirements:** Python 3.12. No GPU, account, API key, or hosted server is required for M1.

### Windows (PowerShell)

~~~powershell
git clone https://github.com/Baglare/AgentVerse.git
cd AgentVerse

py -3.12 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"

& $PY -m pip install -e ".[test]"
& $PY -m pytest -q
~~~

Train a named agent, evaluate it, compare it to a random policy, and generate a report:

~~~powershell
& $PY -m agentverse train --agent Alph --episodes 10000 --seed 42
& $PY -m agentverse evaluate --agent Alph --episodes 100 --seed 123
& $PY -m agentverse baseline --episodes 100 --seed 123
& $PY -m agentverse report --agent Alph
~~~

To view actual decisions from the stored Q-table:

~~~powershell
$env:MPLBACKEND = "TkAgg"
& $PY -m agentverse replay --agent Alph --seed 123
~~~

For the random policy with the same starting state:

~~~powershell
& $PY -m agentverse replay --mode random --seed 123
~~~

The replay window has **Step**, **Play/Pause**, and **Reset** controls. Reset replays the same seed. Close the window to return to the terminal. An interactive Matplotlib backend (such as TkAgg, with Tk/Tkinter available) is required; a headless `Agg` backend cannot display the window.

### Linux / macOS

~~~bash
git clone https://github.com/Baglare/AgentVerse.git
cd AgentVerse

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
python -m agentverse train --agent Alph --episodes 10000 --seed 42
python -m agentverse evaluate --agent Alph --episodes 100 --seed 123
python -m agentverse report --agent Alph
~~~

For replay, use an available interactive Matplotlib backend. GUI support depends on the local Python and windowing setup.

## The M1 environment

The entire current learning task is this fixed map:

~~~text
.....
.##..
...#.
.#...
....G
~~~

- `#` = wall; `G` = goal; `.` = traversable cell.
- An episode begins at a seeded, randomly selected non-goal traversable cell.
- Observations are **discrete position indices**; actions are **up, right, down, left**.
- The goal yields `+1.0`, each step costs `-0.01`, and hitting a wall incurs an extra `-0.05`.
- Episodes stop at the goal or after **100 steps**.
- The training policy is epsilon-greedy tabular Q-learning; evaluation uses greedy `argmax` decisions.
- This is a **fixed, fully observable grid**, not an unseen-map navigation benchmark.

The environment follows the Gymnasium `reset` / `step` API. Future environments may have different observations and actions; compatibility and skill reuse will need to be designed and tested explicitly.

## Reference M1 result

The maintainer's local run used **10,000 training episodes (seed 42)** and **100 evaluation episodes (seed 123)** on the fixed M1 map:

| Metric | Trained Alph | Random policy |
| --- | ---: | ---: |
| Success rate | **100%** | 71% |
| Mean steps | **4.05** | 52.10 |
| Mean reward | **0.9595** | −0.9050 |

The policies were evaluated on the **same seeded starting positions**. Mean steps include all episodes, including those that timed out at 100 steps.

These are results for **one small fixed environment and one reported run**. They do not establish performance on unseen maps, statistically robust cross-seed results, continual learning, or general intelligence. Re-run the commands above to generate local evidence.

Running `report` writes a JSON summary and a `training.png` figure containing rolling success, step, and reward curves. Running `replay` uses the saved policy and the real environment transitions; it does **not** re-train or alter the model.

## Local storage and reproducibility

By default, generated data goes under **`outputs/`**, which is Git-ignored:

~~~text
outputs/
  agents/
    <hash-of-agent-name>/
      agent.json
      navigation/
        <skill-version-uuid>/
          q.npz
          metadata.json
          training.json
      evaluations/
      reports/
  baselines/
~~~

Each agent has a persistent UUID. Saved navigation versions are written to separate directories with a contract and SHA-256 checksum. Q-tables are stored as NumPy NPZ files and loaded with `allow_pickle=False`. Replay and evaluation can load the latest saved version without training again.

These checks help detect incompatible or altered artifacts; they are **not** a guarantee that arbitrary untrusted files are safe. Do not run untrusted code or assume that a community-contributed environment is sandboxed.

CLI commands accept `--output-dir` if you want to keep experiment outputs elsewhere. Keep seeds, parameters, environment contract, and model version when comparing results.

## Repository layout

~~~text
src/agentverse/
  world.py       # Fixed Gymnasium navigation environment
  learning.py    # Q-learning and evaluation
  storage.py     # Agent identity, snapshots, artifact integrity
  cli.py         # train / evaluate / baseline / report / replay
  replay.py      # Read-only interactive policy viewer
  __main__.py    # python -m agentverse entry point
tests/
  test_m1.py
  test_replay.py
~~~

Run all tests with:

~~~bash
python -m pytest -q
~~~

The test suite covers environment behavior, seeded training, paired evaluation, snapshot loading, invalid artifacts, CLI behavior, and replay logic. GUI callbacks are exercised in tests; visual presentation still needs manual verification on a desktop.

## Roadmap

- [x] **M1 — Learning proof:** a seeded navigation task, persistent snapshots, evaluation, and replay.
- [ ] **M2 — Multi-skill continuity:** add a genuinely distinct second task and preserve both skills. Measure whether old performance survives; merely storing a second file is not enough.
- [ ] **M3 — Transfer experiments:** evaluate whether prior learning provides a measurable advantage on related *but changed* tasks, with from-scratch controls.
- [ ] **M4 — Alph vs. Kelt:** train separate agents and compare behavior and performance under shared evaluation rules.
- [ ] **Later — Creator tooling:** portable environment contracts, an approachable training interface, and community-authored tasks.
- [ ] **Later — Multiplayer or hosted arenas:** only after fairness, compatibility, and compute costs are understood.

Roadmap items describe intended investigation, **not guaranteed outcomes**.

## Contributing

Contributions and critical feedback are welcome, especially in:

- Reproducibility, evaluation design, and tests that detect misleading results.
- CPU-efficient learning and skill persistence.
- Environment contracts and safe, portable agent artifacts.
- Small, independently runnable experiments in learning and transfer.

Please open an issue to discuss a substantial change before implementing it. For pull requests, keep changes scoped, add tests where appropriate, and describe the exact commands and results used to validate them. Avoid adding required cloud services or LLM dependencies to the core.

## License and naming

Released under the [MIT License](LICENSE).

**AgentVerse is a working project name**, and the Python distribution is **not presented as an official PyPI release**. The project may be renamed before a wider public launch to avoid confusion with existing projects using a similar name.

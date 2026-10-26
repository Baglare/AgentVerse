# AgentVerse

**An experimental, local-first toolkit for training persistent agents in simulated environments.**

> **Working title · Early prototype (M2 + experimental M3-A/M3-B / v0.1.0)**
>
> The long-term idea is simple: create an agent, train it, keep its identity and learned skills, and eventually test it in different community-built environments. **Today, this repository implements two retained skills, controlled goal transfer, and a Navigation-to-Collection transfer experiment.**

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
| Multiple independent skills per agent | Implemented (M2: Navigation + Resource Collection) |
| Continuing training from an existing skill | **Not implemented** |
| Goal relocation transfer on the Navigation map | **Experimental M3-A; positive and negative transfer observed** |
| Navigation-to-Collection transfer adapter and controls | **M3-B implemented and evaluated; useful learned transfer not established** |
| Multiple competing agents and community worlds | **Future work** |

**An important distinction:** M1 can *save, load, and retain multiple versions* of a navigation skill. Calling `train` again currently starts a **new Q-table from scratch** and saves a new version; it does not continue training the previous one. The separate M3-A command tests goal relocation; ordinary `train` still starts from scratch. General cross-skill transfer and continual learning have **not** been demonstrated.

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
- [x] **M2 — Multi-skill continuity:** separate Navigation and Resource Collection tables; retained identity, checksums, and old-skill evaluation covered by tests.
- [ ] **M3 — Transfer experiments:** M3-A and M3-B implemented and evaluated. M3-B has no demonstrated learned-transfer benefit; M3 remains experimental.
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


## M3-A — Controlled Knowledge Transfer (experimental)

Run the fixed local CPU experiment using an existing saved Navigation model:

~~~powershell
Set-Location C:\AgentVerse\AgentVerse
$PY = ".\.venv\Scripts\python.exe"
& $PY -m agentverse.transfer --agent Alph --source-dir outputs --version bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81 --output-dir outputs/m3-transfer
# Omit --version to use the latest saved Navigation version (recorded in results).
$testRun = Join-Path (Get-Location) ("outputs\tmp\m3-tests-" + [guid]::NewGuid())
& $PY -m pytest -q --basetemp $testRun
~~~

No source training or source writes occur. A missing/incompatible source fails without a result.
`python -m agentverse.transfer --help` describes source/version/output selection. Existing
`agentverse train/evaluate/baseline/report/replay` commands retain their behavior.

**Method fixed before evaluation:** unchanged M1 walls, row-major traversable position
indices and up/right/down/left actions; goals (row, column) (4,3) and (0,4), displaced
(0,-1) and (-4,0) from (4,4). Every non-goal start is checked for reachability.
The separate environment inherits M1 transitions and rewards; M1's contract is unchanged.
Full legacy source contract equality (including map, actions, rewards, time limit and
version), explicit cell/index correspondence, and float64 finite Q-table validation
are required. Shape equality alone is insufficient.

Scratch uses zeros; Transfer copies the source; Shuffled Prior permutes entire source
rows, preserving each action column's values and within-row action preferences but
breaking spatial association. Ten independent target training seeds 0..9 are paired
across arms. All use 500 episodes, at most 100 steps per episode, alpha=.1, gamma=.95,
and epsilon=1-.95*min(episode/400,1), with zero-based episode number. Episode budgets
are equal; actual transition counts can differ because successful episodes end early.
The source's 10,000 episodes are prior learning, not charged to target learning budgets.

NumPy PCG64 streams use SeedSequence([training_seed,101]) for explicit shared starts,
[training_seed,202,episode] for 100 exploration uniforms followed by 100 random actions,
and [training_seed,303] for state permutations. Per-step action draws remain paired even
when trajectories diverge. Evaluation seeds 10000..10009 use [evaluation_seed,404] for
100 explicit starts per seed, shared across arms and checkpoints 0/25/50/100/200/500.
Evaluation is greedy, ties choose the first argmax, and never updates Q. Full start lists,
permutations, episode records, runtime versions and source metadata are in JSON.

Means include failed episodes and 100-step timeouts. AUC uses trapezoids over the six
checkpoint measurements (raw episode units, range 0..500); success threshold is .90.
Time-to-threshold is the first observed qualifying checkpoint, not an interpolated exact
crossing or a claim of sustained success; null means never reached. Curves show every
seed plus mean and sample SD. Paired Transfer-minus-Scratch and Transfer-minus-Shuffled
checkpoint/AUC differences are exported. No hyperparameters were tuned to these outcomes.

**Observed Alph run:** Navigation source `bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81`,
source seed 42, 10,000 episodes. Output run: `outputs/m3-transfer/d526f923-b64b-45d0-8662-f063236c0554`.
All 10 target seeds are included below; AUC values are rounded only for display.

| Goal | Arm | Success AUC mean ± sample SD | Reward AUC mean ± sample SD | First .90 checkpoint median [min,max] |
| --- | --- | ---: | ---: | ---: |
| (4, 3) | scratch | 485.46 ± 3.77 | 389.27 ± 6.91 | 25 [25,50] |
| (4, 3) | transfer | 489.95 ± 0.48 | 459.86 ± 1.84 | 25 [25,25] |
| (4, 3) | shuffled_prior | 458.00 ± 13.76 | 273.34 ± 54.10 | 100 [50,200] |
| (0, 4) | scratch | 487.73 ± 2.71 | 403.37 ± 5.96 | 25 [25,50] |
| (0, 4) | transfer | 470.90 ± 6.95 | 420.59 ± 14.42 | 50 [50,100] |
| (0, 4) | shuffled_prior | 461.99 ± 6.73 | 326.60 ± 39.98 | 100 [50,100] |
| (4, 4) control | scratch | 486.07 ± 2.31 | 390.10 ± 3.97 | 25 [25,50] |
| (4, 4) control | transfer | 500.00 ± 0.00 | 479.17 ± 1.43 | 0 [0,0] |
| (4, 4) control | shuffled_prior | 468.34 ± 7.74 | 338.05 ± 29.12 | 75 [50,100] |

| Training seed | (4,3) Scratch AUC | Transfer AUC | Shuffled AUC | (0,4) Scratch AUC | Transfer AUC | Shuffled AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 487.50 | 490.25 | 456.00 | 487.88 | 474.88 | 455.25 |
| 1 | 479.88 | 490.62 | 466.00 | 483.12 | 474.25 | 458.38 |
| 2 | 487.50 | 489.62 | 447.88 | 483.00 | 473.38 | 459.00 |
| 3 | 487.50 | 489.75 | 468.00 | 486.38 | 475.88 | 467.50 |
| 4 | 487.50 | 489.50 | 469.38 | 489.00 | 456.88 | 454.00 |
| 5 | 485.00 | 489.50 | 470.88 | 490.50 | 466.75 | 455.75 |
| 6 | 487.50 | 490.75 | 468.50 | 488.88 | 466.75 | 466.62 |
| 7 | 477.25 | 489.75 | 455.88 | 489.50 | 464.62 | 461.25 |
| 8 | 487.50 | 490.25 | 426.88 | 488.88 | 480.12 | 474.25 |
| 9 | 487.50 | 489.50 | 450.62 | 490.12 | 475.50 | 467.88 |

Goal (4, 3): paired success-AUC difference +4.49 ± 3.86 (sample SD); all ten differences positive.

Goal (0, 4): paired success-AUC difference -16.82 ± 8.26 (sample SD); all ten differences negative.

Transfer shows a small positive success-AUC effect at (4,3), but negative transfer at
(0,4), where scratch learns faster. All arms reach 100% measured success by checkpoint
200 in these runs. Same-goal (4,4) Transfer starts at 100% success for all seeds; this is
only a mechanism positive control and is excluded from new-goal transfer claims.
Reward-AUC and success-AUC need not rank arms identically: a looping prior can incur
fewer wall penalties while still failing to reach the goal.

The full `results.json` contains every evaluation record, training history, start,
source identity/version and before/after source-file SHA-256 values. Each goal directory
contains `checkpoints.csv`, `summary.csv`, `seed-metrics.csv`, `paired-differences.csv`,
`curves.png`, and separate per-arm/per-seed NPZ checkpoint models. `protocol.json` is
written before running the experiment. `sha256.json` covers all experiment artifacts
except itself. The existing atomic publication helper exposes a run only after successful
completion; failures do not publish a success report. Outputs remain Git-ignored.

For this run, **every Alph file hash and identity stayed unchanged**. The latest original
skills evaluated before and after at independent seed 90000 (100 episodes each) were
identical: Navigation success 1.0, steps 4.15, reward .9585; Resource Collection success
1.0, steps 16, reward 1.24. M1/M2 saved versions were not migrated or overwritten.
Tests use only temporary agents, including a real full-protocol test, and cover source
copy isolation, equal budgets, reproducibility, incompatible contracts, read-only
assessment, unreachable-success reporting, missing/corrupt source rejection, and retained
old-skill behavior. Existing CLI/replay tests run headlessly; no live GUI was opened.

**Limits:** one frozen source model, ten target seeds, two hand-selected goal relocations,
a tiny deterministic map, coarse checkpoints, and 100 sampled evaluation starts with
replacement. Target seeds do not represent independent source training runs. SD and
paired signs are descriptive; no population significance or generalization claim is made.
There is no evidence here of general intelligence, unseen-map generalization, or
Navigation-to-Collection transfer. M3 remains experimental and incomplete.

### Historical M3-B compatibility assessment (before implementation)

Navigation has 21 position states on a 5×5 map; Collection has 30 traversable positions
on a different 6×6 map and 4 inventory masks, hence 120 states. Both have four cardinal
actions, but matching action names cannot align different wall geometry, position indices,
inventory state, base-only starts, resource rewards, or the collect-both-and-return
terminal condition. Direct Q-table copying or row tiling would lack a justified semantic
mapping. This motivated the separate tested adapter and controls below; the historical
M3-A implementation and results remain unchanged.

## M3-B — Cross-Skill Knowledge Transfer

The CPU-only experiment initializes a new Collection learner from frozen Alph
Navigation action advantages. Exact relative subgoal displacement **and** the four
immediate wall indicators must match; otherwise the prior is zero. Subgoals follow
a fixed resource order, then base. There is no shortest-path solver or new expert
training in the adapter. Coverage is only **9/120 states (7.5%)**, or 9/115 eligible
states (7.83%); remote topology equivalence is unverified.

```powershell
Set-Location C:\AgentVerse\AgentVerse
$PY = '.\.venv\Scripts\python.exe'
& $PY -m agentverse.cross_skill_transfer --agent Alph --source-dir outputs --version bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81 --output-dir outputs/m3-cross-skill
```

**Actual run:** `outputs/m3-cross-skill/c6ba838c-9540-4c65-a453-ee4e1190a71a`.
Ten paired target seeds 0..9, **30,000 actual training transitions per arm**,
1.2 million total transitions, approximately 10.53 seconds locally (including
reporting and preservation checks). Checkpoints use transitions rather than episodes
to make actual training-step budgets equal. Each checkpoint evaluates one deterministic
base-start rollout; uncertainty below is across independently trained target seeds.

| Arm | Success AUC mean ± sample SD | Reward AUC mean ± sample SD |
| --- | ---: | ---: |
| A Scratch | 22,900 ± 1,264.91 | 16,378.50 ± 4,814.33 |
| B Cross-Skill Transfer | 22,500 ± 0 | 18,614.37 ± 6,964.99 |
| C Shuffled Prior | 22,900 ± 1,264.91 | 16,710.38 ± 6,765.34 |
| D Untrained Source | 22,900 ± 1,264.91 | 16,378.50 ± 4,814.33 |

AUC uses weighted checkpoint trapezoids over 0..30,000 transitions. B−A, B−C,
and B−D success-AUC differences are each **−400 ± 1,264.91**, descriptive 95% CI
**[−1,304.86, +504.86]**. Reward differences are respectively +2,235.87 ± 8,866.01,
+1,904.00 ± 6,461.14 and +2,235.87 ± 8,866.01; their intervals also include zero.
All four arms finish with 10/10 successful models, reward 1.24 and 16 steps.
Scratch and Untrained Source have identical checkpoint tables and training histories.
The real-source intervention check confirms that modifying a source action value
changes mapped priors by the specified centered amount, without writing the source.

**Conclusion:** mechanistic source influence is verified; a beneficial learned
cross-skill effect is **not established**. Success AUC has a small negative mean
effect and reward AUC a positive mean effect, both uncertain. One source, one map,
ten seeds, sparse coverage and coarse checkpoints limit interpretation. Transfer's
zero success-AUC SD reflects identical sampled success curves, not zero learning
uncertainty. No benchmark or hyperparameter was changed after seeing results.

The required JSON/CSV/PNG artifacts, full episode histories, all checkpoint models,
paired seed differences, source identity/version/hash, mapping reasons and integrity
inventories are in the run directory. **545 retained files** (all agent versions,
old evaluations, baselines and M2/M3-A evidence) have identical before/after SHA-256;
agent identity and old-skill evaluations also match. No skill version was overwritten
or published. The earlier same-protocol run is retained in its own UUID directory.

See [docs/M3_CROSS_SKILL_TRANSFER.md](docs/M3_CROSS_SKILL_TRANSFER.md) for B0 design,
the fixed B1 protocol, all ten seed results, confidence intervals, limitations and
exact verification commands. Existing CLI/replay tests are retained and run headlessly.

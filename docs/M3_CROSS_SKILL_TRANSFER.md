# M3-B — Cross-Skill Knowledge Transfer

## B0: design and fixed protocol

Navigation has 21 traversable positions on a 5×5 map; Resource Collection has
30 positions × 4 inventory masks on a different 6×6 map. The four actions have
the same meanings, but indices, walls, rewards, terminal conditions and inventory
semantics differ. Copying or resizing the Q-table would not align these meanings.
M1, M2 and M3-A environments, training, storage, CLI and replay are unchanged.

The adapter `relative-vector-local-walls-v1` uses a frozen, read-only Navigation
table. Source selection is pinned before the target experiment: Alph/navigation
`bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81` (seed 42, 10,000 episodes). There is no
new source training, source selection by target outcomes, or hyperparameter search.
Source metadata/history describe Navigation training, not Collection training.

For target state `(position, inventory_mask)`, the subgoal is the first uncollected
resource in row-major order, then the base after both resources are collected.
This is a hand-defined fixed ordering, not a shortest-path or nearest-resource
solver. It is the same for all arms (Scratch ignores its prior; D produces zeros).
Let `d = subgoal - position`. The only candidate source position is `(4,4) - d`.
It must be a traversable, nonterminal Navigation cell, with exactly the same
four-bit immediate blocked-neighbour signature as the target position.

For a matching row, initialize `Q_target(state,a) = Q_source(source_state,a) -
mean_a Q_source(source_state,a)` with scale 1.0. This centered action preference
is learned from the actual source table, including its action gaps and wall
penalties. The geometry lookup, centering, scale, resource ordering and fallback
are hand-defined. No target optimal actions, BFS/A*, distance search, evaluation
results or target rewards enter the adapter. Only the independent target learner
subsequently sees target transitions and rewards. Transferred values are used
once for initialization, not as permanent shaping or an ongoing expert policy.

Fallback is a zero row for unavailable displacement, local-wall mismatch,
terminal target states, or encoded states inconsistent with collection-on-entry.
Every row's source correspondence or rejection reason is exported. Terminal and
inconsistent states are excluded from the additional eligible-state denominator;
coverage over all 120 encoded states is also reported. Training visit coverage
is measured separately, after trajectories are observed; it does not tune mapping.
Run-level `coverage` describes the adapter's candidate mapping, including for A,
where that candidate prior is discarded. `nonzero_prior_states` likewise describes
the candidate adapter output; A's actual initialized table is all zeros. D maps
the same nine candidate states but transfers zero values. Initial/checkpoint table
hashes and saved arrays distinguish candidate coverage from applied information.

The bottom-right M1 goal only supplies nonnegative displacement components.
North/west subgoals often cannot map. Matching immediate walls does **not** prove
equivalent remote topology. This deliberately limited prior is a testable candidate,
not a validated spatial abstraction or a reason to silently mirror/rotate a policy.

## B1: controls and equal budgets

| Arm | Source content | Target initialization |
| --- | --- | --- |
| A Scratch | ignored | all zeros |
| B Transfer | retained learned Navigation table | adapter output |
| C Shuffled Prior | same source values, nonterminal entries flattened and permuted | same adapter |
| D Untrained Source | all-zero Navigation table | same adapter |

Shuffling preserves the exact multiset of source nonterminal Q-values, but breaks
both state and action relationships. The terminal row stays untouched and is never
mapped. Centering changes the resulting prior's distribution; C does not guarantee
an identical target-prior distribution to B. The permutation and derived-content
hash are recorded for every seed. D is a synthetic zero-initialized source with the
same contract, not a saved model claimed to have been trained. B/C/D use identical
mapping and centering. Scratch and D should be exactly identical under paired RNG.

Ten independent **target** seeds 0..9 each run all four arms, with 30,000 actual
target transitions per arm. A fixed transition budget implements the requested
equal total training steps: episode counts and lengths may differ. Checkpoints are
0, 250, 500, 1,000, 2,000, 5,000, 10,000 and 30,000 transitions, rather than the
suggested episode checkpoints. Completed episode histories include cumulative
transition indices, and the final partial episode is retained separately. Total
cost is 1,200,000 target training transitions, CPU only.

All arms use alpha=.1, gamma=.95 and epsilon `1 - .95 * min(t/(.8*budget),1)`.
Training RNG is NumPy PCG64, SeedSequence `[seed,202]`: draw `budget` uniforms,
then `budget` integer actions. Draws are indexed by global transition; differing
trajectories cannot shift later draws. Shuffle RNG uses independent `[seed,303]`.
Every reset uses the unchanged M2 base. Greedy ties use first argmax (up, right,
down, left). True completion suppresses bootstrap; time limits retain bootstrap,
consistent with the existing learners. No dependency, GPU, API or server is added.

Evaluation uses exactly **one deterministic greedy base-start rollout** at each
checkpoint, capped at the unchanged 100 steps. It uses no random numbers and
does not update the supplied Q-table. Repeating it 100 times would add no
independent evidence, so no such repetition occurs. Success rate per seed is 0
or 1; the arm mean is a fraction of the ten independently trained target models.
Rewards include failures. Successful step counts are null on failure; failures
and timeouts are explicit. There is no independent evaluation RNG to accidentally
share with training. These are within-task learning measurements, not held-out
test-map results. No additional target configurations were introduced.

## Metrics and interpretation

Raw AUC is the trapezoidal integral
`sum((x[i+1]-x[i]) * (y[i]+y[i+1])/2)`, with x = actual target transitions.
Unequal checkpoint intervals are weighted correctly. Normalized AUC divides by
30,000; raw success AUC ranges 0..30,000. It is a coarse sampled curve integral,
not a count of successful training episodes. Full episode histories support
inspection without pretending interpolated checkpoint values were observed.

All seed metrics and paired B−A, B−C, B−D differences are saved. Summaries give
mean, sample SD (ddof=1), and descriptive 95% Student-t intervals for n=10
(df=9, multiplier 2.262157). The paired differences, not unpaired arm variances,
determine comparison SD/CI. These small-sample intervals assume independent target
seeds and do not cover source-model or map variation, multiple comparisons,
mapping selection or checkpoint discretization. They are descriptive evidence.

Positive transfer means a positive paired learning-curve effect under this
protocol; negative transfer means a harmful prior slows learning. Near-zero or
uncertain comparisons remain inconclusive. A trained-source advantage over both
controls is necessary evidence for a learned-source effect, not sufficient evidence
for general cross-skill learning. Final success alone can hide early negative
transfer. No result is discarded for being negative.

## Storage, integrity and execution

Run from the repository root in PowerShell:

```powershell
Set-Location C:\AgentVerse\AgentVerse
$PY = '.\.venv\Scripts\python.exe'
& $PY -m agentverse.cross_skill_transfer --agent Alph --source-dir outputs --version bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81 --output-dir outputs/m3-cross-skill
& $PY -m pytest -q --basetemp ("outputs/tmp/m3b-tests-" + [guid]::NewGuid())
& $PY -m compileall -q src tests
git diff --check
```

Each invocation publishes a new UUID run directory. `protocol.json` is written
before target training. Required artifacts are `results.json`, `summary.csv`,
`learning_curves.png`, `transfer_comparison.png`, `source_manifest.json`,
`compatibility_report.json` and `integrity_report.json`. Additional files contain
checkpoint metrics, seed AUCs, paired differences/comparisons, before inventories,
per-arm/seed NPZ checkpoint tables, and a complete artifact checksum manifest.
All outputs remain under Git-ignored `outputs/m3-cross-skill/`.

The existing AgentStore validates identity/version/contract, Q checksum, shape,
float64/finite content and training history. The adapter validates both contracts
and action meanings. Old agents (all versions, histories, metadata, evaluations),
baselines, `m2-*` evidence and `m3-transfer` files are hashed before and after.
Original latest-skill performance is also compared with identical evaluation
conditions. No skill is published, migrated or overwritten. The output path is
rejected if it overlaps retained evidence. Missing/corrupt/incompatible models
fail instead of producing a success report.

## Limitations and future work

One source seed, one fixed source goal, one deterministic fixed target map and
ten target seeds cannot support broad generalization claims. Relative vectors
from a corner goal have poor directional coverage. Local-wall matching leaves
long-range topology and reward-scale differences unresolved. The hand-defined
resource ordering can influence usefulness; it is frozen before results.

Future separately registered experiments could train goal-conditioned Navigation
on a training-only goal/map distribution, evaluate held-out goals/maps, and compare
shared representations to this conservative mapping. Those would be new source
training and new benchmarks, not improvements retroactively attributed to M1.
Shared representations would still need action/observation contracts, read-only
source provenance, shuffled/untrained controls and independent source seeds.

## Observed run

Final artifact run: `outputs/m3-cross-skill/c6ba838c-9540-4c65-a453-ee4e1190a71a`.
Elapsed local CPU wall time including plots/integrity: 10.53 seconds. An earlier
same-protocol run (`5d10b203-1a8a-4e7c-845c-30046ae477c3`, 15.52 seconds) is retained.
Export improvements and extra mechanism evidence were added before the final run;
all per-seed learning metrics and tables match across the two runs. No source,
mapping, checkpoint, hyperparameter or budget was tuned between them.

Source agent UUID: `1dd32bf6-05dc-46a2-b7a6-f65aa2271f1e`, name Alph.
Navigation version: `bacb1457-6d0c-4dd5-b202-ea3b8dc0dc81`.
Source NPZ SHA-256:
`4c67c513620ababd60ec31ae2df65a200e5323f5016b5b9013fdcea7cce79c31`.
Two retained M1 versions and the M2 version
`615e9a65-6431-462f-bcc3-2b5725a97d06` remain unchanged.

Compatibility counts: 9 mapped; 88 unavailable relative vectors; 18 local-wall
mismatches; 4 inventory-inconsistent states; 1 terminal. Coverage: 9/120=7.5%,
eligible 9/115=7.83%. All 9 mapped B rows contain nonzero advantages; D has none.
Mean mapped training-visit fractions: A 5.387%, B 8.950%, C 9.596%, D 5.387%.
These fractions reflect different trajectories, not different mapping rules.

| Arm | Success AUC mean ± sample SD | Reward AUC mean ± sample SD |
| --- | ---: | ---: |
| A Scratch | 22,900 ± 1,264.91 | 16,378.50 ± 4,814.33 |
| B Transfer | 22,500 ± 0 | 18,614.37 ± 6,964.99 |
| C Shuffled Prior | 22,900 ± 1,264.91 | 16,710.38 ± 6,765.34 |
| D Untrained Source | 22,900 ± 1,264.91 | 16,378.50 ± 4,814.33 |

| Seed | A success AUC | B success AUC | C success AUC | D success AUC | A reward AUC | B reward AUC | C reward AUC | D reward AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26500 | 22500 | 22500 | 26500 | 20935 | 21775 | 20975 | 20935 |
| 1 | 22500 | 22500 | 22500 | 22500 | 9762.50 | 21925 | 21125 | 9762.50 |
| 2 | 22500 | 22500 | 26500 | 22500 | 21925 | 21925 | 29285 | 21925 |
| 3 | 22500 | 22500 | 22500 | 22500 | 18856.25 | 20637.50 | 10337.50 | 18856.25 |
| 4 | 22500 | 22500 | 22500 | 22500 | 18362.50 | 21925 | 12425 | 18362.50 |
| 5 | 22500 | 22500 | 22500 | 22500 | 17350 | 16950 | 16150 | 17350 |
| 6 | 22500 | 22500 | 22500 | 22500 | 17362.50 | -637.50 | 6862.50 | 17362.50 |
| 7 | 22500 | 22500 | 22500 | 22500 | 17137.50 | 20700 | 10600 | 17137.50 |
| 8 | 22500 | 22500 | 22500 | 22500 | 15731.25 | 18893.75 | 18093.75 | 15731.25 |
| 9 | 22500 | 22500 | 22500 | 22500 | 6362.50 | 22050 | 21250 | 6362.50 |

| Paired comparison | Success AUC difference mean ± SD | Descriptive 95% CI | Reward AUC difference mean ± SD | Descriptive 95% CI |
| --- | ---: | --- | ---: | --- |
| B−A | -400 ± 1264.91 | [-1304.86, 504.86] | 2235.87 ± 8866.01 | [-4106.49, 8578.24] |
| B−C | -400 ± 1264.91 | [-1304.86, 504.86] | 1904.00 ± 6461.14 | [-2718.02, 6526.02] |
| B−D | -400 ± 1264.91 | [-1304.86, 504.86] | 2235.87 ± 8866.01 | [-4106.49, 8578.24] |

Raw per-seed paired differences are in `paired_differences.csv` (60 rows).
All arms have 0/10 successful policies through checkpoint 2000. At 5000 A/C/D
each have 1/10 and B has 0/10; at 10000 and 30000
all arms have 10/10. The A/C early successes occur at different seeds. The final
reward is 1.24 and successful length 16 for every seed/arm; failures/timeouts are
zero there. Coarse sampled B success curves coincide for all ten seeds, producing
zero success-AUC SD despite variable training histories and reward curves.

**Interpretation: inconclusive learned-transfer benefit.** Mean success-AUC
transfer is negative and mean reward-AUC transfer positive; all paired intervals
cross zero. One B seed has negative reward AUC despite eventual completion. Final
success does not establish transfer. This candidate changes the target prior using
learned source content, but it does not demonstrate useful cross-skill learning.

Mechanism checks pass: adding 1 to the real source row's down action **in memory**
changes its mapped target prior by `[-.25,-.25,.75,-.25]`. The source file stays
untouched. B initial tables differ from D; Scratch and D have identical checkpoint
table hashes and complete training histories for every paired seed. Tests also
check source-value multisets under shuffling, source/target mismatch rejection,
missing/corrupt files, zero checkpoint, weighted AUC and honest negative reporting.
There are no failed mechanism checks in the recorded run. Remote-topology
equivalence and broader generalization remain unverified, rather than passed.

Integrity: **545 files** across agents (20), baselines (4), M2 evidence/test
directories (305) and M3-A experiments (216) match before/after. A separate initial
`outputs/m3b-preflight-inventory.json` covers 242 primary evidence files and also
matches after execution. The original agent UUID, all saved versions, histories,
metadata and old evaluations remain unchanged. Old-skill reevaluation before/after
also matches. Local untracked M3-A source/test files were preserved.

The historical M3-A raw record gives (4,3) paired success-AUC mean +4.4875 and
**sample** SD 3.8579 (population SD approximately 3.66); the requested 3.68 does
not exactly match either convention. For (0,4), mean -16.825 and sample SD 8.2641. Those earlier results
are not evidence for this experiment; no historical result was edited.

## Verification

Before changes: **61 passed in 35.29s**. Final full suite: **85 passed in 34.08s**
(61 retained tests + 24 M3-B tests). `python -m compileall -q src tests`,
`git diff --check` and the new module's CLI `--help` completed successfully.
Existing M1/M2/M3-A and replay tests were retained; GUI replay/visual acceptance
was not run. No packages, deployment configuration, commit, push or merge changed.

The first combined-suite attempt had 83 PASS / 2 FAIL: the new deliberately
incompatible-source fixture changed a metadata contract shared with the module
constant. Copying that metadata before corrupting it fixed test isolation. No
production contract validation or pre-existing assertion was weakened; the final
complete rerun above passes. Synthetic sources exercise mechanism/contract checks;
the actual Alph experiment is separate evidence, never a hard-coded test oracle.

The workspace exposes no Vault/router, so no canonical or candidate Vault content
was written. This design/result document records the lasting M3-B decisions locally.

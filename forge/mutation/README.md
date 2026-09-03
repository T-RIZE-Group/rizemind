# Mutation testing

Line and branch coverage say a test *executed* a statement. Mutation testing
says a test *would have noticed* if that statement were wrong. Forge compiles a
deliberately broken copy of a contract — a flipped comparison, a dropped
`require`, a swapped operator — and re-runs the suite. If the suite still
passes, the mutant **survived**, and there is a behaviour the tests never
assert.

## Policy

| Run | Scope | Blocking |
| --- | --- | --- |
| Every PR touching Solidity | Critical targets plus every target the branch changed | Yes |
| Nightly (03:30 UTC) | All production Solidity under `forge/src` | Report-only for now |
| Before a release or audit | All targets, every survivor reviewed by hand | Yes |

Both runs are driven by [`.github/workflows/mutation-testing.yml`](../../.github/workflows/mutation-testing.yml).

## Running it locally

```bash
cd forge
export FOUNDRY_PROFILE=ci        # pins the fuzz seed; results are not reproducible without it
forge soldeer install
forge build

python3 mutation/run.py --target access          # one subsystem
python3 mutation/run.py --all --report-only      # everything, no gate
python3 mutation/run.py --changed-since origin/main   # what the PR gate will run
```

`--report <path>` writes the aggregated JSON report; `--jobs` and `--timeout`
override the values in `targets.toml` for a one-off run.

A full campaign takes noticeably longer than `forge test`: every mutant
recompiles the contract and replays the suite.

## Why a wrapper instead of raw `forge test --mutate`

Forge **exits 0 even when mutants survive**, and its score
(`killed / (killed + survived)`) excludes invalid, skipped and timed-out
mutants. Left alone, three different broken setups would all look green:

* every mutant failing to compile — a perfect score over an empty denominator;
* mutants timing out — quietly dropped from the denominator rather than counted
  as escapes;
* one strong target carrying a weak one under a single repository-wide average.

`run.py` runs one campaign per target and fails the build when any of these
hold:

* `killed + survived == 0` — nothing was actually evaluated;
* `timed_out > 0` — a timeout is a mutant whose fate is unknown, never a kill;
* the invalid-mutant rate exceeds `max_invalid_rate`;
* a target's score is below its `min_score`.

## Current baseline

Foundry `v1.8.1`, profile `ci`, `--mutation-jobs 4`, recorded 2026-09-03.
Reproduce with `python3 mutation/run.py --all`.

| Target | Score | Killed | Survived | Invalid | Skipped | Floor |
| --- | --: | --: | --: | --: | --: | --: |
| `access` | 42.86% | 18 | 24 | 39 | 55 | 40% |
| `compensation` | 92.68% | 38 | 3 | 5 | 10 | 90% |
| `swarm` | 45.83% | 11 | 13 | 36 | 54 | 45% |
| `sampling` | 55.00% | 22 | 18 | 28 | 39 | 52% |
| `randomness` | 87.88% | 116 | 16 | 23 | 113 | 85% |
| `training` | 71.43% | 10 | 4 | 15 | 8 | 70% |

78 mutants survive today. The floors sit just under each measured score, so the
gate blocks regressions from day one while the survivor backlog is worked down.
The whole campaign takes roughly 15 minutes on a 4-vCPU runner, most of it in
`randomness`.

`access` and `swarm` are the two worth attacking first: they are the lowest
scores in the repository *and* the ones where a survivor maps to a privilege or
lifecycle bug rather than a cosmetic one.

## How the PR gate picks targets

`--changed-since <ref>` diffs `forge/src` and `forge/test` against the merge
base and selects:

* every target marked `critical` — `access` and `compensation` run on every PR
  whether or not they were touched, because a survivor in authorization or
  minting logic is the expensive kind;
* every target owning a changed subsystem, where the subsystem is the directory
  under `src/` or `test/`. `src/` and `test/` mirror each other one level deep,
  so editing `test/swarm/registry/SwarmCore.t.sol` re-runs the `swarm` target —
  weakening a test lowers the score exactly as editing the contract does.

A changed file in a subsystem no target owns prints a warning rather than
passing silently, which is the signal to add it to `targets.toml`.

## Adding or changing a target

Targets live in [`targets.toml`](targets.toml).

```toml
[targets.<name>]
paths            = ["src/<subsystem>/<Contract>.sol", ...]
min_score        = 0.0     # floor, established from a baseline campaign
critical         = true    # optional: run on every PR, changed or not
max_invalid_rate = 40.0    # optional: overrides the global ceiling
```

Only production code is listed. Interfaces (`I*.sol`) and the event-only
`types.sol` files are deliberately absent — they hold no behaviour, so every
mutant they generate is noise. Scripts, tests and the vendored `dependencies/`
tree are never mutated.

The whole test suite runs against each mutant. Do **not** narrow it with
`--match-contract`: a mutation in `SwarmV1` should be catchable by the swarm
integration tests, not just by `SwarmV1.t.sol`. Filtering is a last resort for
when runtime becomes unmanageable, and it costs real detection.

## Ratcheting the score

`min_score` is a floor, not a goal. The procedure:

1. Land the target at `min_score = 0.0` and let the nightly campaign record a
   few runs.
2. Set `min_score` to the observed baseline, so the score cannot regress.
3. Raise it as survivors get killed, one step at a time, never above what the
   suite currently clears.

`90%+` is a reasonable objective for authorization and accounting logic. There
is no universal threshold, and a target is gated on its own number precisely so
that a strong score on `randomness` cannot mask survivors in `access`.

## Reviewing survivors

The workflow's job summary lists every survivor as `file:line` with the exact
source rewrite. Each one has to end up as exactly one of:

* **a new behavioural assertion** — the common case; the test called the
  function but never checked what it did;
* **a fuzz boundary test** — the mutant only differs on an edge the fixed
  inputs never reach (`<` vs `<=`, `0`, `type(uint256).max`);
* **an invariant** — the mutant breaks a property no single test owns;
* **documented dead or unreachable code** — then delete the code;
* **a documented equivalent mutant** — the rewrite genuinely cannot change
  behaviour (for example `i++` → `++i` as a statement). Record why in the test
  file next to the relevant assertion.

Never resolve a survivor by disabling the operator that produced it. In
particular `require`, arithmetic and `delegatecall` operators stay on: those are
the mutants that map to real vulnerabilities.

## Determinism

Mutation testing compares a mutated run against a baseline run, so a flaky test
falsely "kills" mutants it never detected. The `ci` profile in
[`foundry.toml`](../foundry.toml) pins the fuzz seed (invariant campaigns derive
from the same seed) and disables `ffi`, and the workflow pins Foundry to an
exact version rather than `nightly` or `stable` — mutation operators and worker
behaviour change between releases, which makes scores incomparable.

Known fragility: `test/randomness/RNG.t.sol::test_randFuzzStatistical` asserts
that 100 samples average within 10% of `max / 2`. That is roughly a 3.5-sigma
bound, so it fails for some `max` values (it does for `max = 102`). The pinned
seed keeps it green and reproducible, but the assertion should be widened or the
sample count raised before the seed is ever changed.

`--mutation-jobs` is likewise pinned in `targets.toml` rather than left to
default to the core count, because `skipped` and `invalid` counts drift with
concurrency and would make baselines runner-dependent.

## Reference

* [Forge mutation testing guide](https://www.getfoundry.sh/guides/mutation-testing)
* [Foundry advanced testing configuration](https://www.getfoundry.sh/config/reference/advanced-testing)

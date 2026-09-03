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

Foundry `v1.8.1`, profile `ci`, `--mutation-jobs 4`. Latest campaign plus the
range observed over 8 full campaigns on 2026-09-03.

| Target | Score | Observed range | Killed | Survived | Invalid | Skipped | Floor |
| --- | --: | --: | --: | --: | --: | --: | --: |
| `access` | 47.50% | 42.86–51.35% | 19 | 21 | 38 | 58 | 32% |
| `compensation` | 92.31% | 91.67–92.68% | 36 | 3 | 7 | 10 | 86% |
| `swarm` | 50.00% | 45.83–53.85% | 13 | 13 | 37 | 51 | 35% |
| `sampling` | 53.85% | 53.85–56.76% | 21 | 18 | 26 | 42 | 48% |
| `randomness` | 87.30% | 86.67–87.88% | 110 | 16 | 20 | 122 | 81% |
| `training` | 76.47% | 71.43–76.47% | 13 | 4 | 13 | 7 | 64% |

75 mutants survive in the latest run. The whole campaign takes roughly 15–20
minutes on a 4-vCPU runner, most of it in `randomness`.

`access` and `swarm` are the two worth attacking first: they are the lowest
scores in the repository *and* the ones where a survivor maps to a privilege or
lifecycle bug rather than a cosmetic one.

### The score is not reproducible across environments

This is the most important caveat on this page, and it shapes every floor above.

Within one machine state, campaigns are **exact**: five consecutive full runs
were bit-identical, and re-running a single target three times gives the same
number every time. Across machine states they are not. Rebuilding `out/`, or
simply coming back later, migrates mutants between `survived`, `skipped` and
`invalid` — `access` alone produced 42.86%, 48.65% and 51.35% from identical
source, and `swarm` moved 8 points the same way. Targets with fewer mutants and
less `invalid` churn (`compensation`, `randomness`, `training`, `sampling`)
stayed within ~2 points.

Pinning `--mutation-jobs` narrows this but does not remove it; the totals stay
constant (`access` is always 136 mutants) while the classification shifts.

Two consequences:

* **Floors carry a margin.** Each `min_score` is the observed *minimum* less a
  margin covering that target's measured spread — 10 points for `access` and
  `swarm`, 5 for the rest. A floor set 1–2 points under a single measurement,
  which is what the tempting "record the baseline and gate on it" reading
  produces, turns CI red without anyone changing a line of code.
* **The survivor list is the durable artifact, not the percentage.** Treat the
  score as a coarse regression alarm — it catches a subsystem falling off a
  cliff — and do the real work from the survivor list in the job summary.

Re-measure the range, rather than a single run, before tightening any floor.

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

1. Land the target at `min_score = 0.0` and let the nightly campaign record
   several runs — several, not one, because of the spread described above.
2. Set `min_score` to the observed *minimum* less a margin covering that
   target's spread, so the score cannot regress but ordinary tool wander cannot
   trip it either.
3. Raise it as survivors get killed, one step at a time, never above what
   repeated campaigns currently clear.

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

`--mutation-jobs` is likewise pinned in `targets.toml` rather than left to
default to the core count, because `skipped` and `invalid` counts drift with
concurrency and would make baselines runner-dependent. The mutation workflow
also sets `cache: false` on `foundry-toolchain`: that cache only holds RPC and
Etherscan responses, which this suite never uses, so here it is pure run-to-run
variance.

This is also why `test/randomness/RNG.t.sol::test_randFuzzStatistical` was
fixed rather than merely pinned. It compared a 100-sample mean against a 10%
tolerance — only 3.46 standard errors, so it failed for roughly one `max` in
1900, or about one plain `forge test` run in eight. Measured over 40 seeds it
failed 3 times; at 300 samples, where the same 10% tolerance is exactly six
standard errors, it failed 0 times. A pinned seed would have hidden that rather
than removed it, and any mutant whose run happened to trip the flake would have
been recorded as killed.

## Reference

* [Forge mutation testing guide](https://www.getfoundry.sh/guides/mutation-testing)
* [Foundry advanced testing configuration](https://www.getfoundry.sh/config/reference/advanced-testing)

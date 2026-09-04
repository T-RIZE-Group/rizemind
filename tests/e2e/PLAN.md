# End-to-end tests for the examples — implementation plan

Status: **plan only, no test code yet.**

Goal: run each of the five apps under `examples/` through the real Flower
simulation engine, the real Rizemind mods and strategies, and a real Anvil
chain — while the data layer returns pre-defined matrices instead of
downloading CIFAR-10/MNIST and running a training loop.

## 1. Scope

The point is not to check that PyTorch can classify digits. It is to check
that the wiring the library owns still works when an example is run the way
its README says: `authentication_mod`, `model_notary_mod`,
`EthAccountStrategy`, `DecentralShapleyValueStrategy`,
`SwarmConfig.get_or_deploy`, `MetricStorageStrategy`.

So the seam we cut is `task.py` (dataset + gradients) and nothing else.
`client.py`, `server.py` and every `pyproject.toml` component reference stay
byte-for-byte the code that ships.

In scope:

- All five examples, two rounds each, via `flower-simulation`.
- Real Ray backend, real message passing, real EIP-712 signing, real contract
  deployment and round transactions on Anvil.
- Assertions on what the run produced: `metrics.csv`, `weights.npz`,
  `config.json`, and on-chain round summaries and contributions.

Not in scope:

- Model quality. Accuracy here is a deterministic function of the parameters.
- Live RizeNet testnet connectivity — `rizenet_testnet` is re-pointed at
  Anvil. A real-network test stays a separate, opt-in `-m network` case.
- Foundry contract behaviour (`forge test` and `tests/integration` cover it).

## 2. Findings that shape the design

Established by probing Flower 1.21's simulation engine with a throwaway
three-file app.

| Finding | Consequence | Status |
| --- | --- | --- |
| `flower-simulation --app DIR` is a public console script that reads `[tool.flwr.app.config]` from that directory's `pyproject.toml` and fuses `--run-config` over it. | We can run a modified *copy* of an example without touching the repo and without the private `_run_simulation` API. `flwr.simulation.run_simulation()` is unusable: it passes no run config, so every example would `KeyError` on `num-server-rounds`. | verified |
| Ray `ClientAppActor` workers inherit the **driver's** cwd, not `--app`. | `server.py` and `client.py` both call `TomlConfig("./pyproject.toml")`. The subprocess must be launched with `cwd` set to the copied app dir, or the chain examples silently read the library's root `pyproject.toml`. Confirmed: with `cwd` set, both the ServerApp thread and the Ray actors resolve `./pyproject.toml` to the copy. | verified |
| **A ClientApp exception does not fail the run.** `FedAvg` defaults to `accept_failures=True`, so a raising client yields `received 0 results and 2 failures` and `flower-simulation` still exits `0`. | Exit code is necessary but nowhere near sufficient. Every test must also scan the log for `ClientAppException` and assert every `received N results and M failures` line has `M == 0`. Without this the suite is green theatre. | verified (trap) |
| `--run-config` is space-separated TOML fragments, not comma-separated: `'num-server-rounds=2 metrics-storage-path="logs"'`. The comma form exits 1. | Overrides built by a helper that quotes strings and joins on spaces. (`examples/README.md` documents the comma form — worth fixing.) | verified |
| Negative Shapley values are clamped by `normalize_contribution_scores` before `swarm.distribute`. | No revert risk — but a clamped score makes the "on-chain contribution equals φ" assertion hold for the wrong reason. Hence the strictly-additive, strictly-positive stub design below. | verified |
| `get_weights()` in the DP examples runs on the Opacus `GradSampleModule`, whose `state_dict` keys may be prefixed. | The examples work today, so order and count must round-trip. Confirmed in Phase 3 with an explicit shape assertion; if it does not hold, the stub keys off the last tensor instead of the first. | to confirm |

## 3. The harness

1. **Copy** — `shutil.copytree("examples/<name>", tmp_path/"app")`.
2. **Displace the data layer** — rename `<pkg>/task.py` to `<pkg>/_real_task.py`
   and write a three-line `task.py`:

   ```python
   """Test stub: real model, pre-defined data and updates."""

   from ._real_task import Net, get_weights, set_weights  # noqa: F401
   from tests.e2e.stubs.fake_task import load_data, test, train  # noqa: F401
   ```

   The model and weight (de)serialisers stay the example's own, so parameter
   shapes, ordering and `state_dict` keys are real. All stub logic lives in a
   normal, lintable module in the test tree, reached by putting the repo root
   on `PYTHONPATH` for the subprocess.
3. **Rewrite config** — patch the copy's `pyproject.toml`: `[tool.web3].url`
   to the session Anvil, `local_factory_deployment_path` to the absolute forge
   broadcast artifact, drop GPU client resources. Write stub settings to
   `e2e_stub.json` beside it; `fake_task` reads that file relative to `cwd`,
   which step 2's cwd guarantee makes reliable (and avoids depending on
   env-var propagation into Ray workers).
4. **Run** — `subprocess.run` on the venv's `flower-simulation`, `cwd=app`,
   hard timeout, env: `FLWR_TELEMETRY_ENABLED=0`, `HF_HUB_OFFLINE=1`,
   `HF_DATASETS_OFFLINE=1`, `HF_HOME`/`HOME` under `tmp_path`,
   `CUDA_VISIBLE_DEVICES=""`. The offline flags enforce "no downloads": if a
   stub is ever bypassed the test fails loudly instead of pulling 170 MB.
5. **Judge** — `assert_clean_run()` checks exit status, absence of
   `ClientAppException`/`ServerAppException`/`Traceback`,
   `Run finished 2 round(s)`, and zero failures on every results/failures
   line. Only then do the artifact assertions run.

Subprocess rather than in-process because process isolation is doing real
work: Ray driver state, Torch global state, the module-level `fds` cache,
`Account.enable_unaudited_hdwallet_features()`, and above all the per-example
cwd. The cost is that assertions come from artifacts, logs and chain state
rather than a returned `Context` — the right trade, since those artifacts are
what a user actually gets.

## 4. Pre-defined matrices, chosen so the answer is computable

A stub returning a constant makes every coalition identical and every Shapley
value zero — the test would pass while the contribution logic did nothing.
Encoding the trainer's identity in the matrix makes the expected result exact.

`train` ignores the data and writes a one-hot signature into the model: all
tensors zeroed, except element `i` of the first tensor's flat view set to
`1.0`, where `i` is the partition id carried on the **dataset** object
(`loader.dataset.partition_id` — an attribute that survives Opacus's
`DPDataLoader` wrapping, unlike anything set on the loader itself).

FedAvg then averages a coalition `S` into a vector whose element `i` is
`1/|S|` for members and `0` otherwise, so `test` recovers exact membership and
scores it as an additive game:

```python
members = [i for i, v in enumerate(w0.ravel()[:n_parts]) if v > 1e-9]
score   = sum(WEIGHT[i] for i in members)     # WEIGHT = (0.12, 0.24, 0.36)
return 1.0 - score, score                     # (loss, accuracy)
```

For an additive game the Shapley value of player `i` is exactly `WEIGHT[i]`:

| Coalition | v(S) | Coalition | v(S) | Trainer | Expected φ |
| --- | --- | --- | --- | --- | --- |
| {1} | 0.12 | {1,2} | 0.36 | trainer 1 | 0.12 |
| {2} | 0.24 | {1,3} | 0.48 | trainer 2 | 0.24 |
| {3} | 0.36 | {2,3} | 0.60 | trainer 3 | 0.36 |
| — | — | {1,2,3} | 0.72 | Σ | 0.72 |

Three things fall out for free: the winning coalition is uniquely `{1,2,3}`,
so `weights.npz` must hold the all-members signature; every φ is strictly
positive and strictly ordered, so a clamped or dropped contribution is
detectable; and the accuracy in `metrics.csv` is a known constant.

`load_data` returns two `DataLoader`s over a tiny in-memory
`FakePartitionDataset` — 64 deterministic random samples per partition, batch
key and tensor shape read from `e2e_stub.json` (`img`/`3x32x32` for the CIFAR
apps, `image`/`1x28x28` for the MNIST ones), equal sizes across partitions so
the FedAvg weighting is the clean `1/|S|`. The dataset is real enough for
Opacus's `make_private` to build its Poisson sampler; with `batch-size=8` the
sample rate is 0.125 and the noise-multiplier search stays fast. No epoch is
ever iterated, because `train` returns before looking at the loader.

**Extra signal for the dynamic-privacy example.** In
`torch_dyn_diff_privacy_shapley` the stub `train` returns
`optimizer.noise_multiplier` as its epsilon instead of a constant. Opacus
derives that from `target_epsilon`, which `FlowerClient.fit` adapts using the
previous round's on-chain contribution — so `average_epsilon` differing
between rounds 1 and 2 is direct evidence the contribution-feedback loop
closed. Round 1 has no prior summary and takes the `-1.0` branch, which is
why every example runs two rounds, not one.

## 5. The five examples

Rounds fixed at two; supernodes at three, giving seven coalitions per round in
the Shapley apps.

| Example | pkg | Batch key / shape | Chain | What only this test covers |
| --- | --- | --- | --- | --- |
| `torch_basic` | `torch_basic` | `img` 3x32x32 | none | Plain `FedAvg` under `MetricStorageStrategy`; the local-disk metric writer end to end. Phase 2 uses it to prove the harness. |
| `torch_diff_privacy` | `src` | `image` 1x28x28 | none | Opacus `PrivacyEngine.make_private` runs for real inside `client.py`; `average_epsilon` via `fit_metrics_aggregation_fn`. |
| `torch_shapley` | `src` | `img` 3x32x32 | anvil | `authentication_mod` + `model_notary_mod` round-trip, `EthAccountStrategy`, `DecentralShapleyValueStrategy`, swarm deploy via factory, `distribute` and `next_round`. |
| `torch_dyn_diff_privacy_shapley` | `src` | `image` 1x28x28 | anvil | The above plus `DynamicPrivacyClient` reading `get_last_contributed_round_summary` and adapting `target_epsilon` — the only path in the repo that reads a contribution back out. |
| `rizenet_testnet` | `src` | `img` 3x32x32 | anvil (re-pointed) | The `mnemonic_store` account path — the test seeds a keystore under a `tmp_path` `HOME`, since `RIZEMIND_HOME` is `Path.home()/".rzmnd"` and honours `$HOME`. Its `[tool.web3].url` and factory address are rewritten to the local chain, so this asserts the app wires up, not that RizeNet is reachable. |

### Run configuration

Common: `--num-supernodes 3`,
`--backend-config '{"client_resources":{"num_cpus":1,"num_gpus":0}}'`, and
overrides `num-server-rounds=2 fraction-fit=1.0 fraction-evaluate=1.0
min-available-clients=3 batch-size=8`.

The three Shapley apps additionally need `num-supernodes=3` *in the run
config*, because `server_fn` reads it from there to build the trainer roster
while the engine reads the CLI flag. The harness derives the run-config value
from the CLI value so they cannot drift.
`torch_dyn_diff_privacy_shapley` also takes `epochs=1`.

## 6. What each test asserts

**Tier 1 — the run was actually clean** (all examples)

- Exit status 0, within the timeout.
- No `ClientAppException`, `ServerAppException` or `Traceback` in the output.
- `Run finished 2 round(s)` present.
- Every `received N results and M failures` line has `M == 0` and `N > 0`.

**Tier 2 — the artifacts a user gets** (all examples)

- `logs/<app-name>/<timestamp>/` exists with exactly one timestamp directory.
- `metrics.csv` parses, covers rounds 1 and 2, and carries the expected keys:
  `accuracy` everywhere, `average_epsilon` for the two DP apps,
  `median_coalition_accuracy` for the three Shapley apps.
- The recorded `accuracy` equals the additive game's prediction to `1e-6`.
- `config.json` round-trips the run config (and, for the chain apps, the TOML
  config `server.py` writes as a second frame).
- `weights.npz` loads (`allow_pickle=True` — saved as a ragged object array)
  and its first tensor's signature is the full trainer set.

**Tier 3 — the chain** (the three chain examples)

- The swarm address is recovered after the run by scanning the known factory
  contract for its `ContractCreated` event, since the proxy address is chosen
  inside the ServerApp with a random salt and never printed.
- `swarm.current_round()` advanced to the expected round.
- `get_last_contributed_round_summary` present for each trainer account, with
  `n_trainers == 3`.
- Per-trainer `get_latest_contribution` strictly ordered
  `trainer1 < trainer2 < trainer3`, none zero. *Open question:* confirm the
  scaling `distribute` applies to a float score before asserting exact
  equality with φ; until then assert ordering and ratios.

## 7. Files

New:

- `tests/e2e/conftest.py` — loads `forge_fixtures`; session fixtures for the
  deployed factory artifact and the sandboxed `HOME`.
- `tests/e2e/examples.py` — `ExampleSpec` and the list of five; the single
  place a new example gets registered.
- `tests/e2e/harness.py` — `prepare_app()`, `run_app()`, `RunResult`,
  `assert_clean_run()`.
- `tests/e2e/stubs/fake_task.py` — `FakePartitionDataset`, `load_data`,
  `train`, `test`, the `WEIGHT` table and the expected-score helpers the
  assertions share.
- `tests/e2e/artifacts.py` — readers and assertions for `metrics.csv`,
  `weights.npz`, `config.json`, and the `ContractCreated` swarm lookup.
- `tests/e2e/test_examples.py` — parametrised over the five specs, plus the
  example-specific extra assertions.
- `tests/e2e/README.md` — how to run it, what it covers, how to add an
  example.

Modified:

- `tests/integration/forge_fixtures.py` — extract the three-script deployment
  currently inlined in `test_swarm_v1_factory.py`'s `factory_config`
  (`SelectorFactory` -> `AlwaysSampled` -> `SwarmV1Factory`) into a session
  fixture both suites use; the integration test switches to it, same
  deployment, no behaviour change.
- `.github/workflows/pytest.yml` — an `e2e-test` job mirroring
  `integration-test` (Foundry + `forge soldeer install`) but with
  `uv sync --group ml`, running `uv run pytest tests/e2e`.
- `pyproject.toml` — register the `e2e` marker so `pytest -m "not e2e"` works
  locally; `testpaths` already covers `tests/`.

Chain-dependent tests skip rather than error when `anvil` or `forge` is
absent, so `pytest tests/e2e` still runs the two chainless examples without
Foundry. Timeouts come from `subprocess.run(timeout=...)` — no new dependency.

## 8. Sequence

1. **Extract the factory fixture.** Done when `pytest tests/integration` is
   unchanged and green.
2. **Harness + `torch_basic`.** Build `prepare_app`, `run_app`,
   `assert_clean_run`, the stub module and the tier-1/2 assertions against the
   simplest example. Done when one example passes *and* a deliberately broken
   client fails the test rather than exiting 0.
3. **`torch_diff_privacy`.** Confirms the MNIST key/shape parameterisation and
   settles the Opacus `state_dict` question. Done when `make_private` runs on
   synthetic data and `average_epsilon` lands in `metrics.csv`.
4. **`torch_shapley` and the chain path.** Anvil-backed config rewriting, two
   on-chain rounds, tier-3 assertions including the `ContractCreated` lookup
   and the φ ordering. Resolve contribution scaling here. Done when
   contributions read back off-chain match the game's ordering and ratios.
5. **`torch_dyn_diff_privacy_shapley`.** Adds the
   `noise_multiplier`-as-epsilon assertion. Done when `average_epsilon`
   differs between rounds 1 and 2.
6. **`rizenet_testnet`.** Keystore seeding under a sandboxed `HOME`, config
   rewrite from testnet to local chain. Done when the `mnemonic_store` path is
   exercised without a network.
7. **CI and docs.** The `e2e-test` job, marker registration,
   `tests/e2e/README.md`, and a wall-clock measurement on a GitHub runner to
   decide whether the job runs on every PR or on merge to `main`.

## 9. Risks and open questions

- **CI cost.** `uv sync --group ml` pulls Torch and Torchvision — roughly 2 GB
  and a few minutes uncached; add `enable-cache: true` to `setup-uv`. Five
  subprocesses each pay a Ray startup and a Torch import; expect three to six
  minutes after install. If that is too much for every push, gate the job to
  `pull_request` like the existing integration job, or to merges on `main`.
- **Coupling to `task.py`'s shape.** The stub assumes each example keeps a
  `task.py` exporting `Net`, `get_weights`, `set_weights`, `load_data`,
  `train`, `test`. All five do, and it is the convention Flower quickstarts
  use. A divergence fails as an `ImportError` at the top of the run — loud,
  not silent.
- **Contribution units on chain.** `swarm.distribute` takes
  `list[tuple[address, float]]` and the contract stores something integral.
  The scaling decides whether tier 3 asserts equality with φ or only ordering
  and ratios. Resolved in Phase 4; ordering is the fallback and already
  catches a broken contribution pipeline.
- **Fixed port 8545.** `start_anvil` defaults to 8545, so a developer with a
  local Anvil running will collide. Worth adding a port parameter to the
  session fixture as a small follow-up; not a blocker for CI.
- **Two incidental bugs found while reading.**
  `examples/rizenet_testnet/src/server.py` passes `"torch-shapley"` as its
  metric-storage app name, so its logs land in the same directory as the
  `torch_shapley` example's — worth a one-word fix. Separately, the
  compatibility table at the bottom of `examples/README.md` lists five
  examples that do not exist in the tree (Basic Signature, Centralized Shapley
  Value, Decentralized TabPFN, RizeNet Deployment, RizeNet Shapley) and omits
  the two DP examples. Both are outside this plan's scope.

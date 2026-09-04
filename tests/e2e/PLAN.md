# End-to-end tests for the examples — implementation plan

Revision 2 — refactor-first. Status: **phase 2 shipped; phases 1 and 3–8 not started.**

Goal: run each example app through the real Flower simulation engine, the real
Rizemind mods and strategies, and a real Anvil chain — without downloading a
dataset or taking a gradient step.

Revision 1 (in this branch's history) built the harness from the outside:
copy each example to a temp dir, rename `task.py`, generate a shim, write a
JSON sidecar, rewrite the `pyproject.toml`, set `PYTHONPATH`. Six steps of
scaffolding. Two measurements showed the scaffolding was paying for two design
problems in the examples, and that fixing those removes all six steps.

## 1. What changed since revision 1

**The examples are near-duplicates.** `task.py` is byte-identical across
`torch_basic`, `torch_shapley` and `rizenet_testnet`, and byte-identical across
the two DP examples. `rizenet_testnet`'s `server.py` differs from
`torch_shapley`'s by one comment line and its `client.py` only by import
spelling. It is not a fifth example; it is the third one pointed at another
chain.

**Nested config tables flatten into overridable run-config keys.** A table at
`[tool.flwr.app.config.web3.swarm.factory_v1]` arrives in `context.run_config`
as the flat key `web3.swarm.factory_v1.name` — exactly the dotted dialect
`rizemind.configuration.transform.unflatten` already speaks. And
`--run-config '"web3.url"="..."'` overrides it.

Together: two seams in the examples remove the whole harness, and both seams
are worth having regardless of testing.

## 2. The two seams

### 2.1 Config comes from the Context, not from `./pyproject.toml`

`TomlConfig("./pyproject.toml")` appears in `server.py` and `client.py` in
three examples. It forces the process's working directory to be the app
directory, which forces a test to copy the app and rewrite its TOML just to
change an RPC URL. It is also a real limitation outside testing: in deployment
mode the ServerApp's cwd is not the app directory.

The fix needs no new dependency and almost no new code. Add `from_run_config()`
beside the existing `from_context()` on `Web3Config`, `AccountConfig` and
`SwarmConfig`:

```python
Web3Config(**unflatten(prefixed(context.run_config, "web3")))
```

Precedence must be documented or this gets confusing fast:
**run_config → state records → environment → TOML fallback**, with one
exception for secrets (section 3).

### 2.2 A `Task` protocol, selected by config

`task.py` fuses model, data loading and the training loop with no point at
which one can be substituted. The seam is one config key, resolved the way
flwr already resolves `serverapp` and `clientapp` — idiomatic rather than an
invented service locator.

```python
# rizemind/tasks/protocol.py
class Task(Protocol):
    def initial_parameters(self) -> NDArrays: ...
    def train(self, parameters: NDArrays, config: dict[str, Scalar]
              ) -> tuple[NDArrays, int, dict[str, Scalar]]: ...
    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]
                 ) -> tuple[float, int, dict[str, Scalar]]: ...

# rizemind/tasks/resolve.py
def load_task(context: Context, key: str = "task") -> Task:
    module, _, attr = str(context.run_config[key]).partition(":")
    return getattr(importlib.import_module(module), attr).from_context(context)

# rizemind/tasks/client.py
class TaskClient(NumPyClient):
    def __init__(self, task: Task) -> None: self.task = task
    def fit(self, parameters, config):      return self.task.train(parameters, config)
    def evaluate(self, parameters, config): return self.task.evaluate(parameters, config)
```

The task's `from_context` classmethod does the real work: the task pulls its
own hyperparameters out of the Context, so the FL layer never mentions
`batch-size`, `local-epochs` or `learning-rate` again. And because
`Task.train` returns the full flwr tuple, each example's `FlowerClient` class
becomes redundant.

## 3. Prerequisite: stop writing secrets to disk

Independent of everything else, and shipped ahead of the rest.

`TomlConfig._load_toml` runs `replace_env_vars` over the parsed document, so
`.data` contains resolved secrets. The three chain examples hand that whole
document to `LocalDiskMetricStorage.write_config`, which writes it to
`logs/<app>/<ts>/config.json`. Today:

- `torch_shapley` and `torch_dyn_diff_privacy_shapley` write the public Anvil
  test mnemonic — harmless in itself, but it is the pattern being taught.
- `rizenet_testnet` writes its **keystore passphrase**.
- Anyone following the README's `mnemonic = "$RIZENET_MNEMONIC"` pattern writes
  a **real seed phrase** to a plain file.

**Shipped.** Redaction lives at the boundary where configuration leaves the
process, in `rizemind.configuration.secrets`: a key-name denylist covering both
nested and flat dot-delimited layouts, applied by
`LocalDiskMetricStorage.write_config` before anything is written. The three
chain examples now hand it `{"web3": toml_config.get("tool.web3")}` rather than
the whole document, so `[tool.eth.account]` is never passed in at all;
redaction is the safety net behind that.

`to_config_record` deliberately does **not** redact, contrary to an earlier
draft of this plan. The client propagates the account into
`context.state.config_records` precisely so `authentication_mod` and
`model_notary_mod` can sign with it; redacting there would break
authentication. Persisting and transmitting are different boundaries, and only
the persisting one gets redacted.

Seam 2.1 would *add* a second copy of this leak — `write_config(context.run_config)`
is already called too — which is why **secrets never enter `run_config`**. The
mnemonic resolves from environment or keystore only; `run_config` carries
`eth.account.default_account_index` and
`eth.account.mnemonic_store.account_name`, nothing sensitive.

## 4. Verified findings

Established in-session by running flwr 1.21 against scratch apps and by
measuring this repository.

| Finding | Consequence | Status |
| --- | --- | --- |
| Nested tables under `[tool.flwr.app.config]` flatten to dotted `run_config` keys (`web3.swarm.factory_v1.name`), and `--run-config '"web3.url"="..."'` overrides them. `rizemind`'s `flatten`/`unflatten` already use `.` as separator. | The whole chain config becomes overridable from the CLI with no file rewriting, and no new parsing code. | verified |
| `flwr run <app-path> <federation>` works from any working directory; run config still comes from the app's own `pyproject.toml`. | Once seam 2.1 lands, tests invoke the real example directory in the repo with no copy at all — only `metrics-storage-path` is redirected to a temp dir. | verified |
| **A ClientApp exception does not fail the run.** `FedAvg` defaults to `accept_failures=True`, so a raising client yields `received 0 results and 2 failures` and the process still exits 0. | Exit code cannot be the oracle. Every test must scan for `ClientAppException` and assert every `received N results and M failures` line has `M == 0`. | verified (trap) |
| `flwr run` resolves `flower-simulation` by name and exits 0 when it is absent — the run never starts, and the only trace is one stdout line. | `PATH` must carry the venv's `bin`. Second independent reason exit status is worthless. | verified (trap) |
| `[dependency-groups]` is not the same as extras. There is no `[project.optional-dependencies]` table, so `pip install rizemind[ml]` fails — PEP 735 groups are not published in wheel metadata. | Any extra needs that table added first. Hatch's `include = ["src/py/rizemind"]` already ships new subpackages, so no build change for code. | verified |
| Base install is **1.1 GB**: `ray` 162 MB, `mlflow` 101 MB, `polars` 124 MB. The library imports `ray` nowhere, `mlflow` in two files, `polars` in one. | See section 5. Independent of testing, and worth more than a tasks extra. | verified |
| `TomlConfig` expands `$ENV_VAR` before exposing `.data`, and the three chain examples call `metrics_storage.write_config(toml_config.data)`. | Secrets are written to `logs/<app>/<ts>/config.json` today. See section 3. | verified (live bug) |
| Upstream's oracle `flwr ls --format=json` on `.runs[0].status == "finished:completed"` returns `success: false` on flwr 1.21 for a local simulation federation — it requires a SuperLink. | The log scan carries the weight. Revisit on a flwr upgrade; it would replace most of tier 1 with one `jq` expression. | verified |
| `task.py` is byte-identical across `torch_basic` / `torch_shapley` / `rizenet_testnet`, and across the two DP examples. `rizenet_testnet`'s app code differs from `torch_shapley`'s by a comment and import spelling. | `rizenet_testnet` stops being a directory. Five examples become four. | verified |

### Prior art: how Flower tests its own examples

`flwrlabs/flower` (read at `35a02bb`) carries `framework/e2e/` and a
`Framework E2E` workflow.

- **Copy: the runner and CI shape.** Their `apps` job runs each example with
  `flwr run --run-config num-server-rounds=1 --stream` from the example's own
  directory — the documented user path, and what this plan uses. They also set
  `FLWR_TELEMETRY_ENABLED: 0` at workflow level, gate everything behind
  `dorny/paths-filter`, and cache the Python install location per example.
- **Copy: the closed-form client.** `framework/e2e/e2e-bare` holds
  `model_params = np.array([1])` and an `objective = 5`; `fit` returns
  `param * (objective / mean(param))`. No dataset, no gradient. Same idea as
  section 6, but it only supports a loose `losses[0]/losses[-1] >= 0.98` band.
- **Understand: why they assert inside the ServerApp.**
  `e2e-pytorch/server_app.py` asserts on `context.history` in `app.main()`
  because a *server* exception fails the run while a client one does not. We
  cannot use that lever without editing the file under test.
- **Don't copy: their answer to slow data.** `e2e-pytorch` downloads CIFAR-10
  and takes `Subset(range(100))` with a shrunken CNN and a cached pre-download
  step. That works when you own the cache; it does not meet the no-download
  requirement and it makes every test depend on Hugging Face being up.

## 5. Packaging: the extras table

Adding `[project.optional-dependencies]` is a prerequisite for any extra.
While it is open, the base install is worth fixing.

| Dependency | Size | Imported by | Proposal |
| --- | --- | --- | --- |
| `ray` | 162 MB | nothing in the library; only flwr's simulation engine at runtime | base becomes `flwr`; add `rizemind[simulation]` |
| `mlflow` | 101 MB | 2 files under `logging/mlflow/` | `rizemind[mlflow]`, subpackage importing lazily |
| `polars` | 124 MB | 1 file — `local_disk_metric_storage.write_config` | replace with stdlib `json`; it is a schema-union dict merge |
| `torch`, `torchvision`, `opacus`, `flwr-datasets` | ~2.5 GB | nothing in the library — examples only | `rizemind[torch]`, **deferred** |

Two naming notes. `[task]` names an internal module; extras should name a
capability the way `flwr[simulation]` and `sqlalchemy[asyncio]` do —
`rizemind[torch]` says what you get and leaves room for `[jax]`. And extras
cannot carry an index, so `rizemind[torch]` pulls CUDA wheels by default;
`--extra-index-url .../whl/cpu` is a pip/uv concern that does not travel in
wheel metadata.

**Why the reference tasks stay in the examples for now.** Shipping `CifarTask`
in the library means taking ownership of ML training code as public API under
semver, and a CIFAR CNN is not what differentiates Rizemind. Flower declines
this deliberately: `flwr` ships no models or datasets, `flwr-datasets` is a
separate distribution, models live in examples. With `rizenet_testnet` reduced
to config, four examples share two task files — acceptable duplication for
self-contained teaching material. Declare `[torch]` when a second consumer asks.

What *does* go in the library, both with zero new dependencies:
`rizemind.tasks` (protocol, resolver, `TaskClient`) because the contract must
be in base or nothing can depend on it; and `rizemind.testing` (the
deterministic task and the pytest fixtures) because the fake is pure numpy,
already a hard dependency.

## 6. The deterministic task

With a protocol in place the fake needs no PyTorch at all, and is far simpler
than revision 1's version, which had to smuggle its signal through the real
model's first tensor.

`rizemind.testing.tasks.AdditiveTask` defines its own parameter vector: a
single 1-D array of length *n*. `train` ignores everything and returns a
one-hot at the trainer's partition index. FedAvg averages a coalition *S* into
a vector holding `1/|S|` at each member index and zero elsewhere, so
`evaluate` recovers exact membership and scores it additively:

```python
members = [i for i, v in enumerate(parameters[0]) if v > 1e-9]
score   = sum(WEIGHT[i] for i in members)     # WEIGHT = (0.12, 0.24, 0.36)
return 1.0 - score, self.num_examples, {"accuracy": score}
```

For an additive game the Shapley value of player *i* is exactly `WEIGHT[i]`:

| Coalition | v(S) | Coalition | v(S) | Trainer | Expected φ |
| --- | --- | --- | --- | --- | --- |
| {1} | 0.12 | {1,2} | 0.36 | trainer 1 | 0.12 |
| {2} | 0.24 | {1,3} | 0.48 | trainer 2 | 0.24 |
| {3} | 0.36 | {2,3} | 0.60 | trainer 3 | 0.36 |
| — | — | {1,2,3} | 0.72 | Σ | 0.72 |

Three properties fall out: the winning coalition is uniquely `{1,2,3}`, so
`weights.npz` must hold the all-members vector; every φ is strictly positive
and strictly ordered, so a contribution clamped by
`normalize_contribution_scores` is detectable rather than silently plausible;
and the accuracy in `metrics.csv` is a known constant, not a range.

**Named coverage gap: the real Nets and Opacus stop being exercised.** In
revision 1 the stub kept the example's real `Net`, so parameter shapes were
real and Opacus's `make_private` ran on synthetic tensors. With a clean
protocol all of that moves inside the example's task and the e2e run never
touches it. Two cheap mitigations: a per-example unit test asserting
`Task.initial_parameters()` round-trips through
`ndarrays_to_parameters`/`parameters_to_ndarrays` with the expected shapes;
and, for the DP examples, a focused test of the task's own `train` against a
small synthetic tensor set. Do not solve this by giving the real task a
"synthetic data" mode to satisfy a test — that is the scaffolding this
revision removed, moved one layer down.

## 7. What it does to an example

`torch_shapley`: 118 lines of `task.py`, 79 of `client.py`, 99 of `server.py`.

`src/task.py` — same code, reorganised, stays in the example:

```python
class Cifar10Task:
    @classmethod
    def from_context(cls, context: Context) -> "Cifar10Task":
        return cls(
            partition_id=int(context.node_config["partition-id"]),
            num_partitions=int(context.node_config["num-partitions"]),
            batch_size=int(context.run_config["batch-size"]),
            local_epochs=int(context.run_config["local-epochs"]),
            learning_rate=float(context.run_config["learning-rate"]),
        )

    def initial_parameters(self): return get_weights(Net())

    def train(self, parameters, config):
        set_weights(self.net, parameters)
        results = _train(self.net, self.trainloader, self.valloader, ...)   # unchanged
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = _test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
```

`src/client.py` — 79 lines to about 22. The 40-line `FlowerClient` deletes
entirely, and every torch import with it:

```python
def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])

    account = AccountConfig.from_run_config(context, default_account_index=partition_id + 1)
    context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = account.to_config_record()
    web3 = Web3Config.from_run_config(context)
    context.state.config_records[WEB3_CONFIG_STATE_KEY] = web3.to_config_record()

    return DecentralShapleyValueClient(TaskClient(load_task(context))).to_client()


Account.enable_unaudited_hdwallet_features()
app = ClientApp(client_fn, mods=[authentication_mod, model_notary_mod])
```

Gone: `import torch`, `TomlConfig("./pyproject.toml")`, three hyperparameter
reads, `load_data`. What remains is exactly the Rizemind story — identity,
chain, the Shapley wrapper, the mods.

`src/server.py` — 99 to about 88, and no ML dependency:

```python
def server_fn(context: Context):
    task = load_task(context)
    strategy = FedAvg(..., initial_parameters=ndarrays_to_parameters(task.initial_parameters()))

    account = AccountConfig.from_run_config(context)
    w3 = Web3Config.from_run_config(context).get_web3()
    aggregator = account.get_account(0)
    trainers = [account.get_account(i).address
                for i in range(1, int(context.run_config["num-supernodes"]) + 1)]
    swarm = SwarmConfig.from_run_config(context).get_or_deploy(
        deployer=aggregator, trainers=trainers, w3=w3)
    ...
```

`from .task import Net, get_weights` and both `TomlConfig` lines go. An
alternative that decouples the server from the ML layer completely is
`initial_parameters=None`, letting flwr request them from a client — costs a
round trip and changes the logs, so the explicit version is the default.

`pyproject.toml` — the config tables move inside:

```diff
 [tool.flwr.app.config]
+task = "src.task:Cifar10Task"
 num-server-rounds = 3
 ...

-[tool.web3.swarm.factory_v1]
+[tool.flwr.app.config.web3]
+url = "http://127.0.0.1:8545"
+
+[tool.flwr.app.config.web3.swarm.factory_v1]
 name = "test_model"
 local_factory_deployment_path = "../../forge/broadcast/.../run-latest.json"
-
-[tool.eth.account]
-mnemonic = "test test ... junk"
-
-[tool.web3]
-url = "http://127.0.0.1:8545"
```

They have to sit under `[tool.flwr.app.config]` — the only table flwr fuses
into `run_config`. The mnemonic does not move there (section 3).

**Relative paths in config resolve against the cwd, not the app.**
`local_factory_deployment_path = "../../forge/broadcast/..."` is handed
straight to `load_forge_artifact(Path(...))`, so it resolves against the
process's working directory. That works today only because you must `cd` into
the example. Once `flwr run examples/torch_shapley` works from the repo root —
verified — that path silently breaks. The config layer should resolve declared
paths against the app directory, or the field should be documented as
cwd-relative and the examples should stop using `../..`. The e2e tests are
unaffected because they override it with an absolute path, which is exactly
why this would go unnoticed.

## 8. The four examples

Two rounds each, three supernodes — seven coalitions per round in the Shapley
apps.

| Example | Chain | Task class | What only this test covers |
| --- | --- | --- | --- |
| `torch_basic` | none | `Cifar10Task` | Plain `FedAvg` under `MetricStorageStrategy`; the local-disk metric writer end to end. The phase that proves the harness. |
| `torch_diff_privacy` | none | `MnistDPTask` | `fit_metrics_aggregation_fn` and the `average_epsilon` path. Opacus itself now sits behind the task — see the coverage gap in section 6. |
| `torch_shapley` | anvil | `Cifar10Task` | `authentication_mod` + `model_notary_mod` round-trip, `EthAccountStrategy`, `DecentralShapleyValueStrategy`, swarm deploy via factory, `distribute` and `next_round`. |
| `torch_dyn_diff_privacy_shapley` | anvil | `MnistDynDPTask` | The above plus `DynamicPrivacyClient` reading `get_last_contributed_round_summary` and adapting `target_epsilon` — the only path in the repo that reads a contribution back out, and the reason every run does two rounds. |
| `rizenet_testnet` | — | — | **Deleted as a directory.** Becomes a README section: `flwr run ../torch_shapley --run-config '"web3.url"="https://testnet.rizenet.io" ...'`. A live-network test stays a separate opt-in `-m network` case. |

`--num-supernodes 3` on the CLI and `num-supernodes=3` in the run config are
independent values that must agree — `server_fn` reads the latter to build the
trainer roster while the engine reads the former. The harness derives one from
the other so they cannot drift.

## 9. What a test looks like now

```python
result = run_example("torch_shapley", overrides={
    "task": '"rizemind.testing.tasks:AdditiveTask"',
    "num-server-rounds": 2, "num-supernodes": 3,
    '"web3.url"': f'"{anvil.url}"',
    '"web3.swarm.factory_v1.local_factory_deployment_path"': f'"{artifact}"',
    "metrics-storage-path": f'"{tmp_path}"',
})
assert_clean_run(result)
```

No copy, no rename, no shim, no sidecar, no `PYTHONPATH`, no TOML rewriting.

**The e2e suite no longer needs the `ml` group.** When `task` points at
`AdditiveTask`, the resolver never imports the example's `src/task.py`, and
after the refactor neither `client.py` nor `server.py` imports torch.
`flwr run` does not install an app's declared dependencies in simulation mode —
only what is actually imported must be present. So the whole suite runs on the
base install: no 2 GB PyTorch download in CI, and no GPU-resource juggling in
the backend config.

### Tier 1 — the run was actually clean

- Exit status 0 within the timeout; no `ClientAppException` /
  `ServerAppException` / `Traceback` in the output.
- `Run finished 2 round(s)` present.
- Every `received N results and M failures` line has `M == 0` and `N > 0`.

### Tier 2 — the artifacts a user gets

- One timestamp directory under the redirected `metrics-storage-path`.
- `metrics.csv` parses, covers both rounds, and carries the expected keys:
  `accuracy` everywhere, `average_epsilon` for the DP apps,
  `median_coalition_accuracy` for the Shapley apps.
- Recorded `accuracy` equals the additive game's prediction to `1e-6`.
- `config.json` round-trips the run config **and contains no secret values**.
- `weights.npz` loads (`allow_pickle=True` — ragged object array) and holds
  the all-members vector.

### Tier 3 — the chain

- The swarm address is recovered after the run by scanning the known factory
  for its `ContractCreated` event — the proxy address is chosen inside the
  ServerApp with a random salt and never printed.
- `swarm.current_round()` advanced to the expected round;
  `get_last_contributed_round_summary` present for each trainer with
  `n_trainers == 3`.
- Per-trainer `get_latest_contribution` strictly ordered and none zero.
  *Open question:* confirm the scaling `distribute` applies to a float score
  before asserting exact equality with φ; ordering and ratios are the fallback
  and already catch a broken pipeline.

## 10. Files

New — library:

- `src/py/rizemind/tasks/{protocol,resolve,client}.py` — `Task`, `load_task`,
  `TaskClient`. No new dependencies.
- `src/py/rizemind/testing/tasks.py` — `AdditiveTask` and the expected-score
  helpers the assertions share. Pure numpy.
- `src/py/rizemind/testing/pytest_plugin.py` — the `anvil`,
  factory-deployment and `run_example` fixtures, and the artifact readers.
  Shipping these makes the harness a product feature rather than test
  scaffolding: a downstream swarm can e2e-test itself the same way.

New — tests:

- `tests/e2e/test_examples.py` — parametrised over the four examples, plus the
  example-specific extras.
- `tests/e2e/README.md` — how to run it, what it covers, how to register an
  example.
- `tests/unit/py/rizemind/tasks/` — resolver, `TaskClient`, `AdditiveTask`,
  and the `from_run_config` dotted-key round-trip.

Modified:

- `src/py/rizemind/{web3,authentication,swarm}/config.py` — add
  `from_run_config()` and a `prefixed()` helper; document precedence; keep
  secrets out.
- `src/py/rizemind/logging/local_disk_metric_storage.py` — redact secret
  fields in `write_config`; drop polars for stdlib `json`.
- `pyproject.toml` — add `[project.optional-dependencies]`; move
  `flwr[simulation]` and `mlflow` out of base; register the `e2e` marker.
- `examples/{torch_basic,torch_diff_privacy,torch_shapley,torch_dyn_diff_privacy_shapley}/`
  — task classes, thinned `client.py`/`server.py`, config tables moved under
  `[tool.flwr.app.config]`.
- `tests/integration/forge_fixtures.py` — extract the three-script deployment
  inlined in `test_swarm_v1_factory.py` into the shared fixture, then
  re-export from `rizemind.testing`.
- `.github/workflows/pytest.yml` — an `e2e-test` job (no `ml` group needed)
  behind a `dorny/paths-filter` gate, with `FLWR_TELEMETRY_ENABLED: 0` at
  workflow level.
- `examples/README.md` — fold `rizenet_testnet` into a config section; fix the
  comma-separated `--run-config` example (the parser wants spaces); correct the
  compatibility table, which lists five examples that do not exist and omits
  both DP ones.

Deleted:

- `examples/rizenet_testnet/src/` — byte-identical to `torch_shapley`'s.
  Survives as a README section and a `--run-config` line.

## 11. Sequence

Ordered so the refactor is never done blind and every phase ends with
something running. Phases 2–5 are independently shippable and none of them
mentions testing.

1. **Capture a golden baseline.** Before touching anything, run each of the
   five examples for one round *with its real dataset* and keep the resulting
   `metrics.csv` and `config.json`. Manual, local, one-time — it needs the
   CIFAR-10 and MNIST downloads, so it is not a CI job. It is the only thing
   that will tell you the refactor preserved behaviour.
   *Done when* five baseline artifact sets are recorded in the PR description.
2. **Stop writing secrets to disk.** `DONE` — `rizemind.configuration.secrets`
   plus redaction in `write_config`, and the examples narrowed to the chain
   config. 35 unit tests, three of which fail without the fix.
3. **Extras table and a smaller base install.** Add
   `[project.optional-dependencies]`; move `flwr[simulation]` and `mlflow` out
   of base with lazy imports in `logging/mlflow/`; replace polars in
   `write_config`. Keep the `ml` dependency group for local development.
   *Done when* the base venv is well under 1 GB and
   `pip install rizemind[simulation]` resolves.
4. **`from_run_config` on the config models.** Library-only, with unit tests,
   no example changes yet. Write down the precedence order and the secrets
   exception. Fix path resolution so declared relative paths resolve against
   the app directory.
   *Done when* unit tests cover the dotted-key round-trip and the precedence
   chain.
5. **`rizemind.tasks` and `rizemind.testing`.** The protocol, resolver,
   `TaskClient`, `AdditiveTask`, and the pytest fixtures. Share the Anvil and
   factory-deployment fixtures with `tests/integration` rather than
   duplicating them.
   *Done when* the new modules have unit tests and `tests/integration` still
   passes on the shared fixtures.
6. **Refactor the examples.** One example per commit: task class, thinned
   client and server, config tables moved. The acceptance bar is a real run
   against the real dataset matching the phase-1 baseline — not a run with the
   fake task.
   *Done when* each example's real run reproduces its baseline metrics, and
   `rizenet_testnet/src/` is gone.
7. **The e2e suite.** Four `run_example` calls and the three assertion tiers.
   Prove the failure detector by temporarily making a task raise and
   confirming the test goes red rather than exiting 0.
   *Done when* all four pass on the base install and a deliberately broken
   client fails the suite.
8. **CI and docs.** The gated `e2e-test` job, marker registration,
   `tests/e2e/README.md`, the `examples/README.md` corrections, and a
   wall-clock measurement on a runner.
   *Done when* the job is green on a PR and its runtime is recorded.

## 12. Risks and open questions

- **Refactoring the examples is now the bulk of the work.** Revision 1 touched
  no shipped code; this one rewrites four examples and adds four public library
  names. That is a real increase in scope and blast radius, and the reason
  phase 1 captures a baseline and phase 6 goes one example per commit. If the
  appetite is not there, revision 1's copy-and-shim harness still works and is
  in this branch's history — it is just more machinery for a worse result.
- **Indirection in teaching code.** A reader of `client.py` can no longer see
  the training loop; they follow `task=` in the pyproject to `src/task.py`. A
  genuine cost for material whose job is to teach. Mitigating: the target is
  named explicitly, sits in the same directory, and what is left in
  `client.py` is now exactly the Rizemind-specific story a reader came for.
- **Four new public names, and two config paths.** `Task`, `TaskClient`,
  `load_task`, `from_run_config` all land under semver. And `from_run_config`
  sits beside the existing `from_context`, so the precedence order has to be
  documented and tested or the two will diverge.
- **Contribution units on chain.** `swarm.distribute` takes
  `list[tuple[address, float]]` and the contract stores something integral.
  The scaling decides whether tier 3 asserts equality with φ or only ordering
  and ratios. Resolved in phase 7; ordering is the fallback.
- **Fixed port 8545.** `start_anvil` defaults to 8545, so a developer with a
  local Anvil running will collide. Add a port parameter when the fixture
  moves into `rizemind.testing`.
- **CI cost, now much lower.** Dropping the `ml` group removes the ~2 GB
  PyTorch install that dominated revision 1's estimate. What remains is four
  `flwr run` subprocesses, each paying a Ray startup, plus Anvil and three
  `forge script` deployments for two of them. The `paths-filter` gate means
  unrelated PRs pay nothing. Measure in phase 8 before deciding between every
  PR and merges to `main`.

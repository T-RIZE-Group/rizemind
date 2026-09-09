# Torch Trainer Scaling

This example lets you measure how global model accuracy changes as you increase the
number of trainers in a Flower simulation.

It uses:

- MNIST by default
- FedAvg with all trainers participating in every round
- Centralized evaluation on one shared test set
- CSV logs for each round and a sweep summary

It also includes a separate coalition-sampling driver that sits above Flower's
simulation layer and lets you choose which coalitions to evaluate.

## What gets logged

Each run creates a timestamped directory under `logs/` with:

- `summary.csv`: one row per trainer-count run
- `trainers-<N>/accuracy_by_round.csv`: round-by-round loss and accuracy for that run

## Run the default experiment

This starts with **10 trainers**, which is the default requested setup:

```bash
cd examples/torch_trainer_scaling
uv run python -m torch_trainer_scaling.experiment
```

## Increase the number of trainers

Run a sweep by passing a comma-separated list:

```bash
cd examples/torch_trainer_scaling
uv run python -m torch_trainer_scaling.experiment --trainer-counts 10,20,30
```

## Useful options

Change the number of rounds:

```bash
uv run python -m torch_trainer_scaling.experiment --trainer-counts 10 --num-rounds 5
```

Speed up a run by capping samples:

```bash
uv run python -m torch_trainer_scaling.experiment \
  --trainer-counts 10,20 \
  --num-rounds 3 \
  --max-train-samples-per-trainer 512 \
  --max-test-samples 2000
```

Run an offline smoke test with synthetic data:

```bash
uv run python -m torch_trainer_scaling.experiment \
  --dataset fake \
  --trainer-counts 10 \
  --num-rounds 1 \
  --max-train-samples-per-trainer 64 \
  --max-test-samples 256
```

## Output example

At the end of a run, the script prints the directory containing the results, for example:

```text
logs/trainer-scaling-2026-03-27-14-30-00/
```

Then inspect:

```bash
cat logs/trainer-scaling-2026-03-27-14-30-00/summary.csv
cat logs/trainer-scaling-2026-03-27-14-30-00/trainers-10/accuracy_by_round.csv
```

## Notes

- The default dataset is `mnist`, so the first run may download MNIST.
- All trainers are sampled every round to make trainer-count comparisons easier.
- Centralized evaluation is used so the reported accuracy is comparable across runs.

## Coalition Sampling

If you want to experiment with coalition selection strategies above Flower, use the
coalition experiment module:

```bash
cd examples/torch_trainer_scaling
uv run python -m torch_trainer_scaling.coalition_experiment
```

This keeps trainer partitions fixed and evaluates only the selected coalition for each
simulation run, which is the correct layer for utility-table generation.

### Available strategies

- `all`: evaluate every coalition
- `uniform`: sample a fixed total number of coalitions
- `stratified`: sample a fixed number of coalitions per coalition size
- `manual`: evaluate exact coalitions you provide

### Example commands

Stratified sampling with 10 trainers and 2 coalitions per size:

```bash
uv run python -m torch_trainer_scaling.coalition_experiment \
  --total-trainers 10 \
  --strategy stratified \
  --budget-per-size 2
```

Uniform sampling with a total budget of 20 coalitions:

```bash
uv run python -m torch_trainer_scaling.coalition_experiment \
  --total-trainers 10 \
  --strategy uniform \
  --budget 20
```

Manual coalition selection:

```bash
uv run python -m torch_trainer_scaling.coalition_experiment \
  --total-trainers 10 \
  --strategy manual \
  --coalitions "empty;0;1;0,1,2;0,1,2,3,4,5,6,7,8,9"
```

Offline smoke test:

```bash
uv run python -m torch_trainer_scaling.coalition_experiment \
  --dataset fake \
  --total-trainers 4 \
  --strategy stratified \
  --budget-per-size 1 \
  --num-rounds 1 \
  --max-train-samples-per-trainer 16 \
  --max-test-samples 64
```

### Coalition outputs

Each coalition experiment creates a timestamped directory under `logs/` with:

- `config.json`: run configuration and selected coalition count
- `coalitions.csv`: utility table with one row per evaluated coalition
- `metrics_by_size.csv`: aggregate accuracy and loss grouped by coalition size

## DP-Shapley Benchmark

If you want to compare approximate DP-Shapley runs against an exact ground-truth
utility table, use the benchmark driver:

```bash
cd examples/torch_trainer_scaling
uv run python -m torch_trainer_scaling.shapley_benchmark
```

By default this:

- runs exhaustive `2^n` ground truth for trainer counts `8,9,10,11,12,13,14,15,16`
- runs `monte_carlo` and `deterministic` approximations for `8..16`
- uses a sampled-mask budget of `837`
- computes Shapley values with the same sampled-mask pairing logic as the Solidity contract
- writes `nrmse_summary.csv` at the benchmark root

### Useful benchmark options

Run a smaller smoke test:

```bash
uv run python -m torch_trainer_scaling.shapley_benchmark \
  --trainer-counts 4 \
  --exact-trainer-counts 4 \
  --sample-budget 8 \
  --dataset fake \
  --num-rounds 1 \
  --max-train-samples-per-trainer 16 \
  --max-test-samples 64
```

Tune CPU-side evaluation for the one-round benchmark:

```bash
uv run python -m torch_trainer_scaling.shapley_benchmark \
  --trainer-counts 8,9,10 \
  --exact-trainer-counts 8,9,10 \
  --num-rounds 1 \
  --device auto \
  --evaluation-device cpu \
  --evaluation-workers 2 \
  --torch-threads-per-worker 1 \
  --train-loader-workers 0 \
  --test-loader-workers 2
```

Use `--evaluation-workers 1` to keep the default serial path. The parallel
evaluation mode only activates when `--num-rounds 1` and `--evaluation-device cpu`.

Change the deterministic mask seed:

```bash
uv run python -m torch_trainer_scaling.shapley_benchmark \
  --deterministic-address 0x0000000000000000000000000000000000000001 \
  --round-id 0
```

Select specific methods:

```bash
uv run python -m torch_trainer_scaling.shapley_benchmark \
  --methods exact,monte_carlo
```

### Benchmark outputs

Each benchmark run creates a timestamped directory under `logs/` with:

- `config.json`: benchmark configuration
- `nrmse_summary.csv`: RMSE and nRMSE per trainer count and method
- `trainers-<N>/<method>/coalitions.csv`: evaluated coalition utilities with `sample_role` and `sample_order`
- `trainers-<N>/<method>/shapley_values.csv`: one Shapley value per trainer

The benchmark summary now also includes:

- `execution_mode`: `serial-mps`, `serial-cpu`, `parallel-cpu-eval`, or `simulation`
- `pretraining_duration_seconds`: one-round local-model cache build time
- `aggregation_duration_seconds`: sum of coalition aggregation time
- `evaluation_duration_seconds`: sum of centralized evaluation time
- `coalitions_per_second`: evaluated-coalition throughput after pretraining

For trainer counts without an exact run, `ground_truth_available=false` and the
`rmse`/`nrmse_pct` fields are left blank.

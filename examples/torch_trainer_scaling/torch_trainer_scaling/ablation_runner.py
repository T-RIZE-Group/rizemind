"""DP-Shapley ablation study runner.

Orchestrates the full experimental grid: methods x trainer counts x budgets x
utility types x game instances. Writes results to CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .ablation_metrics import compute_all_metrics
from .ema import EMAShapleyTracker
from .mask_generators import (
    AntitheticPairingMaskGenerator,
    DeterministicContractMaskGenerator,
    MonteCarloMaskGenerator,
    SampledMaskGenerator,
    StratifiedSeedKeyedMaskGenerator,
)
from .mask_generators_extended import (
    AntitheticMaskGenerator,
    ContractParityStratifiedMaskGenerator,
    StratifiedAntitheticMaskGenerator,
    StratifiedMaskGenerator,
)
from .shapley_estimators import (
    KernelSHAPEstimator,
    MarginalEstimator,
    ShapleyEstimator,
    StratifiedEstimator,
)
from .synthetic_games import (
    GAME_REGISTRY,
    PrecomputedFLGame,
    SyntheticGame,
    build_game,
)

# ---------------------------------------------------------------------------
# Method Registry
# ---------------------------------------------------------------------------

MethodFactory = tuple[
    type[SampledMaskGenerator] | None,
    dict,
    type[ShapleyEstimator],
    dict,
]


def _build_method(
    method_id: str,
    K: int,
    seed: int,
    *,
    ssk_max_missing_trainers: int = 5,
) -> tuple[SampledMaskGenerator, ShapleyEstimator]:
    """Build a (generator, estimator) pair for the given method ID."""
    if method_id == "B":
        gen = DeterministicContractMaskGenerator(K, round_id=0)
        est = MarginalEstimator()
    elif method_id == "MC":
        gen = MonteCarloMaskGenerator(K, seed)
        est = MarginalEstimator()
    elif method_id == "S-prop":
        gen = StratifiedMaskGenerator(K, seed, allocation="proportional")
        est = MarginalEstimator()
    elif method_id == "S-unif":
        gen = StratifiedMaskGenerator(K, seed, allocation="uniform")
        est = MarginalEstimator()
    elif method_id == "A":
        gen = AntitheticPairingMaskGenerator(K, round_id=0)
        est = StratifiedEstimator()
    elif method_id == "A-":
        gen = AntitheticPairingMaskGenerator(K, round_id=0)
        est = MarginalEstimator()
    elif method_id == "S":
        gen = ContractParityStratifiedMaskGenerator(K, round_id=0)
        est = StratifiedEstimator()
    elif method_id == "SSK":
        gen = StratifiedSeedKeyedMaskGenerator(
            K,
            seed,
            max_missing_trainers=ssk_max_missing_trainers,
            min_coalition_size=2,
        )
        est = MarginalEstimator()
    elif method_id == "SA":
        gen = StratifiedAntitheticMaskGenerator(K, seed, allocation="proportional")
        est = StratifiedEstimator()
    elif method_id == "R":
        gen = DeterministicContractMaskGenerator(K, round_id=0)
        est = KernelSHAPEstimator()
    elif method_id == "SAR":
        gen = StratifiedAntitheticMaskGenerator(K, seed, allocation="proportional")
        est = KernelSHAPEstimator()
    else:
        raise ValueError(f"Unknown method '{method_id}'.")
    return gen, est


AVAILABLE_METHODS = (
    "B",
    "MC",
    "S",
    "A",
    "A-",
    "SA",
    "S-prop",
    "S-unif",
    "SSK",
    "R",
    "SAR",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    trainer_counts: tuple[int, ...]
    budgets: tuple[int, ...]
    methods: tuple[str, ...]
    utility_types: tuple[str, ...]
    instances: int
    output_dir: str
    seed: int
    ssk_max_missing_trainers: int
    # Real FL validation
    combined_logs_dir: str | None


# ---------------------------------------------------------------------------
# Single-cell evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    trainer_count: int
    budget: int
    method: str
    utility_type: str
    instance_seed: int
    nrmse_pct: float
    rmse: float
    max_abs_error: float
    kendall_tau: float
    top_k_precision: float
    bias: float
    duration_seconds: float
    sampled_mask_count: int


def _run_single_cell(
    *,
    n: int,
    K: int,
    method_id: str,
    game: SyntheticGame,
    exact_sv: np.ndarray,
    instance_seed: int,
    ssk_max_missing_trainers: int = 5,
) -> CellResult:
    """Run one (n, K, method, game_instance) cell."""
    start = time.perf_counter()

    gen, est = _build_method(
        method_id,
        K,
        instance_seed,
        ssk_max_missing_trainers=ssk_max_missing_trainers,
    )
    masks = gen.generate(n)
    estimated_sv = est.estimate(masks, game.utility, n)

    duration = time.perf_counter() - start

    metrics = compute_all_metrics(exact_sv, estimated_sv, top_k=3)

    return CellResult(
        trainer_count=n,
        budget=K,
        method=method_id,
        utility_type="",  # filled by caller
        instance_seed=instance_seed,
        nrmse_pct=metrics["nrmse_pct"],
        rmse=metrics["rmse"],
        max_abs_error=metrics["max_abs_error"],
        kendall_tau=metrics["kendall_tau"],
        top_k_precision=metrics["top_k_precision"],
        bias=metrics["bias"],
        duration_seconds=duration,
        sampled_mask_count=len(masks),
    )


# ---------------------------------------------------------------------------
# EMA experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EMAResult:
    alpha: float
    trainer_count: int
    budget: int
    method: str
    utility_type: str
    instance_seed: int
    mean_payout_volatility: float
    adaptation_lag: float
    rounds: int


def _run_ema_experiment(
    *,
    n: int,
    K: int,
    method_id: str,
    utility_type: str,
    instance_seed: int,
    n_rounds: int = 50,
    alpha: float = 0.5,
    noise_std: float = 0.1,
    dropout_round: int = 20,
    join_round: int = 30,
) -> EMAResult:
    """Run a multi-round EMA experiment with synthetic utility shift."""
    tracker = EMAShapleyTracker(alpha, n)
    true_values_by_round: list[np.ndarray] = []

    for r in range(n_rounds):
        # Shift game weights with noise
        current_n = n
        game_seed = instance_seed * 1000 + r
        game = build_game(utility_type, current_n, game_seed)

        # Apply noise to utility via seed variation
        gen, est = _build_method(method_id, K, instance_seed + r)
        masks = gen.generate(current_n)
        raw_sv = est.estimate(masks, game.utility, current_n)

        # Pad/truncate to n trainers for tracker
        if len(raw_sv) < n:
            raw_sv = np.pad(raw_sv, (0, n - len(raw_sv)))
        elif len(raw_sv) > n:
            raw_sv = raw_sv[:n]

        true_sv = game.exact_shapley()
        if len(true_sv) < n:
            true_sv = np.pad(true_sv, (0, n - len(true_sv)))

        true_values_by_round.append(true_sv)
        tracker.update(raw_sv)

    volatility = tracker.payout_volatility()
    lag = tracker.adaptation_lag(
        true_values_by_round,
        shift_round=dropout_round,
    )

    return EMAResult(
        alpha=alpha,
        trainer_count=n,
        budget=K,
        method=method_id,
        utility_type=utility_type,
        instance_seed=instance_seed,
        mean_payout_volatility=float(np.mean(volatility)),
        adaptation_lag=lag,
        rounds=n_rounds,
    )


# ---------------------------------------------------------------------------
# Real FL validation
# ---------------------------------------------------------------------------


def _run_fl_validation(
    combined_logs_dir: Path,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    trainer_counts: tuple[int, ...] | None = None,
    *,
    ssk_max_missing_trainers: int = 5,
) -> list[CellResult]:
    """Run methods against precomputed FL utility tables from combined_logs."""
    results: list[CellResult] = []
    allowed_counts = set(trainer_counts) if trainer_counts else None

    for trainer_dir in sorted(combined_logs_dir.iterdir()):
        if not trainer_dir.is_dir() or not trainer_dir.name.startswith("trainers-"):
            continue
        n = int(trainer_dir.name.split("-")[1])
        if allowed_counts is not None and n not in allowed_counts:
            continue
        exact_csv = trainer_dir / "exact" / "coalitions.csv"
        if not exact_csv.exists():
            continue

        print(f"  FL validation: n={n} from {exact_csv.name}")
        game = PrecomputedFLGame(exact_csv, n)
        exact_sv = game.exact_shapley()

        for K in budgets:
            for method_id in methods:
                cell = _run_single_cell(
                    n=n,
                    K=K,
                    method_id=method_id,
                    game=game,
                    exact_sv=exact_sv,
                    instance_seed=0,
                    ssk_max_missing_trainers=ssk_max_missing_trainers,
                )
                results.append(CellResult(
                    trainer_count=cell.trainer_count,
                    budget=cell.budget,
                    method=cell.method,
                    utility_type="realistic-fl",
                    instance_seed=0,
                    nrmse_pct=cell.nrmse_pct,
                    rmse=cell.rmse,
                    max_abs_error=cell.max_abs_error,
                    kendall_tau=cell.kendall_tau,
                    top_k_precision=cell.top_k_precision,
                    bias=cell.bias,
                    duration_seconds=cell.duration_seconds,
                    sampled_mask_count=cell.sampled_mask_count,
                ))

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "trainer_count",
    "budget",
    "method",
    "utility_type",
    "instance_seed",
    "nrmse_pct",
    "rmse",
    "max_abs_error",
    "kendall_tau",
    "top_k_precision",
    "bias",
    "duration_seconds",
    "sampled_mask_count",
]

EMA_COLUMNS = [
    "alpha",
    "trainer_count",
    "budget",
    "method",
    "utility_type",
    "instance_seed",
    "mean_payout_volatility",
    "adaptation_lag",
    "rounds",
]


def _write_results_csv(path: Path, results: list[CellResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_COLUMNS)
        for r in results:
            writer.writerow([
                r.trainer_count,
                r.budget,
                r.method,
                r.utility_type,
                r.instance_seed,
                f"{r.nrmse_pct:.6f}",
                f"{r.rmse:.6f}",
                f"{r.max_abs_error:.6f}",
                f"{r.kendall_tau:.6f}",
                f"{r.top_k_precision:.6f}",
                f"{r.bias:.6f}",
                f"{r.duration_seconds:.6f}",
                r.sampled_mask_count,
            ])


def _write_summary_csv(path: Path, results: list[CellResult]) -> None:
    """Aggregate results: mean ± std across instances per (n, K, method, utility_type)."""
    from collections import defaultdict

    groups: dict[tuple, list[CellResult]] = defaultdict(list)
    for r in results:
        key = (r.trainer_count, r.budget, r.method, r.utility_type)
        groups[key].append(r)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trainer_count", "budget", "method", "utility_type", "instances",
            "nrmse_pct_mean", "nrmse_pct_std",
            "rmse_mean", "rmse_std",
            "max_abs_error_mean", "max_abs_error_std",
            "kendall_tau_mean", "kendall_tau_std",
            "top_k_precision_mean", "top_k_precision_std",
            "bias_mean", "bias_std",
        ])
        for (n, K, method, utype), cells in sorted(groups.items()):
            nrmses = [c.nrmse_pct for c in cells]
            rmses = [c.rmse for c in cells]
            max_errs = [c.max_abs_error for c in cells]
            taus = [c.kendall_tau for c in cells]
            topks = [c.top_k_precision for c in cells]
            biases = [c.bias for c in cells]
            writer.writerow([
                n, K, method, utype, len(cells),
                f"{np.mean(nrmses):.6f}", f"{np.std(nrmses):.6f}",
                f"{np.mean(rmses):.6f}", f"{np.std(rmses):.6f}",
                f"{np.mean(max_errs):.6f}", f"{np.std(max_errs):.6f}",
                f"{np.mean(taus):.6f}", f"{np.std(taus):.6f}",
                f"{np.mean(topks):.6f}", f"{np.std(topks):.6f}",
                f"{np.mean(biases):.6f}", f"{np.std(biases):.6f}",
            ])


def _write_ema_csv(path: Path, results: list[EMAResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EMA_COLUMNS)
        for r in results:
            writer.writerow([
                f"{r.alpha:.2f}",
                r.trainer_count,
                r.budget,
                r.method,
                r.utility_type,
                r.instance_seed,
                f"{r.mean_payout_volatility:.6f}",
                f"{r.adaptation_lag:.2f}",
                r.rounds,
            ])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ablation(config: AblationConfig) -> None:
    """Run the full ablation study."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = Path(config.output_dir) / f"ablation-{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save config
    (output_root / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    all_results: list[CellResult] = []
    total_cells = (
        len(config.trainer_counts)
        * len(config.budgets)
        * len(config.methods)
        * len(config.utility_types)
        * config.instances
    )
    cell_count = 0

    print(f"Ablation study: {total_cells} cells to evaluate")
    print(f"Output: {output_root}")

    for utility_type in config.utility_types:
        for n in config.trainer_counts:
            print(f"\n--- {utility_type}, n={n} ---")

            for instance_seed in range(config.instances):
                game = build_game(utility_type, n, config.seed + instance_seed)
                exact_sv = game.exact_shapley()

                for K in config.budgets:
                    for method_id in config.methods:
                        cell_count += 1
                        cell = _run_single_cell(
                            n=n,
                            K=K,
                            method_id=method_id,
                            game=game,
                            exact_sv=exact_sv,
                            instance_seed=instance_seed,
                            ssk_max_missing_trainers=config.ssk_max_missing_trainers,
                        )
                        all_results.append(CellResult(
                            trainer_count=cell.trainer_count,
                            budget=cell.budget,
                            method=cell.method,
                            utility_type=utility_type,
                            instance_seed=cell.instance_seed,
                            nrmse_pct=cell.nrmse_pct,
                            rmse=cell.rmse,
                            max_abs_error=cell.max_abs_error,
                            kendall_tau=cell.kendall_tau,
                            top_k_precision=cell.top_k_precision,
                            bias=cell.bias,
                            duration_seconds=cell.duration_seconds,
                            sampled_mask_count=cell.sampled_mask_count,
                        ))

                        if cell_count % 100 == 0 or cell_count == total_cells:
                            print(
                                f"  [{cell_count}/{total_cells}] "
                                f"{method_id} n={n} K={K} seed={instance_seed} "
                                f"nRMSE={cell.nrmse_pct:.2f}%"
                            )

    # Write synthetic results
    _write_results_csv(output_root / "ablation_results.csv", all_results)
    _write_summary_csv(output_root / "ablation_summary.csv", all_results)
    print(f"\nSynthetic results: {len(all_results)} cells")

    # Real FL validation
    if config.combined_logs_dir:
        logs_path = Path(config.combined_logs_dir)
        if logs_path.exists():
            print("\n--- Real FL Validation ---")
            fl_results = _run_fl_validation(
                logs_path,
                config.methods,
                config.budgets,
                config.trainer_counts,
                ssk_max_missing_trainers=config.ssk_max_missing_trainers,
            )
            if fl_results:
                all_with_fl = all_results + fl_results
                _write_results_csv(
                    output_root / "ablation_results_with_fl.csv", all_with_fl
                )
                _write_summary_csv(
                    output_root / "ablation_summary_with_fl.csv", all_with_fl
                )
                print(f"FL validation: {len(fl_results)} cells")
        else:
            print(f"Warning: combined_logs_dir not found: {logs_path}")

    # EMA experiments
    print("\n--- EMA Experiments ---")
    ema_results: list[EMAResult] = []
    ema_alphas = [0.3, 0.5, 0.7, 1.0]
    ema_n = max(config.trainer_counts)
    ema_K = 837
    ema_methods = [m for m in config.methods if m in ("SA", "SAR")]
    ema_utility = config.utility_types[0] if config.utility_types else "superadditive-uniform"
    ema_instances = min(config.instances, 20)

    for alpha in ema_alphas:
        for method_id in ema_methods:
            for inst in range(ema_instances):
                result = _run_ema_experiment(
                    n=ema_n,
                    K=ema_K,
                    method_id=method_id,
                    utility_type=ema_utility,
                    instance_seed=config.seed + inst,
                    alpha=alpha,
                )
                ema_results.append(result)

    if ema_results:
        _write_ema_csv(output_root / "ema_results.csv", ema_results)
        print(f"EMA results: {len(ema_results)} cells")

    print(f"\nAll results written to: {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_list(raw: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in raw.split(",") if v.strip())


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in raw.split(",") if v.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DP-Shapley ablation study runner."
    )
    parser.add_argument(
        "--trainer-counts",
        default="4,6,8,10,12,16",
        help="Comma-separated trainer counts.",
    )
    parser.add_argument(
        "--budgets",
        default="200,837,1500",
        help="Comma-separated sample budgets K.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(AVAILABLE_METHODS),
        help=f"Comma-separated methods. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    parser.add_argument(
        "--utility-types",
        default=",".join(GAME_REGISTRY),
        help=f"Comma-separated utility types. Available: {', '.join(GAME_REGISTRY)}",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=100,
        help="Number of random game instances per configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--ssk-max-missing-trainers",
        type=int,
        default=5,
        help=(
            "For the SSK method, sample only coalitions of size n-1 down to "
            "n-k, where k is this value."
        ),
    )
    parser.add_argument(
        "--combined-logs-dir",
        default=None,
        help="Path to combined_logs directory for real FL validation.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    methods = _parse_list(args.methods)
    for m in methods:
        if m not in AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method '{m}'. Available: {', '.join(AVAILABLE_METHODS)}"
            )

    config = AblationConfig(
        trainer_counts=_parse_int_list(args.trainer_counts),
        budgets=_parse_int_list(args.budgets),
        methods=methods,
        utility_types=_parse_list(args.utility_types),
        instances=args.instances,
        output_dir=args.output_dir,
        seed=args.seed,
        ssk_max_missing_trainers=args.ssk_max_missing_trainers,
        combined_logs_dir=args.combined_logs_dir,
    )

    run_ablation(config)


if __name__ == "__main__":
    main()

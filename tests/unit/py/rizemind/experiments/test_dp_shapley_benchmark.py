# ruff: noqa: E402, I001

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "torch_trainer_scaling"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from torch_trainer_scaling.ablation_runner import (  # noqa: E402
    _build_method,
    _run_fl_validation,
)
from torch_trainer_scaling.coalition_experiment import (  # noqa: E402
    ExperimentConfig,
    evaluate_coalitions,
)
from torch_trainer_scaling.combined_logs_nrmse import (  # noqa: E402
    compute_combined_logs_nrmse_rows,
)
from torch_trainer_scaling.mask_generators import (  # noqa: E402
    AntitheticPairingMaskGenerator,
    DEFAULT_DETERMINISTIC_ADDRESS,
    StratifiedSeedKeyedMaskGenerator,
    expand_masks_for_evaluation,
)
from torch_trainer_scaling.mask_generators_extended import (  # noqa: E402
    ContractParityStratifiedMaskGenerator,
    StratifiedAntitheticMaskGenerator,
)
from torch_trainer_scaling.shapley_benchmark import (  # noqa: E402
    BenchmarkConfig,
    _filter_sampled_masks_by_min_size,
    _build_benchmark_config as _build_benchmark_config_from_args,
    _build_parser,
    run_benchmark,
)
from torch_trainer_scaling.shapley_metrics import (  # noqa: E402
    compute_nrmse,
    compute_shapley_values,
    compute_stratified_shapley_values,
)


def _build_benchmark_config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    defaults = dict(
        trainer_counts=(2,),
        exact_trainer_counts=(2,),
        methods=("exact", "monte_carlo", "deterministic"),
        sample_budget=8,
        method_sample_budgets={},
        min_sampled_coalition_size=0,
        utility_source="live",
        precomputed_coalitions_dir=None,
        deterministic_address=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=0,
        ssk_max_missing_trainers=5,
        num_rounds=1,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.001,
        dataset="fake",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "logs"),
        max_train_samples_per_trainer=8,
        max_test_samples=16,
        seed=7,
        device="cpu",
        client_num_cpus=1.0,
        client_num_gpus=0.0,
        train_loader_workers=0,
        test_loader_workers=0,
        persistent_workers=False,
        evaluation_workers=1,
        torch_threads_per_worker=1,
        evaluation_device=None,
        combined_logs_dir=None,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _build_experiment_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    defaults = dict(
        total_trainers=2,
        strategy="test",
        num_rounds=1,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.001,
        dataset="fake",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "logs"),
        max_train_samples_per_trainer=8,
        max_test_samples=16,
        seed=7,
        budget=None,
        budget_per_size=None,
        min_size=0,
        max_size=None,
        coalitions=None,
        coalitions_file=None,
        include_empty=True,
        include_grand=True,
        device="cpu",
        client_num_cpus=1.0,
        client_num_gpus=0.0,
        train_loader_workers=0,
        test_loader_workers=0,
        persistent_workers=False,
        evaluation_workers=1,
        torch_threads_per_worker=1,
        evaluation_device=None,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_compute_nrmse_exact_match() -> None:
    assert compute_nrmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_compute_nrmse_expected_percentage() -> None:
    actual = compute_nrmse([10.0, 20.0], [13.0, 16.0])
    expected = (math.sqrt((9.0 + 16.0) / 2.0) / 15.0) * 100.0
    assert actual == pytest.approx(expected)


def test_compute_nrmse_zero_mean_zero_rmse() -> None:
    assert compute_nrmse([1.0, -1.0], [1.0, -1.0]) == 0.0


def test_compute_nrmse_zero_mean_non_zero_rmse() -> None:
    assert math.isinf(compute_nrmse([1.0, -1.0], [0.0, 0.0]))


def test_compute_nrmse_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        compute_nrmse([1.0, 2.0], [1.0])


def test_compute_shapley_values_matches_solidity_two_player_example() -> None:
    sampled_masks = [0, 1, 2, 3]
    utility_by_mask = {
        0: 0.0,
        1: 300.0,
        2: 600.0,
        3: 1500.0,
    }

    values = compute_shapley_values(sampled_masks, utility_by_mask, total_trainers=2)

    assert values.tolist() == pytest.approx([600.0, 900.0])


def test_compute_shapley_values_skip_missing_masks_uses_only_available_pairs() -> None:
    values = compute_shapley_values(
        [1, 3],
        {1: 0.2, 3: 0.6},
        total_trainers=2,
        missing_policy="skip",
    )

    assert values.tolist() == pytest.approx([0.0, 0.4])


def test_compute_stratified_shapley_values_skip_missing_masks_uses_only_available_pairs() -> None:
    values = compute_stratified_shapley_values(
        [1, 3],
        {1: 0.2, 3: 0.6},
        total_trainers=2,
        missing_policy="skip",
    )

    assert values.tolist() == pytest.approx([0.0, 0.2])


def test_expand_masks_for_evaluation_adds_support_and_upgrades_sampled() -> None:
    plans = expand_masks_for_evaluation([1, 3], 3, sample_role="sampled")
    plans_by_mask = {plan.mask: plan for plan in plans}

    assert set(plans_by_mask) == {0, 1, 2, 3, 5, 7}
    assert plans_by_mask[1].sample_role == "sampled"
    assert plans_by_mask[1].sample_order == 0
    assert plans_by_mask[3].sample_role == "sampled"
    assert plans_by_mask[3].sample_order == 1
    assert plans_by_mask[0].sample_role == "support"
    assert plans_by_mask[0].sample_order is None


def test_run_benchmark_fake_dataset_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    config = _build_benchmark_config(tmp_path, output_dir=str(output_dir))

    rows = run_benchmark(config)

    assert len(rows) == 3
    benchmark_roots = list(output_dir.glob("shapley-benchmark-*"))
    assert len(benchmark_roots) == 1

    summary_path = benchmark_roots[0] / "nrmse_summary.csv"
    assert summary_path.exists()

    approx_rows = [row for row in rows if row.method != "exact"]
    assert all(row.ground_truth_available for row in approx_rows)
    assert all(row.nrmse_pct is not None for row in approx_rows)
    deterministic_row = next(row for row in approx_rows if row.method == "deterministic")
    monte_carlo_row = next(row for row in approx_rows if row.method == "monte_carlo")
    assert deterministic_row.nrmse_pct == pytest.approx(0.0)
    assert monte_carlo_row.nrmse_pct >= 0.0
    assert all(row.execution_mode == "serial-cpu" for row in rows)
    assert all(row.total_duration_seconds >= 0.0 for row in rows)


def test_benchmark_parser_accepts_ssk_and_tail_width() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--methods",
            "exact,ssk",
            "--ssk-max-missing-trainers",
            "1",
            "--utility-source",
            "precomputed",
            "--precomputed-coalitions-dir",
            "combined_logs",
            "--method-sample-budgets",
            "monte_carlo=16823,deterministic=3725",
        ]
    )

    config = _build_benchmark_config_from_args(args)

    assert config.methods == ("exact", "ssk")
    assert config.ssk_max_missing_trainers == 1
    assert config.utility_source == "precomputed"
    assert config.precomputed_coalitions_dir == "combined_logs"
    assert config.method_sample_budgets == {
        "monte_carlo": 16823,
        "deterministic": 3725,
    }


def test_benchmark_parser_accepts_ablation_methods() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--methods",
            "exact,stratified,antithetic,antithetic_stratified,stratified_antithetic",
            "--method-sample-budgets",
            "stratified=111,antithetic=222,antithetic_stratified=222,stratified_antithetic=333",
            "--min-sampled-coalition-size",
            "3",
        ]
    )

    config = _build_benchmark_config_from_args(args)

    assert config.methods == (
        "exact",
        "stratified",
        "antithetic",
        "antithetic_stratified",
        "stratified_antithetic",
    )
    assert config.method_sample_budgets == {
        "stratified": 111,
        "antithetic": 222,
        "antithetic_stratified": 222,
        "stratified_antithetic": 333,
    }
    assert config.min_sampled_coalition_size == 3


def test_filter_sampled_masks_by_min_size_excludes_sizes_below_threshold() -> None:
    sampled_masks = [0, 1, 3, 7, 15]

    filtered = _filter_sampled_masks_by_min_size(
        sampled_masks,
        min_sampled_coalition_size=3,
    )

    assert filtered == [7, 15]


def test_antithetic_generator_pairs_complements() -> None:
    generator = AntitheticPairingMaskGenerator(
        sample_budget=6,
        round_id=5,
    )

    masks = generator.generate(4)
    full_mask = (1 << 4) - 1

    assert len(masks) == 6
    assert masks[1] == (full_mask ^ masks[0])
    assert masks[3] == (full_mask ^ masks[2])
    assert masks[5] == (full_mask ^ masks[4])


def test_contract_parity_stratified_generator_is_deterministic_and_capped() -> None:
    generator = ContractParityStratifiedMaskGenerator(
        sample_budget=20,
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=5,
    )

    masks = generator.generate(4)
    other_masks = ContractParityStratifiedMaskGenerator(
        sample_budget=20,
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=5,
    ).generate(4)
    next_round_masks = ContractParityStratifiedMaskGenerator(
        sample_budget=20,
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=6,
    ).generate(4)

    assert len(masks) == 15
    assert masks == other_masks
    assert masks != next_round_masks
    assert len(set(masks)) == len(masks)


def test_stratified_antithetic_generator_matches_pair_budget_and_seed() -> None:
    generator = StratifiedAntitheticMaskGenerator(
        sample_budget=7,
        allocation="proportional",
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=5,
    )

    masks = generator.generate(5)
    full_mask = (1 << 5) - 1

    assert len(masks) == 6
    assert all(mask.bit_count() <= 2 for mask in masks[::2])
    assert masks[1] == (full_mask ^ masks[0])
    assert masks[3] == (full_mask ^ masks[2])
    assert masks[5] == (full_mask ^ masks[4])

    other_round_masks = StratifiedAntitheticMaskGenerator(
        sample_budget=7,
        allocation="proportional",
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=6,
    ).generate(5)

    assert masks != other_round_masks


def test_run_benchmark_writes_canonical_antithetic_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    precomputed_dir = tmp_path / "combined_logs"
    exact_dir = precomputed_dir / "trainers-3" / "exact"
    exact_dir.mkdir(parents=True)
    exact_dir.joinpath("coalitions.csv").write_text(
        "\n".join(
            [
                "order,coalition,coalition_mask,coalition_size,accuracy,loss,duration_seconds,sample_role,sample_order",
                "0,empty,0,0,0.000000,1.000000,0.100000,exact,0",
                "1,0,1,1,0.250000,0.900000,0.100000,exact,1",
                "2,1,2,1,0.300000,0.900000,0.100000,exact,2",
                '3,"0,1",3,2,0.600000,0.800000,0.100000,exact,3',
                "4,2,4,1,0.350000,0.900000,0.100000,exact,4",
                '5,"0,2",5,2,0.650000,0.800000,0.100000,exact,5',
                '6,"1,2",6,2,0.700000,0.800000,0.100000,exact,6',
                '7,"0,1,2",7,3,1.000000,0.700000,0.100000,exact,7',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _build_benchmark_config(
        tmp_path,
        trainer_counts=(3,),
        exact_trainer_counts=(3,),
        methods=("antithetic",),
        utility_source="precomputed",
        precomputed_coalitions_dir=str(precomputed_dir),
        output_dir=str(output_dir),
        combined_logs_dir=str(precomputed_dir),
    )

    rows = run_benchmark(config)
    canonical_root = precomputed_dir / "trainers-3" / "antithetic"

    assert len(rows) == 1
    assert canonical_root.exists()
    assert (canonical_root / "coalitions.csv").exists()
    assert (canonical_root / "shapley_values.csv").exists()
    assert (canonical_root / "coalition_analysis.json").exists()


def test_run_benchmark_ablation_estimators_and_analysis_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "logs"
    precomputed_dir = tmp_path / "combined_logs"
    exact_dir = precomputed_dir / "trainers-3" / "exact"
    exact_dir.mkdir(parents=True)
    exact_dir.joinpath("coalitions.csv").write_text(
        "\n".join(
            [
                "order,coalition,coalition_mask,coalition_size,accuracy,loss,duration_seconds,sample_role,sample_order",
                "0,empty,0,0,0.000000,1.000000,0.100000,exact,0",
                "1,0,1,1,0.200000,0.900000,0.100000,exact,1",
                "2,1,2,1,0.450000,0.900000,0.100000,exact,2",
                '3,"0,1",3,2,0.650000,0.800000,0.100000,exact,3',
                "4,2,4,1,0.550000,0.900000,0.100000,exact,4",
                '5,"0,2",5,2,0.900000,0.800000,0.100000,exact,5',
                '6,"1,2",6,2,0.950000,0.800000,0.100000,exact,6',
                '7,"0,1,2",7,3,1.500000,0.700000,0.100000,exact,7',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _build_benchmark_config(
        tmp_path,
        trainer_counts=(3,),
        exact_trainer_counts=(),
        methods=("antithetic", "antithetic_stratified", "stratified"),
        sample_budget=4,
        method_sample_budgets={
            "antithetic": 4,
            "antithetic_stratified": 4,
            "stratified": 5,
        },
        utility_source="precomputed",
        precomputed_coalitions_dir=str(precomputed_dir),
        output_dir=str(output_dir),
    )

    run_benchmark(config)

    benchmark_root = next(output_dir.glob("shapley-benchmark-*"))
    anti_path = (
        benchmark_root / "trainers-3" / "antithetic" / "shapley_values.csv"
    )
    anti_strat_path = (
        benchmark_root / "trainers-3" / "antithetic_stratified" / "shapley_values.csv"
    )
    stratified_path = (
        benchmark_root / "trainers-3" / "stratified" / "shapley_values.csv"
    )
    analysis_path = (
        benchmark_root / "trainers-3" / "antithetic_stratified" / "coalition_analysis.json"
    )

    def read_values(path: Path) -> list[float]:
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [float(row["shapley_value"]) for row in reader]

    utility_by_mask = {
        0: 0.0,
        1: 0.2,
        2: 0.45,
        3: 0.65,
        4: 0.55,
        5: 0.9,
        6: 0.95,
        7: 1.5,
    }
    antithetic_masks = AntitheticPairingMaskGenerator(
        sample_budget=4,
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=0,
    ).generate(3)
    stratified_masks = ContractParityStratifiedMaskGenerator(
        sample_budget=5,
        address_seed=DEFAULT_DETERMINISTIC_ADDRESS,
        round_id=0,
    ).generate(3)

    antithetic_utility_by_mask = {
        mask: utility_by_mask[mask] for mask in sorted(set(antithetic_masks))
    }
    stratified_utility_by_mask = {
        mask: utility_by_mask[mask] for mask in sorted(set(stratified_masks))
    }

    assert read_values(anti_path) == pytest.approx(
        compute_shapley_values(
            antithetic_masks,
            antithetic_utility_by_mask,
            3,
            missing_policy="skip",
        ).tolist()
    )
    assert read_values(anti_strat_path) == pytest.approx(
        compute_stratified_shapley_values(
            antithetic_masks,
            antithetic_utility_by_mask,
            3,
            missing_policy="skip",
        ).tolist(),
        abs=1e-6,
    )
    assert read_values(stratified_path) == pytest.approx(
        compute_stratified_shapley_values(
            stratified_masks,
            stratified_utility_by_mask,
            3,
            missing_policy="skip",
        ).tolist(),
        abs=1e-6,
    )

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["method"] == "antithetic_stratified"
    assert analysis["estimator"] == "stratified"
    assert analysis["emitted_sample_count"] == 4
    assert "antithetic_pair_diagnostics" in analysis
    assert "first_sampled_masks" in analysis
    assert analysis["support_size_histogram"] == {}


def test_run_benchmark_precomputed_mode_skips_live_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "logs"
    precomputed_dir = tmp_path / "combined_logs"
    exact_dir = precomputed_dir / "trainers-3" / "exact"
    exact_dir.mkdir(parents=True)
    exact_dir.joinpath("coalitions.csv").write_text(
        "\n".join(
            [
                "order,coalition,coalition_mask,coalition_size,accuracy,loss,duration_seconds,sample_role,sample_order",
                "0,empty,0,0,0.000000,1.000000,0.100000,exact,0",
                "1,0,1,1,0.250000,0.900000,0.100000,exact,1",
                "2,1,2,1,0.300000,0.900000,0.100000,exact,2",
                '3,"0,1",3,2,0.600000,0.800000,0.100000,exact,3',
                "4,2,4,1,0.350000,0.900000,0.100000,exact,4",
                '5,"0,2",5,2,0.650000,0.800000,0.100000,exact,5',
                '6,"1,2",6,2,0.700000,0.800000,0.100000,exact,6',
                '7,"0,1,2",7,3,1.000000,0.700000,0.100000,exact,7',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_evaluate(*args, **kwargs):
        raise AssertionError("precomputed mode should not call live coalition evaluation")

    monkeypatch.setattr(
        "torch_trainer_scaling.shapley_benchmark.evaluate_coalitions",
        fail_evaluate,
    )

    config = _build_benchmark_config(
        tmp_path,
        trainer_counts=(3,),
        exact_trainer_counts=(3,),
        methods=("exact", "ssk"),
        utility_source="precomputed",
        precomputed_coalitions_dir=str(precomputed_dir),
        output_dir=str(output_dir),
        combined_logs_dir=str(tmp_path / "canonical"),
        ssk_max_missing_trainers=1,
    )

    rows = run_benchmark(config)

    assert len(rows) == 2
    assert all(row.execution_mode == "precomputed-exact-lookup" for row in rows)


def test_run_benchmark_honors_method_specific_sample_budgets(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    precomputed_dir = tmp_path / "combined_logs"
    exact_dir = precomputed_dir / "trainers-3" / "exact"
    exact_dir.mkdir(parents=True)
    exact_dir.joinpath("coalitions.csv").write_text(
        "\n".join(
            [
                "order,coalition,coalition_mask,coalition_size,accuracy,loss,duration_seconds,sample_role,sample_order",
                "0,empty,0,0,0.000000,1.000000,0.100000,exact,0",
                "1,0,1,1,0.250000,0.900000,0.100000,exact,1",
                "2,1,2,1,0.300000,0.900000,0.100000,exact,2",
                '3,"0,1",3,2,0.600000,0.800000,0.100000,exact,3',
                "4,2,4,1,0.350000,0.900000,0.100000,exact,4",
                '5,"0,2",5,2,0.650000,0.800000,0.100000,exact,5',
                '6,"1,2",6,2,0.700000,0.800000,0.100000,exact,6',
                '7,"0,1,2",7,3,1.000000,0.700000,0.100000,exact,7',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _build_benchmark_config(
        tmp_path,
        trainer_counts=(3,),
        exact_trainer_counts=(),
        methods=("monte_carlo", "deterministic"),
        sample_budget=1,
        method_sample_budgets={
            "monte_carlo": 4,
            "deterministic": 2,
        },
        utility_source="precomputed",
        precomputed_coalitions_dir=str(precomputed_dir),
        output_dir=str(output_dir),
    )

    rows = run_benchmark(config)
    rows_by_method = {row.method: row for row in rows}

    assert rows_by_method["monte_carlo"].sampled_mask_count == 4
    assert rows_by_method["monte_carlo"].sample_budget == 4
    assert rows_by_method["deterministic"].sampled_mask_count == 2
    assert rows_by_method["deterministic"].sample_budget == 2


def test_run_benchmark_writes_canonical_combined_logs_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    combined_logs_dir = tmp_path / "combined_logs"
    config = _build_benchmark_config(
        tmp_path,
        methods=("exact", "ssk"),
        output_dir=str(output_dir),
        combined_logs_dir=str(combined_logs_dir),
        ssk_max_missing_trainers=1,
    )

    run_benchmark(config)

    canonical_root = combined_logs_dir / "trainers-2" / "ssk-last1"
    coalitions_path = canonical_root / "coalitions.csv"
    shapley_path = canonical_root / "shapley_values.csv"

    assert canonical_root.exists()
    assert coalitions_path.exists()
    assert shapley_path.exists()

    coalitions_lines = coalitions_path.read_text(encoding="utf-8").splitlines()
    shapley_lines = shapley_path.read_text(encoding="utf-8").splitlines()

    assert (
        coalitions_lines[0]
        == "order,coalition,coalition_mask,coalition_size,accuracy,loss,duration_seconds,sample_role,sample_order"
    )
    assert shapley_lines[0] == "trainer_index,method,trainer_count,shapley_value"
    assert all(
        line.split(",")[1] == "ssk-last1" for line in shapley_lines[1:] if line.strip()
    )


def test_run_benchmark_canonical_write_overwrites_only_target_method(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    combined_logs_dir = tmp_path / "combined_logs"
    first_config = _build_benchmark_config(
        tmp_path,
        methods=("exact", "ssk"),
        output_dir=str(output_dir),
        combined_logs_dir=str(combined_logs_dir),
        ssk_max_missing_trainers=1,
    )
    run_benchmark(first_config)

    exact_path = combined_logs_dir / "trainers-2" / "exact" / "shapley_values.csv"
    ssk_path = combined_logs_dir / "trainers-2" / "ssk-last1" / "shapley_values.csv"
    exact_before = exact_path.read_text(encoding="utf-8")
    ssk_path.write_text("stale\n", encoding="utf-8")

    second_config = _build_benchmark_config(
        tmp_path,
        methods=("ssk",),
        exact_trainer_counts=(),
        output_dir=str(output_dir),
        combined_logs_dir=str(combined_logs_dir),
        ssk_max_missing_trainers=1,
    )
    run_benchmark(second_config)

    assert exact_path.read_text(encoding="utf-8") == exact_before
    assert ssk_path.read_text(encoding="utf-8") != "stale\n"


def test_one_round_coalition_evaluation_bypasses_flower_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_experiment_config(tmp_path)

    def fail_start_simulation(*args, **kwargs):
        raise AssertionError("one-round path should not call Flower simulation")

    monkeypatch.setattr(
        "torch_trainer_scaling.coalition_experiment.start_simulation",
        fail_start_simulation,
    )

    evaluations = evaluate_coalitions([(0,), (1,), (0, 1)], config)

    assert len(evaluations) == 3
    assert all(0.0 <= evaluation.accuracy <= 1.0 for evaluation in evaluations)
    assert all(evaluation.execution_mode == "serial-cpu" for evaluation in evaluations)


def test_parallel_one_round_matches_serial_outputs(tmp_path: Path) -> None:
    coalitions = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    serial_config = _build_experiment_config(tmp_path, total_trainers=3)
    parallel_config = _build_experiment_config(
        tmp_path,
        total_trainers=3,
        evaluation_workers=2,
        torch_threads_per_worker=1,
        evaluation_device="cpu",
    )

    serial_evaluations = evaluate_coalitions(coalitions, serial_config)
    parallel_evaluations = evaluate_coalitions(coalitions, parallel_config)

    assert [item.coalition for item in parallel_evaluations] == coalitions
    assert [item.execution_mode for item in parallel_evaluations] == [
        "parallel-cpu-eval"
    ] * len(coalitions)

    for serial_item, parallel_item in zip(serial_evaluations, parallel_evaluations):
        assert parallel_item.coalition == serial_item.coalition
        assert parallel_item.accuracy == pytest.approx(serial_item.accuracy)
        assert parallel_item.loss == pytest.approx(serial_item.loss)


def test_one_round_serial_supports_test_loader_workers(tmp_path: Path) -> None:
    config = _build_experiment_config(tmp_path, test_loader_workers=2)

    evaluations = evaluate_coalitions([(), (0,), (0, 1)], config)

    assert len(evaluations) == 3
    assert all(evaluation.execution_mode == "serial-cpu" for evaluation in evaluations)


def test_stratified_seed_keyed_prefix_uses_upper_strata() -> None:
    generator = StratifiedSeedKeyedMaskGenerator(
        sample_budget=200,
        seed=13,
        max_missing_trainers=5,
        min_coalition_size=2,
    )

    masks = generator.generate(10)

    assert masks
    assert len(masks) == 200
    sizes = {mask.bit_count() for mask in masks}
    assert sizes == {6, 7, 8, 9}
    assert all(size >= 6 for size in sizes)


def test_stratified_seed_keyed_backfills_with_deterministic_masks() -> None:
    n_ten_generator = StratifiedSeedKeyedMaskGenerator(
        sample_budget=873,
        seed=7,
        max_missing_trainers=5,
    )
    n_twelve_generator = StratifiedSeedKeyedMaskGenerator(
        sample_budget=873,
        seed=7,
        max_missing_trainers=5,
    )

    n_ten_masks = n_ten_generator.generate(10)
    n_twelve_masks = n_twelve_generator.generate(12)

    assert len(n_ten_masks) == 873
    assert len(n_twelve_masks) == 873


def test_stratified_seed_keyed_includes_last_one_out_then_fills_budget() -> None:
    generator = StratifiedSeedKeyedMaskGenerator(
        sample_budget=20,
        seed=11,
        max_missing_trainers=1,
    )

    masks = generator.generate(10)

    assert len(masks) == 20
    assert all(mask.bit_count() == 9 for mask in masks[:10])
    assert len(set(masks[:10])) == 10
    assert any(mask.bit_count() != 9 for mask in masks[10:])


def test_build_method_ssk_uses_marginal_estimator() -> None:
    generator, estimator = _build_method("SSK", 32, 17)

    assert isinstance(generator, StratifiedSeedKeyedMaskGenerator)
    assert generator.max_missing_trainers == 5
    assert estimator.name == "marginal"


def test_build_method_ssk_honors_configured_tail_width() -> None:
    generator, _ = _build_method("SSK", 64, 17, ssk_max_missing_trainers=3)

    masks = generator.generate(10)
    sizes = {mask.bit_count() for mask in masks}

    assert generator.max_missing_trainers == 3
    assert sizes == {7, 8, 9}


def test_fl_validation_respects_trainer_count_filter(tmp_path: Path) -> None:
    combined_logs = tmp_path / "combined_logs"
    trainer_3 = combined_logs / "trainers-3" / "exact"
    trainer_4 = combined_logs / "trainers-4" / "exact"
    trainer_3.mkdir(parents=True)
    trainer_4.mkdir(parents=True)

    def write_complete_table(path: Path, n: int) -> None:
        rows = [
            "coalition_mask,accuracy,loss,duration_seconds,sample_role,sample_order"
        ]
        for mask in range(1 << n):
            accuracy = mask.bit_count() / max(n, 1)
            rows.append(f"{mask},{accuracy:.6f},0.0,0.0,exact,{mask}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    write_complete_table(trainer_3 / "coalitions.csv", 3)
    write_complete_table(trainer_4 / "coalitions.csv", 4)

    results = _run_fl_validation(
        combined_logs,
        methods=("SSK",),
        budgets=(8,),
        trainer_counts=(3,),
        ssk_max_missing_trainers=2,
    )

    assert results
    assert {result.trainer_count for result in results} == {3}
    assert all(result.method == "SSK" for result in results)
    assert all(np.isfinite(result.nrmse_pct) for result in results)


def test_run_benchmark_parallel_cpu_smoke(tmp_path: Path) -> None:
    config = _build_benchmark_config(
        tmp_path,
        methods=("exact", "monte_carlo"),
        test_loader_workers=0,
        evaluation_workers=2,
        torch_threads_per_worker=1,
        evaluation_device="cpu",
    )

    rows = run_benchmark(config)

    assert len(rows) == 2
    assert all(row.execution_mode == "parallel-cpu-eval" for row in rows)
    assert all(row.coalitions_per_second > 0.0 for row in rows)


def test_compute_combined_logs_nrmse_rows_uses_exact_baseline(tmp_path: Path) -> None:
    trainer_dir = tmp_path / "combined_logs" / "trainers-3"
    exact_dir = trainer_dir / "exact"
    mc_dir = trainer_dir / "monte_carlo"
    exact_dir.mkdir(parents=True)
    mc_dir.mkdir(parents=True)

    exact_dir.joinpath("shapley_values.csv").write_text(
        "\n".join(
            [
                "trainer_index,method,trainer_count,shapley_value",
                "0,exact,3,1.0",
                "1,exact,3,2.0",
                "2,exact,3,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mc_dir.joinpath("shapley_values.csv").write_text(
        "\n".join(
            [
                "trainer_index,method,trainer_count,shapley_value",
                "0,monte_carlo,3,1.0",
                "1,monte_carlo,3,1.0",
                "2,monte_carlo,3,5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = compute_combined_logs_nrmse_rows(tmp_path / "combined_logs")

    assert [(row.trainer_count, row.method) for row in rows] == [
        (3, "exact"),
        (3, "monte_carlo"),
    ]
    assert rows[0].rmse == 0.0
    assert rows[0].nrmse_pct == 0.0
    assert rows[1].nrmse_pct == pytest.approx(
        (math.sqrt((0.0 + 1.0 + 4.0) / 3.0) / 2.0) * 100.0
    )

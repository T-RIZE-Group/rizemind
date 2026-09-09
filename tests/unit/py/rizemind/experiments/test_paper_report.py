# ruff: noqa: E402, I001

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "torch_trainer_scaling"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

PAPER_REPORT_PATH = (
    EXAMPLE_ROOT / "combined_logs" / "code" / "paper_report.py"
)
SPEC = importlib.util.spec_from_file_location(
    "torch_trainer_scaling_paper_report_test",
    PAPER_REPORT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load report module from {PAPER_REPORT_PATH}")
PAPER_REPORT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAPER_REPORT_MODULE)
generate_paper_report = PAPER_REPORT_MODULE.generate_paper_report

APPROX_METHODS = (
    "monte_carlo",
    "deterministic",
    "antithetic_stratified",
    "stratified",
    "stratified_with_duplicates",
)


def _write_metric_summary(results_dir: Path, filename: str, prefix: str) -> None:
    rows = []
    for trainer_count in (10, 11):
        rows.append(
            {
                "trainer": trainer_count,
                f"{prefix}monte_carlo": 10.0 + trainer_count,
                f"{prefix}deterministic": 5.0 + trainer_count / 10.0,
                f"{prefix}antithetic_stratified": 2.0 + trainer_count / 10.0,
                f"{prefix}stratified": 0.02 * trainer_count,
                f"{prefix}stratified_with_duplicates": 1.0 + trainer_count / 10.0,
            }
        )
    if prefix in {"kendall_tau_", "spearman_rho_", "top_k_precision_"}:
        rows = []
        for trainer_count in (10, 11):
            rows.append(
                {
                    "trainer": trainer_count,
                    f"{prefix}monte_carlo": 0.45,
                    f"{prefix}deterministic": 0.60,
                    f"{prefix}antithetic_stratified": 0.78,
                    f"{prefix}stratified": 0.99,
                    f"{prefix}stratified_with_duplicates": 0.70,
                }
            )
    if prefix == "bias_":
        rows = []
        for trainer_count in (10, 11):
            rows.append(
                {
                    "trainer": trainer_count,
                    f"{prefix}monte_carlo": 1.5,
                    f"{prefix}deterministic": 1.1,
                    f"{prefix}antithetic_stratified": 0.7,
                    f"{prefix}stratified": 0.05,
                    f"{prefix}stratified_with_duplicates": 0.3,
                }
            )
    (results_dir / filename).write_text(json.dumps(rows), encoding="utf-8")


def _make_analysis(trainer_count: int, method: str) -> dict[str, object]:
    sampled_histogram = {
        "0": 1.0,
        "1": 2.0,
        "2": 4.0,
        "3": 5.0,
        str(trainer_count - 1): 2.0,
    }
    evaluated_histogram = {
        "0": 1.0,
        "1": 4.0,
        "2": 6.0,
        "3": 8.0,
        str(trainer_count - 1): 3.0,
        str(trainer_count): 1.0,
    }
    analysis: dict[str, object] = {
        "trainer_count": trainer_count,
        "method": method,
        "completed_seed_count": 2,
        "mean_emitted_sample_count": float(40 + trainer_count),
        "mean_unique_sampled_mask_count": float(35 + trainer_count),
        "mean_duplicate_sampled_mask_count": float(5),
        "mean_evaluated_coalition_count": float(70 + trainer_count),
        "mean_sampled_size_histogram": sampled_histogram,
        "mean_evaluated_size_histogram": evaluated_histogram,
        "mean_support_size_histogram": {"0": 1.0, "1": 2.0},
        "mean_unique_sampled_row_size_histogram": sampled_histogram,
        "representative_first_sampled_masks": [
            {"order": 0, "mask": 0, "coalition": "empty", "coalition_size": 0},
            {"order": 1, "mask": 1, "coalition": "0", "coalition_size": 1},
        ],
        "representative_allocation_by_stratum": {
            "planned_allocation": {"0": 1, "1": 2, "2": 4, "3": 5}
        },
    }
    if method == "antithetic_stratified":
        analysis["mean_antithetic_pair_diagnostics"] = {
            "all_unique_masks_have_complements": 1.0,
            "consecutive_complement_pair_count": 10.0,
            "consecutive_complement_pair_fraction": 1.0,
            "consecutive_pair_count": 10.0,
            "unique_masks_with_complement_present": 20.0,
        }
    return analysis


def _build_synthetic_avg(avg_dir: Path) -> None:
    results_dir = avg_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _write_metric_summary(results_dir, "nrmse_summary.json", "nrmse_")
    _write_metric_summary(results_dir, "kendall_tau_summary.json", "kendall_tau_")
    _write_metric_summary(results_dir, "spearman_rho_summary.json", "spearman_rho_")
    _write_metric_summary(
        results_dir, "top_k_precision_summary.json", "top_k_precision_"
    )
    _write_metric_summary(results_dir, "bias_summary.json", "bias_")

    coalition_summary_rows = []
    for trainer_count in (10, 11):
        trainer_dir = avg_dir / f"trainers-{trainer_count}"
        trainer_dir.mkdir(parents=True, exist_ok=True)
        for method in APPROX_METHODS:
            method_dir = trainer_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            analysis = _make_analysis(trainer_count, method)
            (method_dir / "coalition_analysis.json").write_text(
                json.dumps(analysis),
                encoding="utf-8",
            )
            coalition_summary_rows.append(
                {
                    "trainer": trainer_count,
                    "method": method,
                    "completed_seed_count": analysis["completed_seed_count"],
                    "mean_emitted_sample_count": analysis["mean_emitted_sample_count"],
                    "mean_unique_sampled_mask_count": analysis[
                        "mean_unique_sampled_mask_count"
                    ],
                    "mean_duplicate_sampled_mask_count": analysis[
                        "mean_duplicate_sampled_mask_count"
                    ],
                    "mean_evaluated_coalition_count": analysis[
                        "mean_evaluated_coalition_count"
                    ],
                }
            )
        exact_dir = trainer_dir / "exact"
        exact_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "coalition_analysis_summary.json").write_text(
        json.dumps(coalition_summary_rows),
        encoding="utf-8",
    )


def test_generate_paper_report_creates_markdown_data_and_figures(
    tmp_path: Path,
) -> None:
    avg_dir = tmp_path / "AVG"
    _build_synthetic_avg(avg_dir)
    output_dir = tmp_path / "paper_report"

    report_data = generate_paper_report(avg_dir, output_dir)

    report_path = output_dir / "report.md"
    report_json_path = output_dir / "report_data.json"
    assert report_path.exists()
    assert report_json_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Methodology" in report_text
    assert "## Results" in report_text
    assert "## Blockchain Relevance" in report_text
    assert "### Sample-Budget Table" in report_text
    assert "60,000,000" in report_text
    assert "SR" in report_text
    assert "SSK-L3" not in report_text

    report_json = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert report_json["budget_tables"]["absolute"]["stratified"]["10"] == 666
    assert report_json["budget_tables"]["absolute"]["stratified_with_duplicates"]["11"] == 1046
    assert report_json["budget_tables"]["absolute"]["antithetic_stratified"]["10"] == 1024
    assert "10" in report_json["per_trainer_method"]
    assert "stratified" in report_json["per_trainer_method"]["10"]
    assert report_json["per_trainer_method"]["10"]["stratified"][
        "configured_sample_budget"
    ] == 666
    assert "stratified_with_duplicates" in report_json["budget_tables"]["absolute"]
    assert "antithetic_stratified" in report_json["budget_tables"]["absolute"]
    assert "ssk-last3" not in report_json["budget_tables"]["absolute"]

    expected_figures = {
        "performance_vs_trainers.png",
        "budget_overview.png",
        "accuracy_vs_budget.png",
        "coalition_distribution_focus.png",
        "coverage_heatmaps.png",
        "estimator_sampler_mismatch.png",
        "blockchain_relevance.png",
        "supplemental_normal_fit.png",
    }
    figure_files = {path.name for path in (output_dir / "figures").glob("*.png")}
    assert expected_figures.issubset(figure_files)

    assert report_data["trainer_counts"] == [10, 11]
    assert report_data["focus_methods"] == ("stratified",)

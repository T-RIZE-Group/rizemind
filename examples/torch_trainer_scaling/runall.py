#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path

from torch_trainer_scaling.sampler_budgets import (
    antithetic_stratified_budget_for_trainer_count,
    configured_budget_for_method,
    deterministic_budget_for_trainer_count,
    monte_carlo_budget_for_trainer_count,
    stratified_budget_for_trainer_count,
    stratified_with_duplicates_budget_for_trainer_count,
)

DEFAULT_SAMPLE_BUDGET = 837


def _stratified_budget_for_trainer_count(trainer_count: int) -> int:
    return stratified_budget_for_trainer_count(trainer_count)


def _stratified_with_duplicates_budget_for_trainer_count(trainer_count: int) -> int:
    return stratified_with_duplicates_budget_for_trainer_count(trainer_count)


def _antithetic_stratified_budget_for_trainer_count(trainer_count: int) -> int:
    return antithetic_stratified_budget_for_trainer_count(trainer_count)


def _format_method_sample_budgets(trainer_count: int) -> str:
    method_budgets = {
        method: configured_budget_for_method(
            method=method,
            trainer_count=trainer_count,
            fallback_budget=DEFAULT_SAMPLE_BUDGET,
            ssk_max_missing_trainers=3,
        )
        for method in (
            "monte_carlo",
            "deterministic",
            "antithetic_stratified",
            "stratified",
            "stratified_with_duplicates",
        )
    }
    return ",".join(
        f"{method}={method_budgets[method]}"
        for method in (
            "monte_carlo",
            "deterministic",
            "antithetic_stratified",
            "stratified",
            "stratified_with_duplicates",
        )
    )


def _resolve_combined_logs_workspace(combined_logs_root: Path) -> Path:
    avg_root = combined_logs_root / "AVG"
    if avg_root.is_dir():
        return avg_root
    return combined_logs_root


def main():
    root = Path(__file__).resolve().parent
    combined_logs_root = root / "combined_logs"
    combined_logs = _resolve_combined_logs_workspace(combined_logs_root)
    results_dir = combined_logs / "results"
    
    valid_trainers = []
    print(f"🧹 Cleaning up previous approximation results in {combined_logs.relative_to(root)}...")
    for t_dir in combined_logs.glob("trainers-*"):
        if (t_dir / "exact").is_dir():
            count = t_dir.name.split("-")[1]
            valid_trainers.append(int(count))
        for sub in t_dir.iterdir():
            if sub.is_dir() and sub.name != "exact":
                print(f"🗑️ Deleting {sub.relative_to(root)}")
                shutil.rmtree(sub)

    valid_trainers.sort()
    trainer_counts_str = ",".join(map(str, valid_trainers))
    if not trainer_counts_str:
        print("❌ No exact precomputed utilities found. Exiting.")
        return

    if results_dir.exists():
        for f in results_dir.iterdir():
            print(f"🗑️ Deleting {f.relative_to(root)}")
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
    else:
        results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Recomputing approximate Shapley values from exact precomputed utilities for {trainer_counts_str}...")
    benchmark_methods = (
        "monte_carlo,deterministic,antithetic_stratified,stratified,stratified_with_duplicates"
    )
    benchmark_logs_root = combined_logs_root / "runall"
    for trainer_count in valid_trainers:
        s_budget = _stratified_budget_for_trainer_count(trainer_count)
        sd_budget = _stratified_with_duplicates_budget_for_trainer_count(trainer_count)
        mc_budget = monte_carlo_budget_for_trainer_count(trainer_count)
        det_budget = deterministic_budget_for_trainer_count(trainer_count)
        a_budget = _antithetic_stratified_budget_for_trainer_count(trainer_count)
        method_sample_budgets = _format_method_sample_budgets(trainer_count)
        print(
            "🧮 Running trainer count "
            f"{trainer_count} with hardcoded method budgets "
            f"(mc={mc_budget}, "
            f"det={det_budget}, "
            f"a={a_budget}, "
            f"s={s_budget}, "
            f"s_dup={sd_budget})..."
        )
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "torch_trainer_scaling.shapley_benchmark",
                "--trainer-counts",
                str(trainer_count),
                "--methods",
                benchmark_methods,
                "--utility-source",
                "precomputed",
                "--sample-budget",
                str(DEFAULT_SAMPLE_BUDGET),
                "--method-sample-budgets",
                method_sample_budgets,
                "--ssk-max-missing-trainers",
                "3",
                "--combined-logs-dir",
                str(combined_logs),
                "--output-dir",
                str(benchmark_logs_root / f"trainers-{trainer_count}"),
            ],
            cwd=str(root),
            check=True,
        )
    
    print("\n📊 Running metrics and plots...")
    code_dir = combined_logs_root / "code"
    scripts = [
        "nrmse.py",
        "kendall_tau.py",
        "spearman_rho.py",
        "top_k_precision.py",
        "bias.py",
        "plot_nrmse.py",
        "plot_kendall_tau.py",
        "plot_spearman_rho.py",
        "plot_top_k_precision.py",
        "plot_bias.py"
    ]
    
    for script in scripts:
        script_path = code_dir / script
        if script_path.exists():
            print(f"📈 Running {script}...")
            command = ["uv", "run", "python", str(script_path)]
            if script.startswith("plot_"):
                metric_name = script.removeprefix("plot_").removesuffix(".py")
                command.extend(
                    [
                        "--input",
                        str(results_dir / f"{metric_name}_summary.json"),
                        "--output",
                        str(results_dir / f"{metric_name}_plot.png"),
                    ]
                )
            else:
                metric_name = script.removesuffix(".py")
                command.extend(
                    [
                        "--combined-logs-dir",
                        str(combined_logs),
                        "--output",
                        str(results_dir / f"{metric_name}_summary.json"),
                    ]
                )
            subprocess.run(command, cwd=str(root), check=True)
        else:
            print(f"⚠️  Missing {script_path.relative_to(root)}")

    print(f"\n✅ All done! Results saved in {results_dir.relative_to(root)}/")

if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "forge" / "mutation" / "run.py"
SPEC = importlib.util.spec_from_file_location("rizemind_mutation_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mutation_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mutation_runner
SPEC.loader.exec_module(mutation_runner)


def write_source(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("contract Example {}\n", encoding="utf-8")


def test_validate_policy_rejects_uncovered_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_source(tmp_path, "src/randomness/RNG.sol")
    write_source(tmp_path, "src/randomness/NewRNG.sol")
    monkeypatch.setattr(mutation_runner, "FORGE_ROOT", tmp_path)

    targets = {
        "randomness": mutation_runner.Target(
            name="randomness",
            paths=["src/randomness/RNG.sol"],
            min_score=80.0,
        )
    }

    with pytest.raises(SystemExit, match="NewRNG.sol"):
        mutation_runner.validate_policy(mutation_runner.Defaults(), targets)


def test_validate_policy_accepts_exact_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_source(tmp_path, "src/randomness/RNG.sol")
    write_source(tmp_path, "src/randomness/IRNG.sol")
    monkeypatch.setattr(mutation_runner, "FORGE_ROOT", tmp_path)

    defaults = mutation_runner.Defaults(excluded_paths=["src/randomness/IRNG.sol"])
    targets = {
        "randomness": mutation_runner.Target(
            name="randomness",
            paths=["src/randomness/RNG.sol"],
            min_score=80.0,
        )
    }

    mutation_runner.validate_policy(defaults, targets)

# ruff: noqa: E402, I001

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "torch_trainer_scaling"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from torch_trainer_scaling.coalition_dp_subset_interpolation import (  # noqa: E402
    _interpolate_size_effects,
)
from torch_trainer_scaling.coalition_experiment_dp import (  # noqa: E402
    DPReleaseConfig,
    _noise_std_for_coalition,
)


def test_noise_std_for_coalition_decreases_with_coalition_size() -> None:
    dp_config = DPReleaseConfig(
        clip_norm=1.0,
        noise_multiplier=2.0,
        noise_seed=7,
    )

    sigma_small = _noise_std_for_coalition(5, dp_config)
    sigma_large = _noise_std_for_coalition(10, dp_config)

    assert sigma_large < sigma_small
    assert sigma_small == pytest.approx(0.4)
    assert sigma_large == pytest.approx(0.2)


def test_interpolate_size_effects_fills_gaps_linearly() -> None:
    interpolated = _interpolate_size_effects(
        {0: 0.0, 2: 2.0, 4: 4.0},
        min_size=0,
        max_size=4,
    )

    assert interpolated == pytest.approx(
        {
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
            4: 4.0,
        }
    )

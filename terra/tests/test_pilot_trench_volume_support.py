import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).parents[2] / "tools" / "audit_pilot_trench_volume_support.py"
SPEC = importlib.util.spec_from_file_location("pilot_volume_support", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def test_selects_band_with_supported_interior_midpoint():
    volumes = {
        "straight": list(range(60, 100)),
        "segmented_2": list(range(64, 84)),
        "segmented_3": list(range(66, 86)),
    }
    candidates = support.eligible_bands(
        volumes,
        band_width=10,
        minimum_support_rate=0.10,
    )
    assert candidates
    selected = candidates[0]
    assert selected["minimum_support_rate"] >= 0.10
    assert selected["upper_inclusive"] - selected["lower_inclusive"] == 9


def test_rejects_band_without_common_support():
    volumes = {
        "straight": list(range(10, 20)),
        "segmented_2": list(range(30, 40)),
        "segmented_3": list(range(50, 60)),
    }
    assert not support.eligible_bands(
        volumes,
        band_width=10,
        minimum_support_rate=0.10,
    )


def test_requires_all_three_topologies():
    with pytest.raises(ValueError, match="expected topology keys"):
        support.eligible_bands(
            {"straight": [1], "segmented_2": [1]},
            band_width=1,
        )

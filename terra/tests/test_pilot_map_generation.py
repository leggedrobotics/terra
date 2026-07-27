from types import SimpleNamespace

import numpy as np
import pytest

from tools.pilot_map_generation import (
    APRON_CAPACITY_BANDS,
    APRON_NOMINAL_CAPACITY_RATIOS,
    APRON_SEPARATION_BAND_TILES,
    build_osm_apron_capacity_pair,
    sample_segmented_trench,
)


class _FakeBase:
    @staticmethod
    def rasterize_polyline(points, radius):
        assert points.shape in ((3, 2), (4, 2))
        assert radius == 1
        result = np.zeros((16, 16), dtype=bool)
        result.flat[:70] = True
        return result


class _FakeV3:
    @staticmethod
    def points_inside(points, margin):
        assert margin == 10
        return True


FAKE_V5 = SimpleNamespace(base=_FakeBase, v3=_FakeV3)


@pytest.mark.parametrize(
    ("segment_count", "length_range"),
    [(2, (10.0, 13.0)), (3, (7.0, 9.5))],
)
def test_segmented_sampler_is_deterministic(segment_count, length_range):
    first = sample_segmented_trench(
        FAKE_V5,
        np.random.default_rng(123),
        segment_count,
        length_range,
    )
    second = sample_segmented_trench(
        FAKE_V5,
        np.random.default_rng(123),
        segment_count,
        length_range,
    )
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[1]["audit_generator_draw_count"] == 1
    assert first[1]["audit_generator_rejections"] == {
        "segmented_out_of_bounds": 0,
        "segmented_volume_outside_prefilter": 0,
    }
    assert len(first[1]["segment_lengths_tiles"]) == segment_count
    assert len(first[1]["turn_angles_deg"]) == segment_count - 1


def test_segmented_sampler_rejects_invalid_contract():
    with pytest.raises(ValueError, match="unsupported segment count"):
        sample_segmented_trench(
            FAKE_V5,
            np.random.default_rng(1),
            4,
            (7.0, 9.5),
        )
    with pytest.raises(ValueError, match="invalid segment length range"):
        sample_segmented_trench(
            FAKE_V5,
            np.random.default_rng(1),
            2,
            (10.0, 10.0),
        )


def _foundation_dig():
    dig = np.zeros((64, 64), dtype=np.bool_)
    dig[28:36, 27:37] = True
    return dig


def test_osm_apron_capacity_pair_is_deterministic():
    first = build_osm_apron_capacity_pair(
        _foundation_dig(),
        "pilot:osm:source-001",
    )
    second = build_osm_apron_capacity_pair(
        _foundation_dig(),
        "pilot:osm:source-001",
    )
    assert first["dig_identity_sha256"] == second["dig_identity_sha256"]
    for capacity_token in APRON_CAPACITY_BANDS:
        np.testing.assert_array_equal(
            first["variants"][capacity_token]["target"],
            second["variants"][capacity_token]["target"],
        )
        assert (
            first["variants"][capacity_token]["metadata"]
            == second["variants"][capacity_token]["metadata"]
        )


def test_osm_apron_capacity_pair_preserves_exact_dig_and_source_group():
    dig = _foundation_dig()
    pair = build_osm_apron_capacity_pair(
        dig,
        "pilot:osm:source-002",
    )
    for variant in pair["variants"].values():
        np.testing.assert_array_equal(variant["target"] < 0, dig)
        assert variant["metadata"]["source_family"] == "osm"
        assert variant["metadata"]["source_group_id"] == pair["source_group_id"]
        assert variant["metadata"]["dig_identity_sha256"] == pair["dig_identity_sha256"]


def test_osm_apron_capacity_pair_hits_closed_capacity_and_separation_bands():
    pair = build_osm_apron_capacity_pair(
        _foundation_dig(),
        "pilot:osm:source-003",
    )
    assert (
        pair["variants"]["slcap07_10"]["metadata"]["nominal_single_layer_area_ratio"]
        == 8.5
    )
    for capacity_token, variant in pair["variants"].items():
        metadata = variant["metadata"]
        lower, upper = APRON_CAPACITY_BANDS[capacity_token]
        assert (
            metadata["nominal_single_layer_area_ratio"]
            == APRON_NOMINAL_CAPACITY_RATIOS[capacity_token]
        )
        assert lower <= metadata["achieved_single_layer_area_ratio"] <= upper
        separation_lower, separation_upper = APRON_SEPARATION_BAND_TILES
        assert separation_lower <= metadata["separation_p50_tiles"] <= separation_upper

from types import SimpleNamespace

import numpy as np
import pytest

from tools.pilot_map_generation import sample_segmented_trench


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

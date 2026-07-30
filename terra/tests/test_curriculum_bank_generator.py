from types import SimpleNamespace

import numpy as np
import pytest

from terra.maps_buffer import reset_array_scenario_sha256
from tools.map_generation import generate_curriculum_bank as generator


def _dig(extra_cell=None):
    dig = np.zeros((16, 16), dtype=np.bool_)
    dig[5:9, 5:9] = True
    if extra_cell is not None:
        dig[extra_cell] = True
    return dig


def _sample(**metadata):
    shape = (4, 4)
    return SimpleNamespace(
        target=np.zeros(shape, dtype=np.int8),
        occupancy=np.zeros(shape, dtype=np.bool_),
        dumpability=np.ones(shape, dtype=np.bool_),
        action=np.zeros(shape, dtype=np.int8),
        distance=np.zeros(shape, dtype=np.float32),
        metadata=metadata,
    )


def test_dig_admission_allows_similar_nonidentical_masks():
    bank = object.__new__(generator.DigBankV10)
    bank.t0_levels = frozenset()
    candidate = _dig(extra_cell=(9, 8))
    reference = _dig()

    assert generator.centred_iou(candidate, reference) > 0.9
    assert bank._acceptable("slab", candidate, {}, [(reference, {})]) == ""


def test_dig_admission_rejects_exact_duplicate_and_source_reuse():
    bank = object.__new__(generator.DigBankV10)
    bank.t0_levels = frozenset()
    dig = _dig()

    assert (
        bank._acceptable("slab", dig.copy(), {}, [(dig, {})])
        == "dig_bank_exact_duplicate"
    )
    assert (
        bank._acceptable(
            "slab",
            _dig(extra_cell=(9, 8)),
            {"foundation_source_index": 7},
            [(dig, {"foundation_source_index": 7})],
        )
        == "dig_bank_source_reuse"
    )


def test_sample_indices_do_not_collide_at_large_bank_sizes():
    indices = {
        generator.sample_index_of(condition_index, map_index)
        for condition_index in range(32)
        for map_index in range(256)
    }
    assert len(indices) == 32 * 256
    assert generator.sample_index_of(1, 0) != generator.sample_index_of(0, 128)

    with pytest.raises(ValueError):
        generator.sample_index_of(-1, 0)
    with pytest.raises(ValueError):
        generator.sample_index_of(0, 1000)


def test_review_subset_is_bounded_deterministic_and_spans_bank():
    selected = generator.review_map_indices(64, 16)
    assert len(selected) == 16
    assert min(selected) == 0
    assert max(selected) == 63
    assert selected == generator.review_map_indices(64, 16)
    assert generator.review_map_indices(4, 16) == frozenset(range(4))
    assert generator.review_map_indices(64, 0) == frozenset()


def test_scenario_identity_covers_every_reset_array():
    original = _sample()
    same = _sample()
    changed = _sample()
    changed.dumpability[0, 0] = False

    canonical = reset_array_scenario_sha256(
        {
            name: getattr(original, attribute)
            for name, attribute in generator.ARRAY_FOLDERS.items()
        }
    )
    assert generator.scenario_sha256(original) == canonical
    assert generator.scenario_sha256(original) == generator.scenario_sha256(same)
    assert generator.scenario_sha256(original) != generator.scenario_sha256(changed)


def test_source_group_uses_raw_foundation_source_or_realized_dig():
    raw_source = "b" * 64
    assert (
        generator.source_group_id(
            _sample(
                foundation_source_sha256=raw_source,
                dig_sha256="c" * 64,
            )
        )
        == f"foundation-source:{raw_source}"
    )
    assert (
        generator.source_group_id(_sample(dig_sha256="d" * 64))
        == f"dig:{'d' * 64}"
    )


def test_normalized_dig_identities_separate_translation_from_shape():
    original = _dig()
    translated = np.roll(original, shift=(2, 3), axis=(0, 1))
    reflected = np.fliplr(original)
    changed = _dig(extra_cell=(9, 8))

    assert generator._mask_identity(original) != generator._mask_identity(translated)
    assert (
        generator._translation_normalized_identity(original)
        == generator._translation_normalized_identity(translated)
    )
    assert (
        generator._dihedral_normalized_identity(original)
        == generator._dihedral_normalized_identity(reflected)
    )
    assert (
        generator._dihedral_normalized_identity(original)
        != generator._dihedral_normalized_identity(changed)
    )


def test_digital_perimeter_gives_bounded_compactness_for_square():
    square = np.zeros((8, 8), dtype=np.bool_)
    square[2:6, 2:6] = True
    perimeter = generator._digital_perimeter(square)
    compactness = 4.0 * np.pi * square.sum() / perimeter**2

    assert perimeter == 16
    assert 0.0 < compactness <= 1.0


def test_exact_full_scenario_duplicate_fails_loudly():
    identity = "a" * 64
    with pytest.raises(RuntimeError, match="map-a and map-b"):
        generator.assert_unique_scenario_rows(
            [
                {"map_id": "map-a", "scenario_sha256": identity},
                {"map_id": "map-b", "scenario_sha256": identity},
            ]
        )

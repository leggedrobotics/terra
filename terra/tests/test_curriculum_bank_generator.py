import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.maps_buffer import reset_array_scenario_sha256
from tools.map_generation import curriculum_taxonomy as taxonomy
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


def test_single_cell_dig_is_serviceable_in_all_admission_gates():
    target = np.ones((64, 64), dtype=np.int8)
    target[32, 32] = -1
    occupancy = np.zeros_like(target, dtype=np.bool_)
    dumpability = np.ones_like(target, dtype=np.bool_)

    assert (
        generator.tsvc.direct_service_coverage(
            target,
            occupancy,
            dumpability,
        )
        == 1.0
    )
    coverage = generator.tdump.measure(
        target,
        occupancy,
        dumpability,
        full=False,
    )
    assert coverage["dig_cov_any"] == 1.0
    assert coverage["turn_dump_cov_any"] == 1.0
    plan = generator.tdump.plan_sensitivity(
        target,
        occupancy,
        dumpability,
    )
    assert plan["plan_cost_near"] == 0.0
    assert plan["plan_cost_far"] == 0.0


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


def test_generator_owns_intersections_and_uses_the_15_degree_lattice():
    horizontal = np.asarray([[32.0, 15.0], [32.0, 49.0]])
    vertical = np.asarray([[15.0, 32.0], [49.0, 32.0]])
    dig = generator.v9.rasterize_segments(horizontal, 1.0)
    dig |= generator.v9.rasterize_segments(vertical, 1.0)
    target = np.where(dig, -1, 0).astype(np.int8)
    metadata = {
        "trench_half_width_tiles": 1.0,
        "trench_arms": [horizontal.tolist(), vertical.tolist()],
        "axes_ABC": [
            generator.v3.line_coefficients(horizontal[0], horizontal[-1]),
            generator.v3.line_coefficients(vertical[0], vertical[-1]),
        ],
    }

    owners = generator.v9.trench_axis_owners(target, metadata)

    assert owners.dtype == np.uint8
    assert np.all(owners[target < 0] != 0)
    assert owners[32, 20] == 1
    assert owners[20, 32] == 2
    assert owners[32, 32] == 3
    assert generator.TRENCH_AXES_DEG == tuple(
        float(value) for value in range(0, 180, 15)
    )
    empty = np.zeros((64, 64), dtype=np.bool_)
    open_lane = np.ones((64, 64), dtype=np.bool_)
    for axis_deg in generator.TRENCH_AXES_DEG:
        aligned = generator.tsvc.heading_indices_for_axis(axis_deg)
        expected_count = 2 if axis_deg % 30.0 == 15.0 else 1
        assert len(aligned) == expected_count
        drive = generator.tsvc.backward_drive_check(
            empty,
            empty,
            axis_deg,
            (32.0, 32.0),
            20.0,
            open_lane,
            1,
        )
        assert drive["backward_drive_footprint_clear"]
        assert (
            drive["backward_drive_drift_per_tile"]
            <= generator.v9.BACKWARD_DRIFT_PER_TILE_MAX
        )


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


def test_trench_map_rerolls_layout_after_attempt_exhaustion(monkeypatch):
    condition = SimpleNamespace(
        id="trench-condition",
        planning=False,
        family="trench",
        dig_bank_level="straight",
    )
    dataset = SimpleNamespace()
    layout_indices = []
    salts = []

    class Bank:
        def get(self, _level, _map_index, salt):
            salts.append(salt)
            return _dig(), {}

    def fake_layout(_condition, map_index):
        layout_indices.append(map_index)
        return map_index

    def fake_make_map(_condition, _dataset, _dig, _meta, layout, _rng):
        if layout == 0:
            return None, "plan_start_side_contract"
        return _sample(), ""

    monkeypatch.setattr(generator, "layout_for", fake_layout)
    monkeypatch.setattr(generator, "make_map", fake_make_map)

    samples, rejections, failures = generator.generate_condition(
        condition,
        dataset,
        condition_index=3,
        bank=Bank(),
        n_maps=1,
        max_attempts=2,
    )

    assert failures == []
    assert len(samples) == 1
    assert layout_indices == [0, 1]
    assert salts == [0, 0, 0]
    assert rejections == {
        "plan_start_side_contract": 2,
        "layout_reroll_after_exhaustion": 1,
    }
    assert samples[0].metadata["attempt"] == 2
    assert samples[0].metadata["layout_reroll_round"] == 1
    assert samples[0].metadata["layout_map_index"] == 1


def test_planning_map_layout_search_is_bounded(monkeypatch):
    condition = SimpleNamespace(
        id="planning-condition",
        planning=True,
        family="foundation",
        dig_bank_level="straight",
    )

    class Bank:
        def get(self, _level, _map_index, _salt):
            return _dig(), {}

    monkeypatch.setattr(generator, "layout_for", lambda _condition, index: index)
    monkeypatch.setattr(
        generator,
        "make_map",
        lambda *_args: (None, "plan_start_side_contract"),
    )

    samples, rejections, failures = generator.generate_condition(
        condition,
        SimpleNamespace(),
        condition_index=3,
        bank=Bank(),
        n_maps=1,
        max_attempts=2,
    )

    assert samples == []
    assert rejections == {
        "plan_start_side_contract": 10,
        "layout_reroll_after_exhaustion": 4,
    }
    assert len(failures) == 1
    assert "no accepted sample in 10 attempts" in failures[0]


def test_parallel_collection_replays_only_a_duplicate_rerolled_dig(monkeypatch):
    condition = SimpleNamespace(
        id="condition",
        planning=False,
        family="foundation",
        dig_bank_level="slab",
    )
    first = _dig()
    second = _dig(extra_cell=(9, 8))

    class Bank:
        def get(self, _level, map_index, salt):
            if map_index == 0 or salt == 1:
                return first, {}
            return second, {}

    def fake_make_map(_condition, _dataset, dig, _meta, _layout, _rng):
        sample = _sample()
        sample.target = np.where(dig, -1, 0).astype(np.int8)
        return sample, ""

    monkeypatch.setattr(generator.v9, "SHARED_DIG_ATTEMPTS", 0)
    monkeypatch.setattr(generator.v9, "REROLL_DUMP_ATTEMPTS", 1)
    monkeypatch.setattr(generator, "layout_for", lambda *_args: None)
    monkeypatch.setattr(generator, "make_map", fake_make_map)

    samples, rejections, failures = generator.generate_condition(
        condition,
        SimpleNamespace(),
        condition_index=0,
        bank=Bank(),
        n_maps=2,
        max_attempts=2,
    )

    assert failures == []
    assert len(samples) == 2
    assert not np.array_equal(samples[0].target, samples[1].target)
    assert [sample.metadata["shared_dig"] for sample in samples] == [0, 0]
    assert [sample.metadata["attempt"] for sample in samples] == [0, 1]
    assert rejections == {"condition_dig_exact_duplicate": 1}


def test_current_taxonomy_document_matches_executable_registry():
    repository = Path(__file__).resolve().parents[2]
    document = repository / taxonomy.SPEC_PATH
    assert document.is_file(), taxonomy.SPEC_PATH
    assert taxonomy.spec_path_for("v6-main") == taxonomy.SPEC_PATH
    assert {
        taxonomy.spec_path_for(release)
        for release in ("v3", "v4", "v5-main", "v5-transport")
    } == {taxonomy.HISTORICAL_SPEC_PATH}
    assert generator.CURRENT_TAXONOMY_PATH == taxonomy.SPEC_PATH

    text = document.read_text()
    table = text.split("<!-- taxonomy:v6-main:start -->", 1)[1].split(
        "<!-- taxonomy:v6-main:end -->", 1
    )[0]
    rows = re.findall(
        r"^\| `([^`]+)` \| T(\d+) \| [^|]+ \| (?:`([^`]+)`|-) \|",
        table,
        flags=re.MULTILINE,
    )
    documented = [
        (condition_id, int(tier), anchor or None)
        for condition_id, tier, anchor in rows
    ]
    executable = [
        (condition_id, tier, anchor)
        for _, condition_id, tier, anchor in taxonomy.SPEC_TABLE_V6_MAIN
    ]

    assert documented == executable
    assert len(documented) == 32
    assert {tier: sum(row[1] == tier for row in documented) for tier in range(3)} == {
        0: 9,
        1: 19,
        2: 4,
    }


def test_generated_readme_points_to_current_taxonomy(tmp_path):
    generator.write_readme(tmp_path, generator.DATASETS["main"], {})

    readme = (tmp_path / "README.md").read_text()
    assert f"`{taxonomy.SPEC_PATH}`" in readme
    assert "docs/CURRICULUM_SPEC_V6.md" not in readme

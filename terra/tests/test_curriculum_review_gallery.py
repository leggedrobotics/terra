import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.map_generation import export_review_gallery as gallery


def _registry_row(
    condition_id,
    *,
    tier=0,
    anchor="",
    geometry="slab",
    dump="ring3x",
    capacity="generous",
    site="clean",
    distance="unspec",
    scale="std",
):
    return {
        "condition_id": condition_id,
        "family": "foundation",
        "tier": str(tier),
        "anchor_condition_id": anchor,
        "geometry": geometry,
        "dump": dump,
        "capacity": capacity,
        "site": site,
        "distance": distance,
        "scale": scale,
    }


def test_conditions_are_grouped_by_literal_anchor_delta():
    anchor = _registry_row("anchor")
    registry = {
        "anchor": anchor,
        "capacity": _registry_row("capacity", tier=1, anchor="anchor", capacity="c2x"),
        "distance": _registry_row("distance", tier=1, anchor="anchor", distance="d12"),
        "dump": _registry_row("dump", tier=1, anchor="anchor", dump="side1"),
        "geometry": _registry_row(
            "geometry", tier=1, anchor="anchor", geometry="strips"
        ),
        "site": _registry_row("site", tier=1, anchor="anchor", site="obj"),
        "composed": _registry_row(
            "composed", tier=2, anchor="anchor", dump="side1", site="obj"
        ),
    }

    assert gallery.classify_branch(anchor, registry) == "anchor_easy"
    assert gallery.classify_branch(registry["capacity"], registry) == "dump_capacity"
    assert gallery.classify_branch(registry["distance"], registry) == "dump_distance"
    assert gallery.classify_branch(registry["dump"], registry) == "dump_layout"
    assert (
        gallery.classify_branch(registry["geometry"], registry) == "geometry_topology"
    )
    assert gallery.classify_branch(registry["site"], registry) == "site_constraints"
    assert gallery.classify_branch(registry["composed"], registry) == "composed"

    missing_anchor = _registry_row("bad", tier=1, anchor="absent", site="obj")
    with pytest.raises(ValueError, match="anchor is absent"):
        gallery.classify_branch(missing_anchor, {"bad": missing_anchor})


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_export_copies_or_renders_examples_and_pins_identities(tmp_path):
    bank = tmp_path / "bank"
    output = tmp_path / "review"
    condition_id = "fnd-slab-ring3x"
    condition_dir = bank / condition_id
    previews = condition_dir / "previews"
    previews.mkdir(parents=True)

    registry = _registry_row(condition_id)
    _write_csv(bank / "conditions.csv", [registry])

    maps = []
    root_rows = []
    for index in range(2):
        map_id = f"map-{index}"
        identity = str(index) * 64
        arrays = {}
        for name in ("images", "occupancy", "dumpability"):
            relative = f"dataset/{name}/img_{index}.npy"
            path = bank / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "images":
                value = np.zeros((4, 4), dtype=np.int8)
                value[1, 1] = -1
                value[2, 2] = 1
            elif name == "occupancy":
                value = np.zeros((4, 4), dtype=np.bool_)
            else:
                value = np.ones((4, 4), dtype=np.bool_)
            np.save(path, value)
            arrays[name] = relative

        maps.append(
            {
                "id": map_id,
                "mapIndex": index,
                "sourceGroupId": f"slab:{index}",
                "scenarioSha256": identity,
                "digCells": 1,
                "dumpCells": 1,
                "capacityRatio": 1.0,
                "objectCount": 0,
                "arrays": arrays,
            }
        )
        root_rows.append(
            {
                "condition_id": condition_id,
                "map_id": map_id,
                "map_index": str(index),
                "source_group_id": f"slab:{index}",
                "scenario_sha256": identity,
            }
        )

    copied_preview = Image.new("RGB", (10, 10), "red")
    copied_preview.save(previews / "00__map-0.png")
    source_preview_bytes = (previews / "00__map-0.png").read_bytes()
    _write_csv(bank / "manifest.csv", root_rows)
    (condition_dir / "manifest.json").write_text(
        json.dumps(
            {
                "conditionId": condition_id,
                "family": "foundation",
                "tier": 0,
                "anchorConditionId": None,
                "factorLevels": {
                    "geometry": "slab",
                    "dump": "ring3x",
                    "capacity": "generous",
                    "site": "clean",
                    "distance": "unspec",
                    "scale": "std",
                },
                "mapCount": 2,
                "maps": maps,
            }
        )
    )

    gallery.export_review_gallery(bank, output)

    examples = sorted(
        (output / "00_anchor_easy" / condition_id / "examples").glob("*.png")
    )
    assert len(examples) == 2
    assert examples[0].read_bytes() == source_preview_bytes
    assert Image.open(examples[1]).size == (512, 512)
    assert (output / "00_anchor_easy" / condition_id / "overview.png").is_file()

    with (output / "index.csv").open(newline="") as handle:
        index = list(csv.DictReader(handle))
    assert [row["scenario_sha256"] for row in index] == ["0" * 64, "1" * 64]
    assert all(row["decision"] == "" and row["comment"] == "" for row in index)
    markdown = (output / "index.md").read_text()
    assert "## Anchor / easy" in markdown
    assert "one-axis" not in markdown.lower()


def test_representative_subset_is_bounded_and_spans_the_condition():
    positions = gallery.representative_positions(64)
    assert len(positions) == 16
    assert positions[0] == 0
    assert positions[-1] == 63
    assert gallery.representative_positions(4) == [0, 1, 2, 3]

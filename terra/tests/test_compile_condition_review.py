import hashlib
import json

import pytest

from tools.map_generation import compile_condition_review as review


def _review_data():
    return {
        "schema": "terra_digging_review_v1",
        "release": {
            "id": review.RELEASE_ID,
            "manifestHash": review.MANIFEST_SHA256,
        },
        "cells": [
            {
                "id": condition_id,
                "family": "foundation" if condition_id.startswith("fnd-") else "trench",
            }
            for condition_id in review.CONDITION_IDS
        ],
        "scenarios": [
            {"id": "map-0", "hashes": {"scenario": "b" * 64}},
        ],
    }


def _condition(condition_id, decision="accept"):
    return {
        "schema": review.CONDITION_SCHEMA,
        "datasetId": review.RELEASE_ID,
        "release": review.RELEASE_ID,
        "manifestHash": review.MANIFEST_SHA256,
        "conditionId": condition_id,
        "decision": decision,
        "comment": "",
        "reviewer": "Lorenzo",
        "updatedAt": "2026-07-31T18:00:00Z",
    }


def _all_conditions(overrides=None):
    overrides = overrides or {}
    return [
        _condition(condition_id, overrides.get(condition_id, "accept"))
        for condition_id in review.CONDITION_IDS
    ]


def _write_inputs(tmp_path, monkeypatch, records):
    review_data = tmp_path / "review-data.json"
    decisions = tmp_path / "review.jsonl"
    review_data.write_text(json.dumps(_review_data()))
    decisions.write_text("".join(json.dumps(record) + "\n" for record in records))
    monkeypatch.setattr(
        review,
        "REVIEW_DATA_SHA256",
        hashlib.sha256(review_data.read_bytes()).hexdigest(),
    )
    return review_data, decisions


def test_compile_emits_only_explicit_accepted_conditions(tmp_path, monkeypatch):
    map_record = {
        "schema": review.MAP_SCHEMA,
        "datasetId": review.RELEASE_ID,
        "release": review.RELEASE_ID,
        "manifestHash": review.MANIFEST_SHA256,
        "scenarioId": "map-0",
        "scenarioHash": "b" * 64,
    }
    review_data, decisions = _write_inputs(
        tmp_path,
        monkeypatch,
        [*_all_conditions(), map_record],
    )

    receipt = review.compile_condition_review(
        review_data, decisions, tmp_path / "compiled"
    )

    accepted = sorted(review.CONDITION_IDS)
    assert receipt["accepted_conditions"] == accepted
    assert receipt["map_record_count"] == 1
    assert (tmp_path / "compiled" / "accepted_conditions.txt").read_text() == (
        ",".join(accepted) + "\n"
    )
    persisted = json.loads(
        (tmp_path / "compiled" / "review_admission.json").read_text()
    )
    assert persisted == receipt


def test_compile_rejects_incomplete_or_implicit_condition_review(tmp_path, monkeypatch):
    records = _all_conditions()[:-1]
    review_data, decisions = _write_inputs(tmp_path, monkeypatch, records)
    with pytest.raises(ValueError, match="1 missing"):
        review.compile_condition_review(review_data, decisions, tmp_path / "compiled")


def test_compile_requires_accepted_anchor_for_each_family(tmp_path, monkeypatch):
    overrides = {
        condition_id: "quarantine"
        for condition_id in review.ANCHOR_IDS
        if condition_id.startswith("trn-")
    }
    review_data, decisions = _write_inputs(
        tmp_path, monkeypatch, _all_conditions(overrides)
    )
    with pytest.raises(ValueError, match="00-anchor for: trench"):
        review.compile_condition_review(review_data, decisions, tmp_path / "compiled")


def test_compile_rejects_stale_manifest(tmp_path, monkeypatch):
    records = _all_conditions()
    records[0]["manifestHash"] = "c" * 64
    review_data, decisions = _write_inputs(tmp_path, monkeypatch, records)
    with pytest.raises(ValueError, match="manifest hash does not match"):
        review.compile_condition_review(review_data, decisions, tmp_path / "compiled")


def test_compile_rejects_modified_review_data(tmp_path, monkeypatch):
    review_data, decisions = _write_inputs(tmp_path, monkeypatch, _all_conditions())
    monkeypatch.setattr(review, "REVIEW_DATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="bytes do not match"):
        review.compile_condition_review(review_data, decisions, tmp_path / "compiled")

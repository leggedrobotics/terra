"""Gallery evidence labels and filters work without importing a simulator."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from terra.postprocess.gallery import build


def case(identity, **changes):
    return {
        "id": identity,
        "title": identity,
        "fleet": "mixed",
        "generation": "current",
        "native_status": "success",
        "postprocessed_status": "not_run",
        **changes,
    }


class Cards(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.rows = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.rows.append(attrs)


class GalleryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output" / "index.html"

    def test_all_results_visible_without_recordings_or_javascript(self):
        manifest = {
            "title": "Recorded outcomes",
            "cases": [
                case("native-success"),
                case("missing-failure", native_status="failed"),
                case(
                    "legacy-refinement",
                    generation="legacy",
                    postprocessed_status="failed",
                ),
                case(
                    "synthetic",
                    generation="synthetic",
                    native_status="unknown",
                    postprocessed_status="candidate",
                ),
            ],
        }
        original = deepcopy(manifest)
        self.assertEqual(build(manifest, self.output), self.output)
        self.assertEqual(manifest, original)
        page = self.output.read_text()
        rows = Cards(page).rows
        self.assertEqual(
            [r["data-id"] for r in rows], [c["id"] for c in manifest["cases"]]
        )
        self.assertTrue(all("hidden" not in r for r in rows))
        self.assertIn("Native incomplete", page)
        self.assertIn("Postprocessing unresolved", page)
        self.assertIn("Geometric candidate", page)
        self.assertEqual(page.count("No replay recorded"), 4)
        self.assertNotIn('src="http', page)

    def test_local_evidence_links_rebase_and_labels_are_escaped(self):
        source = self.root / "input"
        source.mkdir()
        (source / "replay & final.html").write_text("recording")
        manifest = {
            "title": "<Fleet>",
            "description": "Native & refined",
            "cases": [
                case(
                    'id"<',
                    title="<script>bad()</script>",
                    native_url="replay & final.html",
                    note="Do not hide <failure>",
                    metrics={"Soil & load": "0 / 230"},
                )
            ],
        }
        path = source / "gallery.json"
        path.write_text(json.dumps(manifest))
        build(path, self.output)
        page = self.output.read_text()
        self.assertIn("../input/replay%20%26%20final.html", page)
        self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", page)
        self.assertNotIn("<script>bad()", page)
        self.assertIn("Soil &amp; load", page)
        self.assertEqual(Cards(page).rows[0]["data-id"], 'id"<')

    def test_template_tokens_in_manifest_stay_literal(self):
        build(
            {
                "title": "Review {{CARDS}}",
                "description": "Read {{SUMMARY}}",
                "cases": [case("only", note="Literal {{TITLE}}")],
            },
            self.output,
        )
        page = self.output.read_text()
        self.assertIn("<h1>Review {{CARDS}}</h1>", page)
        self.assertIn("Read {{SUMMARY}}", page)
        self.assertIn("Literal {{TITLE}}", page)
        self.assertEqual(len(Cards(page).rows), 1)

    def test_bad_outcomes_and_missing_or_unsafe_links_fail_before_writing(self):
        bad_cases = [
            case("bad", native_status="probably complete"),
            case("bad", postprocessed_status="maybe"),
            case("bad", native_url="missing.html"),
            case("bad", native_url="javascript:alert(1)"),
            case("bad", postprocessed_url="https://example.com/replay.html"),
        ]
        for row in bad_cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                build({"cases": [row]}, self.output)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build({"cases": [case("same"), case("same")]}, self.output)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(
        shutil.which("node"), "Node is needed for the browser filter test"
    )
    def test_filters_keep_native_success_separate_from_postprocessing_failure(self):
        build({"cases": []}, self.output)
        script = re.search(
            r"<script>(.*?)</script>", self.output.read_text(), re.S
        ).group(1)
        check = """
const assert = require('node:assert/strict');
const filters = {fleet:'all', generation:'all', stage:'all', outcome:'all', query:''};
const row = {fleet:'mixed', generation:'legacy', native:'success', postprocessed:'failed', search:'case161 u1500'};
assert.equal(matchesCase(row, filters), true);
assert.equal(matchesCase(row, {...filters, stage:'native', outcome:'success'}), true);
assert.equal(matchesCase(row, {...filters, stage:'native', outcome:'failed'}), false);
assert.equal(matchesCase(row, {...filters, stage:'postprocessed', outcome:'failed'}), true);
assert.equal(matchesCase(row, {...filters, generation:'current'}), false);
assert.equal(matchesCase(row, {...filters, fleet:'two_excavators'}), false);
assert.equal(matchesCase(row, {...filters, query:' U1500 '}), true);
assert.equal(matchesCase(row, {...filters, query:'u16867'}), false);
const noRecording = {...row, native:'failed', postprocessed:''};
assert.equal(matchesCase(noRecording, {...filters, outcome:'failed'}), true);
assert.equal(matchesCase(noRecording, {...filters, stage:'postprocessed'}), false);
const candidate = {...row, native:'unknown', postprocessed:'unverified'};
assert.equal(matchesCase(candidate, {...filters, stage:'postprocessed', outcome:'success'}), false);
assert.equal(matchesCase(candidate, {...filters, stage:'postprocessed', outcome:'unverified'}), true);
"""
        result = subprocess.run(
            ["node", "-e", script + check], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

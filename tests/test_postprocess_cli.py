"""The portable command writes reviewable results for accepted and failed plans."""

import json
from pathlib import Path
import subprocess
import sys

from terra.postprocess.cli import main

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "terra/postprocess/examples/fleet_two_excavators.json"
)


def test_fleet_command_writes_both_pages_and_preserves_source(tmp_path, capsys):
    out = tmp_path / "fleet"
    assert main(["fleet", str(EXAMPLE), "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "GEOMETRIC_CANDIDATE"
    assert report["material_events"] == 4
    for name in ("original_page", "postprocessed_page"):
        text = Path(report["files"][name]).read_text()
        assert "terra.postprocessed.v1" in text
        assert "__BUNDLE__" not in text
    source = json.loads(EXAMPLE.read_text())
    assert len(source["agents"]) == 2
    assert (out / "original.json.gz").is_file()
    assert (out / "postprocessed.json.gz").is_file()


def test_failed_refinement_still_writes_diagnostic_pages(tmp_path, capsys):
    source = json.loads(EXAMPLE.read_text())
    for agent in source["agents"]:
        agent["geometry"]["tool_width_m"] = 3.0
    input_path = tmp_path / "source.json"
    input_path.write_text(json.dumps(source))
    out = tmp_path / "rejected"
    assert main(["fleet", str(input_path), "--out", str(out)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INCOMPLETE_WORKSPACE_REFINEMENT"
    assert result["material_events"] == 4
    assert (out / "original.html").is_file()
    assert "INCOMPLETE_WORKSPACE_REFINEMENT" in (out / "postprocessed.html").read_text()


def test_render_rejects_unknown_schema_without_creating_output(tmp_path, capsys):
    source = tmp_path / "invalid.json"
    source.write_text('{"schema": "unknown"}')
    out = tmp_path / "invalid.html"
    assert main(["render", str(source), "--out", str(out)]) == 1
    assert not out.exists()
    assert (
        "Expected terra.viewer3d.v1 or terra.postprocessed.v1"
        in capsys.readouterr().err
    )


def test_help_does_not_import_simulator_or_renderer():
    code = """
import sys
from terra.postprocess.cli import parser
parser().parse_args(['render', 'recording.json', '--out', 'page.html'])
assert not any(name.split('.')[0] in {'jax', 'rclpy', 'numpy', 'scipy'} for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

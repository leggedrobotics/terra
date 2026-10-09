"""Build an offline fleet gallery from explicit native/postprocessing outcomes."""

from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

FLEETS = {
    "mixed": "Excavator + skid steer",
    "two_excavators": "Two excavators",
    "solo": "One excavator",
}
GENERATIONS = {
    "current": "Current checkpoint",
    "legacy": "Earlier example",
    "synthetic": "Synthetic example",
}
NATIVE = {
    "success": ("Native complete", "success"),
    "failed": ("Native incomplete", "failed"),
    "unknown": ("Native outcome unknown", "unverified"),
}
POSTPROCESSED = {
    "not_run": ("Postprocessing not run", ""),
    "candidate": ("Geometric candidate", "unverified"),
    "valid": ("Valid under modeled rules", "success"),
    "failed": ("Postprocessing unresolved", "failed"),
}


def _text(value):
    return escape(str(value), quote=True)


def _url(value, base, destination):
    """Keep web links; rebase local links relative to the generated HTML."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Gallery links must be nonempty strings")
    parts = urlsplit(value)
    if parts.scheme in ("http", "https") and parts.netloc:
        return value
    if parts.scheme or parts.netloc:
        raise ValueError(f"Unsupported gallery link: {value}")
    if not parts.path:
        return value
    path = Path(unquote(parts.path))
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"Gallery link does not exist: {path}")
    relative = os.path.relpath(path.resolve(), destination.parent)
    return urlunsplit(("", "", quote(relative, safe="/"), parts.query, parts.fragment))


def _link(label, value, base, destination, css=""):
    href = _text(_url(value, base, destination))
    return f'<a class="{css}" href="{href}">{_text(label)}</a>'


def _evaluations(evaluations, base, destination):
    if not isinstance(evaluations, list):
        raise ValueError("Gallery evaluations must be a list")
    if not evaluations:
        return ""
    panels = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ValueError("Gallery evaluations must be objects")
        title = evaluation.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Each gallery evaluation requires a nonempty title")
        completed, total = evaluation.get("completed"), evaluation.get("total")
        if (
            type(completed) is not int
            or type(total) is not int
            or not 0 <= completed <= total
            or total <= 0
        ):
            raise ValueError(
                "Evaluation counts must be integers with 0 <= completed <= total and total > 0"
            )
        note = evaluation.get("note", "")
        if not isinstance(note, str):
            raise ValueError("Gallery evaluation notes must be strings")
        source = _link("Panel results", evaluation.get("source_url"), base, destination)
        panels.append(
            f'<div class="evaluation"><h3>{_text(title)}</h3>'
            f'<p class="evaluation-rate"><strong>{completed} / {total}</strong> complete'
            f" <span>({100 * completed / total:.1f}%)</span></p>"
            f'<p class="note">{_text(note)}</p>{source}</div>'
        )
    return (
        '<section class="evaluations" aria-labelledby="evaluations-title">'
        '<h2 id="evaluations-title">Evaluation panels</h2>'
        '<p class="evaluation-context">Full-panel results; gallery filters do not change these rates.</p>'
        f'<div class="evaluation-grid">{"".join(panels)}</div></section>'
    )


def _card(case, base, destination):
    identity, title = case["id"], case["title"]
    native_label, native_outcome = NATIVE[case["native_status"]]
    post_label, post_outcome = POSTPROCESSED[case["postprocessed_status"]]
    checkpoint = str(case.get("checkpoint", ""))
    note = str(case.get("note", ""))
    search = " ".join(
        (identity, title, checkpoint, note, FLEETS[case["fleet"]])
    ).lower()
    attrs = {
        "id": identity,
        "fleet": case["fleet"],
        "generation": case["generation"],
        "native": native_outcome,
        "postprocessed": post_outcome,
        "search": search,
    }
    attributes = " ".join(f'data-{k}="{_text(v)}"' for k, v in attrs.items())
    thumbnail = ""
    if case.get("thumbnail"):
        src = _text(_url(case["thumbnail"], base, destination))
        thumbnail = (
            f'<img class="thumbnail" src="{src}" alt="{_text(title)}" loading="lazy">'
        )
    metrics = "".join(
        f"<div><dt>{_text(label)}</dt><dd>{_text(value)}</dd></div>"
        for label, value in case.get("metrics", {}).items()
    )
    primary = []
    if case.get("native_url"):
        primary.append(
            _link(
                case.get("native_label", "Native replay"),
                case["native_url"],
                base,
                destination,
                "open",
            )
        )
    if case.get("postprocessed_url"):
        primary.append(
            _link(
                "Postprocessed replay",
                case["postprocessed_url"],
                base,
                destination,
                "open secondary",
            )
        )
    links = " · ".join(
        _link(link["label"], link["href"], base, destination)
        for link in case.get("links", [])
    )
    availability = "" if primary else '<p class="unavailable">No replay recorded</p>'
    return f"""<article class="card" {attributes}>
{thumbnail}<div class="card-body">
<div class="card-top"><span>{_text(FLEETS[case['fleet']])}</span>
<span class="generation">{_text(GENERATIONS[case['generation']])}</span></div>
<h2>{_text(title)}</h2><p class="checkpoint">{_text(checkpoint)}</p>
<div class="badges"><span class="badge {native_outcome}">{native_label}</span>
<span class="badge {post_outcome or 'neutral'}">{post_label}</span></div>
<dl>{metrics}</dl><p class="note">{_text(note)}</p>{availability}
<div class="primary-links">{''.join(primary)}</div>
<p class="evidence-links">{links}</p></div></article>"""


def build(manifest, out):
    """Write an offline gallery and return its path.

    ``manifest`` is a JSON path or a mapping with ``title``, ``description`` and
    ``cases``. Each case declares id/title, fleet, generation, native_status and
    postprocessed_status. Optional replay URLs, thumbnail, metrics, note and
    labeled links add evidence without changing its declared outcome. A case's
    optional native_label changes only its replay button. Optional evaluations
    list full-panel title/completed/total/source_url and an optional note. Relative
    local links resolve from the manifest directory (cwd for an in-memory dict).
    Cases without recordings stay visible. No simulator or replay is executed.
    """
    destination = Path(out).expanduser().resolve()
    if isinstance(manifest, (str, os.PathLike)):
        path = Path(manifest).expanduser().resolve()
        content = json.loads(path.read_text())
        base = path.parent
    else:
        content, base = manifest, Path.cwd()
    if not isinstance(content, dict) or not isinstance(content.get("cases"), list):
        raise ValueError("Gallery manifest requires a cases list")
    seen = set()
    for case in content["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Gallery cases must be objects")
        for field in ("id", "title"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Each gallery case requires a nonempty {field}")
        if case["id"] in seen:
            raise ValueError(f"Duplicate gallery case id: {case['id']}")
        seen.add(case["id"])
        for field, choices in (
            ("fleet", FLEETS),
            ("generation", GENERATIONS),
            ("native_status", NATIVE),
            ("postprocessed_status", POSTPROCESSED),
        ):
            if case.get(field) not in choices:
                raise ValueError(f"Invalid {field} for gallery case {case['id']}")
        if not isinstance(case.get("metrics", {}), dict):
            raise ValueError("Gallery metrics must be a label/value mapping")
        links = case.get("links", [])
        if not isinstance(links, list) or any(
            not isinstance(link, dict) or not link.get("label") or not link.get("href")
            for link in links
        ):
            raise ValueError("Gallery links require label and href")
        if case.get("postprocessed_url") and case["postprocessed_status"] == "not_run":
            raise ValueError("A postprocessed replay requires its processing outcome")
        if "native_label" in case and (
            not isinstance(case["native_label"], str)
            or not case["native_label"].strip()
        ):
            raise ValueError("Gallery native_label must be a nonempty string")
    cases = content["cases"]
    counts = (
        (len(cases), "entries"),
        (sum(c["native_status"] == "success" for c in cases), "native complete"),
        (sum(c["native_status"] == "failed" for c in cases), "native incomplete"),
        (
            sum(c["postprocessed_status"] == "failed" for c in cases),
            "postprocessing unresolved",
        ),
    )
    summary = "".join(
        f"<span><strong>{count}</strong> {label}</span>" for count, label in counts
    )
    page = (Path(__file__).parent / "assets" / "gallery.html").read_text()
    replacements = {
        "TITLE": _text(content.get("title", "Terra fleet gallery")),
        "DESCRIPTION": _text(content.get("description", "")),
        "EVALUATIONS": _evaluations(content.get("evaluations", []), base, destination),
        "SUMMARY": summary,
        "CARDS": "\n".join(_card(case, base, destination) for case in cases),
    }
    page = re.sub(
        r"\{\{(TITLE|DESCRIPTION|EVALUATIONS|SUMMARY|CARDS)\}\}",
        lambda match: replacements[match[1]],
        page,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)
    return destination

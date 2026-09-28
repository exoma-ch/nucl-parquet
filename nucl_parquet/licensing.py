"""Read `data/licenses.toml` and render generated licence artifacts from it.

`data/licenses.toml` is the single source of truth for data provenance and
redistribution terms. The generated artifacts are:

  ATTRIBUTION.md          the AUTO:libraries table (prose stays hand-written)
  data/<library>/LICENSE.txt   per-directory sidecars

`scripts/build_notices.py` is the CLI that writes them and the check-mode gate
that CI runs.

Manifest entry keys are ``"<section>.<name>"`` (e.g. ``"libraries.jeff-4_0"``,
``"stopping.catima"``).

Stdlib only — no numpy / duckdb / polars — because the check-mode gate must
run in pre-commit and CI without pulling the project venv.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "licenses.toml"

#: Sections of the manifest whose sub-tables describe bundled data.
DATA_SECTIONS = ("libraries", "stopping", "meta")

#: Name of the generated per-directory sidecar. Excluded from coverage — the
#: sidecar is a generated artifact, not third-party data.
SIDECAR_NAME = "LICENSE.txt"

_RISK_ICON = {"green": "🟢", "amber": "🟡", "red": "🔴"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path | None = None) -> dict:
    """Parse the manifest TOML."""
    return tomllib.loads((path or MANIFEST).read_text(encoding="utf-8"))


def entries(manifest: dict) -> dict[str, dict]:
    """Flatten the data-bearing sections to ``{"section.name": entry}``."""
    out: dict[str, dict] = {}
    for section in DATA_SECTIONS:
        for name, entry in manifest.get(section, {}).items():
            out[f"{section}.{name}"] = entry
    return out


def copyright_line(manifest: dict) -> str:
    """The single copyright line stamped into every generated artifact.

    Sourced from ``[manifest].copyright`` so a copyright-holder change (e.g.
    the pending flip to "ETH Zürich" tracked in #239) is a one-field edit
    that flows into every sidecar without touching the code.
    """
    return manifest["manifest"]["copyright"]


# ---------------------------------------------------------------------------
# Path matching
#
# Two pattern forms, deliberately narrow so a typo fails loudly rather than
# over-matching and silently "covering" an unaudited tree:
#   "data/foo/**"          -> everything beneath data/foo/
#   "data/meta/psf_*.pq"   -> segment-wise fnmatch; `*` never crosses a `/`
# ---------------------------------------------------------------------------


def match(pattern: str, path: str) -> bool:
    """True if the repo-relative *path* is claimed by *pattern*."""
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    pat_parts = pattern.split("/")
    path_parts = path.split("/")
    if len(pat_parts) != len(path_parts):
        return False
    return all(fnmatch.fnmatchcase(sp, pp) for sp, pp in zip(path_parts, pat_parts))


def owners(manifest: dict, path: str) -> list[str]:
    """Every entry key claiming *path* (normally exactly one)."""
    return [key for key, entry in entries(manifest).items() if any(match(p, path) for p in entry.get("paths", []))]


@dataclass(frozen=True)
class Coverage:
    """Result of checking tracked data files against the manifest."""

    covered: dict[str, list[str]]  # path -> owning entry keys
    uncovered: list[str]  # tracked data files no entry claims
    unused_patterns: list[str]  # "key: pattern" that matched nothing

    @property
    def ok(self) -> bool:
        return not self.uncovered and not self.unused_patterns


def check_coverage(manifest: dict, tracked: list[str]) -> Coverage:
    """Classify *tracked* repo-relative paths against the manifest.

    Project metadata (catalog, schema, the manifest itself, suppliers) and
    generated sidecars are excluded — they are MIT project files, not
    bundled third-party data.
    """
    metadata = set(manifest["manifest"].get("project_metadata", []))
    ent = entries(manifest)

    data_files = [p for p in tracked if p not in metadata and Path(p).name != SIDECAR_NAME]

    covered: dict[str, list[str]] = {}
    uncovered: list[str] = []
    hit: set[tuple[str, str]] = set()

    for path in data_files:
        keys = []
        for key, entry in ent.items():
            for pattern in entry.get("paths", []):
                if match(pattern, path):
                    keys.append(key)
                    hit.add((key, pattern))
                    break
        if keys:
            covered[path] = keys
        else:
            uncovered.append(path)

    unused = [
        f"{key}: {pattern}"
        for key, entry in ent.items()
        for pattern in entry.get("paths", [])
        if (key, pattern) not in hit
    ]
    return Coverage(covered=covered, uncovered=sorted(uncovered), unused_patterns=sorted(unused))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _sorted_entries(manifest: dict) -> list[tuple[str, dict]]:
    return sorted(entries(manifest).items())


def _redistributable(value: bool | str) -> str:
    """TOML booleans render as Python `True`; normalise for human-facing text."""
    if isinstance(value, bool):
        return "yes" if value else "NO — must not be bundled"
    return str(value)


def render_attribution_table(manifest: dict) -> str:
    """Markdown summary table for ``ATTRIBUTION.md`` (AUTO section)."""
    rows = [
        "| Library | Custodian | Terms | Cite |",
        "|---|---|---|---|",
    ]
    for _key, e in _sorted_entries(manifest):
        icon = _RISK_ICON.get(e.get("risk", ""), "")
        cite = e["citation"].split(";")[0].strip().rstrip(".")
        if len(cite) > 90:
            cite = cite[:87].rstrip() + "…"
        rows.append(f"| {icon} {e['name']} | {e['custodian']} | {e['license']} | {cite} |")
    return "\n".join(rows)


def sidecar_dirs(manifest: dict, tracked: list[str]) -> dict[str, list[str]]:
    """Map each bundled data directory to the entry keys covering it.

    Sidecars are written one level below the data root (``data/tendl-2025/``,
    ``data/jeff-4.0/``, …) — the granularity at which someone is likely to
    copy a subtree out of the repo, and the same granularity ``catalog.json``
    uses for its data-dir-relative paths.
    """
    roots = manifest["manifest"]["data_roots"]
    metadata = set(manifest["manifest"].get("project_metadata", []))
    out: dict[str, set[str]] = {}

    for path in tracked:
        if path in metadata or Path(path).name == SIDECAR_NAME:
            continue
        parts = path.split("/")
        if parts[0] not in roots:
            continue
        # `data/jeff-4.0/xs/n_Cu.parquet` -> `data/jeff-4.0`. A file sitting
        # directly in the root has no library dir of its own, so it is grouped
        # under the root itself.
        directory = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        out.setdefault(directory, set()).update(owners(manifest, path))

    return {d: sorted(keys) for d, keys in sorted(out.items())}


def render_sidecar(manifest: dict, directory: str, keys: list[str]) -> str:
    """Per-directory ``LICENSE.txt`` explaining what governs that subtree."""
    ent = entries(manifest)
    rule = "-" * 78
    lines = [
        f"{directory}/ — redistribution terms",
        "",
        "GENERATED FILE — do not edit. Source of truth: data/licenses.toml",
        "Regenerate: python scripts/build_notices.py --write",
        "",
        "The MIT licence of nucl-parquet covers the code and the ENDF-6 -> Parquet",
        "conversion ONLY. The data in this directory is third-party material governed",
        "by the terms below.",
        "",
        copyright_line(manifest),
        "",
    ]
    for key in keys:
        e = ent[key]
        lines += [
            rule,
            e["name"],
            rule,
            f"  custodian       : {e['custodian']}",
            f"  licence         : {e['license']}",
            f"  redistributable : {_redistributable(e['redistributable'])}",
            f"  terms           : {e['terms_url']}",
            "  cite            : " + e["citation"],
        ]
        if e.get("notes"):
            lines.append(f"  notes           : {e['notes']}")
        lines.append("")

    lines += [
        rule,
        "When you redistribute this directory you must keep this file, keep the",
        "top-level NOTICE, cite the works above, and state that the data was",
        "reformatted from its original format to Apache Parquet by eXoma.",
        "",
    ]
    return "\n".join(lines)

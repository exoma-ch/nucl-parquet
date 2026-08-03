# SPDX-FileCopyrightText: 2024-2026 ETH Zürich (eXoma — Exotic Matter Applications)
# SPDX-FileContributor: Lars Gerchow
# SPDX-License-Identifier: MIT
"""Read `data/licenses.toml` and render every licence artifact from it.

`data/licenses.toml` is the single source of truth for data provenance,
redistribution terms, and the ETH Zürich institutional position (RSETHZ 440.4).
Everything else — ``NOTICE``, the ``ATTRIBUTION.md`` tables, the per-dataset
``LICENSE.txt`` sidecars, and the SPDX headers on source files — is generated
from it so the four can never disagree.

This module is the library half; ``scripts/build_notices.py`` is the CLI that
writes the artifacts and the check-mode gate that CI runs.

Entry keys are ``"<section>.<name>"`` (e.g. ``"libraries.jeff-4_0"``,
``"stopping.catima"``), matching the ``applies_to`` references in ``[[notices]]``.
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

#: Name of the generated per-directory sidecar. Excluded from coverage.
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

    Project metadata (catalog, schema, the manifest itself) and generated
    sidecars are excluded — they are MIT project files, not bundled data.
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
# ETH Zürich / RSETHZ 440.4
# ---------------------------------------------------------------------------


def eth_copyright_line(manifest: dict) -> str:
    """The Art. 27(1)(e) copyright line: ETH as holder, creators named."""
    eth = manifest["eth"]
    line = f"Copyright (c) {eth['copyright_years']} {eth['copyright_holder']}"
    group = eth.get("research_group")
    if group:
        line += f" ({group})"
    return line


def _redistributable(value: bool | str) -> str:
    """TOML booleans render as Python `True`; normalise for human-facing text."""
    if isinstance(value, bool):
        return "yes" if value else "NO — must not be bundled"
    return str(value)


def spdx_header(manifest: dict, comment: str = "#") -> str:
    """SPDX header block stamped onto source files."""
    eth = manifest["eth"]
    lines = [f"SPDX-FileCopyrightText: {eth['spdx_copyright']}"]
    lines += [f"SPDX-FileContributor: {c}" for c in eth.get("creators", [])]
    lines.append(f"SPDX-License-Identifier: {eth['spdx_license']}")
    return "\n".join(f"{comment} {line}" for line in lines)


def eth_notice_block(manifest: dict) -> str:
    """The ETH institutional block rendered into NOTICE."""
    eth = manifest["eth"]
    creators = ", ".join(eth.get("creators", [])) or "see CITATION.cff"
    return f"""{eth_copyright_line(manifest)}
Creators: {creators}

This software is released by ETH Zürich under {eth["regulation"]}, the
{eth["regulation_title"]}, in force since {eth["in_force"]}.
Under Art. 5(3) ETH Zürich holds the exclusive economic exploitation rights in
software created by its employees in the course of their duties; the moral
rights remain with the creators named above. Art. 27(1)(e) requires that the
open-source distribution carry the licence together with this notice.

{eth["regulation_url"]}"""


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _sorted_entries(manifest: dict) -> list[tuple[str, dict]]:
    return sorted(entries(manifest).items())


def render_notice(manifest: dict) -> str:
    """The complete ``NOTICE`` file."""
    m = manifest["manifest"]
    rule = "=" * 80
    parts = [
        "nucl-parquet",
        "",
        "GENERATED FILE — do not edit. Source of truth: data/licenses.toml",
        "Regenerate: python scripts/build_notices.py --write",
        "",
        rule,
        "ETH Zürich — copyright and institutional terms",
        rule,
        eth_notice_block(manifest),
        "",
        rule,
        "Scope of the MIT licence",
        rule,
        """This product's CODE and the ENDF-6 -> Parquet conversion are licensed under the
MIT License (see LICENSE). The MIT License does NOT apply to the bundled
evaluated nuclear data, which is third-party material redistributed under the
terms of its respective custodians. Per-library provenance, licenses, and
required citations are recorded in data/licenses.toml and ATTRIBUTION.md.

The nuclear data has been reformatted (ENDF-6 -> Apache Parquet) by eXoma; this
is a format conversion only. Required third-party notices follow.""",
    ]

    for notice in manifest.get("notices", []):
        parts += ["", rule, notice["title"], rule, notice["body"].strip()]

    # Copyleft posture — generated from [code_dependencies] so the policy and
    # the dependency metadata can never drift apart.
    deps = manifest.get("code_dependencies", {})
    if deps:
        body = [
            "Copyleft obligations attach to DISTRIBUTION (GPL) or to network interaction",
            "with a modified version (AGPL-3.0 section 13) — not to 'commercial use'.",
            "Data computed by a program is not a derivative work of that program, so the",
            "computed tables above are redistributed freely. The programs themselves are",
            "not redistributed here:",
            "",
        ]
        for dep in deps.values():
            body.append(f"  - {dep['name']} ({dep['spdx']}) — {dep['role']}.")
            body.append(f"    Runtime dependency: {'yes' if dep.get('runtime_dependency') else 'no'}.")
        parts += ["", rule, "Copyleft-licensed tooling (not redistributed)", rule, "\n".join(body).rstrip()]

    parts += [
        "",
        rule,
        f"""For full per-library terms and required citations, see data/licenses.toml and
ATTRIBUTION.md. This data carries inherent evaluation uncertainties — see
DISCLAIMER.md.

Redistribution posture: {m["usage"]}""",
        "",
    ]
    return "\n".join(parts)


def render_attribution_table(manifest: dict) -> str:
    """Markdown summary table for ``ATTRIBUTION.md``."""
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


def render_dependency_table(manifest: dict) -> str:
    """Markdown table of copyleft-relevant code dependencies."""
    rows = [
        "| Dependency | SPDX | Role | Runtime dep? |",
        "|---|---|---|---|",
    ]
    for dep in manifest.get("code_dependencies", {}).values():
        runtime = "**no**" if not dep.get("runtime_dependency") else "yes"
        rows.append(f"| [{dep['name']}]({dep['upstream']}) | `{dep['spdx']}` | {dep['role']} | {runtime} |")
    return "\n".join(rows)


def render_eth_table(manifest: dict) -> str:
    """Markdown checklist of the RSETHZ 440.4 Art. 27(1) release conditions."""
    eth = manifest["eth"]
    decided = eth.get("oss_decision_recorded", False)
    mark = "☑" if decided else "☐"
    return "\n".join(
        [
            "| Art. 27(1) condition | Status |",
            "|---|---|",
            f"| (a) Release decided by the responsible Forschungsgruppenleiter:in | {mark} {eth['group_leader']} "
            f"(decision {eth['oss_decision_date']}) |",
            "| (b) Complies with applicable law incl. export control | ☑ see COMPLIANCE.md §1 |",
            "| (c) No conflict with ETH/third-party IP; dependency licences compatible | ☑ see COMPLIANCE.md §3 |",
            "| (d) Published without delay on a public platform | ☑ github.com/exoma-ch/nucl-parquet |",
            f"| (e) Carries the OSS licence and © notice naming ETH Zürich + creators | ☑ LICENSE, NOTICE "
            f"(holder: {eth['copyright_holder']}) |",
            f"| Art. 27(3) — no CLA without ETH transfer approval | ☑ policy: `{eth['cla_policy']}` |",
            f"| Art. 24 — software disclosure to ETH transfer | {'due' if eth['software_disclosure_required'] else '— not due (non-commercial)'} |",
            f"| Art. 34 — research-data disclosure to ETH transfer | {'due' if eth['data_disclosure_required'] else '— not due (non-commercial)'} |",
        ]
    )


def sidecar_dirs(manifest: dict, tracked: list[str]) -> dict[str, list[str]]:
    """Map each bundled data directory to the entry keys covering it.

    Sidecars are written one level below the data root (``data/tendl-2025/``,
    ``data/jeff-4.0/``, …) — the granularity at which someone is likely to copy
    a subtree out of the repo, and the same granularity ``catalog.json`` uses
    for its data-dir-relative paths.
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
        eth_copyright_line(manifest),
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
            f"  SPDX            : {e.get('spdx', 'NOASSERTION')}",
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

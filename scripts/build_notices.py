#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 ETH Zürich (eXoma — Exotic Matter Applications)
# SPDX-FileContributor: Lars Gerchow
# SPDX-License-Identifier: MIT
"""Generate every licence artifact from `data/licenses.toml`.

The manifest is the single source of truth. This script renders:

  NOTICE                  fully generated
  ATTRIBUTION.md          AUTO:libraries, AUTO:dependencies sections
  COMPLIANCE.md           AUTO:eth section
  data/<library>/LICENSE.txt  per-directory sidecars
  nucl_parquet/**/*.py    SPDX headers
  scripts/*.py            SPDX headers

and verifies that every tracked data file is claimed by exactly one manifest
entry — so adding a dataset without recording its provenance fails CI.

Usage:
    python scripts/build_notices.py            # check mode (exit 1 on drift)
    python scripts/build_notices.py --write    # write the artifacts
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

# Load nucl_parquet/licensing.py directly rather than importing the package:
# nucl_parquet/__init__ pulls duckdb/zstandard, and this gate must stay
# stdlib-only so pre-commit and CI can run it without the project venv.
_SPEC = importlib.util.spec_from_file_location(
    "nucl_parquet_licensing",
    Path(__file__).resolve().parent.parent / "nucl_parquet" / "licensing.py",
)
licensing = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = licensing  # @dataclass resolves types via sys.modules
_SPEC.loader.exec_module(licensing)

ROOT = licensing.ROOT
NOTICE = ROOT / "NOTICE"
ATTRIBUTION = ROOT / "ATTRIBUTION.md"
COMPLIANCE = ROOT / "COMPLIANCE.md"

#: Source trees that get SPDX headers. Tests are deliberately excluded — the
#: distributed package and its build tooling carry the notice; the test suite
#: is covered by the top-level LICENSE.
HEADER_TREES = ("nucl_parquet", "scripts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tracked_files(paths: list[str]) -> list[str]:
    """`git ls-files` restricted to *paths*, repo-relative."""
    out = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def inject_sections(text: str, sections: dict[str, str]) -> str:
    """Replace `<!-- AUTO:tag -->...<!-- /AUTO:tag -->` blocks."""
    for tag, content in sections.items():
        pattern = re.compile(rf"(<!-- AUTO:{tag} -->\n).*?(<!-- /AUTO:{tag} -->)", re.DOTALL)
        if not pattern.search(text):
            raise SystemExit(f"missing marker pair for AUTO:{tag} — add it to the target file")
        text = pattern.sub(lambda m: f"{m.group(1)}{content}\n{m.group(2)}", text)
    return text


def stamp_header(text: str, header: str) -> str:
    """Insert/replace the SPDX block, preserving a leading shebang."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    if lines and lines[0].startswith("#!"):
        out.append(lines[0])
        i = 1
    while i < len(lines) and lines[i].startswith("# SPDX-"):
        i += 1
    out.extend(header.split("\n"))
    out.extend(lines[i:])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Artifact planning
#
# Every generator returns {path: desired_content}; check and write modes share
# the same plan so they can never diverge.
# ---------------------------------------------------------------------------


def plan(manifest: dict) -> tuple[dict[Path, str], licensing.Coverage]:
    roots = manifest["manifest"]["data_roots"]
    tracked = tracked_files(roots)
    coverage = licensing.check_coverage(manifest, tracked)

    artifacts: dict[Path, str] = {NOTICE: licensing.render_notice(manifest)}

    artifacts[ATTRIBUTION] = inject_sections(
        ATTRIBUTION.read_text(encoding="utf-8"),
        {
            "libraries": licensing.render_attribution_table(manifest),
            "dependencies": licensing.render_dependency_table(manifest),
        },
    )
    artifacts[COMPLIANCE] = inject_sections(
        COMPLIANCE.read_text(encoding="utf-8"),
        {"eth": licensing.render_eth_table(manifest)},
    )

    for directory, keys in licensing.sidecar_dirs(manifest, tracked).items():
        artifacts[ROOT / directory / licensing.SIDECAR_NAME] = licensing.render_sidecar(manifest, directory, keys)

    header = licensing.spdx_header(manifest)
    for source in sorted(tracked_files([f"{t}/**/*.py" for t in HEADER_TREES] + [f"{t}/*.py" for t in HEADER_TREES])):
        path = ROOT / source
        artifacts[path] = stamp_header(path.read_text(encoding="utf-8"), header)

    return artifacts, coverage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(write: bool) -> int:
    manifest = licensing.load_manifest()
    artifacts, coverage = plan(manifest)

    failed = False

    if coverage.uncovered:
        failed = True
        print(f"{len(coverage.uncovered)} tracked data file(s) not claimed by data/licenses.toml:", file=sys.stderr)
        for path in coverage.uncovered[:20]:
            print(f"  {path}", file=sys.stderr)
        if len(coverage.uncovered) > 20:
            print(f"  ... and {len(coverage.uncovered) - 20} more", file=sys.stderr)
        print("  -> add a `paths` glob to the owning entry, or add a new entry.", file=sys.stderr)

    if coverage.unused_patterns:
        failed = True
        print("manifest `paths` globs matching nothing (stale entry?):", file=sys.stderr)
        for pattern in coverage.unused_patterns:
            print(f"  {pattern}", file=sys.stderr)

    stale = [p for p, content in artifacts.items() if not p.exists() or p.read_text(encoding="utf-8") != content]

    if write:
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(artifacts[path], encoding="utf-8")
        print(f"wrote {len(stale)} artifact(s); {len(artifacts) - len(stale)} already current")
        # Coverage problems are real even in write mode — the generator cannot
        # invent provenance for an unclaimed file.
        return 1 if failed else 0

    if stale:
        failed = True
        print(f"{len(stale)} licence artifact(s) out of date:", file=sys.stderr)
        for path in stale:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        print("  -> run: python scripts/build_notices.py --write", file=sys.stderr)

    if not failed:
        print(f"licence artifacts current; {len(coverage.covered)} data files covered by the manifest")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write artifacts instead of checking")
    sys.exit(main(parser.parse_args().write))

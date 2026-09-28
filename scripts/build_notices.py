#!/usr/bin/env python3
"""Generate licence artifacts from `data/licenses.toml` (check-mode by default).

The manifest is the single source of truth. This script renders:

  ATTRIBUTION.md              AUTO:libraries table (prose stays hand-written)
  data/<library>/LICENSE.txt  per-directory sidecars

and verifies that every tracked data file is claimed by exactly one manifest
entry — so adding a dataset without recording its provenance fails CI.

`NOTICE` and `COMPLIANCE.md` are hand-written and are NOT touched here.

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
ATTRIBUTION = ROOT / "ATTRIBUTION.md"


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
        pattern = re.compile(
            rf"(<!-- AUTO:{tag} -->\n).*?(<!-- /AUTO:{tag} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"missing marker pair for AUTO:{tag} — add it to the target file")
        text = pattern.sub(lambda m: f"{m.group(1)}{content}\n{m.group(2)}", text)
    return text


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

    artifacts: dict[Path, str] = {}

    artifacts[ATTRIBUTION] = inject_sections(
        ATTRIBUTION.read_text(encoding="utf-8"),
        {"libraries": licensing.render_attribution_table(manifest)},
    )

    for directory, keys in licensing.sidecar_dirs(manifest, tracked).items():
        artifacts[ROOT / directory / licensing.SIDECAR_NAME] = licensing.render_sidecar(manifest, directory, keys)

    return artifacts, coverage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run(write: bool) -> int:
    manifest = licensing.load_manifest()
    artifacts, coverage = plan(manifest)

    failed = False

    if coverage.uncovered:
        failed = True
        print(
            f"{len(coverage.uncovered)} tracked data file(s) not claimed by data/licenses.toml:",
            file=sys.stderr,
        )
        for path in coverage.uncovered[:20]:
            print(f"  {path}", file=sys.stderr)
        if len(coverage.uncovered) > 20:
            print(f"  ... and {len(coverage.uncovered) - 20} more", file=sys.stderr)
        print(
            "  -> add a `paths` glob to the owning entry, or add a new entry.",
            file=sys.stderr,
        )

    if coverage.unused_patterns:
        failed = True
        print(
            "manifest `paths` globs matching nothing (stale entry?):",
            file=sys.stderr,
        )
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
        print(
            "  -> run: python scripts/build_notices.py --write",
            file=sys.stderr,
        )

    if not failed:
        print(f"licence artifacts current; {len(coverage.covered)} data files covered by the manifest")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately from running it (#363).

    Extracted so `tests/test_repo_layout.py::test_every_script_exposes_build_parser`
    can inspect the CLI without executing it.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    ap.add_argument(
        "--write",
        action="store_true",
        help=(
            "regenerate the ATTRIBUTION.md AUTO section and the per-library "
            "LICENSE.txt sidecars in place (default: check-only, exit 1 on drift)"
        ),
    )
    return ap


def main() -> int:
    return _run(build_parser().parse_args().write)


if __name__ == "__main__":
    sys.exit(main())

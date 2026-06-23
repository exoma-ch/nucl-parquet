#!/usr/bin/env python3
"""Auto-generate README sections from catalog.json.

Reads catalog.json and produces markdown for:
  - Libraries table (from catalog.libraries)
  - Views / data tables inventory (from catalog.views)

The README uses marker comments to delimit auto-generated sections:
  <!-- AUTO:libraries -->  ...  <!-- /AUTO:libraries -->
  <!-- AUTO:views -->      ...  <!-- /AUTO:views -->

Usage:
    python scripts/build_readme.py          # check mode (exit 1 if drift)
    python scripts/build_readme.py --write  # overwrite README sections
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
README = ROOT / "README.md"


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text())


# ---------------------------------------------------------------------------
# Libraries table
# ---------------------------------------------------------------------------


def build_libraries_table(catalog: dict) -> str:
    """Build markdown table of cross-section libraries from catalog."""
    lines = [
        "| Library | Projectiles | Source |",
        "|---------|------------|--------|",
    ]
    # Sort libraries by name for stable output
    for lib_id, lib in sorted(catalog["libraries"].items(), key=lambda kv: kv[1]["name"]):
        if not lib.get("projectiles"):
            continue
        name = lib["name"]
        url = lib.get("source_url", "")
        proj_display = ", ".join(_projectile_display(p) for p in lib["projectiles"])
        # Extract organization from description or use lib_id
        desc = lib.get("description", "")
        # Try to extract org from parenthetical at end of description
        org_match = re.search(r"\(([^)]+)\)\s*$", desc)
        org = org_match.group(1) if org_match else ""
        if url:
            lines.append(f"| [{name}]({url}) | {proj_display} | {org} |")
        else:
            lines.append(f"| {name} | {proj_display} | {org} |")
    return "\n".join(lines)


def _projectile_display(p: str) -> str:
    """Convert projectile code to display name."""
    return {
        "n": "n",
        "p": "p",
        "d": "d",
        "t": "t",
        "h": "\u00b3He",
        "a": "\u03b1",
        "g": "\u03b3",
    }.get(p, p)


# ---------------------------------------------------------------------------
# Views / data inventory table
# ---------------------------------------------------------------------------


def build_views_table(catalog: dict) -> str:
    """Build markdown table of all registered views from catalog.views."""
    views = catalog.get("views", {})
    lines = [
        "| View | Path | Type |",
        "|------|------|------|",
    ]
    for name, vdef in sorted(views.items()):
        path = vdef["path"]
        vtype = vdef.get("type", "file")
        # Show glob views with wildcard for clarity. A directory glob gets a
        # trailing `/*.parquet`; a path that already contains a wildcard (a
        # file-glob pattern, e.g. stopping/catima_*.parquet) is shown as-is.
        if vtype == "glob" and "*" not in path:
            path = f"`{path}/*.parquet`"
        else:
            path = f"`{path}`"
        lines.append(f"| `{name}` | {path} | {vtype} |")

    # Add derived views not in catalog
    lines.append("| `xs` | union of all XS libraries | derived |")
    lines.append("| `ground_states` | filtered from `nuclides` | derived |")
    lines.append("| `eadl_transitions` | alias for `atomic_relaxation` | derived |")
    lines.append("| `fluorescence` | filtered from `atomic_relaxation` | derived |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section injection
# ---------------------------------------------------------------------------

AUTO_RE = re.compile(
    r"(<!-- AUTO:(\w+) -->\n).*?(<!-- /AUTO:\2 -->)",
    re.DOTALL,
)


def inject_sections(readme_text: str, sections: dict[str, str]) -> str:
    """Replace content between AUTO markers with generated sections."""

    def replacer(m: re.Match) -> str:
        tag = m.group(2)
        if tag in sections:
            return f"{m.group(1)}{sections[tag]}\n{m.group(3)}"
        return m.group(0)

    return AUTO_RE.sub(replacer, readme_text)


def check_or_write(write: bool = False) -> bool:
    """Check for drift or write updated README. Returns True if OK."""
    catalog = load_catalog()
    sections = {
        "libraries": build_libraries_table(catalog),
        "views": build_views_table(catalog),
    }

    readme_text = README.read_text()
    updated = inject_sections(readme_text, sections)

    if readme_text == updated:
        print("README is up to date with catalog.json")
        return True

    if write:
        README.write_text(updated)
        print(f"Updated {len(sections)} README section(s)")
        return True
    else:
        print("README has drifted from catalog.json!", file=sys.stderr)
        print("Run: python scripts/build_readme.py --write", file=sys.stderr)
        # Show which sections changed
        for tag in sections:
            marker = f"<!-- AUTO:{tag} -->"
            if marker not in readme_text:
                print(f"  Missing marker: {marker}", file=sys.stderr)
            else:
                old_match = re.search(
                    rf"<!-- AUTO:{tag} -->\n(.*?)<!-- /AUTO:{tag} -->",
                    readme_text,
                    re.DOTALL,
                )
                if old_match and old_match.group(1).strip() != sections[tag].strip():
                    print(f"  Section '{tag}' needs update", file=sys.stderr)
        return False


if __name__ == "__main__":
    write_mode = "--write" in sys.argv
    ok = check_or_write(write=write_mode)
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""Check supplier URLs and update `<data-dir>/suppliers.json` with status + timestamp.

Exit codes:
  0 — no changes
  1 — error
  2 — data changed (signals the workflow to create a release)

Usage:
    python scripts/update_suppliers.py
    python scripts/update_suppliers.py --dry-run   # probe every URL, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR  # noqa: E402

SUPPLIERS_REL = Path("suppliers.json")
TIMEOUT = 15  # seconds


def check_url(url: str) -> str:
    """Return 'ok', 'redirect', or 'down'."""
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "nucl-parquet-bot/1.0"})
        resp = urlopen(req, timeout=TIMEOUT)
        if resp.status < 400:
            return "ok"
        return "down"
    except HTTPError as e:
        # Some sites block HEAD; try GET
        if e.code == 405:
            try:
                req = Request(url, headers={"User-Agent": "nucl-parquet-bot/1.0"})
                resp = urlopen(req, timeout=TIMEOUT)
                return "ok" if resp.status < 400 else "down"
            except Exception:
                return "down"
        return "down" if e.code >= 400 else "ok"
    except URLError:
        return "down"
    except Exception:
        return "down"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it.

    This script took no arguments and wrote `data/suppliers.json` unconditionally
    on every run — including bumping `last_checked` when nothing had changed
    (#363). It also spelled its own `Path(__file__).parent.parent / "data"`, a
    form the #349 guard did not match because it never mentions `ROOT`.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data directory holding suppliers.json (default: {DATA_DIR.name}/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe every supplier URL and report, but do not write suppliers.json.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    suppliers_path = args.data_dir / SUPPLIERS_REL

    if not suppliers_path.exists():
        print(f"ERROR: {suppliers_path} not found", file=sys.stderr)
        return 1

    suppliers = json.loads(suppliers_path.read_text())
    today = date.today().isoformat()
    changed = False

    for s in suppliers:
        url = s["url"]
        old_status = s.get("status", "unknown")
        new_status = check_url(url)
        print(f"  {s['name']:40s} {url:50s} → {new_status}")

        if new_status != old_status:
            changed = True
            s["status"] = new_status

        s["last_checked"] = today

    if args.dry_run:
        print(f"\ndry run: would {'update' if changed else 'rewrite'} {suppliers_path}")
        return 2 if changed else 0

    # Write back (always update last_checked)
    suppliers_path.write_text(json.dumps(suppliers, indent=2, ensure_ascii=False) + "\n")

    if changed:
        print("\n⚠ Supplier status changed — diff detected")
        return 2
    else:
        print("\n✓ All suppliers unchanged")
        return 0


if __name__ == "__main__":
    sys.exit(main())

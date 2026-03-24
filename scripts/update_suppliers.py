#!/usr/bin/env python3
"""Check supplier URLs and update data/suppliers.json with status + timestamp.

Exit codes:
  0 — no changes (or --force)
  1 — error
  2 — data changed (signals the workflow to create a release)
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SUPPLIERS_PATH = Path(__file__).parent.parent / "data" / "suppliers.json"
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


def main() -> int:
    if not SUPPLIERS_PATH.exists():
        print(f"ERROR: {SUPPLIERS_PATH} not found", file=sys.stderr)
        return 1

    suppliers = json.loads(SUPPLIERS_PATH.read_text())
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

    # Write back (always update last_checked)
    SUPPLIERS_PATH.write_text(json.dumps(suppliers, indent=2, ensure_ascii=False) + "\n")

    if changed:
        print("\n⚠ Supplier status changed — diff detected")
        return 2
    else:
        print("\n✓ All suppliers unchanged")
        return 0


if __name__ == "__main__":
    sys.exit(main())

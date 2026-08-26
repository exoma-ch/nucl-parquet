"""Report libraries whose committed parquets no longer match their builder.

The parquets under `data/` are committed artefacts, so fixing a builder is not
the same as fixing the data it produced. Between #260 and #334 those two drifted
apart for thirteen months with CI green. Each library's `manifest.json` now
records the digest of the script that built it; this reports every library where
that digest no longer matches the script on disk, plus every library whose
provenance cannot be checked at all.

The check itself lives in `nucl_parquet/builder_stamp.py` and is enforced by
`tests/test_builder_staleness.py`. This script is the operator-facing view of
the same code — run it to see the report without running pytest.

Usage:
    nix develop -c uv run python scripts/check_builder_staleness.py
    nix develop -c uv run python scripts/check_builder_staleness.py --show-exempt

Exit status is 1 when anything is unexplained, 0 when every library is either
verified or carries an entry in `data/builder_stamp_exemptions.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet.builder_stamp import EXEMPTIONS_FILE, audit, format_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363).

    Read-only: this script audits and reports, it writes nothing.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument(
        "--show-exempt",
        action="store_true",
        help="also list the libraries currently excused, and the issue that removes each",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    findings = audit(args.data_dir, args.repo_root)
    print(format_report(findings))

    if args.show_exempt:
        path = args.data_dir / EXEMPTIONS_FILE
        exemptions = json.loads(path.read_text()).get("exemptions", {}) if path.exists() else {}
        print(f"\n{len(exemptions)} exemption(s) in {path}:")
        for key, entry in sorted(exemptions.items()):
            print(f"  {key:<20} {entry.get('reason'):<18} removed by #{entry.get('issue')}")

    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()

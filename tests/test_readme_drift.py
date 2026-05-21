"""CI test: README auto-generated sections must match catalog.json.

Catches drift when new data tables are added to catalog.json but README
is not regenerated. Fix: `python scripts/build_readme.py --write`.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_matches_catalog():
    """README auto-sections must be up to date with catalog.json."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_readme.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"README has drifted from catalog.json.\nRun: python scripts/build_readme.py --write\nstderr: {result.stderr}"
    )

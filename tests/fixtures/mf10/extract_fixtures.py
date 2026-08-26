"""Regenerate the MF=10 oracle fixtures from the IAEA mirror.

    nix develop -c uv run python tests/fixtures/mf10/extract_fixtures.py

Needs network; run by hand when the fixture set changes, never from the test
suite. `tests/test_mt_residuals.py` reads the committed output offline.

Every retained line is byte-for-byte the evaluator's, so the `IZAP` product
identifiers the oracle reads are the evaluator's own and not a transcription of
the table under test. Only MF=10 sections are kept — MF=1's descriptive text
alone is three times the size of everything else here, and the oracle does not
read it. The head record of each MF=10 section carries `ZA`, so the fixture
still names its own target without the filename having to be parsed.

See README.md for what each file witnesses and why it was chosen.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from fetch_endf_libs import IAEA_MIRROR, LIBRARIES  # noqa: E402

HERE = Path(__file__).parent

#: (library key, sublibrary code, filename on the mirror). Chosen to cover 34 of
#: the 35 MT numbers that any MF=10 section in a 74-evaluation sample named a
#: single product for, in as few bytes as possible, across three libraries.
SOURCES = [
    ("jeff-4.0", "n", "n_049-In-114_4928.zip"),
    ("fendl-3.2", "n", "n_024-Cr-50_2425.zip"),
    ("jeff-4.0", "n", "n_033-As-75_3325.zip"),
    ("tendl-2025", "n", "n_032-Ge-76_3243.zip"),
]


def _mf_mt(line: str) -> tuple[int, int]:
    """The MF and MT fields of an ENDF-6 record (columns 71-72 and 73-75)."""
    return int(line[70:72] or 0), int(line[72:75] or 0)


def extract_mf10(text: str) -> str:
    """Return a minimal ENDF-6 material holding only `text`'s MF=10 sections."""
    lines = [line for line in text.splitlines() if len(line) >= 75]
    mat = int(lines[1][66:70])
    kept = [line for line in lines[1:] if _mf_mt(line)[0] == 10]
    if not kept:
        raise ValueError("no MF=10 sections in this evaluation")

    out = [lines[0], *kept]  # TPID, then every MF=10 record including its SENDs
    if _mf_mt(out[-1])[1] != 0:
        out.append(f"{'':<66}{mat:>4d}{10:>2d}{0:>3d}{99999:>5d}")  # SEND
    out.append(f"{'':<66}{mat:>4d}{0:>2d}{0:>3d}{0:>5d}")  # FEND
    out.append(f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}")  # MEND
    out.append(f"{'':<66}{-1:>4d}{0:>2d}{0:>3d}{0:>5d}")  # TEND
    return "\n".join(out) + "\n"


def main() -> None:
    session = requests.Session()
    for lib_key, sublib, filename in SOURCES:
        lib = LIBRARIES[lib_key]
        url = f"{IAEA_MIRROR}/{lib.iaea_path}/{lib.sublibraries[sublib]}/{filename}"
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            text = zf.read(zf.namelist()[0]).decode("ascii", errors="replace")

        fixture = extract_mf10(text)
        dest = HERE / lib_key / re.sub(r"\.zip$", ".endf", filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(fixture)
        mts = sorted({_mf_mt(line)[1] for line in fixture.splitlines() if _mf_mt(line)[0] == 10} - {0})
        print(f"{dest.relative_to(HERE)}: {len(fixture) / 1024:.1f} KiB, MF=10 MTs {mts}")


if __name__ == "__main__":
    main()

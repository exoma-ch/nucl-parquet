"""Unit tests for `scripts/migrate_xs_schema.py`.

The migration lifts legacy per-library tables into `CANONICAL_XS_SCHEMA`. It ran
once over 35.4M rows, and every defect it had was a silent one: a stem it could
not parse meant a file skipped, and a column it did not carry forward meant data
deleted. Both happened. These pin the two failure modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_xs_schema import _RENAMES, parse_stem  # noqa: E402


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("p_Cu", ("p", 1, 1, 29)),
        ("n_Fe", ("n", 0, 1, 26)),
        ("a_U", ("a", 2, 4, 92)),
        # Heavy ions carry their own mass in the projectile token. He-3 goes down
        # the same path as a heavy ion rather than the light-ion table, and must
        # still land on (Z=2, A=3) rather than being read as helium-natural.
        ("he3_C", ("he3", 2, 3, 6)),
        ("ar40_Ac", ("ar40", 18, 40, 89)),
        ("c12_Au", ("c12", 6, 12, 79)),
        # Elements the builders have no symbol for are written Z<number>.
        # 183 files used this form and were skipped by the first regex.
        ("p_Z61", ("p", 1, 1, 61)),
        ("d_Z105", ("d", 1, 2, 105)),
        ("ar40_Z99", ("ar40", 18, 40, 99)),
        # Unparseable must be None, never a partial identity.
        ("garbage", None),
        ("p_Xx", None),
        ("", None),
    ],
)
def test_parse_stem(stem: str, expected: tuple | None) -> None:
    """Every stem a builder emits must parse.

    A stem that returns None is a file the migration skips, and skipping is
    invisible: the file stays in its legacy shape and only surfaces when a
    consumer unions it.
    """
    assert parse_stem(stem) == expected


def test_legacy_provenance_column_is_carried_forward() -> None:
    """`exfor_entry` was renamed, not dropped.

    An early migration selected only the canonical column names, which silently
    deleted the provenance chain back to the measurement for every EXFOR row --
    real data loss, recovered only because the legacy tree was still in git.
    """
    assert _RENAMES["exfor_entry"] == "source_entry"

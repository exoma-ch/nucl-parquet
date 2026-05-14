"""Python golden-file parity tests.

Mirrors the Rust + TS golden tests so all three clients are checked against
the same fixtures. Sub-issue #176 / parent #173.

Re-run `uv run python tests/golden/generate.py` to refresh fixtures after a
schema-additive change (per ADR-0002).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generate import (
    gen_co60_beta_gamma,
    gen_co60_gamma_gamma,
    gen_identify_gamma_1173,
    gen_ni60_emissions,
    gen_sr90_y90_negative,
    gen_y86_kshell_xray_gamma,
)

import nucl_parquet

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

CASES = [
    ("co60_beta_gamma", gen_co60_beta_gamma),
    ("y86_kshell_xray_gamma", gen_y86_kshell_xray_gamma),
    ("co60_gamma_gamma", gen_co60_gamma_gamma),
    ("sr90_y90_negative", gen_sr90_y90_negative),
    ("identify_gamma_1173keV", gen_identify_gamma_1173),
    ("ni60_emissions", gen_ni60_emissions),
]


@pytest.mark.data
@pytest.mark.parametrize("name,generator", CASES)
def test_python_matches_golden(name: str, generator) -> None:
    fixture_path = FIXTURE_DIR / f"{name}.json"
    if not fixture_path.exists():
        pytest.skip(f"fixture {fixture_path.name} missing — run generate.py")
    golden = json.loads(fixture_path.read_text())

    db = nucl_parquet.connect()
    actual = generator(db)

    assert actual == golden, (
        f"Python output differs from golden {name}.\n"
        f"Re-run `uv run python tests/golden/generate.py` if the schema changed."
    )

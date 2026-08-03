"""Catch whole-file magnitude errors by comparing libraries against each other.

A shipped `p_Pb.parquet` once carried cross sections 1000x too large -- barns
written into an `xs_mb` column. Nothing caught it: the values were finite,
positive, non-null and well under every ceiling the suite asserted, and no
single library can tell you its own units are wrong.

Peer libraries can. Fourteen evaluated libraries overlap heavily, and while they
disagree channel-by-channel -- different level schemes, isomer conventions,
threshold treatments -- they do not disagree *systematically across a whole
file*. Empirically the worst file in the corpus sits within 1.6x of its peers,
so a 10x gate has ~60x of headroom before it touches real physics while still
catching a unit error by two orders of magnitude.

This is deliberately a per-file median, not a per-channel bound: a unit error is
a property of a file (one builder, one conversion), so the median over a file's
channels is exactly the statistic that isolates it, and it is robust to the
handful of channels where evaluations genuinely differ by 20x.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"

# Sample where most libraries have data and thresholds are behind us. 14.5 MeV is
# the D-T point, the most heavily evaluated energy in the corpus.
SAMPLE_MeV = 14.5
WINDOW = (13.5, 15.5)

# Only compare channels well above the noise floor. Below-threshold tails run to
# 1e-31 mb, where a ratio is meaningless -- unfloored, they produce 1e29x
# "disagreements" that say nothing about units.
FLOOR_mb = 1.0

# A channel needs this many independent libraries before its median is a
# trustworthy reference.
MIN_LIBS_PER_CHANNEL = 3

# A file needs this many overlapping channels before its median ratio means
# anything.
MIN_CHANNELS_PER_FILE = 5

# Worst observed in the corpus is 1.65x. A unit error is 1000x.
MAX_RATIO = 10.0


def _evaluated_libraries() -> list[tuple[str, Path]]:
    """Evaluated production libraries only -- the ones measuring the same thing.

    Experimental (EXFOR) and transport-channel tables are excluded: they carry a
    different observable, so a ratio against an evaluation is not a defect.
    """
    catalog_path = DATA_DIR / "catalog.json"
    if not catalog_path.exists():
        return []
    catalog = json.loads(catalog_path.read_text())
    out = []
    for key, info in catalog.get("libraries", {}).items():
        if info.get("data_type") != "cross_sections" or "path" not in info:
            continue
        d = DATA_DIR / info["path"]
        if d.exists() and any(d.glob("*.parquet")):
            out.append((key, d))
    return out


pytestmark = pytest.mark.skipif(
    len(_evaluated_libraries()) < 2,
    reason="need at least two evaluated libraries to cross-check",
)


def _sampled(con: duckdb.DuckDBPyConnection, scale: dict[str, float] | None = None) -> None:
    """Create `pt`: one point per (library, channel) near SAMPLE_MeV.

    `scale` multiplies one library's xs_mb, used by the self-test below to inject
    a synthetic unit error.
    """
    scale = scale or {}
    union = " UNION ALL BY NAME ".join(
        f"""SELECT '{key}' AS lib, projectile, target_Z, target_A,
                   residual_Z, residual_A, energy_MeV,
                   xs_mb * {scale.get(key, 1.0)} AS xs_mb
            FROM read_parquet('{d}/*.parquet')"""
        for key, d in _evaluated_libraries()
    )
    con.sql(f"CREATE OR REPLACE VIEW allxs AS {union}")
    con.sql(f"""
        CREATE OR REPLACE VIEW pt AS
        SELECT lib, projectile, target_Z, target_A, residual_Z, residual_A,
               arg_min(xs_mb, abs(energy_MeV - {SAMPLE_MeV})) AS xs
        FROM allxs
        WHERE residual_Z IS NOT NULL
          AND energy_MeV BETWEEN {WINDOW[0]} AND {WINDOW[1]}
          AND xs_mb > {FLOOR_mb}
        GROUP BY ALL
    """)


_OUTLIER_SQL = f"""
WITH ref AS (
    SELECT projectile, target_Z, target_A, residual_Z, residual_A, median(xs) AS m
    FROM pt
    GROUP BY ALL
    HAVING count(*) >= {MIN_LIBS_PER_CHANNEL}
)
SELECT p.lib, p.projectile, p.target_Z, count(*) AS chans,
       median(p.xs / ref.m) AS ratio
FROM pt p JOIN ref USING (projectile, target_Z, target_A, residual_Z, residual_A)
GROUP BY ALL
HAVING count(*) >= {MIN_CHANNELS_PER_FILE}
   AND (median(p.xs / ref.m) > {MAX_RATIO} OR median(p.xs / ref.m) < {1 / MAX_RATIO})
ORDER BY abs(log10(median(p.xs / ref.m))) DESC
"""


@pytest.mark.data
def test_no_library_file_is_systematically_off_scale() -> None:
    """No file may sit an order of magnitude off its peers across its channels."""
    con = duckdb.connect()
    _sampled(con)
    bad = con.sql(_OUTLIER_SQL).fetchall()
    assert not bad, "files systematically off-scale versus peer libraries:\n  " + "\n  ".join(
        f"{lib}/{proj}_Z{tz}: median ratio {ratio:.4g} over {n} shared channels" for lib, proj, tz, n, ratio in bad
    )


@pytest.mark.data
def test_the_gate_would_catch_a_unit_error() -> None:
    """Self-test: a barns-for-millibarns slip must trip the check above.

    Without this, a gate that silently stopped discriminating -- a renamed
    column, an empty join -- would keep passing and look like evidence of health.
    """
    con = duckdb.connect()
    # 1000x on one library only, the shape of a mb/b confusion in one builder.
    # Pick the broadest library so the self-test does not depend on catalog order
    # landing on one with too few shared channels to judge.
    victim = max(_evaluated_libraries(), key=lambda kd: len(list(kd[1].glob("*.parquet"))))[0]
    _sampled(con, scale={victim: 1000.0})
    bad = con.sql(_OUTLIER_SQL).fetchall()
    assert bad, "the consistency gate no longer detects a 1000x unit error"
    assert any(row[0] == victim for row in bad), f"{victim} was scaled 1000x but was not flagged"

"""`LIBRARIES` in `scripts/fetch_endf_libs.py` against `data/catalog.json`.

The registry says what a re-ingest will go and fetch. The catalog says what the
repo ships. Nothing kept the two honest, and they drifted: `iaea-medical`
declared a neutron sublibrary that is a 404 on the IAEA mirror and always was.
`catalog.json` had already been corrected to `['p','d']` in #321 and
`data/iaea-medical/xs/` holds only `d_*` and `p_*` files, so the registry was
the last thing still claiming neutrons (#356).

That survived because `list_endf_files` logged a warning and returned `[]` on an
empty listing. A re-ingest walked the 404, wrote one line into the middle of a
multi-hour run, and exited 0 — #334's zero-file ingest wearing a different hat.

These checks are deliberately **offline**. A reachability test against the
mirror is the obvious thing to write and the wrong thing to write: it fails in
PR CI, it fails on a train, and a test that cannot run is not a check (#355).
Whether a URL 200s is a question for a human running the audit in the PR that
changes the registry; whether the registry agrees with what we ship is a
question a test can answer every time, for free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_endf_libs import LIBRARIES, UNSHIPPED_SUBLIBRARIES  # noqa: E402

CATALOG = ROOT / "data" / "catalog.json"


@pytest.fixture(scope="module")
def catalog() -> dict:
    if not CATALOG.exists():
        pytest.skip("data/catalog.json not available")
    return json.loads(CATALOG.read_text())


def _projectiles(catalog: dict, key: str) -> set[str] | None:
    entry = catalog["libraries"].get(key)
    if entry is None:
        return None
    return set(entry.get("projectiles") or [])


def test_every_registry_library_is_catalogued(catalog):
    """A library you can fetch but that the catalog does not describe is drift
    in the other direction — data would appear under `data/<key>/` that no
    consumer can discover."""
    missing = sorted(key for key in LIBRARIES if _projectiles(catalog, key) is None)
    assert not missing, f"in LIBRARIES but not in catalog.json: {missing}"


def test_everything_shipped_can_be_refetched(catalog):
    """The direction that costs data.

    If the catalog ships a projectile the registry cannot fetch, the next
    rebuild produces nothing for it — and a rebuild that writes over `data/`
    deletes it. This is the check that matters most; the reverse direction only
    wastes a download.
    """
    unfetchable: list[str] = []
    for key, lib in sorted(LIBRARIES.items()):
        shipped = _projectiles(catalog, key)
        if shipped is None:
            continue
        for code in sorted(shipped - set(lib.sublibraries)):
            unfetchable.append(f"{key}: catalog ships '{code}', LIBRARIES cannot fetch it")
    assert not unfetchable, "\n".join(unfetchable)


def test_everything_declared_is_shipped_or_explained(catalog):
    """The direction that hid #356.

    A declared sublibrary that the repo does not ship is either deliberate
    (endfb-8.1's neutron data, retired in #263 in favour of endfb-8.0) or a
    mistake (iaea-medical's, which 404s). Those look identical in a diff, so
    the deliberate ones must say so in `UNSHIPPED_SUBLIBRARIES` and anything
    else fails here.
    """
    unexplained: list[str] = []
    for key, lib in sorted(LIBRARIES.items()):
        shipped = _projectiles(catalog, key)
        if shipped is None:
            continue
        for code in sorted(set(lib.sublibraries) - shipped):
            if (key, code) not in UNSHIPPED_SUBLIBRARIES:
                unexplained.append(f"{key}: LIBRARIES declares '{code}', catalog.json does not ship it")
    assert not unexplained, (
        "\n".join(unexplained) + "\n\nEither remove the sublibrary from LIBRARIES, or add it to "
        "UNSHIPPED_SUBLIBRARIES with the reason it is declared but not shipped."
    )


def test_iaea_medical_declares_no_neutron_sublibrary():
    """The specific regression. IAEA-Medical/n/ is a 404 on the mirror and
    always was; the catalog and `data/iaea-medical/xs/` have said p and d for
    some time. Asserted by name, so re-adding it fails loudly rather than
    costing someone a multi-hour run that exits 0."""
    assert "n" not in LIBRARIES["iaea-medical"].sublibraries
    assert set(LIBRARIES["iaea-medical"].sublibraries) >= {"p", "d"}


def test_the_check_would_have_caught_the_iaea_medical_declaration(catalog, monkeypatch):
    """Negative control. Put the 404ing sublibrary back and the check must
    object — otherwise it is agreeing with whatever the registry happens to say,
    which is what it exists to stop."""
    monkeypatch.setitem(LIBRARIES["iaea-medical"].sublibraries, "n", "n")
    with pytest.raises(AssertionError, match="iaea-medical: LIBRARIES declares 'n'"):
        test_everything_declared_is_shipped_or_explained(catalog)


def test_the_refetch_check_would_have_caught_a_dropped_projectile(catalog, monkeypatch):
    """Negative control for the other direction: remove a projectile the repo
    actually ships and the rebuild-deletes-data check must fire."""
    monkeypatch.delitem(LIBRARIES["iaea-medical"].sublibraries, "p")
    with pytest.raises(AssertionError, match="catalog ships 'p'"):
        test_everything_shipped_can_be_refetched(catalog)


def test_a_sweep_skips_the_sublibraries_the_repo_does_not_ship(monkeypatch, tmp_path):
    """`--sublibrary n --all` must not write endfb-8.1 neutron parquets.

    They were retired in #263 and `catalog.json` does not list them, so a sweep
    that fetched them would create exactly the registry/catalog drift the checks
    above forbid — and, per #341, a stray tree under `data/` that the next
    person has to `git clean` away.
    """
    import fetch_endf_libs as m

    fetched: list[tuple[str, str]] = []
    monkeypatch.setattr(m, "fetch_library", lambda key, sub, out, session: fetched.append((key, sub)))
    monkeypatch.setattr(sys, "argv", ["fetch_endf_libs.py", "--all", "--sublibrary", "n", "--output", str(tmp_path)])
    m.main()

    assert fetched, "the sweep fetched nothing at all"
    assert ("endfb-8.1", "n") not in fetched, "a sweep ingested a retired sublibrary"
    assert ("irdff-2", "n") in fetched, "a sweep skipped a library it should have fetched"


def test_naming_a_retired_sublibrary_explicitly_still_fetches_it(monkeypatch, tmp_path):
    """Skipped in a sweep, not removed. The counterpart to the test above —
    without it, 'skip' and 'delete the capability' are indistinguishable."""
    import fetch_endf_libs as m

    fetched: list[tuple[str, str]] = []
    monkeypatch.setattr(m, "fetch_library", lambda key, sub, out, session: fetched.append((key, sub)))
    argv = ["fetch_endf_libs.py", "--library", "endfb-8.1", "--sublibrary", "n", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()

    assert fetched == [("endfb-8.1", "n")]


def test_unshipped_entries_are_real_and_reasoned():
    """The allowlist must describe the registry as it is — a stale entry would
    quietly widen what the check above permits."""
    for (key, code), reason in sorted(UNSHIPPED_SUBLIBRARIES.items()):
        assert key in LIBRARIES, f"UNSHIPPED_SUBLIBRARIES names unknown library {key!r}"
        assert code in LIBRARIES[key].sublibraries, f"{key} no longer declares {code!r} — drop the entry"
        assert len(reason) > 30, f"{key}/{code}: give an actual reason, got {reason!r}"


def test_on_disk_files_match_the_catalogued_projectiles(catalog):
    """A third witness, so the two tables above cannot agree with each other and
    both be wrong: `data/<key>/xs/<projectile>_<Element>.parquet`."""
    checked = 0
    for key in sorted(LIBRARIES):
        shipped = _projectiles(catalog, key)
        xs_dir = ROOT / "data" / key / "xs"
        if shipped is None or not xs_dir.is_dir():
            continue
        on_disk = {path.name.split("_", 1)[0] for path in xs_dir.glob("*_*.parquet")}
        assert on_disk, f"data/{key}/xs holds no parquet files"
        assert on_disk == shipped, f"data/{key}/xs holds {sorted(on_disk)}, catalog says {sorted(shipped)}"
        checked += 1
    assert checked >= len(LIBRARIES) - 1, (
        f"only {checked} of {len(LIBRARIES)} registry libraries had an xs/ directory to check — "
        "this test passes trivially if the data tree is not there"
    )


# ---------------------------------------------------------------------------
# list_endf_files must fail rather than report an empty success
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def get(self, url: str, timeout: int | None = None) -> _FakeResponse:  # noqa: ARG002
        return self._response


def test_a_404_listing_raises():
    """What a re-ingest of iaea-medical/n did before #356: 404, one warning,
    exit code 0."""
    from fetch_endf_libs import list_endf_files

    session = _FakeSession(_FakeResponse("<html>Not Found</html>", status_code=404))
    with pytest.raises(RuntimeError, match="cannot list"):
        list_endf_files(LIBRARIES["irdff-2"], "n", session)


def test_an_empty_but_successful_listing_raises():
    """HTTP 200 with no .zip links — a mirror that reorganised its directory
    layout. Indistinguishable from success in the old code."""
    from fetch_endf_libs import list_endf_files

    session = _FakeSession(_FakeResponse("<html><body>Index of /n</body></html>"))
    with pytest.raises(RuntimeError, match="listed no"):
        list_endf_files(LIBRARIES["irdff-2"], "n", session)


def test_an_undeclared_sublibrary_raises():
    from fetch_endf_libs import list_endf_files

    session = _FakeSession(_FakeResponse(""))
    with pytest.raises(KeyError, match="declares no"):
        list_endf_files(LIBRARIES["iaea-medical"], "n", session)


def test_a_real_listing_still_comes_back():
    """The positive case — without it the three above pass on a function that
    raises unconditionally."""
    from fetch_endf_libs import list_endf_files

    html = '<a href="n_013-Al-27_1325.zip">a</a> <a href="n_026-Fe-56_2631.zip">b</a>'
    session = _FakeSession(_FakeResponse(html))
    assert list_endf_files(LIBRARIES["irdff-2"], "n", session) == [
        "n_013-Al-27_1325.zip",
        "n_026-Fe-56_2631.zip",
    ]


# ---------------------------------------------------------------------------
# The parser must be able to read what the registry says it will fetch (#372)
# ---------------------------------------------------------------------------
#
# `parse_endf_filename` came in with #334, which re-ingested seven libraries with
# `--sublibrary n` only. No `he3_`/`he4_` filename was ever fed to it, and both
# its regexes began `[a-z]+_`, which cannot match a projectile code containing a
# digit. Every file in five shipped sublibraries — endfb-8.1 h/a, jendl-5 a,
# tendl-2025 h/a, 242 shards — returned None, so a rebuild aborted on the
# empty-ingest guard and could not regenerate any of them.
#
# The registry already knows the codes. Nothing checked that the parser can read
# what they serve, so these do.

#: The naming conventions observed on the IAEA mirror, as
#: (suffix-after-the-prefix, expected (Z, A, isomer)). The prefix is supplied
#: per sublibrary — it is the sublibrary's *directory* name, which is what the
#: mirror uses to prefix every file in it.
_FILENAME_CONVENTIONS: tuple[tuple[str, tuple[int, int, str]], ...] = (
    # Z zero-padded to 3 — most libraries.
    ("029-Cu-63_2925.zip", (29, 63, "")),
    # Z to 2, and to 1 for Z < 10 — IRDFF-II.
    ("79-Au-197_7925.zip", (79, 197, "")),
    ("3-Li-6_0325.zip", (3, 6, "")),
    # Element symbol UPPERCASE — ENDF/B-VIII.1's he3/he4 files, and nothing else.
    ("002-HE-4_0228.zip", (2, 4, "")),
    # Isomeric state suffix on A — JEFF/JENDL/TENDL.
    ("095-Am-242M_9547.zip", (95, 242, "m")),
    # MAT first, then Z, unpadded — BROND-3.1, and the he4 files.
    ("9640_96-Cm-245.zip", (96, 245, "")),
    ("1325_13-Al-27.zip", (13, 27, "")),
)


def _sublibrary_prefixes() -> list[tuple[str, str, str]]:
    """(library key, sublibrary code, mirror directory) for everything declared."""
    return [
        (key, code, directory)
        for key, lib in sorted(LIBRARIES.items())
        for code, directory in sorted(lib.sublibraries.items())
    ]


def test_every_declared_sublibrary_can_have_its_filenames_parsed():
    """The gate that would have caught #372.

    For every (library, sublibrary) the registry declares, a filename in each
    convention the mirror uses must parse. `he3`/`he4` are the cases that
    mattered: `[a-z]+_` matched neither, so the parser was blind to two of the
    six projectile codes the registry has always declared.
    """
    from fetch_endf_libs import parse_endf_filename

    entries = _sublibrary_prefixes()
    assert entries, "the registry declares no sublibraries at all"

    failures: list[str] = []
    checked = 0
    for key, code, directory in entries:
        for suffix, expected in _FILENAME_CONVENTIONS:
            filename = f"{directory}_{suffix}"
            checked += 1
            got = parse_endf_filename(filename)
            if got != expected:
                failures.append(f"{key}/{code}: parse_endf_filename({filename!r}) -> {got!r}, expected {expected!r}")

    assert not failures, "the ingest cannot read filenames it will be served:\n  " + "\n  ".join(failures)
    # Positive assertion: a loop that matched nothing would pass silently.
    assert checked == len(entries) * len(_FILENAME_CONVENTIONS)
    assert checked >= 100, f"only {checked} (library, sublibrary, convention) combinations checked"


def test_the_digit_bearing_projectile_codes_are_actually_exercised():
    """`he3` and `he4` are the whole point — assert they are in the sweep above,
    so a registry edit that dropped them could not quietly narrow this test."""
    directories = {directory for _key, _code, directory in _sublibrary_prefixes()}
    assert {"he3", "he4"} <= directories, f"he3/he4 missing from the registry: {sorted(directories)}"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # The three real he3/he4 shapes named in #372, verbatim.
        ("he4_002-HE-4_0228.zip", (2, 4, "")),
        ("he3_002-He-3_0225.zip", (2, 3, "")),
        ("he4_003-Li-6_0325.zip", (3, 6, "")),
        ("he4_1325_13-Al-27.zip", (13, 27, "")),
        # ENDF/B-VIII.1 is the only mirror that uppercases the element symbol,
        # so it gets its own case rather than riding on the sweep.
        ("he4_002-HE-4_0228.zip", (2, 4, "")),
        ("he3_002-HE-3_0225.zip", (2, 3, "")),
    ],
)
def test_helium_filenames_parse(filename, expected):
    from fetch_endf_libs import parse_endf_filename

    assert parse_endf_filename(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        # A MAT number where the projectile code belongs. `[a-z0-9]+_` — the
        # other candidate prefix — reads this as Z=29, A=63 and attributes the
        # file to copper. `[a-z]+\d*_` requires the shape a projectile code has.
        "9640_029-Cu-63_2925.zip",
        "42_029-Cu-63_2925.zip",
        "0_029-Cu-63_2925.zip",
        # No prefix at all.
        "029-Cu-63_2925.zip",
        "_029-Cu-63_2925.zip",
        "readme.txt",
    ],
)
def test_names_that_are_not_evaluations_are_rejected(filename):
    """Returning None reaches the empty-ingest guard, which is loud. Inventing
    an attribution for a malformed name is the quiet failure this file keeps
    being bitten by."""
    from fetch_endf_libs import parse_endf_filename

    assert parse_endf_filename(filename) is None


def test_all_sublibs_skips_the_sublibraries_the_repo_does_not_ship(monkeypatch, tmp_path):
    """`--library <x> --all-sublibs` is a sweep too (#372).

    #360 wired the skip into `--sublibrary <x> --all` only, so the two paths
    disagreed about what is in scope — and every `rebuild_command` in
    catalog.json uses the per-library form, so a rebuild drove the one path
    without the skip and attempted iaea-medical/a, which `UNSHIPPED_SUBLIBRARIES`
    records as never ingested.
    """
    import fetch_endf_libs as m

    fetched: list[tuple[str, str]] = []
    monkeypatch.setattr(m, "fetch_library", lambda key, sub, out, session: fetched.append((key, sub)))
    argv = ["fetch_endf_libs.py", "--library", "iaea-medical", "--all-sublibs", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()

    assert fetched, "the sweep fetched nothing at all"
    assert ("iaea-medical", "a") not in fetched, "--all-sublibs ingested an unshipped sublibrary"
    assert ("iaea-medical", "h") not in fetched, "--all-sublibs ingested an unshipped sublibrary"
    assert ("iaea-medical", "p") in fetched, "--all-sublibs skipped a sublibrary it should have fetched"
    assert ("iaea-medical", "d") in fetched


def test_all_sublibs_and_all_agree_on_scope(monkeypatch, tmp_path):
    """The two sweep spellings must select the same (library, sublibrary) set.

    Stated as an invariant rather than a list, so a future skip rule added to
    one path cannot drift from the other the way #360's did.
    """
    import fetch_endf_libs as m

    def run(argv: list[str]) -> set[tuple[str, str]]:
        fetched: set[tuple[str, str]] = set()
        monkeypatch.setattr(m, "fetch_library", lambda key, sub, out, session: fetched.add((key, sub)))
        monkeypatch.setattr(sys, "argv", ["fetch_endf_libs.py", *argv, "--output", str(tmp_path)])
        m.main()
        return fetched

    per_library: set[tuple[str, str]] = set()
    for key in LIBRARIES:
        per_library |= run(["--library", key, "--all-sublibs"])

    everything: set[tuple[str, str]] = set()
    for code in sorted({c for lib in LIBRARIES.values() for c in lib.sublibraries}):
        everything |= run(["--all", "--sublibrary", code])

    assert per_library == everything, (
        "--all-sublibs and --all disagree about scope:\n"
        f"  only --all-sublibs: {sorted(per_library - everything)}\n"
        f"  only --all:         {sorted(everything - per_library)}"
    )
    assert per_library, "both sweeps selected nothing"

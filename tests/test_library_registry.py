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

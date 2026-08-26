"""The committed parquets must not drift away from the builder that made them.

`scripts/fetch_endf_libs.py` was fixed in July 2026 (#260) and only `endfb-8.1`
was rebuilt. Seven neutron libraries shipped pre-fix data for thirteen months —
Au-197(n,γ) at 1 MeV reading ~3,953 mb instead of ~83 — and every test passed
the whole time, because nothing in the repository related a library to the code
that produced it. #334 fixed the data; #342 is this, the missing relation.

Two layers here:

  1. **The guard on the real tree** — every library either verifies against its
     builder or carries an explicit, issue-bearing entry in
     `data/builder_stamp_exemptions.json`.
  2. **The guard on the guard** — synthetic fixtures that prove the mechanism
     actually goes red when a builder moves. A staleness check that quietly
     matches nothing is the exact failure mode this file exists to prevent: the
     first version of the Au-197 capture gate asserted `xs_mb == 0`, found no
     rows, and passed while the data was wrong by nineteen orders of magnitude.

Not marked `@pytest.mark.data`: the check reads `catalog.json`, the manifests
and the builder scripts, all of which are in every checkout. It runs in PR CI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nucl_parquet import builder_stamp
from nucl_parquet.builder_stamp import (
    EXEMPTION_REASONS,
    EXEMPTIONS_FILE,
    audit,
    format_report,
    make_stamp,
    manifest_path_for,
    script_digest,
    write_builder_stamp,
)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_EXEMPTIONS = _DATA_DIR / EXEMPTIONS_FILE


def _exemptions() -> dict[str, dict]:
    return json.loads(_EXEMPTIONS.read_text())["exemptions"]


# ---------------------------------------------------------------------------
# Layer 1: the guard on the real data tree
# ---------------------------------------------------------------------------


def test_no_library_has_drifted_from_its_builder() -> None:
    """The load-bearing gate.

    Fails when a library's stamped builder digest disagrees with the script on
    disk, when a library ships without saying what built it, or when the
    exemption ledger has drifted from reality.

    If this goes red because a builder legitimately changed and the data has not
    been re-ingested yet, that is the check working. Either re-ingest (the
    failure message prints the command) or add a `stale-accepted` entry naming
    the issue that will.
    """
    findings = audit(_DATA_DIR, _REPO_ROOT)
    assert not findings, "\n" + format_report(findings)


def test_every_library_that_ships_parquets_declares_a_builder() -> None:
    """`catalog.json` must say what produced each library it declares a `path` for.

    Five libraries were dropped in by data-only commits with no ingest script
    anywhere in the tree (#346). That is worth knowing, and `"builder": null`
    says it out loud instead of leaving the field absent, where "nobody recorded
    it" and "there is nothing to record" look identical.
    """
    catalog = json.loads((_DATA_DIR / "catalog.json").read_text())
    missing = [
        key
        for key, info in sorted(catalog["libraries"].items())
        if "path" in info and info.get("path") and "builder" not in info
    ]
    assert not missing, (
        "libraries with a `path` but no `builder` in catalog.json: "
        + ", ".join(missing)
        + "\n  Set it to the producing script's repo-relative path, or to null if the "
        "library was produced outside this repository (see #346)."
    )


def test_declared_builders_exist() -> None:
    """A `builder` path that names nothing is worse than none — it looks checked."""
    catalog = json.loads((_DATA_DIR / "catalog.json").read_text())
    dangling = [
        f"{key} -> {info['builder']}"
        for key, info in sorted(catalog["libraries"].items())
        if info.get("builder") and not (_REPO_ROOT / info["builder"]).exists()
    ]
    assert not dangling, "catalog.json names builders that do not exist: " + ", ".join(dangling)


def test_a_rebuild_command_that_would_downgrade_the_schema_chains_the_migration() -> None:
    """The remedy this PR prints must not itself be a defect.

    `scripts/build_neutron_njoy.py` and `nucl_parquet/build_hi_xs.py` still emit
    the pre-migration 6-column form (#359). Their committed data is 18-column
    canonical only because `scripts/migrate_xs_schema.py` was run once, after the
    fact, and nothing chains the two. So running the bare ingest on any library
    they build drops 12 of 18 columns — `library`, `kind`, `projectile`,
    `target_Z`, `MT` and all provenance — which is `CLAUDE.md` principle 5 undone
    by the normal maintenance operation, silently, with the ingest exiting 0.

    That knowledge lived in no file. It lives in `rebuild_command` now, and this
    keeps it there: the requirement is **derived from the builder's source**, not
    from a list someone has to remember to update. Add a library whose builder
    does not emit canonical rows and forget the migration, and this fails.

    `scripts/fetch_endf_libs.py` was the third such builder and the one behind
    nine of the eleven libraries this originally checked. #347/#359 fixed it, so
    it now writes canonical directly and drops out of the loop by itself —
    exactly the "the requirement lifts" path this docstring described. The floor
    below moved 11 -> 2 to match; raise it back if a new legacy builder lands,
    and retire the test entirely once these last two are converted.

    The source grep is a heuristic. The load-bearing proof that `fetch_endf_libs`
    really emits canonical rows is
    `test_fetch_endf_libs.py::test_written_file_is_exactly_the_canonical_schema`,
    which ingests a material and compares the written columns and dtypes.
    """
    catalog = json.loads((_DATA_DIR / "catalog.json").read_text())
    problems: list[str] = []
    checked = 0
    for key, info in sorted(catalog["libraries"].items()):
        builder, cmd = info.get("builder"), info.get("rebuild_command")
        if not builder or not cmd:
            continue
        src = (_REPO_ROOT / builder).read_text()
        # Either spelling counts: `build_channels.py` builds its frame straight
        # from CANONICAL_XS_SCHEMA, `fetch_endf_libs.py` goes through
        # `_canonical.canonical_frame`. Both land in the same shape.
        if "CANONICAL_XS_SCHEMA" in src or "canonical_frame" in src:
            continue  # emits canonical form directly; nothing to migrate
        checked += 1
        if f"migrate_xs_schema.py --library {key}" not in cmd:
            problems.append(
                f"{key}: built by {builder}, which writes the legacy 6-column schema, but\n      {cmd!r}\n      does not chain scripts/migrate_xs_schema.py --library {key}"
            )

    # Positive assertion. If the loop matched nothing this test would pass by
    # finding nothing — the exact failure mode #342 exists to prevent.
    assert checked >= 2, (
        f"only {checked} librarie(s) matched the legacy-schema builders; this test has stopped "
        "checking what it was written for. Two remain after #347/#359 — endfb-8.0 "
        "(build_neutron_njoy.py) and hi-xs (build_hi_xs.py). If both have since been "
        "converted, retire this test rather than lowering the floor again."
    )
    assert not problems, "rebuild_command would revert these libraries to the legacy schema (#359):\n  " + "\n  ".join(
        problems
    )


# ---------------------------------------------------------------------------
# Layer 1b: the exemption ledger cannot become a dumping ground
# ---------------------------------------------------------------------------


def test_every_exemption_names_a_reason_and_an_issue() -> None:
    """Each excused library must say *why* and *what removes it*.

    An allowlist without an owner is how a thirteen-month gap starts. The issue
    reference is the part that matters: it is the only thing that turns "we know
    about this" into work someone can pick up.
    """
    problems: list[str] = []
    for key, entry in sorted(_exemptions().items()):
        if entry.get("reason") not in EXEMPTION_REASONS:
            problems.append(f"{key}: reason {entry.get('reason')!r} not one of {sorted(EXEMPTION_REASONS)}")
        issue = entry.get("issue")
        if not isinstance(issue, int) or issue <= 0:
            problems.append(f"{key}: `issue` must be the number of the issue that removes this entry, got {issue!r}")
        note = entry.get("note")
        if not isinstance(note, str) or len(note.strip()) < 20:
            problems.append(f"{key}: `note` must explain the exemption in a sentence, got {note!r}")
        unknown = set(entry) - {"reason", "issue", "note"}
        if unknown:
            problems.append(f"{key}: unrecognised field(s) {sorted(unknown)}")
    assert not problems, f"malformed entries in data/{EXEMPTIONS_FILE}:\n  " + "\n  ".join(problems)


def test_external_builder_exemptions_agree_with_the_catalog() -> None:
    """`reason: external-builder` must correspond to `"builder": null`, both ways.

    Otherwise the two files can disagree about which libraries have no in-repo
    pipeline, and the ledger stops describing the catalog it excuses.
    """
    catalog = json.loads((_DATA_DIR / "catalog.json").read_text())
    null_builder = {
        key
        for key, info in catalog["libraries"].items()
        if info.get("path") and "builder" in info and info["builder"] is None
    }
    excused = {key for key, e in _exemptions().items() if e.get("reason") == "external-builder"}
    assert null_builder == excused, (
        f'catalog.json says `"builder": null` for {sorted(null_builder)} but '
        f"data/{EXEMPTIONS_FILE} excuses {sorted(excused)} as external-builder — see #346"
    )


def test_the_ledger_shrinks_it_does_not_grow() -> None:
    """A ceiling on the excused set, so adding one is a visible decision.

    19 libraries are grandfathered today (14 unstamped, 5 with no in-repo
    builder). Anything above that means an exemption was added rather than a
    library re-ingested, and the number in this assertion is the thing a
    reviewer sees move.
    """
    assert len(_exemptions()) <= 19, (
        f"data/{EXEMPTIONS_FILE} has grown to {len(_exemptions())} entries. "
        "Exemptions are debt (#345, #346) — re-ingest instead, or lower this ceiling "
        "in the same PR that justifies raising it."
    )


# ---------------------------------------------------------------------------
# Layer 2: the guard on the guard
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_tree(tmp_path: Path) -> Path:
    """A minimal repo: one builder script, one library, one stamped manifest."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "data" / "lib-a" / "xs").mkdir(parents=True)

    script = repo / "scripts" / "make_lib_a.py"
    script.write_text("# v1\nprint('build')\n")
    (repo / "data" / "lib-a" / "xs" / "n_Fe.parquet").write_bytes(b"PAR1")
    (repo / "data" / "catalog.json").write_text(
        json.dumps(
            {
                "libraries": {
                    "lib-a": {
                        "name": "Library A",
                        "data_type": "cross_sections",
                        "builder": "scripts/make_lib_a.py",
                        "rebuild_command": "python scripts/make_lib_a.py",
                        "path": "lib-a/xs/",
                    }
                }
            }
        )
    )
    write_builder_stamp(repo / "data" / "lib-a" / "manifest.json", script, files_written=1, repo_root=repo)
    return repo


def _audit(repo: Path) -> list:
    return audit(repo / "data", repo)


def test_a_stamped_library_that_matches_its_builder_is_clean(fake_tree: Path) -> None:
    """The baseline. If this ever fails, every other case below is meaningless."""
    assert _audit(fake_tree) == []


def test_editing_the_builder_makes_the_library_stale(fake_tree: Path) -> None:
    """The whole point: code moves, data does not, and the check says so.

    This is #260 -> #334 in miniature.
    """
    (fake_tree / "scripts" / "make_lib_a.py").write_text("# v2 — a correctness fix\nprint('build')\n")

    findings = _audit(fake_tree)
    assert [f.kind for f in findings] == ["stale"]
    assert findings[0].library == "lib-a"


def test_the_stale_report_names_the_builder_the_digests_and_the_fix(fake_tree: Path) -> None:
    """A failure nobody can act on is how thirteen months go by.

    The report must carry the library, the builder, both digests and the literal
    re-ingest command — asserted positively, because a message that degrades to
    "something is wrong" is the failure this issue is about.
    """
    script = fake_tree / "scripts" / "make_lib_a.py"
    before = script_digest(script)
    script.write_text("# v2\n")
    after = script_digest(script)

    report = format_report(_audit(fake_tree))
    for expected in ("lib-a", "scripts/make_lib_a.py", before, after, "python scripts/make_lib_a.py"):
        assert expected in report, f"{expected!r} missing from staleness report:\n{report}"
    assert before != after


def test_reformatting_the_builder_also_counts_as_stale(fake_tree: Path) -> None:
    """Deliberate, documented polarity.

    A whitespace-only edit marks the output stale. The alternative — deciding
    which edits "really" affect output — can be wrong in the silent direction,
    and silence is what this guard exists to end. Cost of a false positive: one
    `stale-accepted` line. Cost of a false negative: thirteen months.
    """
    script = fake_tree / "scripts" / "make_lib_a.py"
    script.write_text(script.read_text() + "\n\n")
    assert [f.kind for f in _audit(fake_tree)] == ["stale"]


def test_a_stale_accepted_exemption_silences_it(fake_tree: Path) -> None:
    """The documented escape hatch: knowingly shipping pre-change data.

    #340 lands a correctness fix in `fetch_endf_libs.py` with no re-ingest. Once
    those libraries carry stamps, this is the entry that keeps `main` green
    while naming the issue that re-ingests them.
    """
    (fake_tree / "scripts" / "make_lib_a.py").write_text("# v2\n")
    _write_exemptions(fake_tree, {"lib-a": {"reason": "stale-accepted", "issue": 340, "note": "x" * 30}})
    assert _audit(fake_tree) == []


def test_an_unstamped_exemption_does_not_cover_a_stale_library(fake_tree: Path) -> None:
    """Reasons are not interchangeable.

    Otherwise the grandfathering entries added today would silently become
    blanket permission for real staleness the moment a library is stamped.
    """
    (fake_tree / "scripts" / "make_lib_a.py").write_text("# v2\n")
    _write_exemptions(fake_tree, {"lib-a": {"reason": "unstamped", "issue": 345, "note": "x" * 30}})

    findings = _audit(fake_tree)
    assert [f.kind for f in findings] == ["stale"]
    # And it says why the excuse did not apply, rather than reporting the
    # unrelated "this exemption is no longer needed".
    assert "does not cover" in findings[0].detail


def test_a_library_with_no_stamp_is_flagged_unstamped(fake_tree: Path) -> None:
    manifest = fake_tree / "data" / "lib-a" / "manifest.json"
    manifest.write_text(json.dumps({"library": "lib-a", "files": 1}))
    assert [f.kind for f in _audit(fake_tree)] == ["unstamped"]


def test_a_stamp_naming_a_different_script_is_flagged(fake_tree: Path) -> None:
    """Catches a copy-pasted stamp, and a builder that was renamed under the data."""
    manifest = fake_tree / "data" / "lib-a" / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["builder"]["script"] = "scripts/some_other_builder.py"
    manifest.write_text(json.dumps(doc))
    kinds = [f.kind for f in _audit(fake_tree)]
    assert kinds == ["builder-mismatch"], kinds


def test_a_library_without_a_declared_builder_is_flagged(fake_tree: Path) -> None:
    catalog = fake_tree / "data" / "catalog.json"
    doc = json.loads(catalog.read_text())
    del doc["libraries"]["lib-a"]["builder"]
    catalog.write_text(json.dumps(doc))
    assert [f.kind for f in _audit(fake_tree)] == ["no-builder-declared"]


def test_a_null_builder_needs_an_external_builder_exemption(fake_tree: Path) -> None:
    catalog = fake_tree / "data" / "catalog.json"
    doc = json.loads(catalog.read_text())
    doc["libraries"]["lib-a"]["builder"] = None
    catalog.write_text(json.dumps(doc))
    assert [f.kind for f in _audit(fake_tree)] == ["external-builder"]

    _write_exemptions(fake_tree, {"lib-a": {"reason": "external-builder", "issue": 346, "note": "x" * 30}})
    assert _audit(fake_tree) == []


def test_an_exemption_that_is_no_longer_needed_fails(fake_tree: Path) -> None:
    """Self-cleaning.

    Re-ingesting a library must force its grandfathering entry out, or the
    ledger silently accumulates and stops meaning anything. This is what makes
    #345 and #346 finite.
    """
    _write_exemptions(fake_tree, {"lib-a": {"reason": "unstamped", "issue": 345, "note": "x" * 30}})
    findings = _audit(fake_tree)
    assert [f.kind for f in findings] == ["dead-exemption"]
    assert "345" in findings[0].detail


def test_an_exemption_for_an_unknown_library_fails(fake_tree: Path) -> None:
    _write_exemptions(fake_tree, {"lib-zzz": {"reason": "unstamped", "issue": 345, "note": "x" * 30}})
    assert [f.kind for f in _audit(fake_tree)] == ["unknown-library"]


def test_the_check_works_without_git(fake_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Correctness must not depend on git history.

    `actions/checkout` fetches depth 1, the release artefact is a tarball, and
    the sdist has no `.git` at all. A guard that only works inside a full clone
    is not a guard on the published data. Git is used only to enrich the
    message, and its absence is reported rather than read as "clean".
    """
    monkeypatch.setattr(builder_stamp, "_git", lambda *a, **k: None)
    (fake_tree / "scripts" / "make_lib_a.py").write_text("# v2\n")

    findings = _audit(fake_tree)
    assert [f.kind for f in findings] == ["stale"]
    assert findings[0].commits is None
    assert "unavailable" in findings[0].render()


def test_a_stamp_made_without_git_records_unknown_not_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`commit` and `dirty` are nullable on purpose.

    A build outside a checkout must say "I do not know which commit this was"
    rather than claim a clean tree it never inspected.
    """
    monkeypatch.setattr(builder_stamp, "_git", lambda *a, **k: None)
    script = tmp_path / "b.py"
    script.write_text("x = 1\n")

    stamp = make_stamp(script, tmp_path)
    assert stamp["script"] == "b.py"
    assert stamp["sha256"] == script_digest(script)
    assert stamp["commit"] is None
    assert stamp["dirty"] is None


def test_a_malformed_stamp_is_a_finding_not_a_pass(fake_tree: Path) -> None:
    """A hand-edited manifest must not be able to fake a clean bill of health.

    `builder.sha256` missing, null, or not a digest has to be distinguishable
    from `builder.sha256` matching. Getting this wrong is the same class of
    mistake as the `xs_mb == 0` gate that passed by finding nothing.
    """
    manifest = fake_tree / "data" / "lib-a" / "manifest.json"
    original = json.loads(manifest.read_text())

    for broken in ({}, {"script": "scripts/make_lib_a.py"}, {"script": "scripts/make_lib_a.py", "sha256": None}):
        manifest.write_text(json.dumps({**original, "builder": broken}))
        kinds = [f.kind for f in _audit(fake_tree)]
        assert kinds and kinds != ["unstamped"], f"{broken!r} slipped through as {kinds}"

    # A scalar where an object belongs must be a finding, not an AttributeError.
    manifest.write_text(json.dumps({**original, "builder": "scripts/make_lib_a.py"}))
    assert [f.kind for f in _audit(fake_tree)] == ["builder-mismatch"]


def test_a_builder_path_escaping_the_repo_is_rejected(fake_tree: Path) -> None:
    """`../` would point the digest at a file outside the checkout.

    That is a guard whose verdict depends on the machine running it — it would
    pass or fail for reasons nobody can reproduce.
    """
    catalog = fake_tree / "data" / "catalog.json"
    original = json.loads(catalog.read_text())
    for escape in ("../outside.py", "/etc/hostname", ""):
        doc = json.loads(json.dumps(original))
        doc["libraries"]["lib-a"]["builder"] = escape
        catalog.write_text(json.dumps(doc))
        assert [f.kind for f in _audit(fake_tree)] == ["builder-mismatch"], f"{escape!r} was accepted"


def test_make_stamp_refuses_a_script_outside_the_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="outside"):
        make_stamp(outside, repo)


def test_a_missing_exemption_ledger_makes_the_check_stricter_not_weaker(fake_tree: Path) -> None:
    """Deleting the ledger must not be a way to silence anything.

    `load_exemptions` returns `{}` for an absent file, so every excused library
    goes back to being a finding. Malformed JSON raises rather than being read
    as "no exemptions apply to anyone" — either way, never a silent pass.
    """
    manifest = fake_tree / "data" / "lib-a" / "manifest.json"
    manifest.write_text(json.dumps({"library": "lib-a"}))
    _write_exemptions(fake_tree, {"lib-a": {"reason": "unstamped", "issue": 345, "note": "x" * 30}})
    assert _audit(fake_tree) == []

    (fake_tree / "data" / EXEMPTIONS_FILE).unlink()
    assert [f.kind for f in _audit(fake_tree)] == ["unstamped"]

    (fake_tree / "data" / EXEMPTIONS_FILE).write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        _audit(fake_tree)


def test_a_run_that_wrote_nothing_must_not_stamp(tmp_path: Path) -> None:
    """The guard on the mitigation itself.

    A builder whose fetches all failed leaves the *previous* build's parquets on
    disk. Stamping there would attest that the current builder produced them —
    re-manufacturing #260 -> #334 inside the thing meant to detect it. Checking
    the output directory would not catch it, because the directory looks full;
    only "how many files did *this run* write" does.
    """
    script = tmp_path / "b.py"
    script.write_text("x = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"library": "lib-a", "files": 97}))

    for count in (0, -1):
        with pytest.raises(RuntimeError, match="refusing to stamp"):
            write_builder_stamp(manifest, script, files_written=count, repo_root=tmp_path)

    # And the manifest is untouched — no half-written artefact claiming success.
    assert "builder" not in json.loads(manifest.read_text())

    write_builder_stamp(manifest, script, files_written=1, repo_root=tmp_path)
    assert json.loads(manifest.read_text())["builder"]["sha256"] == script_digest(script)


def test_write_builder_stamp_preserves_the_rest_of_the_manifest(tmp_path: Path) -> None:
    """The row and file counts `build_manifests.py` derives must survive a stamp."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"library": "lib-a", "files": 97, "total_rows": 2_812_689}))
    script = tmp_path / "b.py"
    script.write_text("x = 1\n")

    write_builder_stamp(manifest, script, files_written=97, repo_root=tmp_path)

    doc = json.loads(manifest.read_text())
    assert doc["files"] == 97
    assert doc["total_rows"] == 2_812_689
    assert doc["builder"]["sha256"] == script_digest(script)


def test_build_manifests_never_writes_a_stamp(tmp_path: Path) -> None:
    """`build_manifests.py` regenerates from data that may be years old.

    Stamping there would attest that today's builder produced yesterday's
    parquets — precisely the lie the stamp exists to detect. Only a real ingest
    may stamp, so the round-trip through `build_manifests.py` must neither add a
    stamp nor drop one.
    """
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from build_manifests import build_manifest

    pq_dir = tmp_path / "lib-a" / "xs"
    pq_dir.mkdir(parents=True)
    shutil.copy(_DATA_DIR / "endfb-8.1" / "xs" / "a_Be.parquet", pq_dir / "a_Be.parquet")

    fresh = build_manifest("lib-a", pq_dir)
    assert "builder" not in fresh
    assert fresh["total_rows"] > 0  # positive assertion: it really read the file

    existing = {"builder": {"script": "scripts/x.py", "sha256": "0" * 64, "commit": None, "dirty": None}}
    assert {**existing, **fresh}["builder"] == existing["builder"]


def test_an_empty_ingest_leaves_no_artefact_to_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BROND-3.1's silent success, from the provenance side.

    Every filename used a shape the parser did not recognise, each was skipped
    with a warning, and the run exited 0. #334 added a `RuntimeError` and left an
    older `logger.warning(); return` on the same condition above it, so the guard
    was unreachable. This branch found that independently while wiring the stamp
    — a builder that can report success on an empty ingest can stamp one — but
    #354 landed the identical fix first, at the identical position, with two
    further guards on top. Both placements were compared; that one wins and this
    branch keeps none of its own.

    What stays is the half `test_empty_ingest_guard_raises_rather_than_returning`
    does not assert: that **nothing was written**. "It raised" and "it raised
    before leaving a manifest claiming 97 files" are different guarantees, and
    only the second is what a provenance stamp rests on.
    """
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import fetch_endf_libs

    # Files are found, and every one of them is skipped — the BROND-3.1 shape.
    monkeypatch.setattr(fetch_endf_libs, "list_endf_files", lambda *a: ["n_9640_96-Cm-245.zip"] * 3)
    monkeypatch.setattr(fetch_endf_libs, "download_and_parse", lambda *a: fetch_endf_libs.ParsedFile(rows=[]))

    with pytest.raises(RuntimeError, match="Refusing to report success on an empty ingest"):
        fetch_endf_libs.fetch_library("brond-3.1", "n", tmp_path, session=None)

    assert not list(tmp_path.rglob("*.json")), "an empty ingest left a manifest behind"
    assert not list(tmp_path.rglob("*.parquet"))


def test_manifest_path_rule_matches_the_committed_layout() -> None:
    """The rule the builders and `build_manifests.py` now share.

    `endfb-8.0/xs/` and `endfb-8.0/channels/` are two libraries under one root;
    walking up unconditionally made the second silently overwrite the first.
    """
    assert manifest_path_for(_DATA_DIR / "endfb-8.0" / "xs", "endfb-8.0") == _DATA_DIR / "endfb-8.0" / "manifest.json"
    assert (
        manifest_path_for(_DATA_DIR / "endfb-8.0" / "channels", "endfb-8.0-channels")
        == _DATA_DIR / "endfb-8.0" / "channels" / "manifest.json"
    )
    assert manifest_path_for(_DATA_DIR / "exfor", "exfor") == _DATA_DIR / "exfor" / "manifest.json"
    for path in (
        manifest_path_for(_DATA_DIR / "endfb-8.0" / "xs", "endfb-8.0"),
        manifest_path_for(_DATA_DIR / "endfb-8.0" / "channels", "endfb-8.0-channels"),
        manifest_path_for(_DATA_DIR / "exfor", "exfor"),
    ):
        assert path.exists(), f"{path} should already be committed"


def _write_exemptions(repo: Path, exemptions: dict) -> None:
    (repo / "data" / EXEMPTIONS_FILE).write_text(json.dumps({"exemptions": exemptions}))

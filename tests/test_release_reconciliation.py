"""Gates on the weekly release reconciliation (#364).

#344 was found by a human noticing an absence, as #274 and #308 were before it.
#350 closed the merge path — the tag push and the release trigger are now one
operation, with a confirmation step. What it did not close is everything that
happens outside that merge: a hand-pushed tag, `release-data.yml` failing after
it started, a release or asset deleted later, or the confirmation step itself
being edited into uselessness. All of those land back in #344's state.

So `scripts/reconcile_data_release.sh` asserts the invariant directly — the
version `data/catalog.json` claims is published, complete and signed — rather
than asserting that the machinery which maintains it looks right. A check that
fails for the same reasons as the thing it checks is not a check.

Two layers, as in `test_data_signing.py`:

1. **Wiring** — parse `reconcile-data-release.yml` and assert the parts that are
   load-bearing and silent when wrong: that it is scheduled at all, that the
   reconcile step cannot abort the job before the issue is filed, that a failure
   *files an issue* rather than only going red, that a repeat failure updates
   instead of duplicating, and that a recovery closes it.

2. **Behaviour** — run the real script against a fake `gh`, driving each way a
   release can be wrong: no tag, no release, a draft, a missing asset, a
   zero-byte asset, a bad signature. Each must fail, and the report must name
   what to do about it.

The second layer is the one that matters. Layer 1 only proves the YAML reads
right; a detector whose detection logic is broken is worse than none, because
its silence is now evidence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "reconcile-data-release.yml"
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile_data_release.sh"
_VERIFY_SCRIPT = _REPO_ROOT / "scripts" / "verify_data_release.sh"
_CATALOG = _REPO_ROOT / "data" / "catalog.json"

# The remedy a reader of the filed issue needs. Same string #350 put into
# auto-tag-data.yml's failure paths, for the same reason: before #344 it existed
# only in somebody's head.
_REMEDY = "gh workflow run release-data.yml -f tag="

_needs_jq = pytest.mark.skipif(
    shutil.which("jq") is None,
    reason="jq not on PATH — run inside the devShell (nix develop)",
)


def _wf() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def _steps() -> list[dict]:
    return _wf()["jobs"]["reconcile"]["steps"]


def _step(fragment: str) -> dict:
    frag = fragment.lower()
    matches = [s for s in _steps() if frag in str(s.get("name", "")).lower()]
    assert matches, (
        f"reconcile-data-release.yml has no step named like {fragment!r}. "
        f"Steps: {[s.get('name') or s.get('uses') for s in _steps()]}"
    )
    assert len(matches) == 1, f"expected one step matching {fragment!r}, got {[s['name'] for s in matches]}"
    return matches[0]


def _index(fragment: str) -> int:
    frag = fragment.lower()
    return next(i for i, s in enumerate(_steps()) if frag in str(s.get("name", "")).lower())


# -- Layer 1: the workflow is wired to actually tell someone -----------------


def test_reconciliation_is_scheduled() -> None:
    """It runs on a schedule, or it is not a detector at all.

    Everything else in this file is downstream of the job actually firing
    without anyone asking it to.
    """
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1).
    triggers = _wf().get("on") or _wf().get(True)
    schedule = triggers.get("schedule")
    assert schedule, "reconcile-data-release.yml must declare a `schedule:` trigger"
    cron = schedule[0]["cron"]
    assert len(cron.split()) == 5, f"malformed cron {cron!r}"
    assert "workflow_dispatch" in triggers, (
        "keep workflow_dispatch: the first thing anyone does with a failing detector is re-run it by hand"
    )


def test_the_job_can_file_an_issue() -> None:
    """`issues: write`, and deliberately not the App token.

    The nucl-parquet-release-bot installation has no `issues` scope, so an App
    token cannot open the issue this job exists to open. #350's lesson was
    "confirm the credential holds the permission", not "always use the App".
    """
    perms = _wf()["permissions"]
    assert perms.get("issues") == "write", (
        "GITHUB_TOKEN needs `issues: write` — a failure that only goes red is the failure mode #364 is about"
    )
    assert perms.get("contents") == "read"

    body = _WORKFLOW.read_text()
    assert "create-github-app-token" not in body, (
        "do not mint an App token here: the installation carries no `issues` scope, "
        "so it could not file the issue, and reaching for it anyway would repeat "
        "#344's mistake of assuming a credential's permissions"
    )


def test_the_reconcile_step_cannot_abort_the_job_before_the_issue_is_filed() -> None:
    """`continue-on-error` on the check, or nothing downstream ever runs.

    This is the single most breakable thing in the workflow: without it the
    script's non-zero exit ends the job immediately, the issue is never filed,
    and the whole detector degrades to exactly the red-run-nobody-reads that
    #364 says is insufficient. It fails *silently* — the run is red either way.
    """
    check = _step("Reconcile")
    assert check.get("id") == "check"
    assert check.get("continue-on-error") is True, (
        "the reconcile step must be continue-on-error, otherwise a failure aborts "
        "the job before the issue-filing step and the detector goes quiet exactly "
        "when it has something to say"
    )
    assert "reconcile_data_release.sh" in check["run"]
    assert "--report" in check["run"], "the report file is what becomes the issue body"


def test_failure_files_an_issue_before_the_run_is_failed() -> None:
    """Order matters: file first, go red second.

    If the failing step came first, `exit 1` would end the job and the issue
    would never be created — the workflow would look like it reports, and not.
    """
    file_idx = _index("Open or update the tracking issue")
    fail_idx = _index("Fail the run")
    assert file_idx < fail_idx, (
        f"the issue-filing step (index {file_idx}) must precede the step that fails "
        f"the run (index {fail_idx}); otherwise the job dies before filing"
    )

    # `always()` matters as much as the ordering. Without it these steps carry an
    # implicit success() gate, so any earlier step failing — an apt hiccup, an
    # API blip in the issue lookup — suppresses the filing while the job still
    # goes red. That is "the run was red and nobody looked" reintroduced one
    # layer out from where #364 removed it.
    for name in ("Open or update the tracking issue", "Fail the run"):
        cond = str(_step(name).get("if", ""))
        assert cond == "always() && steps.check.outcome == 'failure'", (
            f"{name!r} has `if: {cond}`. It must be "
            "`always() && steps.check.outcome == 'failure'`, or an unrelated earlier "
            "failure silently suppresses the only output that reaches a person."
        )


def test_the_issue_lookup_cannot_fail_the_job() -> None:
    """A transient lookup error must not swallow a real finding.

    Not knowing whether an issue is already open risks a duplicate. Not filing
    at all is the failure this workflow exists to prevent, so the lookup
    degrades to "none open" rather than failing.
    """
    run = _step("Find any open reconciliation issue")["run"]
    # Match the directive, not the word — the step's own comment explains why
    # `set -e` is absent, and a substring check would trip over that.
    errexit = [ln for ln in run.splitlines() if re.match(r"\s*set\s+-[a-zA-Z]*e", ln)]
    assert not errexit, f"errexit here turns an API blip into a swallowed finding: {errexit}"
    assert "|| true" in run, "the lookup must tolerate failure"
    assert re.search(r'\[\[ "\$\{NUM\}" =~ \^\[0-9\]\+\$ \]\] \|\| NUM=""', run), (
        "a failed lookup must yield an empty number, not gh's error text, or the filing step branches on garbage"
    )


def test_a_repeat_failure_updates_rather_than_duplicating() -> None:
    """One issue per outage, not one per Monday.

    A detector that opens a fresh issue every week for the same unfixed problem
    teaches people to filter it, which is the same end state as not having it.
    """
    run = _step("Open or update the tracking issue")["run"]
    assert "gh issue comment" in run, "an existing issue must be commented on, not duplicated"
    assert "gh issue create" in run, "the first failure must open the issue"
    assert "EXISTING" in run, "the two paths must branch on whether an issue is already open"

    lookup = _step("Find any open reconciliation issue")["run"]
    assert "--state open" in lookup, "a closed issue from a previous outage must not suppress a new one"
    assert "TRACKING_LABEL" in lookup, (
        "dedupe on a label: titles carry the version and change, and body search is "
        "indexed asynchronously, so both would file duplicates"
    )


def test_recovery_closes_the_issue() -> None:
    """The label has to mean 'currently broken' for the dedupe to be correct."""
    step = _step("Close the tracking issue")
    cond = str(step.get("if", ""))
    assert "steps.check.outcome == 'success'" in cond
    assert "steps.tracking.outputs.number != ''" in cond, (
        "only close when an issue is actually open, or every green run tries to close nothing and goes red doing it"
    )
    assert "gh issue close" in step["run"]


def test_the_job_cannot_race_itself() -> None:
    """Two runs filing the same issue is the duplicate this dedupe cannot catch.

    The lookup and the create are separate API calls, so a dispatch racing the
    schedule can interleave between them.
    """
    concurrency = _wf().get("concurrency")
    assert concurrency, "reconcile-data-release.yml must declare a concurrency group"
    assert concurrency.get("cancel-in-progress") is False, (
        "cancel-in-progress must be false: cancelling a run mid-flight can drop the "
        "issue-filing step, which is the one output that reaches a person"
    )


def test_minisign_is_installed() -> None:
    """The signature check is the only one that needs a binary the runner lacks.

    Without it verify_data_release.sh fails on every run for an environment
    reason, and a detector that cries wolf weekly gets muted.
    """
    assert any("minisign" in str(s.get("run", "")) for s in _steps()), (
        "the runner must install minisign; scripts/verify_data_release.sh needs it"
    )


def _run_step(step_name: str, tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess, str]:
    """Execute one workflow step's real `run:` body against a recording `gh`.

    The issue-filing step is the only output of this workflow that reaches a
    person, and it is the piece a schedule-only workflow cannot be tried out
    on before merging. Asserting the YAML mentions `gh issue create` proves
    nothing about whether the branch is reachable or the quoting survives, so
    the body is lifted out and run — the same technique test_data_signing.py
    uses on the signing step.
    """
    body = _step(step_name)["run"]
    assert "${{" not in body, f"{step_name!r} interpolates into the shell; bind through env instead (#289)"

    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    calls = tmp_path / "gh-calls.txt"
    gh = bindir / "gh"
    gh.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{calls}"\nexit 0\n')
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok=True)
    proc = subprocess.run(
        ["bash", "-e", "-c", body],
        env={
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "RUNNER_TEMP": str(runner_temp),
            "TRACKING_LABEL": "release-reconciliation",
            "RUN_URL": "https://example.invalid/run/1",
            **env,
        },
        capture_output=True,
        text=True,
    )
    return proc, (calls.read_text() if calls.exists() else "")


def test_the_first_failure_opens_an_issue(tmp_path: Path) -> None:
    """No issue open yet — the step must create one, labelled for the dedupe."""
    (tmp_path / "runner-temp").mkdir()
    (tmp_path / "runner-temp" / "report.md").write_text("- no published release for `data-2026.8.3`.\n")
    proc, calls = _run_step("Open or update the tracking issue", tmp_path, {"EXISTING": ""})
    assert proc.returncode == 0, proc.stderr
    assert "issue create" in calls, f"expected an issue to be created, gh saw: {calls!r}"
    assert "release-reconciliation" in calls, "the new issue must carry the dedupe label"
    assert "issue comment" not in calls


def test_a_second_failure_comments_on_the_open_issue(tmp_path: Path) -> None:
    """An issue is already open — comment, never open a second one."""
    (tmp_path / "runner-temp").mkdir()
    (tmp_path / "runner-temp" / "report.md").write_text("- still broken\n")
    proc, calls = _run_step("Open or update the tracking issue", tmp_path, {"EXISTING": "42"})
    assert proc.returncode == 0, proc.stderr
    assert "issue comment 42" in calls, f"expected a comment on #42, gh saw: {calls!r}"
    assert "issue create" not in calls, "a duplicate issue every Monday is how a detector gets muted"


def test_an_empty_report_still_files_something_honest(tmp_path: Path) -> None:
    """The script died before it could diagnose — file that, not an empty issue.

    A blank issue body is indistinguishable from a formatting bug, and the
    reader has no way to tell that the diagnosis is missing rather than absent.
    """
    (tmp_path / "runner-temp").mkdir()
    (tmp_path / "runner-temp" / "report.md").write_text("")
    proc, calls = _run_step("Open or update the tracking issue", tmp_path, {"EXISTING": ""})
    assert proc.returncode == 0, proc.stderr
    body = (tmp_path / "runner-temp" / "issue.md").read_text()
    assert "failed before it could diagnose" in body
    assert "issue create" in calls


def test_the_issue_body_carries_the_report_and_the_run_link(tmp_path: Path) -> None:
    """Whoever reads the issue needs both the finding and the run that found it."""
    (tmp_path / "runner-temp").mkdir()
    (tmp_path / "runner-temp" / "report.md").write_text("- asset `x.tar.zst` is 0 bytes.\n")
    proc, _ = _run_step("Open or update the tracking issue", tmp_path, {"EXISTING": ""})
    assert proc.returncode == 0, proc.stderr
    body = (tmp_path / "runner-temp" / "issue.md").read_text()
    assert "is 0 bytes" in body, "the report must survive into the issue body"
    assert "https://example.invalid/run/1" in body, "and the run that produced it must be linked"


def test_recovery_closes_with_a_comment(tmp_path: Path) -> None:
    """Closing silently leaves the reader guessing whether it was fixed or dropped."""
    proc, calls = _run_step("Close the tracking issue", tmp_path, {"EXISTING": "42"})
    assert proc.returncode == 0, proc.stderr
    assert "issue close 42" in calls
    assert "--comment" in calls and "verifies against" in calls


def test_no_workflow_opts_back_into_node_20() -> None:
    """`ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION` stops working on 2026-09-23.

    GitHub flipped JavaScript actions to Node 24 by default on 2026-06-16 and
    removes the Node 20 runtime on 2026-09-23; that variable is the opt-back-in
    and dies with it. Nothing here sets it, which is what made deleting
    release-please.yml's stale FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 opt-in safe
    (#365) — this keeps it that way rather than leaving the reasoning in a
    commit message.
    """
    wf_dir = _REPO_ROOT / ".github" / "workflows"
    offenders = [
        p.name for p in sorted(wf_dir.glob("*.yml")) if "ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION" in p.read_text()
    ]
    assert not offenders, (
        f"{offenders} pin JavaScript actions back to Node 20. That runtime is removed "
        "on 2026-09-23; the workflow will fail then. Update the action instead."
    )


# -- Layer 2: run the real script -------------------------------------------


def _fake_gh(bindir: Path, fixtures: Path) -> None:
    """Install a `gh` stub that answers from fixture files.

    The script's entire view of GitHub is two `gh api` calls, which makes this a
    clean seam: every branch below is driven by what these fixtures do or do not
    contain, with no network and no live release to depend on.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    gh = bindir / "gh"
    gh.write_text(
        f"""#!/usr/bin/env bash
FIX="{fixtures}"
case "$*" in
  *git/ref/tags/*)
    if [ -f "$FIX/tag.txt" ]; then cat "$FIX/tag.txt"; exit 0; fi
    echo "gh: Not Found (HTTP 404)" >&2; exit 1 ;;
  *releases/tags/*)
    if [ -f "$FIX/release.json" ]; then cat "$FIX/release.json"; exit 0; fi
    echo "gh: Not Found (HTTP 404)" >&2; exit 1 ;;
esac
echo "fake gh: unexpected call: $*" >&2
exit 99
"""
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _assets(version: str, *, omit: str | None = None, zero: str | None = None) -> list[dict]:
    names = [
        f"nucl-parquet-data-{version}.tar.zst",
        f"nucl-parquet-data-{version}.tar.zst.minisig",
        f"nucl-parquet-data-{version}.manifest.json",
        f"nucl-parquet-data-{version}.manifest.json.minisig",
    ]
    out = []
    for n in names:
        if omit and n.endswith(omit):
            continue
        out.append({"name": n, "size": 0 if (zero and n.endswith(zero)) else 4242})
    return out


def _run(
    tmp_path: Path,
    *,
    version: str = "2026.8.3",
    tag: str | None = "a" * 40,
    release: dict | None = None,
    verify_rc: int = 0,
    args: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess, str]:
    """Execute the real reconcile script against fixtures. Returns (proc, report)."""
    fixtures = tmp_path / "fix"
    fixtures.mkdir(parents=True, exist_ok=True)
    if tag is not None:
        (fixtures / "tag.txt").write_text(tag + "\n")
    if release is not None:
        (fixtures / "release.json").write_text(json.dumps(release))

    bindir = tmp_path / "bin"
    _fake_gh(bindir, fixtures)

    # Stand in for verify_data_release.sh. Exercising this script's handling of
    # a signature failure must not require a 727MB download or a real bad
    # signature; VERIFY_SCRIPT is the documented seam, as PUBKEY_FILE is in the
    # script it stands in for.
    stub = tmp_path / "verify-stub.sh"
    stub.write_text(f'#!/usr/bin/env bash\necho "stub verify $*"\nexit {verify_rc}\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    report = tmp_path / "report.md"
    proc = subprocess.run(
        ["bash", str(_SCRIPT), version, "--report", str(report), *(args or [])],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "VERIFY_SCRIPT": str(stub),
            "GITHUB_REPOSITORY": "exoma-ch/nucl-parquet",
        },
        capture_output=True,
        text=True,
    )
    return proc, (report.read_text() if report.exists() else "")


def _good_release(version: str = "2026.8.3", **kw) -> dict:
    return {"draft": False, "published_at": "2026-08-26T09:54:24Z", "assets": _assets(version, **kw)}


@_needs_jq
def test_a_healthy_release_reconciles(tmp_path: Path) -> None:
    proc, report = _run(tmp_path, release=_good_release())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "is published, complete and signed" in proc.stdout
    assert report == "", "a passing run must leave no report behind to be filed as an issue"


@_needs_jq
def test_a_missing_tag_is_reported(tmp_path: Path) -> None:
    proc, report = _run(tmp_path, tag=None, release=_good_release())
    assert proc.returncode == 1
    assert "no tag `data-2026.8.3`" in report


@_needs_jq
def test_the_344_state_is_reported(tmp_path: Path) -> None:
    """Tag present, release absent — precisely what #344 left behind."""
    proc, report = _run(tmp_path, release=None)
    assert proc.returncode == 1
    assert "no published release" in report
    assert 'resolving "latest" gets the previous version' in report or "latest" in report
    assert _REMEDY in report, "the report must carry the remedy, not just the diagnosis"


@_needs_jq
def test_a_draft_release_does_not_count_as_published(tmp_path: Path) -> None:
    """A draft is invisible to consumers and to the download URLs.

    It is also the shape a half-finished manual recovery leaves behind, so
    treating "a release object exists" as success would bless exactly that.
    """
    rel = _good_release()
    rel["draft"] = True
    proc, report = _run(tmp_path, release=rel)
    assert proc.returncode == 1
    assert "draft" in report


@_needs_jq
@pytest.mark.parametrize(
    "missing",
    [".tar.zst.minisig", ".manifest.json", ".manifest.json.minisig"],
)
def test_a_missing_asset_is_reported(tmp_path: Path, missing: str) -> None:
    proc, report = _run(tmp_path, release=_good_release(omit=missing))
    assert proc.returncode == 1
    assert "missing the asset" in report and missing in report


@_needs_jq
def test_a_zero_byte_asset_is_reported(tmp_path: Path) -> None:
    """Worse than missing: it downloads fine and fails at whatever consumes it."""
    proc, report = _run(tmp_path, release=_good_release(zero=".tar.zst"))
    assert proc.returncode == 1
    assert "0 bytes" in report


@_needs_jq
def test_a_non_numeric_asset_size_is_reported(tmp_path: Path) -> None:
    """`[ null -eq 0 ]` is a syntax error, and `set -e` is off here.

    Left unguarded the script falls through to the else branch and reports the
    asset fine — a check whose failure mode is "reports OK", which is the exact
    species of bug this whole file exists to catch.
    """
    rel = _good_release()
    rel["assets"][0]["size"] = None
    proc, report = _run(tmp_path, release=rel)
    assert proc.returncode == 1
    assert "non-numeric size" in report


@_needs_jq
def test_report_flag_requires_a_path(tmp_path: Path) -> None:
    """A bare trailing `--report` must not silently write to /dev/null."""
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "2026.8.3", "--skip-signature", "--report"],
        env={**os.environ, "GITHUB_REPOSITORY": "exoma-ch/nucl-parquet"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, "a usage error must exit 2, not proceed with the flag inert"
    assert "--report needs a path" in proc.stderr


@_needs_jq
def test_an_unsigned_by_design_release_says_so_rather_than_not_verified(tmp_path: Path) -> None:
    """ "Skipped" and "does not apply" are different claims.

    A release below the signing floor was checked and found not to need a
    signature — a complete answer. Reporting that as "NOT verified this run"
    would read as an open question every week for a release that has none.
    """
    floor = re.search(r'^FIRST_SIGNED_VERSION="([^"]+)"', _VERIFY_SCRIPT.read_text(), re.M).group(1)
    assert floor == "2026.8.2", "floor moved; update this test's fixture version"
    old = "2026.6.0"
    rel = {"draft": False, "published_at": "x", "assets": [{"name": f"nucl-parquet-data-{old}.tar.zst", "size": 9}]}
    proc, _ = _run(tmp_path, version=old, release=rel)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "carries no signature by design" in proc.stdout
    assert "NOT verified" not in proc.stdout


@_needs_jq
def test_a_failing_signature_is_reported(tmp_path: Path) -> None:
    """The bytes are there but do not verify against the key consumers pin."""
    proc, report = _run(tmp_path, release=_good_release(), verify_rc=1)
    assert proc.returncode == 1
    assert "verify_data_release.sh" in report


@_needs_jq
def test_all_findings_are_collected_not_just_the_first(tmp_path: Path) -> None:
    """A report naming one of two problems sends the reader down one branch."""
    proc, report = _run(tmp_path, tag=None, release=None)
    assert proc.returncode == 1
    assert "no tag" in report and "no published release" in report


@_needs_jq
def test_the_report_is_actionable(tmp_path: Path) -> None:
    """It must say what to run, not merely what is wrong."""
    _, report = _run(tmp_path, release=None)
    assert _REMEDY + "data-2026.8.3" in report
    assert "reconcile_data_release.sh" in report, "the reader must be able to reproduce it locally"
    assert "Auto-tag data release" in report, "and be told which workflow re-creates a missing tag"


@_needs_jq
def test_skip_signature_does_not_claim_the_signature_was_checked(tmp_path: Path) -> None:
    """#283's lesson: never report having done something that was skipped."""
    proc, _ = _run(tmp_path, release=_good_release(), args=["--skip-signature"])
    assert proc.returncode == 0
    assert "Signature NOT verified" in proc.stdout
    assert "complete and signed" not in proc.stdout


@_needs_jq
def test_releases_below_the_manifest_floor_do_not_require_the_manifest(tmp_path: Path) -> None:
    """The manifest pair first shipped in FIRST_MANIFEST_VERSION (#296).

    Requiring it unconditionally would report every historical release as
    broken, which is the fastest way to get a detector switched off.
    """
    floor = re.search(r'^FIRST_MANIFEST_VERSION="([^"]+)"', _VERIFY_SCRIPT.read_text(), re.M).group(1)
    assert floor == "2026.8.3", "floor moved; update this test's fixture version"
    old = "2026.8.2"  # signed, but predates the manifest
    rel = {"draft": False, "published_at": "x", "assets": _assets(old, omit=".manifest.json")}
    rel["assets"] = [a for a in rel["assets"] if "manifest" not in a["name"]]
    proc, _ = _run(tmp_path, version=old, release=rel)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@_needs_jq
def test_the_calver_floor_comparison_is_numeric_not_lexical(tmp_path: Path) -> None:
    """2026.10.0 is above 2026.8.3, and a string compare says otherwise.

    Lexically "2026.10.0" < "2026.8.3", so a naive comparison would stop
    requiring the signature and manifest the month October lands — silently
    weakening the check exactly when nobody is looking at it.
    """
    v = "2026.10.0"
    rel = {"draft": False, "published_at": "x", "assets": _assets(v, omit=".manifest.json.minisig")}
    proc, report = _run(tmp_path, version=v, release=rel)
    assert proc.returncode == 1, (
        "2026.10.0 is above the manifest floor, so the manifest signature is still "
        "required; a lexical comparison would have skipped the check"
    )
    assert "manifest.json.minisig" in report


@_needs_jq
def test_the_default_version_comes_from_the_catalog(tmp_path: Path) -> None:
    """No argument means "whatever data/catalog.json currently claims".

    That is the invariant being policed, so reading it from anywhere else — or
    needing to be told — would let the catalog and the check drift apart.
    """
    expected = json.loads(_CATALOG.read_text())["data_version"]
    fixtures = tmp_path / "fix"
    fixtures.mkdir()
    bindir = tmp_path / "bin"
    _fake_gh(bindir, fixtures)  # no fixtures: everything 404s
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "--skip-signature"],
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert f"data-{expected}" in proc.stdout + proc.stderr

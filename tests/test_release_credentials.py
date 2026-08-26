"""Gates on how the release pipeline authenticates and how it fails (#344).

Three expired-credential incidents, one shape each time:

* **#274** — `RELEASE_PLEASE_TOKEN` expired; every package release blocked.
* **#308** — `CARGO_REGISTRY_TOKEN` invalid; 403 on crates.io, half-published.
* **#344** — `DATA_RELEASE_PAT` expired; `data-2026.8.3` tagged, never released.

All three were found the same way: a human went looking for something
downstream that had not happened. So there are two things to hold closed, and
they are separate properties:

1. **No workflow depends on a credential that expires.** GitHub App
   installation tokens are minted per run and have no expiry to forget, and the
   repo already has the App (`RP_APP_ID` / `RP_APP_PRIVATE_KEY`, added in #274).
   The allowlist below is deliberately a positive assertion, not a `*_PAT`
   pattern match: adding any new secret to any workflow has to come past this
   test, which is the review step the three incidents never got.

2. **No half-completed release is reachable, and none is silent.** #344's real
   damage was not the 401. It was that the tag push had already succeeded, so
   `data/catalog.json` claimed 2026.8.3, the tag existed, no tarball or
   signature did, consumers resolving "latest" silently got 2026.8.2 — and the
   obvious retry was refused by the workflow's own "tag already exists" guard.
   `auto-tag-data.yml` now performs a single write (the App-token tag push,
   which fires `release-data.yml` by itself), checks its credentials before
   that write, reconciles rather than refuses when a tag has no release, and
   names the manual remedy verbatim on every path that can leave one needed.

As in `test_data_signing.py`, the wiring assertions are backed by executing the
workflow's real `run:` body: string checks still pass with a missing `then` or
a stray quote, and running it does not.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_AUTO_TAG = _WORKFLOW_DIR / "auto-tag-data.yml"
_RELEASE_DATA = _WORKFLOW_DIR / "release-data.yml"

# The exact string a human needs when a version is tagged with no release. It
# lived only in somebody's head until #344, which is why the tag sat unreleased.
_REMEDY = "gh workflow run release-data.yml -f tag="

# Every secret any workflow is allowed to reference, and why it cannot be an
# App token. Adding an entry is a deliberate act with a reviewer attached.
_ALLOWED_SECRETS = {
    # The release-bot GitHub App. Minted per run; nothing to expire.
    "RP_APP_ID",
    "RP_APP_PRIVATE_KEY",
    # Not GitHub credentials — a GitHub App cannot authenticate to these.
    "CARGO_REGISTRY_TOKEN",
    "HF_TOKEN",
    # Offline signing key material (#289), rotated by ceremony, not by expiry.
    "DATA_SIGNING_KEY",
    "DATA_SIGNING_KEY_PASSWORD",
}

_SECRET_REF = re.compile(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)")

# `gh` as the first word of a line, after an optional `if ` and any number of
# leading `VAR=value` environment assignments. See _invokes_dispatch.
_DISPATCH_CALL = re.compile(
    r"^[ \t]*(?:if[ \t]+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S*[ \t]+)*gh[ \t]+workflow[ \t]+run\b",
    re.M,
)


def _workflows() -> list[Path]:
    paths = sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml"))
    assert paths, f"no workflows found under {_WORKFLOW_DIR} — this test would assert nothing"
    return paths


def _steps(workflow: Path, job: str) -> list[dict]:
    wf = yaml.safe_load(workflow.read_text())
    return wf["jobs"][job]["steps"]


def _auto_tag_steps() -> list[dict]:
    return _steps(_AUTO_TAG, "auto-tag")


def _step(name_fragment: str) -> dict:
    """Return the single auto-tag step whose name contains `name_fragment`."""
    frag = name_fragment.lower()
    matches = [s for s in _auto_tag_steps() if frag in str(s.get("name", "")).lower()]
    assert matches, (
        f"auto-tag-data.yml has no step named like {name_fragment!r}. "
        f"Steps present: {[s.get('name') or s.get('uses') for s in _auto_tag_steps()]}"
    )
    assert len(matches) == 1, f"expected one step matching {name_fragment!r}, got {[s['name'] for s in matches]}"
    return matches[0]


def _index_of(predicate) -> int:
    steps = _auto_tag_steps()
    return next(i for i, s in enumerate(steps) if predicate(s))


def _invokes_dispatch(step: dict) -> bool:
    """True if the step *runs* `gh workflow run`, rather than quoting it.

    Several error messages name `gh workflow run release-data.yml -f tag=…`
    verbatim on purpose — that is the manual remedy #344 needed and nobody had
    written down. Substring-matching the whole body would count those quoted
    remedies as dispatch calls and make the single-write test vacuous.

    So match the shape of a command rather than the presence of a string: `gh`
    must be the first word of the line, give or take an `if ` and any leading
    `VAR=value` assignments. A quoted continuation, a comment, a heredoc line
    and a `printf` argument all fail that test, which is the point.
    """
    return bool(_DISPATCH_CALL.search(str(step.get("run", ""))))


# -- 1. No expiring credentials ---------------------------------------------


@pytest.mark.parametrize("workflow", _workflows(), ids=lambda p: p.name)
def test_no_workflow_references_an_unreviewed_secret(workflow: Path) -> None:
    """Every `secrets.X` in every workflow is on the allowlist above.

    A `*_PAT` pattern check would have caught #344 and missed #274 (a PAT named
    `RELEASE_PLEASE_TOKEN`) and #308 (`CARGO_REGISTRY_TOKEN`). Enumerating what
    is allowed catches the next one whatever it is called.
    """
    referenced = set(_SECRET_REF.findall(workflow.read_text()))
    unknown = referenced - _ALLOWED_SECRETS - {"GITHUB_TOKEN"}
    assert not unknown, (
        f"{workflow.name} references un-allowlisted secret(s) {sorted(unknown)}. "
        "If it is a GitHub credential, use the release-bot App "
        "(actions/create-github-app-token with RP_APP_ID / RP_APP_PRIVATE_KEY) — "
        "long-lived PATs have expired silently three times (#274, #308, #344). "
        "If it genuinely cannot be an App token, add it to _ALLOWED_SECRETS with "
        "a comment saying why."
    )


@pytest.mark.parametrize("workflow", _workflows(), ids=lambda p: p.name)
def test_no_workflow_uses_a_personal_access_token(workflow: Path) -> None:
    """The specific #344 regression: `DATA_RELEASE_PAT`, or anything shaped like it.

    Kept alongside the allowlist rather than folded into it because it names the
    incident. A future edit that reintroduces a PAT should fail against a test
    that explains what happened, not only against a set-difference.
    """
    referenced = set(_SECRET_REF.findall(workflow.read_text()))
    pats = {s for s in referenced if s.endswith("_PAT") or s == "DATA_RELEASE_PAT"}
    assert not pats, (
        f"{workflow.name} references {sorted(pats)}. `DATA_RELEASE_PAT` expired and "
        "auto-tag-data.yml pushed the data-2026.8.3 tag and then failed to publish "
        "anything behind it (#344). Use the release-bot App token instead — it is "
        "minted per run and does not expire."
    )


def test_auto_tag_mints_the_release_bot_app_token() -> None:
    """The credential is the App, wired the same way release-please.yml wires it."""
    minters = [s for s in _auto_tag_steps() if "create-github-app-token" in str(s.get("uses", ""))]
    assert len(minters) == 1, (
        "auto-tag-data.yml must mint exactly one App installation token via "
        f"actions/create-github-app-token, found {len(minters)}"
    )
    with_ = minters[0].get("with", {})
    assert "secrets.RP_APP_ID" in str(with_.get("app-id", ""))
    assert "secrets.RP_APP_PRIVATE_KEY" in str(with_.get("private-key", ""))
    assert minters[0].get("id") == "app-token", "the token step must be addressable as steps.app-token"


def test_the_tag_is_pushed_with_the_app_token() -> None:
    """`origin` must carry the App credential, or the tag triggers nothing.

    A tag pushed with GITHUB_TOKEN does not fire `on: push` — that restriction
    is the reason the old workflow needed a second, separately-authenticated
    dispatch at all. An App installation token does fire it, which is what lets
    the release be one operation instead of two.
    """
    checkouts = [s for s in _auto_tag_steps() if "actions/checkout" in str(s.get("uses", ""))]
    assert len(checkouts) == 1, f"expected one checkout step, got {len(checkouts)}"
    token = str(checkouts[0].get("with", {}).get("token", ""))
    assert "steps.app-token.outputs.token" in token, (
        "actions/checkout must be given the App token so `git push origin <tag>` "
        "authenticates as the App. With the default GITHUB_TOKEN the tag lands and "
        "release-data.yml never starts — the exact half-state #344 left behind."
    )

    push = _step("Push the data tag")
    assert "git push origin" in push["run"]


# -- 2. The half-completed state ---------------------------------------------


def test_credentials_are_verified_before_any_ref_moves() -> None:
    """Preflight precedes the tag push.

    #344 failed *after* the write. Checking first is the difference between
    "re-run the job" and a repository claiming a version it cannot serve.
    """
    verify_idx = _index_of(lambda s: "verify credentials" in str(s.get("name", "")).lower())
    push_idx = _index_of(lambda s: "push the data tag" in str(s.get("name", "")).lower())
    mint_idx = _index_of(lambda s: "create-github-app-token" in str(s.get("uses", "")))

    assert mint_idx < push_idx, "the App token must be minted before the tag is pushed"
    assert verify_idx < push_idx, (
        f"credential verification (step {verify_idx}) must run before the tag push "
        f"(step {push_idx}); otherwise a bad credential leaves a tag with no release."
    )

    run = _step("Verify credentials")["run"]
    assert "steps.app-token" not in run, "no ${{ }} interpolation in the step body — bind through env (#289)"
    assert "gh api" in run and "repos/${GITHUB_REPOSITORY}" in run, (
        "preflight must actually call the API with the App token; minting a token "
        "only proves the PEM parses, not that the installation still covers this repo"
    )
    assert "release-data.yml" in run, (
        "preflight must confirm release-data.yml is enabled — a disabled workflow "
        "turns the tag push into the same silent no-op the App token was meant to fix"
    )


def test_the_release_is_a_single_write() -> None:
    """No routine dispatch step: the tag push *is* the trigger.

    Two writes is what made a partial failure possible. The only `gh workflow
    run` left must be the recovery path, gated on a tag that already exists
    without a release — never on the normal merge path.
    """
    dispatchers = [s for s in _auto_tag_steps() if _invokes_dispatch(s)]
    assert len(dispatchers) == 1, (
        f"expected exactly one `gh workflow run` step (the recovery path), got {[s.get('name') for s in dispatchers]}"
    )
    cond = str(dispatchers[0].get("if", ""))
    assert "steps.state.outputs.tag_exists == 'true'" in cond, (
        "the dispatch must be reachable only when the tag already exists; on the "
        "normal path the App-token tag push is what starts release-data.yml"
    )
    assert "steps.state.outputs.release_exists == 'false'" in cond, (
        "the dispatch must be reachable only when no release exists, or a re-run "
        "would rebuild and re-upload an already-published release"
    )

    push_cond = str(_step("Push the data tag").get("if", ""))
    assert "steps.state.outputs.tag_exists == 'false'" in push_cond
    assert "steps.state.outputs.release_exists == 'false'" in push_cond


def test_a_tagged_but_unreleased_version_is_retried_not_refused() -> None:
    """The #344 state must be recoverable by re-running the job.

    The old workflow's response to an existing tag was `exit 1, refusing to
    re-tag`, which is right about not moving the tag and wrong about what to do
    instead — so the one state that most needed a retry was the one state that
    could not have one.
    """
    step = _step("Re-trigger the release")
    run = step["run"]
    assert "gh workflow run release-data.yml" in run
    assert "::warning::" in run, "re-triggering is a recoverable condition, not an error"
    assert _REMEDY in run, (
        "the fallback must name the manual command verbatim: dispatching needs "
        "Actions: write on the App installation, which the normal path deliberately "
        "does not require, so this step can legitimately fail"
    )


def test_state_is_resolved_from_both_the_tag_and_the_release() -> None:
    """Idempotency needs both facts, not just "does the tag exist".

    Tag-and-release is a green no-op; tag-without-release is the failure to
    recover from; release-without-tag must be refused rather than re-tagged onto
    whatever `main` is now. Keying only on the tag collapses all three.
    """
    run = _step("Resolve what already exists")["run"]
    assert "git ls-remote" in run and "refs/tags/" in run
    assert "releases/tags/" in run, "the release lookup must query the release for this exact tag"

    # "No release" must mean a 404, not "the lookup failed somehow". Every other
    # failure has to stop the job: a transient error read as "no release" would
    # route an already-published version into the recovery dispatch and
    # re-publish it over the top of itself.
    assert "HTTP 404" in run, (
        "the release lookup must distinguish a 404 from any other failure; "
        "treating every error as 'no release' makes the idempotency checks lie"
    )
    assert "::error::could not determine" in run and "exit 1" in run, (
        "an indeterminate release lookup must fail the job rather than guess"
    )
    # Both facts must reach `$GITHUB_OUTPUT`, not merely the run log. Every
    # branch below keys off `steps.state.outputs.*`, and an output that is never
    # written reads as the empty string — which matches neither 'true' nor
    # 'false', so every branch silently skips and the job goes green having done
    # nothing at all.
    for key in ("tag_exists", "release_exists"):
        assert re.search(rf'^\s*echo "{key}=\$\{{\w+}}" >> "\$GITHUB_OUTPUT"\s*$', run, re.M), (
            f"the state step must write {key} to $GITHUB_OUTPUT; the downstream "
            f"`if:` conditions read steps.state.outputs.{key}"
        )

    already = _step("Already released")
    assert "steps.state.outputs.release_exists == 'true'" in str(already.get("if", ""))
    assert "exit 1" not in str(already.get("run", "")), (
        "an already-released version is a no-op, not a failure — otherwise every "
        "re-run of this job goes red and stops being a safe first response"
    )

    refuse = _step("Refuse to re-tag a published release")
    assert "exit 1" in refuse["run"]
    assert "::error::" in refuse["run"]


def test_the_tag_push_is_confirmed_to_have_started_the_release() -> None:
    """Assert the trigger fired; do not assume it.

    This is the detector the three incidents lacked. Everything else makes the
    half-state unreachable by construction, but "unreachable by construction" is
    exactly what was believed before each of them.
    """
    step = _step("Confirm release-data.yml started")
    assert str(step.get("if", "")) == "steps.push-tag.outcome == 'success'", (
        "the confirmation must run when — and only when — this job pushed the tag"
    )
    run = step["run"]
    assert "actions/workflows/release-data.yml/runs" in run, "must query for a real release-data.yml run"
    assert "$ENV.TAG" in run, "the run must be matched against this tag, not merely against 'any recent run'"
    assert "exit 1" in run and "::error::" in run, "a tag with no release must fail the job loudly"
    assert _REMEDY in run, "the error must name the manual remedy verbatim"


@pytest.mark.parametrize(
    "step_name",
    ["Refuse to re-tag a published release", "Re-trigger the release", "Confirm release-data.yml started"],
)
def test_every_post_tag_failure_names_the_remedy(step_name: str) -> None:
    """Once a tag exists, a red job must tell the reader how to finish the release."""
    run = _step(step_name)["run"]
    assert "exit 1" in run, f"{step_name!r} is expected to have a failure path"
    assert _REMEDY in run or "Restore the tag" in run, (
        f"{step_name!r} can fail after a tag exists but does not say what to do about it. "
        f"Name the remedy: {_REMEDY}data-<version>"
    )


def test_release_data_still_accepts_a_tag_push() -> None:
    """The push trigger is now the entire release mechanism.

    Deleting `on: push: tags:` from release-data.yml would leave auto-tag-data.yml
    pushing tags into silence, with nothing else in the repository objecting.
    """
    wf = yaml.safe_load(_RELEASE_DATA.read_text())
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1).
    triggers = wf.get("on") or wf.get(True)
    assert "data-*" in triggers["push"]["tags"], (
        "release-data.yml must trigger on `data-*` tag pushes — that push, made with "
        "the release-bot App token, is what publishes a data release (#344)"
    )
    assert "tag" in triggers["workflow_dispatch"]["inputs"], (
        f"release-data.yml must keep its `tag` dispatch input: `{_REMEDY}data-<version>` "
        "is the documented manual remedy and the recovery path auto-tag-data.yml uses"
    )


def test_auto_tag_can_read_actions_state() -> None:
    """GITHUB_TOKEN needs `actions: read` for the preflight and the confirmation.

    Both degrade to a hard failure without it, so a tightened `permissions:`
    block would turn a working release into a red one on every merge.
    """
    wf = yaml.safe_load(_AUTO_TAG.read_text())
    perms = wf["permissions"]
    assert perms.get("actions") == "read", (
        "auto-tag-data.yml needs `actions: read` to check that release-data.yml is "
        "enabled and that the tag push actually started it"
    )
    assert perms.get("contents") == "read", (
        "GITHUB_TOKEN should not carry write access here — the App token does every "
        "write, and `contents: read` is what makes that explicit rather than incidental"
    )


# -- Layer 2: run the real `run:` body ---------------------------------------


def test_detect_step_does_not_interpolate_into_the_shell() -> None:
    """No `${{ }}` in the detect body — a precondition for executing it below.

    Same house rule as the signing step (#289): an expansion is textual
    substitution before bash parses the line. It is also what makes the
    execution tests real rather than a re-implementation.
    """
    step = _step("Detect the data_version")
    assert "${{" not in step["run"]
    # The harness below runs the body as `bash -e`, which is what a `run:` block
    # with no `shell:` key gets. An explicit `shell:` here would change the flags
    # the runner uses and quietly desynchronise the test from production.
    assert "shell" not in step, "the execution harness assumes the runner's default `bash -e`"


def test_the_job_cannot_race_itself() -> None:
    """Two concurrent runs must not both decide the tag does not exist yet.

    The state check and the tag push are separate steps, so a merge racing a
    manual dispatch (or two dispatches) can interleave between them. Cancelling
    is the wrong remedy — a run cancelled after its push but before its
    confirmation abandons exactly the unwatched half-state this workflow closes.
    """
    wf = yaml.safe_load(_AUTO_TAG.read_text())
    concurrency = wf.get("concurrency")
    assert concurrency, "auto-tag-data.yml must declare a concurrency group"
    assert concurrency.get("cancel-in-progress") is False, (
        "cancel-in-progress must be false: cancelling a run that has already pushed "
        "the tag leaves it unconfirmed and unreleased, which is the #344 state"
    )


def _run_detect(
    workdir: Path, *, old: str | None, new: str, event_name: str
) -> tuple[subprocess.CompletedProcess, dict]:
    """Execute auto-tag-data.yml's detect step against a two-commit repo.

    `old` is the `data_version` in HEAD~1 (None writes no catalog there), `new`
    the one in HEAD — the same shape as a merge that bumps the catalog.
    """
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(workdir),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=workdir, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (data / "catalog.json").write_text(f'{{"data_version": "{old or "0000.0.0"}"}}\n')
    git("add", "-A")
    git("commit", "-qm", "base")
    (data / "catalog.json").write_text(f'{{"data_version": "{new}"}}\n')
    git("add", "-A")
    # --allow-empty: the unchanged-version case is a real merge shape (a commit
    # that touches data/catalog.json without moving data_version), and here the
    # two catalogs are byte-identical.
    git("commit", "-qm", "bump", "--allow-empty")

    github_output = workdir / "github-output"
    github_output.touch()
    proc = subprocess.run(
        # `-e`, because that is what the runner does. A `run:` block with no
        # `shell:` key executes as `bash -e {0}`; only an explicit `shell: bash`
        # adds `-o pipefail`. Running it any other way would let a body that
        # aborts on the runner pass here.
        ["bash", "-e", "-c", _step("Detect the data_version")["run"]],
        cwd=workdir,
        env={**env, "EVENT_NAME": event_name, "GITHUB_OUTPUT": str(github_output)},
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in github_output.read_text().splitlines() if "=" in line)
    return proc, outputs


def test_detect_emits_the_tag_when_the_version_changes(tmp_path: Path) -> None:
    proc, outputs = _run_detect(tmp_path, old="2026.8.2", new="2026.8.3", event_name="push")
    assert proc.returncode == 0, proc.stderr
    assert outputs["tag"] == "data-2026.8.3"
    assert outputs["version"] == "2026.8.3"


def test_detect_is_a_no_op_when_the_version_is_unchanged(tmp_path: Path) -> None:
    """A merge that touches catalog.json without bumping the version tags nothing."""
    proc, outputs = _run_detect(tmp_path, old="2026.8.3", new="2026.8.3", event_name="push")
    assert proc.returncode == 0, proc.stderr
    assert outputs["tag"] == ""
    assert "version" not in outputs


def test_manual_dispatch_targets_the_current_catalog_version(tmp_path: Path) -> None:
    """The escape hatch must work in the case it exists for.

    The old detect step diffed HEAD against HEAD~1 on every event, so dispatching
    it to repair an unreleased version reported "nothing to tag" — the bump it
    was asked about is several commits back by then. On a dispatch the current
    catalog version is the target, and the state checks decide what to do with it.
    """
    proc, outputs = _run_detect(tmp_path, old="2026.8.3", new="2026.8.3", event_name="workflow_dispatch")
    assert proc.returncode == 0, proc.stderr
    assert outputs["tag"] == "data-2026.8.3", (
        "a manual dispatch must target data/catalog.json's current data_version, not the diff"
    )


@pytest.mark.parametrize("bad", ["2026.8", "v2026.8.3", "2026.8.3-rc1", "latest"])
def test_detect_rejects_a_non_calver_version(tmp_path: Path, bad: str) -> None:
    """A malformed version must not become a tag release-data.yml would then reject."""
    proc, outputs = _run_detect(tmp_path, old="2026.8.2", new=bad, event_name="push")
    assert proc.returncode != 0, f"detect accepted {bad!r} and emitted {outputs}"
    assert "CalVer" in proc.stderr


def test_detect_rejects_an_empty_version(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    proc, _ = _run_detect(tmp_path, old="2026.8.2", new="", event_name="push")
    assert proc.returncode != 0
    assert "empty" in proc.stderr

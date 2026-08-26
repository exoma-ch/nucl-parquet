"""Provenance stamps that tie committed parquets to the builder that made them.

The parquets under `data/` are committed artefacts. Fixing a builder is not the
same as fixing the data it produced, and nothing in the repository expressed
that: `scripts/fetch_endf_libs.py` and its output could diverge arbitrarily far
while every test stayed green.

That is not hypothetical. PR #260 fixed `mt_to_residual` in July 2026 — MT=2
elastic was being mapped onto the (n,γ) residual — and only `endfb-8.1` was
rebuilt. Seven neutron libraries shipped pre-fix parquets for **thirteen
months**, making Au-197(n,γ) at 1 MeV read ~3,953 mb instead of ~83, until #334
re-ingested them. See #342.

## The invariant

Every library records, in its `manifest.json`, *which* builder produced it and
*what that builder looked like* at the time:

    "builder": {
      "script":  "scripts/fetch_endf_libs.py",
      "sha256":  "<sha-256 of that file's bytes at build time>",
      "commit":  "8f3061d7ffb86701a7e112b945e811d6a9708dfe",
      "dirty":   false
    }

A library is **stale** when `sha256` no longer matches the script on disk.

## Why a content digest and not "the last commit touching that path"

The obvious formulation is "fail when the stamp's commit predates the most
recent commit touching the builder", and that is what #342 proposed. It is the
wrong primitive for this repository:

  - `actions/checkout` fetches depth 1. `git log <stamp>..HEAD -- <script>` in a
    shallow clone cannot even resolve `<stamp>`, so the check would degrade in
    exactly the environment it exists to run in. Deepening is not free either —
    the pack is 1.9 GiB because the data is committed.
  - The published artefact is a tarball, and the PyPI sdist has no `.git` at
    all. A guard that only works inside a full clone is not a guard on the data.

A digest of the script's bytes needs no history, no network and no remote: it is
one `open().read()`. It answers the same question — "has the builder moved since
this data was made?" — and it answers it identically in a shallow clone, a
tarball and a working tree. Git history is still used, when it happens to be
available, but only to *enrich the failure message* with the list of commits
that landed in between. Correctness never depends on it.

The digest is deliberately coarse: reformatting the builder marks its output
stale. That polarity is the safe one. A guard that tries to decide which edits
"really" affect output is a guard that can be wrong in the silent direction, and
silence is the failure mode this module exists to end. The escape hatch is
explicit and committed — see `data/builder_stamp_exemptions.json`.

## Exemptions

No manifest carried a stamp before this landed, and re-ingesting nineteen
libraries (multi-GB downloads each) to bootstrap a guard is not a thing to do in
one PR. So every unstamped library is grandfathered in an explicit, committed
ledger where each entry names the issue that will remove it.

The ledger is self-cleaning: an exemption that no longer covers a real finding
is itself a failure (`dead-exemption`), so re-ingesting a library forces its
entry out.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Filename of the committed exemption ledger, relative to `data/`.
EXEMPTIONS_FILE = "builder_stamp_exemptions.json"

#: What a `builder.sha256` must look like. Lowercase only, because that is what
#: `hashlib.sha256().hexdigest()` emits — a hand-written uppercase digest is
#: reported as a mismatch rather than accepted, which is the polarity that fails
#: toward "look at this" instead of toward silence.
_HEX64 = re.compile(r"[0-9a-f]{64}")

#: The `data_type` values that denote a cross-section library shipping a
#: `manifest.json`. `scripts/build_manifests.py` walks the same set through
#: `library_dirs` rather than keeping its own copy.
XS_TYPES = {
    "cross_sections",
    "transport_cross_sections",
    "production_cross_sections",
    "total_reaction_cross_sections",
    "experimental_cross_sections",
}

#: Finding kind -> the exemption `reason` that is allowed to cover it.
#:
#: The mapping is deliberately not many-to-one: an `unstamped` exemption must
#: not silently start covering a `stale` finding once the library is finally
#: stamped. Widening an exemption is an edit someone has to make on purpose.
EXEMPTIBLE: dict[str, str] = {
    "unstamped": "unstamped",
    "external-builder": "external-builder",
    "stale": "stale-accepted",
}

#: Every reason the ledger accepts. Anything else is a typo or a new category
#: that should be argued for in review rather than smuggled in.
EXEMPTION_REASONS = frozenset(EXEMPTIBLE.values())


# ---------------------------------------------------------------------------
# Emitting a stamp (called by the builders)
# ---------------------------------------------------------------------------


def script_digest(script: Path) -> str:
    """SHA-256 of a builder script's bytes.

    Bytes, not a parsed/normalised form: the point is to notice *any* change,
    and anything cleverer is a chance to be wrong in the silent direction.
    """
    return hashlib.sha256(script.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    """Run git, returning stripped stdout, or None if git cannot answer.

    Every caller treats None as "unknown", never as "clean" — this runs in
    tarballs and shallow clones where git legitimately has nothing to say.
    """
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.strip()


def make_stamp(script: Path, repo_root: Path = REPO_ROOT) -> dict:
    """Build the `builder` block for a manifest.

    `script` may be absolute or repo-relative; the stamp records it relative to
    `repo_root` so the same manifest reads the same from any checkout.

    `commit` and `dirty` are provenance for humans and for the failure message.
    They are nullable on purpose: a build run outside a git checkout should say
    "I do not know which commit this was" rather than invent one. `sha256` is
    the field the guard actually compares, and it is always available.
    """
    repo_root = repo_root.resolve()
    script = (repo_root / script).resolve()
    if not script.is_relative_to(repo_root):
        raise ValueError(
            f"cannot stamp {script}: it is outside {repo_root}. A stamp names a builder in "
            "this repository, so the audit can find the same file from any checkout."
        )
    rel = script.relative_to(repo_root).as_posix()
    head = _git(repo_root, "rev-parse", "HEAD")
    # `--` disambiguates the path from a ref of the same name.
    diff = _git(repo_root, "status", "--porcelain", "--", rel)
    return {
        "script": rel,
        "sha256": script_digest(script),
        "commit": head,
        # None (not False) when git could not be asked — an unknown is not a
        # clean bill of health.
        "dirty": bool(diff) if diff is not None else None,
    }


def manifest_path_for(pq_dir: Path, library: str) -> Path:
    """Where `library`'s manifest lives, given the directory holding its parquets.

    The established convention is `data/<lib>/manifest.json` beside `xs/`, but
    two libraries can share a root (`endfb-8.0/xs/` and `endfb-8.0/channels/`)
    and some hold parquets directly (`exfor/`). Walking up unconditionally makes
    the second library silently overwrite the first — so only walk up when the
    parent directory *is* this library.
    """
    parent = pq_dir.parent
    return (parent if parent.name == library else pq_dir) / "manifest.json"


def write_builder_stamp(
    manifest_path: Path,
    script: Path,
    *,
    files_written: int,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Record `script`'s stamp in `manifest_path`, preserving everything else.

    Read-modify-write rather than overwrite: the manifest also carries counts
    that `scripts/build_manifests.py` derives from disk, and a builder must not
    clobber them.

    `files_written` is how many files **this run** produced, and a run that
    produced none must not stamp. That is the whole invariant, so it is enforced
    here rather than at each of the five call sites:

      - Checking the output directory instead would not do. A run whose fetches
        all failed leaves the *previous* build's parquets sitting there, so the
        directory looks populated while nothing was regenerated.
      - A stamp on a failed run is worse than no stamp: it attests that the
        current builder produced parquets it never touched, which is exactly the
        lie the stamp exists to detect. The guard would then be re-manufacturing
        the #260 -> #334 failure inside its own mitigation.

    Likewise deliberately never called by `build_manifests.py`, which
    regenerates manifests from data that may be years old.
    """
    if files_written <= 0:
        raise RuntimeError(
            f"refusing to stamp {manifest_path}: this run wrote {files_written} files. "
            "A stamp on an empty build claims the current builder produced whatever is "
            "already on disk — fix the ingest, do not stamp it."
        )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["builder"] = make_stamp(script, repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["builder"]


# ---------------------------------------------------------------------------
# Auditing the stamps (called by the tests and by scripts/check_builder_staleness.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One thing wrong with one library's (or one exemption's) provenance."""

    kind: str
    library: str
    detail: str
    #: Rendered remediation — what the operator should actually type.
    fix: str = ""
    #: Commits that landed on the builder since the stamp, when git can say.
    #: `None` means "history unavailable here" and is reported as such rather
    #: than rendered as "no commits".
    commits: tuple[str, ...] | None = None

    def render(self) -> str:
        lines = [f"{self.library}: {self.detail}"]
        if self.kind == "stale":
            if self.commits is None:
                lines.append(
                    "    commits since: unavailable (shallow clone or no .git) — "
                    "the digest comparison above is authoritative"
                )
            elif self.commits:
                lines.append("    commits since:")
                lines.extend(f"        {c}" for c in self.commits)
            else:
                lines.append(
                    "    commits since: none — the builder differs from the stamp but the "
                    "change is uncommitted in this working tree"
                )
        for line in self.fix.splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines)


@dataclass
class _Library:
    key: str
    builder: str | None
    builder_declared: bool
    rebuild_command: str | None
    manifest: dict | None
    manifest_path: Path
    stamp: dict | None = field(default=None)


def library_dirs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Yield (library key, parquet dir, manifest path) for every xs library present.

    The single definition of "which libraries have a manifest, and where".
    `scripts/build_manifests.py` imports this rather than restating the walk, so
    a rule change — a new `data_type`, a different parquet extension — cannot
    drift between the thing that writes manifests and the thing that audits
    them.

    Libraries absent from this checkout are skipped: a sparse clone is not a
    provenance defect.
    """
    catalog = json.loads((data_dir / "catalog.json").read_text())
    out = []
    for key, info in sorted(catalog.get("libraries", {}).items()):
        if info.get("data_type") not in XS_TYPES or "path" not in info:
            continue
        pq_dir = data_dir / info["path"]
        if not pq_dir.exists() or not any(pq_dir.glob("*.parquet")):
            continue
        out.append((key, pq_dir, manifest_path_for(pq_dir, key)))
    return out


def _libraries(data_dir: Path) -> list[_Library]:
    """Every catalog library that ships parquets, with its provenance fields."""
    catalog = json.loads((data_dir / "catalog.json").read_text())
    out: list[_Library] = []
    for key, _pq_dir, mpath in library_dirs(data_dir):
        info = catalog["libraries"][key]
        manifest = json.loads(mpath.read_text()) if mpath.exists() else None
        out.append(
            _Library(
                key=key,
                builder=info.get("builder"),
                builder_declared="builder" in info,
                rebuild_command=info.get("rebuild_command"),
                manifest=manifest,
                manifest_path=mpath,
                stamp=(manifest or {}).get("builder"),
            )
        )
    return out


def load_exemptions(data_dir: Path) -> dict[str, dict]:
    path = data_dir / EXEMPTIONS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("exemptions", {})


def commits_since(repo_root: Path, commit: str | None, script: str) -> tuple[str, ...] | None:
    """Commits touching `script` after `commit`, or None if git cannot say.

    Purely cosmetic — it turns "the builder changed" into "the builder changed
    in these commits", which is the difference between a failure someone can act
    on and one they have to go archaeology for. A shallow clone returns None and
    the caller says so out loud.
    """
    if not commit:
        return None
    # `cat-file -e` first: in a shallow clone the stamped commit is usually not
    # present at all, and `git log` would fail with a confusing message.
    if _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        return None
    log = _git(repo_root, "log", "--format=%h %s", f"{commit}..HEAD", "--", script)
    if log is None:
        return None
    return tuple(line for line in log.splitlines() if line.strip())


def _fix_hint(lib: _Library, kind: str) -> str:
    """The literal command to run. Not a hint at one.

    The thirteen-month gap persisted partly because nothing said what to do
    about it, so every failure names the exact invocation and the exact escape
    hatch, with the reason string already filled in.
    """
    script = lib.builder
    cmd = lib.rebuild_command or (
        f"nix develop -c uv run python {script} --help   # for the exact flags" if script else ""
    )
    lines = []
    if cmd:
        lines.append(f"re-ingest    : {cmd}")
        # Only builders that write to a throwaway directory need the copy step;
        # the ones that write into `data/` in place do not.
        if "<scratch>" in cmd:
            lines.append(f"               then copy the output over data/{lib.key}/ and re-run")
        else:
            lines.append("               then re-run")
        lines.append("               nix develop -c uv run python scripts/build_manifests.py")
    reason = EXEMPTIBLE.get(kind)
    if reason:
        lines.append(
            f"or exempt    : add {lib.key!r} to data/{EXEMPTIONS_FILE} with "
            f'"reason": "{reason}" and the issue number that removes the entry'
        )
    else:
        lines.append(f"not exemptible: fix {lib.key} in data/catalog.json — this is one line, not a migration")
    return "\n".join(lines)


def audit(data_dir: Path, repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Every provenance defect not covered by a matching exemption.

    Kinds emitted:

    ``no-builder-declared``
        A library ships parquets but `catalog.json` does not say what made it.
        Never exemptible — writing one line in the catalog is not a migration.
    ``external-builder``
        `catalog.json` says `"builder": null`: produced by a pipeline that is
        not in this repository, so no stamp is possible until it is brought in.
    ``unstamped``
        Built before stamping existed. Grandfathered; leaves on re-ingest.
    ``builder-mismatch``
        The stamp names a different script than the catalog declares, or one
        that no longer exists. Never exemptible — one of the two is simply wrong.
    ``stale``
        The stamped digest disagrees with the builder on disk. *The* finding.
    ``dead-exemption`` / ``unknown-library``
        The ledger has drifted from reality. Keeps it from becoming a dumping
        ground for entries nobody ever removes.
    """
    exemptions = load_exemptions(data_dir)
    libraries = _libraries(data_dir)
    findings: list[Finding] = []
    #: Libraries that had *something* wrong — whether or not an exemption
    #: covered it. An exemption whose library is not in here is describing a
    #: problem that no longer exists, which is what makes the ledger shrink.
    imperfect: set[str] = set()

    for lib in libraries:
        kind, detail, commits = _classify(lib, repo_root)
        if kind is None:
            continue
        imperfect.add(lib.key)
        exemption = exemptions.get(lib.key)
        if exemption:
            if exemption.get("reason") == EXEMPTIBLE.get(kind):
                continue
            # An exemption that names a different reason is not permission. Say
            # so on the finding itself rather than emitting a second one — the
            # library has one problem, and "your excuse does not cover it" is
            # part of that problem, not another.
            detail += (
                f"\n    exempted as {exemption.get('reason')!r} (#{exemption.get('issue')}), which does not "
                f"cover a {kind!r} finding — widen it deliberately, or fix the library"
            )
        findings.append(
            Finding(
                kind=kind,
                library=lib.key,
                detail=detail,
                fix=_fix_hint(lib, kind),
                commits=commits,
            )
        )

    known = {lib.key for lib in libraries}
    for key, entry in sorted(exemptions.items()):
        if key not in known:
            findings.append(
                Finding(
                    kind="unknown-library",
                    library=key,
                    detail=(
                        f"exempted in data/{EXEMPTIONS_FILE} but no such library ships parquets "
                        "in this checkout — delete the entry"
                    ),
                )
            )
        elif key not in imperfect:
            findings.append(
                Finding(
                    kind="dead-exemption",
                    library=key,
                    detail=(
                        f"exempted as {entry.get('reason')!r} (see #{entry.get('issue')}) but the library "
                        f"now passes on its own — delete the entry from data/{EXEMPTIONS_FILE}"
                    ),
                )
            )

    return findings


def _classify(lib: _Library, repo_root: Path) -> tuple[str | None, str, tuple[str, ...] | None]:
    if not lib.builder_declared:
        return (
            "no-builder-declared",
            "ships parquets but catalog.json declares no `builder` — add the producing "
            "script's repo-relative path, or `null` if it was produced outside this repository",
            None,
        )
    if lib.builder is None:
        return (
            "external-builder",
            'catalog.json declares `"builder": null` — produced by a pipeline that is not in '
            "this repository, so its output cannot be checked against anything here",
            None,
        )

    # A builder path must name a file *inside* the repository. `../` would let a
    # catalog edit point the digest at an arbitrary file on the machine running
    # the check, which is a guard that passes for reasons nobody can reproduce.
    script = (repo_root / lib.builder).resolve()
    if not str(lib.builder).strip() or Path(lib.builder).is_absolute() or not script.is_relative_to(repo_root):
        return (
            "builder-mismatch",
            f"catalog.json names builder {lib.builder!r}, which is not a path inside the repository",
            None,
        )
    if not script.is_file():
        return (
            "builder-mismatch",
            f"catalog.json names builder {lib.builder!r}, which does not exist",
            None,
        )
    if lib.stamp is None:
        return (
            "unstamped",
            f"manifest.json carries no `builder` stamp, so there is nothing to compare {lib.builder} against",
            None,
        )

    # Every field below is read from a committed JSON file that a human may have
    # hand-edited. A malformed stamp must be a finding, not an AttributeError
    # and above all not a pass — "the digest field is missing" and "the digest
    # matches" have to be distinguishable.
    if not isinstance(lib.stamp, dict):
        return (
            "builder-mismatch",
            f"manifest.json `builder` is {type(lib.stamp).__name__}, expected an object with `script` and `sha256`",
            None,
        )
    stamped_script = lib.stamp.get("script")
    if stamped_script != lib.builder:
        return (
            "builder-mismatch",
            f"stamped by {stamped_script!r} but catalog.json declares {lib.builder!r} — one of the two is wrong",
            None,
        )
    stamped_digest = lib.stamp.get("sha256")
    if not isinstance(stamped_digest, str) or not _HEX64.fullmatch(stamped_digest):
        return (
            "builder-mismatch",
            f"manifest.json `builder.sha256` is {stamped_digest!r}, not a SHA-256 hex digest — "
            "re-ingest so a real stamp is written",
            None,
        )

    current = script_digest(script)
    if stamped_digest == current:
        return (None, "", None)

    commit = lib.stamp.get("commit")
    dirty = lib.stamp.get("dirty")
    where = f" at {commit[:8]}" if commit else " (build not in a git checkout)"
    if dirty:
        where += ", from a dirty working tree"
    return (
        "stale",
        (
            f"built by {lib.builder}{where}, which has changed since\n"
            f"    stamped      : {lib.stamp.get('sha256')}\n"
            f"    on disk now  : {current}"
        ),
        commits_since(repo_root, commit, lib.builder),
    )


def format_report(findings: list[Finding]) -> str:
    if not findings:
        return "builder stamps: all libraries verified or explicitly exempted"
    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)
    blocks = [f"{len(findings)} library provenance problem(s):"]
    for kind, group in sorted(by_kind.items()):
        blocks.append(f"\n[{kind}]")
        blocks.extend("  " + f.render().replace("\n", "\n  ") for f in group)
    return "\n".join(blocks)

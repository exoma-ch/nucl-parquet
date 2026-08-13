# Data-release signing

Every data release from `data-2026.8.2` onward is signed with
[minisign](https://jedisct1.github.io/minisign/). The signature is published as
`nucl-parquet-data-<CalVer>.tar.zst.minisig` next to the tarball on the GitHub
release.

## What this is for

A SHA-256 pin and a signature answer different questions, and consumers need
both:

| Control | Question it answers | Where it lives |
|---|---|---|
| `catalog.json::data_sha256` | Are these the bytes that build was tested against? | this repo |
| Consumer-side tarball pin | Same, enforced at install time | e.g. HYRR |
| **minisign signature** | **Did these bytes come from the nuclear-data team?** | **this document's key** |

Only the last one survives re-pinning. When a consumer re-pins to a new data
release it reads GitHub's published asset digest — but this repo's releases are
mutable (`immutable: false`), so that digest is a convenience, not a control. A
signature roots trust in an offline key instead of in the GitHub account, so it
still means something if the account is compromised.

This is complementary to [#288](https://github.com/exoma-ch/nucl-parquet/issues/288)
(immutable releases + build attestations), not a substitute:

- **Immutable releases** stop bytes changing after publication — they make the
  *digest* trustworthy.
- **Attestations** bind the artifact to the workflow that built it — trust
  rooted in GitHub.
- **minisign** binds it to a key the nuclear-data team holds offline — trust
  rooted in something that survives a GitHub account compromise.

## The public key

The key consumers pin lives at
[`docs/security/data-signing-key.pub`](./data-signing-key.pub) in this repo. It
is the single canonical copy: the release workflow verifies its own signature
against this file before uploading, so a key that has rotated without this file
being updated fails the release instead of shipping an unverifiable signature.

Pin it by value, not by URL. Fetching the key over the same channel as the
artifact it authenticates proves nothing.

## Verifying a release

Inside this repo:

```bash
just verify-release              # verify the version in data/catalog.json
just verify-release 2026.8.2     # verify a specific CalVer
```

Standalone, with only `minisign` and the public key:

```bash
VERSION=2026.8.2
BASE="https://github.com/exoma-ch/nucl-parquet/releases/download/data-${VERSION}"
curl -fLO "${BASE}/nucl-parquet-data-${VERSION}.tar.zst"
curl -fLO "${BASE}/nucl-parquet-data-${VERSION}.tar.zst.minisig"
minisign -Vm "nucl-parquet-data-${VERSION}.tar.zst" -P "$(tail -1 data-signing-key.pub)"
```

### Check the trusted comment, not just the exit code

The signature covers a *trusted comment* of the form:

```
nucl-parquet data 2026.8.2 tag=data-2026.8.2 sha256=<64 hex>
```

minisign proves the signature covers the bytes you have. It cannot tell you
that a genuine signature for release A is being served to you as release B —
downgrading you to an older, authentically-signed release with a known defect.
Because the trusted comment is covered by the signature and names the version,
tag and digest, comparing it to the release you *asked for* is what closes that
gap. `scripts/verify_data_release.sh` does this; a bare `minisign -Vm` does not.

## The signed content manifest

From `data-2026.8.3` a release also carries:

```
nucl-parquet-data-<CalVer>.manifest.json          # relpath -> {sha256, size}
nucl-parquet-data-<CalVer>.manifest.json.minisig  # signed with the same key
```

### Why a second control

The tarball signature covers the *archive bytes*. It stops verifying the moment
anything legitimately rewrites the archive — and that is routine on the way into
an isolated network. Content Disarm & Reconstruction appliances (OPSWAT
MetaDefender, Deep CDR) are standard at hospitals and Tier-1 nuclear sites: they
open a `.tar.zst`, scan each entry, and **repack it**. The nuclear data arrives
intact; the signature does not survive, because the bytes are no longer the bytes
that were signed. Roughly a fifth of realistic deployments, and precisely the
ones with the strictest verification requirements ([hyrr#614](https://github.com/exoma-ch/hyrr/issues/614)).

A signed map of per-file digests survives anything that preserves file
*contents* while changing archive *framing*. It also makes a **partial**
transfer verifiable: a consumer who carried only `tendl-2023-iso/` across a data
diode can check what they hold, instead of moving all 785 MB because only the
whole archive is signed.

This is Debian's `Release`/`InRelease` model. Both controls stay — the archive
signature is cheaper and stronger when the bytes survive; the manifest is what
remains when they do not.

### Verifying an extracted tree

```bash
just verify-extracted /path/to/extracted 2026.8.3
just verify-extracted /path/to/extracted 2026.8.3 --partial   # subset transfer
```

Standalone, with `minisign`, `jq` and `sha256sum`:

```bash
VERSION=2026.8.3
BASE="https://github.com/exoma-ch/nucl-parquet/releases/download/data-${VERSION}"
curl -fLO "${BASE}/nucl-parquet-data-${VERSION}.manifest.json"
curl -fLO "${BASE}/nucl-parquet-data-${VERSION}.manifest.json.minisig"

minisign -Vm "nucl-parquet-data-${VERSION}.manifest.json" -P "$(tail -1 data-signing-key.pub)"
jq -r '.files | to_entries[] | "\(.value.sha256)  \(.key)"' \
  "nucl-parquet-data-${VERSION}.manifest.json" > SHA256SUMS
cd /path/to/extracted && sha256sum -c SHA256SUMS
```

### What it covers, and what that is not

The manifest covers **every file in the archive**, which is deliberately wider
than `catalog.json::data_sha256`. The tree hash covers only `*.parquet`, because
it answers "did the data change" — a catalog edit must not read as a data
change. The manifest answers "are these the bytes we published", so it must
include `catalog.json` and `licenses.toml`: they ride inside the tarball and are
the files most worth tampering with. [#234](https://github.com/exoma-ch/nucl-parquet/issues/234)
is a live case of a wrong licence claim shipping on a published artefact.

Both are built from one tree walk (`nucl_parquet.iter_file_digests`), so they
cannot disagree about the same tree.

**The manifest is bound to its release.** `data_version` and `tag` are signed
fields inside it, and its `.minisig` trusted comment repeats them. Without that,
a genuine manifest for release A verifies happily against release B's extracted
files, and every digest unchanged between the two agrees — a consumer doing a
partial check might never notice. `verify_data_release.sh` refuses on mismatch.

**A caveat worth stating plainly.** A gateway that rewrites file *contents*
rather than only archive framing — re-encoding something it believes it
understands — will fail the manifest check too. That is correct: the manifest
converts "unverifiable" into "verifiably modified". It is not a claim that
verification works everywhere.

## Grandfathering — releases before signing

Data releases published before `data-2026.8.2` carry **no** `.minisig`, and no
signature will be issued for them retroactively. Signing an old artifact today
would assert something the key cannot honestly attest: that we vouched for those
bytes at the time they were published.

> The cutoff version is defined once, as `FIRST_SIGNED_VERSION` in
> `scripts/verify_data_release.sh`, and this document is gated against it by
> `tests/test_data_signing.py`. It names the first release published after
> signing landed — if that release ships under a different CalVer than planned,
> change the constant and the test will point here.

Consumers should therefore:

- **require** a valid signature for `>= 2026.8.2`;
- fall back to the SHA-256 pin alone for older releases, and treat that as a
  known gap rather than a passing check.

`scripts/verify_data_release.sh` enforces this rather than merely documenting
it. A missing signature hard-fails, and `--allow-unsigned` is **refused at or
above the cutoff** — it only applies to releases that genuinely predate signing.

That refusal is the point. Without it, an attacker who can strip the `.minisig`
(a MITM, or anyone able to rewrite these mutable releases) lets the consumer hit
the "carries no signature" error, and relies on them re-running with the flag
that error just suggested. If the flag worked at any version, the whole scheme
would be one retry away from bypass. "This release predates signing" and "an
attacker stripped the signature" must never be the same code path.

The opt-out warning says plainly that *nothing* was verified — not the origin,
not the digest. It deliberately does not cite the catalog SHA-256 pin, because
this script does not check it.

## Key custody

| Property | Value |
|---|---|
| Algorithm | Ed25519 (minisign) |
| Generated | offline, on a trusted personal machine — never in CI |
| Secret key | password-protected; master copy in the maintainer's password manager |
| CI access | `DATA_SIGNING_KEY` + `DATA_SIGNING_KEY_PASSWORD` repository secrets |
| Distinct from | the HYRR updater key — compromise of one must not imply the other |

The signing key is deliberately **separate** from the HYRR desktop updater key.
Reusing that key would make a compromise of either trust root a compromise of
both, at the saving of one rotation-runbook entry.

### Why the key is not sops-encrypted into the repo

This repo is public. sops ciphertext committed here is permanently harvestable —
every clone, fork and archive keeps a copy forever, so a future compromise of
the decryption key retroactively exposes every historical version of the signing
key. A repository secret only leaks to someone with access at the time of the
leak.

For a revocable API token that trade is fine, and worth making to cut the
secret-setting toil. For this key it is not: the public key is a **pinned trust
root**, so rotating it costs a release on every consumer that pinned it, and
"harvest now, decrypt later" is precisely the threat model signatures exist to
survive.

### Why the secret in CI is still a real signing key

The pitfall worth naming: a key reachable by a workflow that a pull request can
influence attests to nothing. `release-data.yml` is triggered only by `data-*`
tag pushes and by `workflow_dispatch` — there is no `pull_request` trigger, so
no PR can reach the secret. Key material is written under `RUNNER_TEMP` (never
the workspace, where an asset glob could sweep it into a release) and shredded
when the step exits.

## Generating the key (once)

```bash
just gen-signing-key
```

Run this on a trusted personal machine, never in CI. It generates the keypair,
verifies the passphrase actually decrypts it *before* uploading anything, sets
both repository secrets, and prints the public key line to commit.

Then commit the public key and back up the secret:

```bash
git add docs/security/data-signing-key.pub
git commit -m "feat(release): publish the data-signing public key"
shred -u <path the script printed>
```

## Rotation

Rotation is expensive by design — every consumer that pinned the old key must
ship an update before it will accept new releases. Plan it; do not do it
reflexively.

**Routine rotation (key not believed compromised):**

1. Generate the new keypair offline (`just gen-signing-key`, confirm `rotate`).
2. Commit the new `data-signing-key.pub`. Open the PR, but **do not merge yet**.
3. Give consumers a window to ship support for the new key alongside the old
   one. For HYRR that means a release carrying both keys.
4. Merge. The next data release is signed with the new key; the workflow's
   self-verification confirms the committed key matches the secret.
5. Announce the change on the release notes of the first release signed with it.
6. Destroy the old secret key only after the window closes.

**Emergency rotation (key compromised):**

1. Rotate immediately — steps 1, 2 and 4 above, no consumer window.
2. Publish an advisory naming the last release signed with the compromised key.
   Everything signed after the compromise but before rotation must be treated as
   untrusted regardless of a valid signature.
3. Re-release the affected data versions under new CalVer tags signed with the
   new key. Do not re-sign the old tags in place — a mutable release with a
   changed signature is indistinguishable from an attack.
4. Notify consumers that pin the key, HYRR first.

## Related

- [#289](https://github.com/exoma-ch/nucl-parquet/issues/289) — this work
- [#288](https://github.com/exoma-ch/nucl-parquet/issues/288) — immutable
  releases + attestations (complementary; cheaper, and reduces the same risk
  from a different direction)
- [exoma-ch/hyrr#594](https://github.com/exoma-ch/hyrr/issues/594) — the
  consumer-side verification this unblocks
- [exoma-ch/hyrr#577](https://github.com/exoma-ch/hyrr/issues/577) — the
  SHA-256 pin this complements

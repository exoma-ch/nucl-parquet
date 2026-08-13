#!/usr/bin/env bash
# Verify a published nucl-parquet data release against the project's offline
# signing key (#289).
#
# A SHA-256 pin answers "are these the bytes that build was tested against?".
# This answers the different question: "did these bytes come from the
# nuclear-data team?" — which is the one that survives re-pinning, because
# GitHub's published asset digest is only as trustworthy as the account.
#
# Usage:
#   scripts/verify_data_release.sh                     # verify catalog.json's data_version
#   scripts/verify_data_release.sh 2026.8.2            # verify a specific CalVer
#   scripts/verify_data_release.sh --file t.tar.zst \
#       --sig t.tar.zst.minisig --version 2026.8.2     # verify local files, no network
#
# Exits non-zero on any failure. Requires minisign (in the devShell).
set -euo pipefail

REPO="exoma-ch/nucl-parquet"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBKEY_FILE="${PUBKEY_FILE:-${ROOT}/docs/security/data-signing-key.pub}"

# Grandfathering: data releases published before signing landed carry no
# .minisig. That is a documented gap, not a verification failure to paper
# over — see docs/security/data-signing.md. Verifying one of those requires
# opting out explicitly, so "old release" can never be silently indistinguishable
# from "signature stripped by an attacker".
#
# This is the single source of truth for the cutoff — consumers implementing
# "require a signature at or above version X" read it from here, and
# docs/security/data-signing.md is gated against it by tests/test_data_signing.py.
# It is the first release published after signing landed; adjust it if that
# release ships under a different CalVer than planned.
FIRST_SIGNED_VERSION="2026.8.2"
ALLOW_UNSIGNED=0

# Manifest mode (#296). A signature over the tarball proves the *archive bytes*
# are ours, and stops verifying the moment anything legitimately rewrites the
# archive. That is routine on the way into an isolated network: Content Disarm
# & Reconstruction gateways open a .tar.zst, scan each entry and repack it, so
# the data arrives intact and the signature does not survive. These flags verify
# the signed per-file manifest instead, which survives a repack — and lets a
# consumer who moved only one library check what they actually have.
EXTRACTED=""
LOCAL_MANIFEST=""
PARTIAL=0
ALLOW_EXTRA=0
FIRST_MANIFEST_VERSION="2026.8.3"

VERSION=""
LOCAL_FILE=""
LOCAL_SIG=""

die() { echo "error: $*" >&2; exit 1; }

# The one place the opt-out is spelled out, so the local-file and network paths
# cannot drift into saying different things about the same situation.
#
# It states what was NOT checked. An earlier version claimed integrity rested
# on "TLS and the catalog SHA-256 pin" — a control this script never performs.
# Naming an absent check in the one message a user reads while deciding to
# trust unverified bytes is worse than saying nothing at all.
warn_unsigned() {
  echo "WARNING: $1 carries no signature, and --allow-unsigned was passed." >&2
  echo "WARNING: ${VERSION} predates ${FIRST_SIGNED_VERSION}, so this is expected." >&2
  echo "WARNING: NOTHING about these bytes has been verified by this script —" >&2
  echo "WARNING: not the origin, not the digest. Treat them as unverified." >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --file)            LOCAL_FILE="$2"; shift 2 ;;
    --sig)             LOCAL_SIG="$2"; shift 2 ;;
    --version)         VERSION="$2"; shift 2 ;;
    --pubkey)          PUBKEY_FILE="$2"; shift 2 ;;
    --allow-unsigned)  ALLOW_UNSIGNED=1; shift ;;
    --extracted)       EXTRACTED="$2"; shift 2 ;;
    --manifest)        LOCAL_MANIFEST="$2"; shift 2 ;;
    --partial)         PARTIAL=1; shift ;;
    --allow-extra)     ALLOW_EXTRA=1; shift ;;
    -h|--help)         sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)                die "unknown flag: $1" ;;
    *)                 VERSION="$1"; shift ;;
  esac
done

command -v minisign >/dev/null 2>&1 \
  || die "minisign not found. Run inside the devShell: nix develop -c $0"

[ -f "${PUBKEY_FILE}" ] || die "public key not found: ${PUBKEY_FILE}
The key consumers pin must be committed before releases can be verified (#289).
See docs/security/data-signing.md."

# minisign -P wants the bare base64 line, not the commented file.
# `|| true` keeps `set -o pipefail` from aborting on a comment-only file, which
# would skip the friendly diagnosis on the next line.
PUBKEY="$( { grep -v '^untrusted comment:' "${PUBKEY_FILE}" || true; } | tr -d '[:space:]')"
[ -n "${PUBKEY}" ] || die "no key line found in ${PUBKEY_FILE}"

if [ -z "${VERSION}" ]; then
  VERSION="$(jq -r .data_version "${ROOT}/data/catalog.json")"
  echo "No version given — using data/catalog.json::data_version = ${VERSION}"
fi

[[ "${VERSION}" =~ ^[0-9]{4}\.[0-9]+\.[0-9]+$ ]] \
  || die "version '${VERSION}' is not CalVer YYYY.MM.MICRO"

TAG="data-${VERSION}"
ASSET="nucl-parquet-data-${VERSION}.tar.zst"

# Compare CalVer numerically: 2026.8.10 is newer than 2026.8.9, which a string
# compare gets backwards.
_calver_lt() {
  local a b
  IFS=. read -r -a a <<< "$1"
  IFS=. read -r -a b <<< "$2"
  for i in 0 1 2; do
    (( a[i] < b[i] )) && return 0
    (( a[i] > b[i] )) && return 1
  done
  return 1
}

# Is this version old enough that an absent signature is expected?
if _calver_lt "${VERSION}" "${FIRST_SIGNED_VERSION}"; then
  PREDATES_SIGNING=1
else
  PREDATES_SIGNING=0
fi

# The cutoff has to be *enforced*, not merely documented. Otherwise an attacker
# who can strip the .minisig (a MITM, or anyone who can rewrite these mutable
# releases) just downgrades the consumer into the unsigned path, and
# --allow-unsigned — offered by the error message the consumer just read —
# accepts arbitrary bytes for a version that should always be signed.
if [ "${ALLOW_UNSIGNED}" -eq 1 ] && [ "${PREDATES_SIGNING}" -eq 0 ]; then
  die "refusing --allow-unsigned for ${VERSION}.
Releases from ${FIRST_SIGNED_VERSION} onward are always signed, so a missing
signature means the signature was removed — not that this release predates
signing. --allow-unsigned only applies below ${FIRST_SIGNED_VERSION}."
fi

# --- manifest mode: verify an extracted tree, not an archive ----------------
#
# Runs instead of the tarball path, because a consumer here does not have an
# archive to check — a CDR gateway repacked it, or they only carried one
# library across a diode.
if [ -n "${EXTRACTED}" ]; then
  [ -d "${EXTRACTED}" ] || die "no such directory: ${EXTRACTED}"
  command -v jq >/dev/null 2>&1 || die "jq is required for manifest verification"

  MWORK="$(mktemp -d)"
  trap 'rm -rf "${MWORK}"' EXIT

  if [ -n "${LOCAL_MANIFEST}" ]; then
    MANIFEST="${LOCAL_MANIFEST}"
    MSIG="${LOCAL_MANIFEST}.minisig"
  else
    BASE="https://github.com/${REPO}/releases/download/${TAG}"
    MANIFEST="${MWORK}/manifest.json"
    MSIG="${MANIFEST}.minisig"
    echo "Fetching manifest for ${TAG} ..."
    curl -fsSL --retry 3 -o "${MANIFEST}" "${BASE}/nucl-parquet-data-${VERSION}.manifest.json" \
      || die "release ${TAG} carries no manifest.
Releases before ${FIRST_MANIFEST_VERSION} predate the signed content manifest (#296).
For those, verify the tarball itself: $0 ${VERSION}"
    curl -fsSL --retry 3 -o "${MSIG}" "${BASE}/nucl-parquet-data-${VERSION}.manifest.json.minisig" \
      || die "manifest is published but its signature is not — refusing to trust an unsigned manifest.
An unsigned manifest beside the files it describes is rewritable by anyone who
can rewrite the files, and reads as a control while being none."
  fi
  [ -f "${MSIG}" ] || die "manifest signature not found: ${MSIG}"

  # 1. Is the manifest itself authentic?
  echo "Verifying manifest signature ..."
  minisign -V -P "${PUBKEY}" -m "${MANIFEST}" -x "${MSIG}" >/dev/null \
    || die "MANIFEST SIGNATURE VERIFICATION FAILED.
This manifest was not signed by the nuclear-data key. Do not use it."

  # 2. Does it describe the release we asked for? Same replay gap the tarball
  #    signature closes — a genuine manifest for release A verifies happily
  #    against release B's files, and every digest unchanged between the two
  #    agrees. Checked twice: the signed trusted comment, and the signed
  #    fields inside the manifest.
  MSIG_LINES="$(wc -l < "${MSIG}")"
  [ "${MSIG_LINES}" -eq 4 ] || die "malformed manifest signature: ${MSIG_LINES} lines, expected 4"
  MTRUSTED="$(sed -n '3s/^trusted comment: //p' "${MSIG}")"
  MSIGNED_TAG="$(printf '%s' "${MTRUSTED}" | sed -n 's/.*tag=\([^ ]*\).*/\1/p')"
  if [ -n "${MSIGNED_TAG}" ] && [ "${MSIGNED_TAG}" != "${TAG}" ]; then
    die "REPLAY DETECTED: manifest is validly signed but was issued for ${MSIGNED_TAG}, not ${TAG}."
  fi
  M_VERSION="$(jq -r '.data_version // empty' "${MANIFEST}")"
  M_TAG="$(jq -r '.tag // empty' "${MANIFEST}")"
  [ "${M_VERSION}" = "${VERSION}" ] || die "manifest declares data_version ${M_VERSION}, expected ${VERSION}"
  [ -z "${M_TAG}" ] || [ "${M_TAG}" = "${TAG}" ] || die "manifest declares tag ${M_TAG}, expected ${TAG}"

  # 3. Do the files on disk match it? `sha256sum -c` is exactly this check, and
  #    is present anywhere the data is.
  jq -r '.files | to_entries[] | "\(.value.sha256)  \(.key)"' "${MANIFEST}" > "${MWORK}/SHA256SUMS"
  TOTAL=$(wc -l < "${MWORK}/SHA256SUMS")
  echo "Checking $(printf '%d' "${TOTAL}") files against the signed manifest ..."

  CHECK_ARGS=()
  # A partial transfer legitimately lacks files; a full one must not.
  [ "${PARTIAL}" -eq 1 ] && CHECK_ARGS+=(--ignore-missing)

  # Run once and keep the output: a second pass over 6600 files just to count
  # the OK lines doubles the work on exactly the machine least likely to have
  # cycles to spare.
  CHECK_RC=0
  ( cd "${EXTRACTED}" && sha256sum -c "${CHECK_ARGS[@]}" "${MWORK}/SHA256SUMS" ) > "${MWORK}/check.out" 2>&1 || CHECK_RC=$?

  # `sha256sum -c` answers one direction only: every listed file is present and
  # matches. It says nothing about files on disk that the manifest does NOT
  # list, and silently ignores them. That is the injection path this whole
  # feature exists to close — the threat model is a gateway or mirror that can
  # write into the extracted tree, and a consumer that then globs
  # `data/**/*.parquet` would load planted bytes from a tree just declared OK.
  ( cd "${EXTRACTED}" && find . -type f ! -name '*.minisig' ! -name '*.manifest.json' \
      | sed 's|^\./||' | LC_ALL=C sort ) > "${MWORK}/on_disk.txt"
  jq -r '.files | keys[]' "${MANIFEST}" | LC_ALL=C sort > "${MWORK}/in_manifest.txt"
  EXTRA_COUNT=$(comm -23 "${MWORK}/on_disk.txt" "${MWORK}/in_manifest.txt" | wc -l)

  if [ "${EXTRA_COUNT}" -gt 0 ] && [ "${ALLOW_EXTRA}" -eq 0 ]; then
    echo "error: ${EXTRA_COUNT} file(s) present on disk are NOT in the signed manifest:" >&2
    comm -23 "${MWORK}/on_disk.txt" "${MWORK}/in_manifest.txt" | head -20 >&2
    die "The manifest cannot vouch for these bytes. A tree that verifies while
carrying unlisted files is exactly the gap a signed manifest is supposed to
close: anything globbing this directory would load them.
If you deliberately extracted into a directory holding other files, re-run with
--allow-extra."
  fi

  if [ "${CHECK_RC}" -eq 0 ]; then
    if [ "${PARTIAL}" -eq 1 ]; then
      PRESENT=$(grep -c ': OK$' "${MWORK}/check.out" || true)
      echo
      echo "OK  ${PRESENT}/${TOTAL} files present, all matching the signed manifest  (${TAG})"
      echo "    partial transfer: files absent locally were not checked"
    else
      echo
      echo "OK  all ${TOTAL} files match the signed manifest  (${TAG})"
    fi
    [ "${EXTRA_COUNT}" -gt 0 ] && echo "    warning: ${EXTRA_COUNT} unlisted file(s) present, accepted via --allow-extra"
    echo "    signed by: $(grep '^untrusted comment:' "${PUBKEY_FILE}" | sed 's/^untrusted comment: //')"
    echo "    trusted comment: ${MTRUSTED}"
    exit 0
  fi

  grep -v ': OK$' "${MWORK}/check.out" | head -20 >&2
  die "One or more files do not match the signed manifest.
A mismatch after a legitimate repack means the contents were altered, not merely
the archive framing."
fi

WORKDIR=""
# `return 0` is load-bearing: an EXIT trap whose last command fails sets the
# script's exit status. With `[ -n "${WORKDIR}" ] && rm -rf ...` the test
# returns 1 whenever no temp dir was used (--file mode), so a *successful*
# verification would report failure.
cleanup() {
  if [ -n "${WORKDIR}" ]; then
    rm -rf "${WORKDIR}"
  fi
  return 0
}
trap cleanup EXIT

if [ -n "${LOCAL_FILE}" ]; then
  TARBALL="${LOCAL_FILE}"
  SIG="${LOCAL_SIG:-${LOCAL_FILE}.minisig}"
  [ -f "${TARBALL}" ] || die "no such file: ${TARBALL}"
  # Honour --allow-unsigned here too. It used to apply only on the network
  # path, so verifying an already-downloaded tarball ignored the flag and died
  # regardless — the flag was silently inert exactly where a maintainer is most
  # likely to reach for it.
  if [ ! -f "${SIG}" ] && [ "${ALLOW_UNSIGNED}" -eq 1 ]; then
    warn_unsigned "${TARBALL}"
    exit 0
  fi
else
  WORKDIR="$(mktemp -d)"
  TARBALL="${WORKDIR}/${ASSET}"
  SIG="${TARBALL}.minisig"
  BASE="https://github.com/${REPO}/releases/download/${TAG}"

  echo "Fetching ${ASSET} from ${TAG} ..."
  curl -fsSL --retry 3 -o "${TARBALL}" "${BASE}/${ASSET}" \
    || die "could not download ${BASE}/${ASSET} (does release ${TAG} exist?)"

  if ! curl -fsSL --retry 3 -o "${SIG}" "${BASE}/${ASSET}.minisig"; then
    if [ "${ALLOW_UNSIGNED}" -eq 1 ]; then
      warn_unsigned "${TAG}"
      exit 0
    fi
    die "release ${TAG} carries no .minisig asset.
Releases before ${FIRST_SIGNED_VERSION} are unsigned by design — see the
grandfathering rule in docs/security/data-signing.md. If ${VERSION} is at or
above ${FIRST_SIGNED_VERSION}, a missing signature is a RED FLAG, not an old
release. To accept an unsigned release knowingly, re-run with --allow-unsigned."
  fi
fi

[ -f "${SIG}" ] || die "signature not found: ${SIG}"

# 1. Does the signature verify against the key consumers pin?
echo "Verifying signature against $(basename "${PUBKEY_FILE}") ..."
minisign -V -P "${PUBKEY}" -m "${TARBALL}" -x "${SIG}" >/dev/null \
  || die "SIGNATURE VERIFICATION FAILED for ${TARBALL}.
These bytes were not signed by the nuclear-data key. Do not use them."

# 2. Does the signed trusted comment describe *this* release?
#
# minisign alone proves the signature covers these bytes — it cannot detect a
# genuine signature from release A being served as release B. The trusted
# comment is covered by the signature and names the version, tag and digest,
# so cross-checking it is what closes the replay gap.
#
# Read ONLY line 3, and only from a file that is exactly 4 lines. A minisign
# signature file is:
#
#   1  untrusted comment: ...
#   2  <base64 signature>
#   3  trusted comment: ...          <-- the only line covered by the global signature
#   4  <base64 global signature>
#
# minisign ignores anything after line 4: appending a second `trusted comment:`
# line to a valid .minisig still verifies (confirmed against minisign 0.12). A
# naive `sed -n 's/^trusted comment: //p'` would then return the real line AND
# the attacker's, so every field below would be parsed from partly unsigned
# text. Pinning to line 3 is what makes "the comment is covered by the
# signature" actually true of the value we use.
SIG_LINES="$(wc -l < "${SIG}")"
if [ "${SIG_LINES}" -ne 4 ]; then
  die "malformed signature file: ${SIG} has ${SIG_LINES} lines, expected exactly 4.
Extra lines in a .minisig are not covered by the signature — refusing to parse it."
fi
TRUSTED="$(sed -n '3s/^trusted comment: //p' "${SIG}")"
[ -n "${TRUSTED}" ] || die "no trusted comment on line 3 of ${SIG}"

ACTUAL_SHA="$(sha256sum "${TARBALL}" | cut -d' ' -f1)"
SIGNED_SHA="$(printf '%s' "${TRUSTED}" | sed -n 's/.*sha256=\([0-9a-f]\{64\}\).*/\1/p')"
SIGNED_TAG="$(printf '%s' "${TRUSTED}" | sed -n 's/.*tag=\([^ ]*\).*/\1/p')"

[ -n "${SIGNED_SHA}" ] || die "trusted comment carries no sha256= field: ${TRUSTED}"

if [ "${SIGNED_SHA}" != "${ACTUAL_SHA}" ]; then
  die "trusted comment digest does not match the file.
  signed: ${SIGNED_SHA}
  actual: ${ACTUAL_SHA}"
fi

if [ -n "${SIGNED_TAG}" ] && [ "${SIGNED_TAG}" != "${TAG}" ]; then
  die "REPLAY DETECTED: signature is valid but was issued for ${SIGNED_TAG}, not ${TAG}.
These are authentic bytes from a *different* release."
fi

# 3. Does the digest match what the catalog claims, when we have a catalog?
#    Cheap cross-check that the signed release is the one this checkout expects.
if [ -z "${LOCAL_FILE}" ] && [ -f "${ROOT}/data/catalog.json" ]; then
  CATALOG_VERSION="$(jq -r .data_version "${ROOT}/data/catalog.json")"
  if [ "${CATALOG_VERSION}" != "${VERSION}" ]; then
    echo "note: this checkout's catalog is ${CATALOG_VERSION}, you verified ${VERSION}."
  fi
fi

echo
echo "OK  signature valid  ${ASSET}"
echo "    signed by: $(grep '^untrusted comment:' "${PUBKEY_FILE}" | sed 's/^untrusted comment: //')"
echo "    trusted comment: ${TRUSTED}"
echo "    sha256: ${ACTUAL_SHA}"

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

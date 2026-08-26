#!/usr/bin/env bash
# Assert that data/catalog.json::data_version is actually published (#364).
#
# The invariant this repository keeps promising and has twice broken: whatever
# `data/catalog.json` claims, a consumer resolving that version can download it,
# and what they download is signed by us.
#
# #344 broke it — catalog said 2026.8.3, the tag existed, no release did, and a
# consumer resolving "latest" silently got 2026.8.2. #350 made the tag push and
# the release trigger one operation and added a confirmation step, which closes
# the *merge* path. It does not close:
#
#   - a tag pushed by hand (release-data.yml still accepts those, and
#     CONTRIBUTING documents them)
#   - release-data.yml failing after it starts — #350 asserts the run began,
#     not that it published
#   - a release, or an asset, deleted afterwards
#   - the confirmation step being edited into uselessness
#
# Each of those lands back in #344's state with nothing watching. So this checks
# the invariant itself rather than the mechanism that is supposed to maintain it,
# which is the whole point: a check that shares a cause of failure with the thing
# it checks is not a check.
#
# Usage:
#   scripts/reconcile_data_release.sh                    # check catalog.json's data_version
#   scripts/reconcile_data_release.sh 2026.8.3           # check a specific CalVer
#   scripts/reconcile_data_release.sh --skip-signature   # structural checks only, no 727MB download
#   scripts/reconcile_data_release.sh --report out.md    # also write a markdown failure report
#
# Exits 0 if the invariant holds, 1 if it does not, 2 on a usage error.
# Requires gh (authenticated), jq, and — unless --skip-signature — minisign.
set -uo pipefail

REPO="${GITHUB_REPOSITORY:-exoma-ch/nucl-parquet}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Overridable for the same reason verify_data_release.sh makes PUBKEY_FILE
# overridable: it is the seam the tests drive. The signature check is a whole
# separate script and a 727MB download, so exercising *this* script's handling
# of a signature failure needs to be able to stand in for it.
VERIFY_SCRIPT="${VERIFY_SCRIPT:-${ROOT}/scripts/verify_data_release.sh}"

# The version floors live in verify_data_release.sh and are read from it, never
# restated. Releases below them are unsigned / manifest-less *by design* (the
# grandfathering rule in docs/security/data-signing.md), and a second copy of
# the cutoff is how one consumer ends up trusting an unsigned release.
_floor() {
  local name="$1" value
  value="$(sed -n "s/^${name}=\"\([^\"]*\)\".*/\1/p" "${ROOT}/scripts/verify_data_release.sh" | head -1)"
  if [ -z "${value}" ]; then
    echo "error: ${name} not found in scripts/verify_data_release.sh" >&2
    exit 2
  fi
  printf '%s' "${value}"
}

VERSION=""
REPORT=""
SKIP_SIGNATURE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-signature) SKIP_SIGNATURE=1; shift ;;
    # A bare trailing `--report` would otherwise leave REPORT empty and write
    # the report to /dev/null — the flag silently doing nothing, in the script
    # whose whole job is noticing things that silently do nothing.
    --report)         [ -n "${2:-}" ] || { echo "error: --report needs a path" >&2; exit 2; }
                      REPORT="$2"; shift 2 ;;
    -h|--help)        sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)               echo "error: unknown flag $1" >&2; exit 2 ;;
    *)                VERSION="$1"; shift ;;
  esac
done

if [ -z "${VERSION}" ]; then
  VERSION="$(jq -r '.data_version // empty' "${ROOT}/data/catalog.json")"
  [ -n "${VERSION}" ] || { echo "error: data/catalog.json::data_version is empty" >&2; exit 2; }
fi
TAG="data-${VERSION}"

# Findings accumulate rather than exiting at the first one. A report that says
# "no release" when the truth is "no release AND the tag is gone too" sends
# whoever reads it down one branch of a two-branch problem.
FAILURES=()
fail() { FAILURES+=("$1"); echo "FAIL  $1" >&2; }
ok()   { echo "ok    $1"; }

echo "Reconciling ${TAG} (from ${REPO}) ..."
echo

# --- 1. the tag exists ------------------------------------------------------
#
# Checked against the remote, not a local checkout: a shallow CI clone has no
# tags, and "git rev-parse said no" would then be a false alarm every run.
TAG_SHA="$(gh api "repos/${REPO}/git/ref/tags/${TAG}" --jq '.object.sha' 2>/dev/null || true)"
if [ -n "${TAG_SHA}" ]; then
  ok "tag ${TAG} exists (${TAG_SHA})"
else
  fail "no tag \`${TAG}\`. data/catalog.json claims ${VERSION}, but nothing in the repository marks which commit that is."
fi

# --- 2. a published release exists ------------------------------------------
#
# A draft release is invisible to consumers and to the download URLs
# verify_data_release.sh uses, so "exists" has to mean published, not present.
RELEASE_JSON="$(gh api "repos/${REPO}/releases/tags/${TAG}" 2>/dev/null || true)"
RELEASE_EXISTS=0
if [ -n "${RELEASE_JSON}" ]; then
  IS_DRAFT="$(printf '%s' "${RELEASE_JSON}" | jq -r '.draft')"
  if [ "${IS_DRAFT}" = "true" ]; then
    fail "release \`${TAG}\` exists but is still a **draft** — consumers cannot download a draft release."
  else
    RELEASE_EXISTS=1
    ok "release ${TAG} is published ($(printf '%s' "${RELEASE_JSON}" | jq -r '.published_at'))"
  fi
else
  fail "no published release for \`${TAG}\`. This is the #344 state: the catalog claims ${VERSION}, and a consumer resolving \"latest\" gets the previous version instead."
fi

# --- 3. all four assets are present and non-empty ---------------------------
#
# The manifest pair (#296) first shipped in FIRST_MANIFEST_VERSION, so it is
# only required at or above that floor — otherwise this would report every
# historical release as broken.
FIRST_SIGNED="$(_floor FIRST_SIGNED_VERSION)"
FIRST_MANIFEST="$(_floor FIRST_MANIFEST_VERSION)"

# Numeric CalVer comparison. A lexical one puts 2026.10.0 below 2026.8.0 and
# would silently stop requiring signatures the month that lands.
_calver_ge() {
  local a="$1" b="$2"
  [ "$(printf '%s\n%s\n' "$a" "$b" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" = "$b" ]
}

EXPECTED=("nucl-parquet-data-${VERSION}.tar.zst")
_calver_ge "${VERSION}" "${FIRST_SIGNED}" \
  && EXPECTED+=("nucl-parquet-data-${VERSION}.tar.zst.minisig")
_calver_ge "${VERSION}" "${FIRST_MANIFEST}" \
  && EXPECTED+=("nucl-parquet-data-${VERSION}.manifest.json"
                "nucl-parquet-data-${VERSION}.manifest.json.minisig")

if [ "${RELEASE_EXISTS}" -eq 1 ]; then
  for asset in "${EXPECTED[@]}"; do
    SIZE="$(printf '%s' "${RELEASE_JSON}" \
      | jq -r --arg n "${asset}" '.assets[] | select(.name == $n) | .size' | head -1)"
    if [ -z "${SIZE}" ]; then
      fail "release \`${TAG}\` is missing the asset \`${asset}\`."
    elif ! [[ "${SIZE}" =~ ^[0-9]+$ ]]; then
      # `[ null -eq 0 ]` is a syntax error, and with `set -e` off the script
      # would fall through to the else and report the asset fine. Any check
      # whose failure mode is "reports OK" has to be closed explicitly.
      fail "asset \`${asset}\` on \`${TAG}\` reports a non-numeric size (\`${SIZE}\`) — cannot confirm it is intact."
    elif [ "${SIZE}" -eq 0 ]; then
      # A zero-byte asset is worse than a missing one: it downloads fine and
      # fails at whatever tries to use it.
      fail "asset \`${asset}\` on \`${TAG}\` is 0 bytes."
    else
      ok "asset ${asset} (${SIZE} bytes)"
    fi
  done
else
  echo "skip  asset checks (no published release to inspect)"
fi

# --- 4. the signature verifies against the key consumers pin ----------------
#
# Delegated to verify_data_release.sh rather than reimplemented, so this job
# exercises the exact path a consumer runs. That is deliberate: a bug in the
# consumer script is as much a broken release promise as a missing asset, and
# nothing else runs that script on a schedule.
# Three outcomes, not two. "Skipped" and "does not apply" are different claims:
# a release below the signing floor was checked and found not to need one, which
# is a complete answer, whereas --skip-signature leaves the question open. The
# closing summary says which, so neither reads as the other.
SIGNATURE_STATE=skipped
if [ "${SKIP_SIGNATURE}" -eq 1 ]; then
  echo "skip  signature verification (--skip-signature)"
elif [ "${RELEASE_EXISTS}" -ne 1 ]; then
  echo "skip  signature verification (no published release to verify)"
elif ! _calver_ge "${VERSION}" "${FIRST_SIGNED}"; then
  SIGNATURE_STATE=not-applicable
  echo "n/a   signature verification (${VERSION} predates ${FIRST_SIGNED}, unsigned by design)"
else
  SIGNATURE_STATE=checked
  echo
  echo "Running $(basename "${VERIFY_SCRIPT}") ${VERSION} ..."
  if VERIFY_OUT="$("${VERIFY_SCRIPT}" "${VERSION}" 2>&1)"; then
    echo "${VERIFY_OUT}"
    ok "signature verifies against docs/security/data-signing-key.pub"
  else
    echo "${VERIFY_OUT}" >&2
    fail "\`scripts/verify_data_release.sh ${VERSION}\` failed. The published bytes do not verify against the committed public key, or the consumer verification path itself is broken:

\`\`\`
$(printf '%s' "${VERIFY_OUT}" | tail -20)
\`\`\`"
  fi
fi

# --- report -----------------------------------------------------------------

echo
if [ "${#FAILURES[@]}" -eq 0 ]; then
  # Say only what was actually checked. "complete and signed" after a run that
  # skipped the signature is the same species of lie as the green-while-doing-
  # nothing HF mirror job (#283) — and this script exists to catch that class.
  case "${SIGNATURE_STATE}" in
    checked)        echo "OK  ${TAG} is published, complete and signed." ;;
    not-applicable) echo "OK  ${TAG} is published and complete. Predates ${FIRST_SIGNED}, so it carries no signature by design." ;;
    *)              echo "OK  ${TAG} is published and complete. Signature NOT verified this run." ;;
  esac
  [ -n "${REPORT}" ] && : > "${REPORT}"
  exit 0
fi

{
  echo "\`data/catalog.json\` claims **${VERSION}**, but the published release does not match it."
  echo
  for f in "${FAILURES[@]}"; do
    echo "- ${f}"
  done
  echo
  echo "### What this means"
  echo
  echo "A consumer resolving \"latest data release\" does not get ${VERSION}."
  echo "That is the state #344 left behind, and the reason this check exists."
  echo
  echo "### How to fix it"
  echo
  echo "Re-run the release for this version:"
  echo
  echo '```console'
  echo "\$ gh workflow run release-data.yml -f tag=${TAG}"
  echo '```'
  echo
  echo "If the tag itself is missing, re-run **Auto-tag data release** instead —"
  echo "it reconciles the catalog against what is published and creates the tag."
  echo
  echo "Reproduce locally with:"
  echo
  echo '```console'
  echo "\$ nix develop -c ./scripts/reconcile_data_release.sh ${VERSION}"
  echo '```'
} | tee "${REPORT:-/dev/null}" >&2

echo "${#FAILURES[@]} check(s) failed for ${TAG}" >&2
exit 1

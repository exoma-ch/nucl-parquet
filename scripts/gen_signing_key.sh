#!/usr/bin/env bash
# Generate the nucl-parquet data-signing keypair and install it as repository
# secrets (#289). Run this ONCE, on a trusted personal machine — never in CI.
#
#   just gen-signing-key
#
# What it does:
#   1. Generates a password-protected minisign keypair.
#   2. Sets DATA_SIGNING_KEY + DATA_SIGNING_KEY_PASSWORD as repo secrets via gh.
#   3. Writes the public key to docs/security/data-signing-key.pub for you to commit.
#   4. Leaves the secret key on disk at a path it prints, for you to move into
#      your password manager — and reminds you to shred it.
#
# Why not sops-encrypt the key into the repo? This repo is public: sops
# ciphertext there is permanently harvestable by every clone, fork and archive,
# and the signing key is a *pinned trust root* — consumers bake the public key
# in, so rotating it costs a release on every consumer. A plain repository
# secret only leaks to someone with access at the time. That trade is worth it
# for a trust root; it would not be for a revocable API token.
set -euo pipefail

REPO="${REPO:-exoma-ch/nucl-parquet}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBKEY_FILE="${ROOT}/docs/security/data-signing-key.pub"
OUTDIR="${OUTDIR:-$(mktemp -d)}"
SECKEY="${OUTDIR}/data-signing.key"

die() { echo "error: $*" >&2; exit 1; }

# If the script aborts anywhere after `minisign -G` — a wrong passphrase, a gh
# failure, Ctrl-C — a freshly generated secret key is sitting in a temp dir at
# 0600 that nobody has been told about. Shred it on any unsuccessful exit: an
# abandoned run must not leave an undocumented signing key on the filesystem.
# On success the key is deliberately kept, because the operator still has to
# move it into their password manager; the closing message names the path and
# the shred command.
COMPLETED=0
on_exit() {
  if [ "${COMPLETED}" -eq 0 ] && [ -f "${SECKEY}" ]; then
    shred -u "${SECKEY}" 2>/dev/null || rm -f "${SECKEY}"
    echo >&2
    echo "aborted before completion — the partially-provisioned secret key at" >&2
    echo "  ${SECKEY}" >&2
    echo "has been shredded. Nothing was left on disk. Re-run to start over," >&2
    echo "and check whether either repository secret was already updated:" >&2
    echo "  gh secret list --repo ${REPO}" >&2
  fi
}
trap on_exit EXIT

command -v minisign >/dev/null 2>&1 || die "minisign not found (nix develop -c $0)"
command -v gh >/dev/null 2>&1 || die "gh not found — needed to set repository secrets"

FORCE=""
if [ -f "${PUBKEY_FILE}" ]; then
  echo "WARNING: ${PUBKEY_FILE} already exists."
  echo "Generating a NEW key rotates the trust root: every consumer that pinned"
  echo "the old key must ship an update before it will accept new releases."
  echo "Follow the rotation runbook in docs/security/data-signing.md instead of"
  echo "overwriting blind."
  read -r -p "Type 'rotate' to continue: " confirm
  [ "${confirm}" = "rotate" ] || die "aborted"
  FORCE="-f"
fi

gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run 'gh auth login'"

echo
echo "Choose a strong passphrase. It protects the offline master copy of the key"
echo "(the copy you put in your password manager). Store BOTH in the same entry."
echo

# Prompt once, here, and feed minisign — rather than letting minisign prompt
# and then asking again for the copy to upload. Three prompts for the same
# secret invites a typo that would put a passphrase in CI which cannot decrypt
# the key, and it breaks entirely when stdin is not a terminal (minisign drains
# stdin, so the follow-up `read` hits EOF and `set -e` aborts mid-generation,
# leaving a keypair on disk and nothing uploaded).
read -r -s -p "Passphrase for the new signing key: " PASSPHRASE
echo
read -r -s -p "Confirm passphrase: " PASSPHRASE_CONFIRM
echo
[ -n "${PASSPHRASE}" ] || die "empty passphrase — the offline master copy would be unprotected"
[ "${PASSPHRASE}" = "${PASSPHRASE_CONFIRM}" ] || die "passphrases do not match"

umask 077
mkdir -p "$(dirname "${PUBKEY_FILE}")"
# shellcheck disable=SC2086  # FORCE is intentionally word-split (empty or -f)
printf '%s\n%s\n' "${PASSPHRASE}" "${PASSPHRASE}" \
  | minisign -G ${FORCE} -p "${PUBKEY_FILE}" -s "${SECKEY}"

# Prove the passphrase actually decrypts the key before uploading anything.
# Otherwise CI would hold a password that cannot sign, and the first signed
# release is what discovers it — after the tag has already been pushed.
TESTFILE="${OUTDIR}/.keycheck"
echo keycheck > "${TESTFILE}"
printf '%s\n' "${PASSPHRASE}" | minisign -S -s "${SECKEY}" -m "${TESTFILE}" >/dev/null 2>&1 \
  || die "that passphrase does not decrypt the key you just generated — nothing was uploaded"

echo "Passphrase verified. Setting repository secrets on ${REPO} ..."
gh secret set DATA_SIGNING_KEY --repo "${REPO}" < "${SECKEY}"
printf '%s' "${PASSPHRASE}" | gh secret set DATA_SIGNING_KEY_PASSWORD --repo "${REPO}"

# Past this point the run succeeded: the key is provisioned and the operator
# needs the on-disk copy to back up, so the EXIT trap must stop shredding it.
COMPLETED=1

cat <<EOF

================================================================================
Done. Two things left, both yours:

1. COMMIT the public key:
       git add ${PUBKEY_FILE#"${ROOT}/"}
       git commit -m "feat(release): publish the data-signing public key"

   Public key line (this is what consumers pin):
       $(grep -v '^untrusted comment:' "${PUBKEY_FILE}")

2. BACK UP the secret key, then destroy this copy. It exists in exactly two
   places right now: the repository secret, and:
       ${SECKEY}

   Put it in your password manager alongside the passphrase, then:
       shred -u ${SECKEY}

   If you lose it you cannot sign new releases without rotating the trust
   root — which costs a release on every consumer that pinned the old key.
================================================================================
EOF

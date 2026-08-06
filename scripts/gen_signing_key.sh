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

command -v minisign >/dev/null 2>&1 || die "minisign not found (nix develop -c $0)"
command -v gh >/dev/null 2>&1 || die "gh not found — needed to set repository secrets"

if [ -f "${PUBKEY_FILE}" ]; then
  echo "WARNING: ${PUBKEY_FILE} already exists."
  echo "Generating a NEW key rotates the trust root: every consumer that pinned"
  echo "the old key must ship an update before it will accept new releases."
  echo "Follow the rotation runbook in docs/security/data-signing.md instead of"
  echo "overwriting blind."
  read -r -p "Type 'rotate' to continue: " confirm
  [ "${confirm}" = "rotate" ] || die "aborted"
fi

gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run 'gh auth login'"

echo
echo "Choose a strong passphrase. It protects the offline master copy of the key"
echo "(the copy you put in your password manager). Store BOTH in the same entry."
echo

umask 077
mkdir -p "$(dirname "${PUBKEY_FILE}")"
minisign -G -p "${PUBKEY_FILE}" -s "${SECKEY}"

echo
read -r -s -p "Re-enter the passphrase so it can be stored as a repo secret: " PASSPHRASE
echo

# Fail before touching secrets if the passphrase is wrong — otherwise CI would
# hold a password that cannot decrypt the key, and the first signed release
# would be the thing that discovers it.
TESTFILE="${OUTDIR}/.keycheck"
echo keycheck > "${TESTFILE}"
printf '%s\n' "${PASSPHRASE}" | minisign -S -s "${SECKEY}" -m "${TESTFILE}" >/dev/null 2>&1 \
  || die "that passphrase does not decrypt the key you just generated — nothing was uploaded"

echo "Passphrase verified. Setting repository secrets on ${REPO} ..."
gh secret set DATA_SIGNING_KEY --repo "${REPO}" < "${SECKEY}"
printf '%s' "${PASSPHRASE}" | gh secret set DATA_SIGNING_KEY_PASSWORD --repo "${REPO}"

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

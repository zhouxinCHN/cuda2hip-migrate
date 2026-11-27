#!/usr/bin/env bash
# Fetch the pinned CUDA Samples v12.9 checkout without vendoring it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/third-party/cuda-samples}"
REPO="https://github.com/NVIDIA/cuda-samples.git"
TAG="v12.9"
COMMIT="cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997"

mkdir -p "$(dirname "${DEST}")"

if [[ ! -d "${DEST}/.git" ]]; then
    git clone --filter=blob:none "${REPO}" "${DEST}"
fi

git -C "${DEST}" fetch --tags origin
git -C "${DEST}" checkout --detach "${COMMIT}"

actual="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${actual}" != "${COMMIT}" ]]; then
    echo "ERROR: expected ${COMMIT}, found ${actual}" >&2
    exit 1
fi

echo "CUDA Samples ${TAG} ready at ${DEST}"
echo "Commit: ${actual}"

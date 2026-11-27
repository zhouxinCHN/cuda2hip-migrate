#!/usr/bin/env bash
# Fetch the pinned CUDALibrarySamples checkout without vendoring it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/third-party/CUDALibrarySamples}"
REPO="https://github.com/NVIDIA/CUDALibrarySamples.git"
COMMIT="692a738fbf481d52788da8c9685848bb73022743"

mkdir -p "$(dirname "${DEST}")"

if [[ ! -d "${DEST}/.git" ]]; then
    git clone --filter=blob:none "${REPO}" "${DEST}"
fi

git -C "${DEST}" fetch origin
git -C "${DEST}" checkout --detach "${COMMIT}"

actual="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${actual}" != "${COMMIT}" ]]; then
    echo "ERROR: expected ${COMMIT}, found ${actual}" >&2
    exit 1
fi

echo "CUDALibrarySamples ready at ${DEST}"
echo "Commit: ${actual}"

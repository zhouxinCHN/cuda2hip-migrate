"""Verify the pinned CUDA Samples checkout before anything is migrated.

The checkout is read-only input. Nothing in this project writes to it: sources
are copied into an artifact directory first, and every later stage works on the
copy.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

EXPECTED_TAG = "v12.9"
EXPECTED_COMMIT = "cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997"
DEFAULT_CHECKOUT = Path("third-party/cuda-samples")
SAMPLES_SUBDIR = "Samples"
COMMON_SUBDIR = "Common"


class CheckoutError(RuntimeError):
    """The checkout is missing or does not match the pinned commit."""


@dataclass(frozen=True)
class Checkout:
    root: Path
    commit: str
    tag: str

    @property
    def samples(self) -> Path:
        return self.root / SAMPLES_SUBDIR

    @property
    def common(self) -> Path:
        return self.root / COMMON_SUBDIR


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CheckoutError(
            f"git {' '.join(args)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_checkout(root: Path | None = None) -> Checkout:
    """Return the checkout only when it is exactly the pinned v12.9 commit."""
    candidate = Path(root) if root is not None else DEFAULT_CHECKOUT
    if not (candidate / ".git").exists():
        raise CheckoutError(
            f"{candidate} is not a git checkout; run ./scripts/fetch_cuda_samples.sh"
        )
    if not (candidate / SAMPLES_SUBDIR).is_dir():
        raise CheckoutError(f"{candidate} has no {SAMPLES_SUBDIR}/ directory")

    commit = _git(candidate, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise CheckoutError(
            f"checkout HEAD is {commit}, expected {EXPECTED_COMMIT} ({EXPECTED_TAG})"
        )

    dirty = _git(candidate, "status", "--porcelain")
    if dirty:
        raise CheckoutError(
            "checkout has local modifications; the pinned input must stay pristine:\n"
            + dirty
        )

    tag = _git(candidate, "describe", "--tags", "--exact-match")
    if tag != EXPECTED_TAG:
        raise CheckoutError(f"checkout describes as {tag}, expected {EXPECTED_TAG}")

    return Checkout(root=candidate.resolve(), commit=commit, tag=tag)


def sample_directory(checkout: Checkout, sample_path: str) -> Path:
    """Resolve a manifest sample path to its directory inside the checkout."""
    directory = checkout.samples / sample_path
    if not directory.is_dir():
        raise CheckoutError(f"sample path not found in checkout: {sample_path}")
    return directory

#!/usr/bin/env python3
"""Apply local patches to the vsrvrt package.

Detects the vsrvrt install location, checks each patch, and applies
only those that haven't been applied yet. Safe to run repeatedly.

Usage:
    python apply_patches.py          # apply all pending patches
    python apply_patches.py --check  # dry-run, report status only
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

PATCHES_DIR = Path(__file__).resolve().parent / "patches"

# Patches listed in order; each is applied against the vsrvrt package root
PATCHES = [
    "001-preview-chunk-cache-eviction.patch",
]


def get_vsrvrt_path() -> Path:
    """Return the installed vsrvrt package directory."""
    try:
        dist = importlib.metadata.distribution("vsrvrt")
    except importlib.metadata.PackageNotFoundError:
        sys.exit("ERROR: vsrvrt is not installed in the current environment.")
    # dist._path is the .dist-info dir; package is a sibling
    pkg_dir = Path(dist._path).parent / "vsrvrt"
    if not pkg_dir.is_dir():
        sys.exit(f"ERROR: expected vsrvrt at {pkg_dir} but directory not found.")
    return pkg_dir


def patch_status(patch_file: Path, pkg_root: Path) -> str:
    """Return 'applied', 'pending', or 'conflict'."""
    # --reverse --check: succeeds if the patch is already applied
    rev = subprocess.run(
        ["patch", "--dry-run", "--reverse", "--silent", "-p1",
         "-d", str(pkg_root.parent), "-i", str(patch_file)],
        capture_output=True,
    )
    if rev.returncode == 0:
        return "applied"

    # --check (forward): succeeds if the patch can be cleanly applied
    fwd = subprocess.run(
        ["patch", "--dry-run", "--silent", "-p1",
         "-d", str(pkg_root.parent), "-i", str(patch_file)],
        capture_output=True,
    )
    if fwd.returncode == 0:
        return "pending"

    return "conflict"


def apply_patch(patch_file: Path, pkg_root: Path) -> bool:
    """Apply a patch. Returns True on success."""
    result = subprocess.run(
        ["patch", "-p1", "-d", str(pkg_root.parent), "-i", str(patch_file)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Applied: {patch_file.name}")
        return True
    else:
        print(f"  FAILED:  {patch_file.name}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False


def main():
    check_only = "--check" in sys.argv

    pkg_root = get_vsrvrt_path()
    version = importlib.metadata.version("vsrvrt")
    print(f"vsrvrt {version} at {pkg_root}")

    if not PATCHES_DIR.is_dir():
        sys.exit(f"ERROR: patches directory not found: {PATCHES_DIR}")

    all_ok = True
    for name in PATCHES:
        patch_file = PATCHES_DIR / name
        if not patch_file.exists():
            print(f"  MISSING: {name}", file=sys.stderr)
            all_ok = False
            continue

        status = patch_status(patch_file, pkg_root)

        if status == "applied":
            print(f"  Already applied: {name}")
        elif status == "pending":
            if check_only:
                print(f"  Needs applying:  {name}")
            else:
                if not apply_patch(patch_file, pkg_root):
                    all_ok = False
        else:
            print(f"  CONFLICT: {name} — cannot apply cleanly", file=sys.stderr)
            all_ok = False

    if not all_ok:
        sys.exit(1)
    elif check_only:
        print("Check complete.")
    else:
        print("All patches applied.")


if __name__ == "__main__":
    main()

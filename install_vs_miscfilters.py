#!/usr/bin/env python3
"""Build and install the VapourSynth MiscFilters plugin from source.

MiscFilters provides misc.SCDetect (scene change detection) and other
filters used by vsrife and other VapourSynth scripts. It is not bundled
with conda-installed VapourSynth and has no pip/conda package, so it
must be built from the upstream source.

Requirements:
    - meson, ninja, pkg-config, and a C++ compiler (g++)
    - VapourSynth development headers (included with conda vapoursynth)
    - git (to clone the source)

Usage:
    python install_vs_miscfilters.py           # build and install
    python install_vs_miscfilters.py --check   # check if already installed
    python install_vs_miscfilters.py --force   # rebuild even if installed
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/vapoursynth/vs-miscfilters-obsolete.git"


def get_vs_plugin_dir() -> Path:
    """Return the VapourSynth plugin directory for the current environment."""
    # Try to find it via the vapoursynth module location
    spec = importlib.util.find_spec("vapoursynth")
    if spec is None or spec.origin is None:
        sys.exit("ERROR: vapoursynth is not installed in the current environment.")

    vs_site = Path(spec.origin).resolve().parent
    # conda installs put plugins in <env>/lib/vapoursynth/
    env_prefix = Path(sys.prefix)
    plugin_dir = env_prefix / "lib" / "vapoursynth"
    if plugin_dir.is_dir():
        return plugin_dir

    # Fallback: site-packages/vapoursynth/plugins/
    alt = vs_site / "plugins"
    if alt.is_dir():
        return alt

    sys.exit(f"ERROR: could not locate VapourSynth plugin directory.\n"
             f"  Searched: {plugin_dir}\n"
             f"           {alt}")


def is_installed(plugin_dir: Path) -> bool:
    """Check if libmiscfilters.so is already installed."""
    return (plugin_dir / "libmiscfilters.so").is_file()


def check_misc_loads() -> bool:
    """Verify that core.misc is accessible in VapourSynth."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import vapoursynth as vs; core = vs.core; core.misc; print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


def check_build_deps():
    """Verify required build tools are available."""
    missing = []
    for tool in ["meson", "ninja", "pkg-config", "g++"]:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        sys.exit(f"ERROR: missing build dependencies: {', '.join(missing)}\n"
                 f"  Install with: sudo apt install {' '.join(missing)} "
                 f"(or equivalent for your distro)")


def build_and_install(plugin_dir: Path):
    """Clone, build, and install MiscFilters."""
    check_build_deps()

    # PKG_CONFIG_PATH must include the conda env so meson finds vapoursynth
    env = os.environ.copy()
    env_prefix = Path(sys.prefix)
    pkg_paths = [
        str(env_prefix / "lib" / "pkgconfig"),
        env.get("PKG_CONFIG_PATH", ""),
    ]
    env["PKG_CONFIG_PATH"] = ":".join(p for p in pkg_paths if p)

    with tempfile.TemporaryDirectory(prefix="vs-miscfilters-") as tmpdir:
        src_dir = Path(tmpdir) / "vs-miscfilters"

        print(f"Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(src_dir)],
            check=True, capture_output=True,
        )

        build_dir = src_dir / "build"
        print("Configuring build ...")
        subprocess.run(
            ["meson", "setup", str(build_dir),
             f"--prefix={env_prefix}"],
            cwd=str(src_dir), env=env, check=True,
        )

        print("Building ...")
        subprocess.run(
            ["ninja", "-C", str(build_dir)],
            check=True,
        )

        # Install the .so directly to the plugin dir (avoids needing sudo)
        so_file = build_dir / "libmiscfilters.so"
        if not so_file.is_file():
            sys.exit("ERROR: build succeeded but libmiscfilters.so not found")

        dest = plugin_dir / "libmiscfilters.so"
        print(f"Installing {so_file.name} → {dest}")
        shutil.copy2(str(so_file), str(dest))

    # Verify it loads
    if check_misc_loads():
        print("Verified: core.misc is now available")
    else:
        print("WARNING: installed but core.misc did not load — "
              "check VapourSynth plugin autoload paths", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build and install VapourSynth MiscFilters plugin from source",
    )
    parser.add_argument("--check", action="store_true",
                        help="Check if MiscFilters is installed, don't build")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild and reinstall even if already present")
    args = parser.parse_args()

    plugin_dir = get_vs_plugin_dir()
    installed = is_installed(plugin_dir)

    if args.check:
        if installed:
            loads = check_misc_loads()
            print(f"MiscFilters: installed at {plugin_dir / 'libmiscfilters.so'}")
            print(f"core.misc:   {'OK' if loads else 'FAILED to load'}")
        else:
            print(f"MiscFilters: NOT installed (plugin dir: {plugin_dir})")
        sys.exit(0 if installed else 1)

    if installed and not args.force:
        print(f"MiscFilters already installed at {plugin_dir / 'libmiscfilters.so'}")
        print("Use --force to rebuild and reinstall.")
        return

    build_and_install(plugin_dir)
    print("Done.")


if __name__ == "__main__":
    main()

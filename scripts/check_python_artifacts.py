"""Build and test wheels, including a wheel rebuilt from an isolated sdist."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def check_wheel(wheel: Path, directory: Path, tests: Path) -> None:
    environment = directory / "environment"
    run(
        "uv",
        "venv",
        "--clear",
        "--python",
        sys.executable,
        str(environment),
        cwd=directory,
    )
    executable = environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    run(
        "uv",
        "pip",
        "install",
        "--python",
        str(executable),
        "--no-deps",
        str(wheel),
        cwd=directory,
    )
    run(str(executable), "-I", str(tests), cwd=directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", action="store_true", help="Also rebuild an sdist.")
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    tests = repository / "packages/python/tests/test_package.py"
    with tempfile.TemporaryDirectory(prefix="shrinkage-artifacts-") as temporary:
        directory = Path(temporary)
        artifacts = directory / "dist"
        run(
            "uv",
            "build",
            "--project",
            str(repository / "packages/python"),
            "--wheel",
            "--out-dir",
            str(artifacts),
            cwd=directory,
        )
        check_wheel(next(artifacts.glob("*.whl")), directory, tests)
        if options.sdist:
            run(
                "uv",
                "build",
                "--project",
                str(repository / "packages/python"),
                "--sdist",
                "--out-dir",
                str(artifacts),
                cwd=directory,
            )
            unpacked = directory / "unpacked"
            with tarfile.open(next(artifacts.glob("*.tar.gz"))) as archive:
                archive.extractall(unpacked, filter="data")
            source = next(unpacked.iterdir())
            rebuilt = directory / "rebuilt"
            # An isolated build must resolve every path dependency within the archive.
            run(
                "uv",
                "build",
                "--project",
                str(source),
                "--wheel",
                "--out-dir",
                str(rebuilt),
                cwd=directory,
            )
            check_wheel(next(rebuilt.glob("*.whl")), directory, tests)


if __name__ == "__main__":
    main()

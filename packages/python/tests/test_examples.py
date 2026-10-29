"""Run the public workflow against whichever distribution is installed."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExampleTests(unittest.TestCase):
    def test_hybrid_example_uses_the_installed_package(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "hybrid.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", str(example)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(len(result.stdout.splitlines()), 6)


if __name__ == "__main__":
    unittest.main()

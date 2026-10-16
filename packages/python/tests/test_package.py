"""Check the same public and native package surface in every build mode."""

import importlib.machinery
import importlib.metadata
import importlib.resources
import unittest

import shrinkage
from shrinkage import _shrinkage


class PackageTests(unittest.TestCase):
    def test_versions_match_distribution(self) -> None:
        expected = importlib.metadata.version("shrinkage")
        self.assertEqual(shrinkage.__version__, expected)
        self.assertEqual(_shrinkage.__version__, expected)

    def test_extension_calls_rust_core(self) -> None:
        self.assertTrue(
            any(
                _shrinkage.__file__.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
        )
        self.assertEqual(_shrinkage._core_smoke_test(), 1.5)

    def test_typing_files_are_installed(self) -> None:
        package = importlib.resources.files("shrinkage")
        self.assertTrue(package.joinpath("py.typed").is_file())
        self.assertIn(
            "def _core_smoke_test() -> float:",
            package.joinpath("_shrinkage.pyi").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

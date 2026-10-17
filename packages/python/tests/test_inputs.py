"""Exercise the native dense boundary shared by fitting and prediction."""

from __future__ import annotations

import gc
import unittest
import weakref
from typing import Any

import numpy as np
import numpy.typing as npt

from shrinkage import _shrinkage

FloatArray = npt.NDArray[np.float64]


class DenseInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.arange(12, dtype=np.float64).reshape(4, 3)
        self.y = np.arange(4, dtype=np.float64)

    def assert_snapshot(self, x: FloatArray, y: FloatArray) -> None:
        data = _shrinkage._prepare_dense_inputs(x, y)
        self.assertEqual(data.shape, x.shape)
        columns, response = data._snapshot()
        np.testing.assert_array_equal(columns, x.ravel(order="F"))
        np.testing.assert_array_equal(response, y)

    def test_layouts_preserve_logical_values(self) -> None:
        base = np.arange(48, dtype=np.float64).reshape(8, 6)
        for x in (
            self.x,
            np.asfortranarray(self.x),
            self.x.T,
            base[::2, ::2],
            base[::-2, ::-2],
            np.broadcast_to(self.x[:1], (4, 3)),
        ):
            with self.subTest(strides=x.strides):
                y = np.arange(2 * x.shape[0], dtype=np.float64)[::-2]
                self.assert_snapshot(x, y)
        self.assert_snapshot(self.x, np.broadcast_to(self.y[:1], (4,)))

    def test_arrays_backed_by_python_buffers_are_copied(self) -> None:
        storage = bytearray(self.x.tobytes())
        x = np.frombuffer(storage, dtype=np.float64).reshape(4, 3)
        y = np.frombuffer(self.y.tobytes(), dtype=np.float64)
        data = _shrinkage._prepare_dense_inputs(x, y)
        storage[:] = bytes(len(storage))
        del storage, x, y
        columns, response = data._snapshot()
        np.testing.assert_array_equal(columns, self.x.T.ravel())
        np.testing.assert_array_equal(response, self.y)

    def test_readonly_and_writable_arrays_are_unchanged(self) -> None:
        for writable in (True, False):
            with self.subTest(writable=writable):
                x, y = self.x.copy(), self.y.copy()
                x.flags.writeable = y.flags.writeable = writable
                self.assert_snapshot(x, y)
                np.testing.assert_array_equal(x, self.x)
                np.testing.assert_array_equal(y, self.y)
                self.assertEqual(x.flags.writeable, writable)
                self.assertEqual(y.flags.writeable, writable)

    def test_snapshot_is_independent_of_writable_aliases(self) -> None:
        x, y = self.x.view(), self.y.view()
        x.flags.writeable = y.flags.writeable = False
        data = _shrinkage._prepare_dense_inputs(x, y)
        self.x[:] = -100
        self.y[:] = -200
        columns, response = data._snapshot()
        np.testing.assert_array_equal(columns, np.arange(12).reshape(4, 3).T.ravel())
        np.testing.assert_array_equal(response, np.arange(4))

    def test_snapshot_outlives_sources_without_retaining_them(self) -> None:
        x_owner = np.arange(48, dtype=np.float64)
        y_owner = np.arange(8, dtype=np.float64)
        x, y = x_owner.reshape(8, 6)[::2, ::2], y_owner[::2]
        references = [weakref.ref(array) for array in (x_owner, y_owner, x, y)]
        data = _shrinkage._prepare_dense_inputs(x, y)
        del x_owner, y_owner, x, y
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        columns, response = data._snapshot()
        np.testing.assert_array_equal(
            columns, [0, 12, 24, 36, 2, 14, 26, 38, 4, 16, 28, 40]
        )
        np.testing.assert_array_equal(response, [0, 2, 4, 6])

    def test_native_storage_is_immutable_from_python(self) -> None:
        data = _shrinkage._prepare_dense_inputs(self.x, self.y)
        with self.assertRaises(AttributeError):
            data.__setattr__("shape", (1, 12))
        columns, response = data._snapshot()
        columns[0] = response[0] = -100
        self.assertEqual(data._snapshot()[0][0], 0.0)
        self.assertEqual(data._snapshot()[1][0], 0.0)

    def test_no_numpy_zero_copy_mode(self) -> None:
        for x in (self.x, np.asfortranarray(self.x)):
            x.flags.writeable = False
            with self.assertRaisesRegex(ValueError, "copy=False.*Rust-owned"):
                _shrinkage._prepare_dense_inputs(x, self.y, copy=False)

    def test_wrong_dtypes_are_rejected_without_coercion(self) -> None:
        for dtype in (np.float32, np.int64, np.bool_, np.complex128, object, "U4"):
            for name in ("X", "y"):
                with self.subTest(dtype=dtype, name=name):
                    x = self.x.astype(dtype) if name == "X" else self.x
                    y = self.y.astype(dtype) if name == "y" else self.y
                    with self.assertRaisesRegex(TypeError, name + ".*float64"):
                        _shrinkage._prepare_dense_inputs(x, y)

    def test_nonnative_byte_order_is_rejected(self) -> None:
        swapped = np.dtype(np.float64).newbyteorder("S")
        for x, y in (
            (self.x.astype(swapped), self.y),
            (self.x, self.y.astype(swapped)),
        ):
            with self.assertRaisesRegex(TypeError, "native.*float64"):
                _shrinkage._prepare_dense_inputs(x, y)

    def test_object_scalar_conversion_is_not_called(self) -> None:
        class Scalar:
            def __float__(self) -> float:
                raise AssertionError("Python scalar conversion called")

        with self.assertRaisesRegex(TypeError, "y.*float64"):
            _shrinkage._prepare_dense_inputs(self.x, np.array([Scalar()] * 4))

    def test_unaligned_buffers_are_rejected(self) -> None:
        x = np.ndarray((4, 3), dtype=np.float64, buffer=bytearray(97), offset=1)
        y = np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1)
        for design, response in ((x, self.y), (self.x, y)):
            with self.assertRaisesRegex(ValueError, "aligned"):
                _shrinkage._prepare_dense_inputs(design, response)

    def test_wrong_dimensions_are_rejected(self) -> None:
        for x in (np.array(1.0), self.x.ravel(), self.x[:, :, None]):
            with (
                self.subTest(shape=x.shape),
                self.assertRaisesRegex(ValueError, "X.*two-dimensional"),
            ):
                _shrinkage._prepare_dense_inputs(x, self.y)
        for y in (np.array(1.0), self.y[:, None], self.y[None, :, None]):
            with (
                self.subTest(shape=y.shape),
                self.assertRaisesRegex(ValueError, "y.*one-dimensional"),
            ):
                _shrinkage._prepare_dense_inputs(self.x, y)

    def test_empty_observations_and_mismatched_responses_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            _shrinkage._prepare_dense_inputs(np.empty((0, 3)), np.empty(0))
        for y in (np.empty(0), self.y[:-1], np.arange(5, dtype=np.float64)):
            with self.assertRaisesRegex(ValueError, "y.*length.*X.*rows"):
                _shrinkage._prepare_dense_inputs(self.x, y)

    def test_zero_features_and_singleton_axes_are_supported(self) -> None:
        self.assert_snapshot(np.empty((4, 0)), self.y)
        self.assert_snapshot(self.x[:1], self.y[:1])
        self.assert_snapshot(self.x[:, :1], self.y)

    def test_nonfinite_values_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            for name in ("X", "y"):
                with self.subTest(value=value, name=name):
                    x, y = self.x.copy(), self.y.copy()
                    if name == "X":
                        x[2, 1] = value
                    else:
                        y[2] = value
                    with self.assertRaisesRegex(ValueError, name + ".*finite"):
                        _shrinkage._prepare_dense_inputs(x, y)

    def test_array_protocols_and_subclasses_are_not_called(self) -> None:
        class ArrayProtocol:
            def __array__(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("array protocol called")

        class ArraySubclass(np.ndarray[Any, np.dtype[np.float64]]):
            def __getitem__(self, index: Any) -> Any:
                raise AssertionError("Python indexing called")

        for invalid in (
            None,
            [[1.0]],
            self.x.data,
            ArrayProtocol(),
            self.x.view(ArraySubclass),
            np.ma.array(self.x),
        ):
            for name in ("X", "y"):
                with self.subTest(type=type(invalid), name=name):
                    x = invalid if name == "X" else self.x
                    y = invalid if name == "y" else self.y
                    with self.assertRaisesRegex(TypeError, name + ".*numpy.ndarray"):
                        _shrinkage._prepare_dense_inputs(x, y)

    def test_failed_validation_releases_buffers(self) -> None:
        x, y = self.x.copy(), self.y.copy()
        y[0] = np.nan
        with self.assertRaises(ValueError):
            _shrinkage._prepare_dense_inputs(x, y)
        x.resize((12,), refcheck=True)
        y.resize((8,), refcheck=True)

    def test_shape_error_releases_buffers_before_copying(self) -> None:
        x = self.x.copy()
        y = np.arange(3, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "y.*length"):
            _shrinkage._prepare_dense_inputs(x, y)
        x.resize((12,), refcheck=True)
        y.resize((8,), refcheck=True)


if __name__ == "__main__":
    unittest.main()

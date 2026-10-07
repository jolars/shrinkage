//! Buffered solver parity, read failures, and caller-owned workspace reuse.

#![cfg(all(feature = "experimental-block-reader", feature = "zarrs_v0_22"))]

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::experimental::{BufferedDesign, fit_buffered};
use shrinkage::lazymatrix::{MatTransposeVecInto, MatVecInto, ZarrMatrix};
use shrinkage::{FitError, Gaussian, Normalization, Problem, ProximalGradient, Ridge};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs::array::{Array, ArrayBuilder, DataType};
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
};

const ROWS: [[f64; 3]; 5] = [
    [0.0, 1.0, 5.0],
    [2.0, 0.0, 5.0],
    [4.0, 3.0, 5.0],
    [1.0, 2.0, 5.0],
    [3.0, 4.0, 5.0],
];
const RESPONSE: [f64; 5] = [1.0, 3.0, 2.0, 5.0, 4.0];

struct Store {
    inner: Arc<MemoryStore>,
    reads: AtomicUsize,
    fail_after: AtomicUsize,
}

impl Store {
    fn record(&self) -> Result<(), StorageError> {
        let index = self.reads.fetch_add(1, Ordering::Relaxed);
        if index >= self.fail_after.load(Ordering::Relaxed) {
            return Err(StorageError::Other("injected block failure".into()));
        }
        Ok(())
    }

    fn reset(&self, fail_after: usize) {
        self.reads.store(0, Ordering::Relaxed);
        self.fail_after.store(fail_after, Ordering::Relaxed);
    }
}

impl ReadableStorageTraits for Store {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.record()?;
        self.inner.get(key)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.record()?;
        self.inner.get_partial_many(key, ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

fn fixture() -> (ZarrMatrix<Store>, Arc<Store>) {
    let inner = Arc::new(MemoryStore::new());
    let array = ArrayBuilder::new(vec![5, 3], vec![2, 2], DataType::Float64, 0.0_f64)
        .build(inner.clone(), "/x")
        .unwrap();
    array.store_metadata().unwrap();
    array
        .store_array_subset_elements(
            &array.subset_all(),
            &ROWS.into_iter().flatten().collect::<Vec<_>>(),
        )
        .unwrap();
    let store = Arc::new(Store {
        inner,
        reads: AtomicUsize::new(0),
        fail_after: AtomicUsize::new(usize::MAX),
    });
    let array = Array::open(store.clone(), "/x").unwrap();
    store.reset(usize::MAX);
    (ZarrMatrix::try_new(array).unwrap(), store)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn buffered_fits_match_dense_fits_and_reuse_the_same_buffer() {
    let (matrix, _) = fixture();
    let dense =
        matrix::Matrix::from_rows(&ROWS.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
    let options = ProximalGradient::<Vec<f64>>::new()
        .initial_step(0.01)
        .tolerance(1e-8);
    let mut buffer = [f64::NAN; 14];
    let address = buffer.as_ptr();
    for normalization in [Normalization::None, Normalization::Standardize] {
        for intercept in [false, true] {
            let datafit = Gaussian::new(&RESPONSE).fit_intercept(intercept);
            let expected = Problem::new(&dense, datafit, Ridge::new(0.2))
                .normalize(normalization)
                .fit_with(&options)
                .unwrap();
            for shape in [[2, 2], [4, 3], [1, 1]] {
                buffer.fill(f64::NAN);
                let fit = fit_buffered(
                    &Problem::new(&matrix, datafit, Ridge::new(0.2)).normalize(normalization),
                    &options,
                    shape,
                    &mut buffer,
                )
                .unwrap();
                assert_eq!(fit.termination(), expected.termination());
                close(fit.objective(), expected.objective());
                close(fit.intercept(), expected.intercept());
                for (&actual, &expected) in fit.coefficients().iter().zip(expected.coefficients()) {
                    close(actual, expected);
                }
                assert_eq!(buffer.as_ptr(), address);
                assert!(buffer[shape[0] * shape[1]..].iter().all(|x| x.is_nan()));
            }
        }
    }
}

#[test]
fn failures_during_statistics_and_later_products_stop_and_allow_retry() {
    let (matrix, store) = fixture();
    let problem = Problem::new(&matrix, Gaussian::new(&RESPONSE), Ridge::new(0.2))
        .normalize(Normalization::Standardize);
    let options = ProximalGradient::<Vec<f64>>::new()
        .initial_step(0.01)
        .tolerance(1e-6);
    let mut buffer = [999.0; 14];
    let address = buffer.as_ptr();
    // Each validation, normalization, or product scan reads six storage chunks.
    for (fail_after, operation) in [
        (1, "validating buffered matrix entries"),
        (7, "computing training normalization statistics"),
        (13, "computing training normalization statistics"),
        (19, "computing a forward design product"),
        (25, "computing a transposed design product"),
        (31, "computing a forward design product"),
        (37, "computing a forward design product"),
    ] {
        store.reset(fail_after);
        buffer.fill(999.0);
        let error = fit_buffered(&problem, &options, [4, 3], &mut buffer).unwrap_err();
        assert!(
            matches!(error, FitError::Backend { operation: actual, .. } if actual == operation)
        );
        assert!(
            std::error::Error::source(&error)
                .unwrap()
                .to_string()
                .contains("injected block failure")
        );
        assert_eq!(store.reads.load(Ordering::Relaxed), fail_after + 1);
        assert_eq!(&buffer[12..], &[999.0; 2]);
        assert_eq!(buffer.as_ptr(), address);
        store.reset(usize::MAX);
        assert!(fit_buffered(&problem, &options, [4, 3], &mut buffer).is_ok());
    }
}

#[test]
fn buffered_products_overwrite_poisoned_outputs_and_match_the_adjoint() {
    let (matrix, _) = fixture();
    let mut buffer = [f64::NAN; 12];
    let design = BufferedDesign::new(&matrix, [4, 3], &mut buffer);
    let x = vec![1.0, 2.0, -1.0];
    let y = vec![2.0, -1.0, 0.0, 3.0, 1.0];
    let mut forward = vec![f64::NAN; 5];
    let mut transpose = vec![f64::NAN; 3];
    design.matvec_into(&x, &mut forward).unwrap();
    design.mat_transpose_vec_into(&y, &mut transpose).unwrap();
    close(
        forward.iter().zip(&y).map(|(a, b)| a * b).sum(),
        transpose.iter().zip(&x).map(|(a, b)| a * b).sum(),
    );
    for (row, &actual) in ROWS.iter().zip(&forward) {
        close(actual, row.iter().zip(&x).map(|(a, b)| a * b).sum());
    }
}

#[test]
fn failed_blocks_are_never_accumulated_into_products() {
    let (matrix, store) = fixture();
    let mut buffer = [999.0; 12];
    let design = BufferedDesign::new(&matrix, [4, 3], &mut buffer);
    for fail_after in [1, 5] {
        store.reset(fail_after);
        let mut forward = vec![f64::NAN; 5];
        assert!(design.matvec_into(&vec![1.0; 3], &mut forward).is_err());
        let rows = if fail_after == 1 { 0 } else { 4 };
        let expected: Vec<_> = ROWS
            .iter()
            .enumerate()
            .map(|(i, row)| if i < rows { row.iter().sum() } else { 0.0 })
            .collect();
        assert_eq!(forward, expected);
        store.reset(fail_after);
        let mut transpose = vec![f64::NAN; 3];
        assert!(
            design
                .mat_transpose_vec_into(&vec![1.0; 5], &mut transpose)
                .is_err()
        );
        for (j, &actual) in transpose.iter().enumerate() {
            close(actual, ROWS[..rows].iter().map(|row| row[j]).sum());
        }
        store.reset(usize::MAX);
        design.matvec_into(&vec![1.0; 3], &mut forward).unwrap();
        assert_eq!(forward[4], ROWS[4].iter().sum::<f64>());
    }
}

#[test]
fn buffered_options_and_empty_training_are_validated_before_reads() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let (matrix, store) = fixture();
    for (shape, capacity) in [([0, 2], 4), ([2, 2], 3), ([usize::MAX, 2], 4)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                BufferedDesign::new(&matrix, shape, &mut vec![0.0; capacity]);
            }))
            .is_err()
        );
    }
    let mut buffer = [0.0; 4];
    let design = BufferedDesign::new(&matrix, [2, 2], &mut buffer);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            design.matvec_into(&vec![1.0], &mut vec![0.0; 5]).unwrap();
        }))
        .is_err()
    );
    assert_eq!(store.reads.load(Ordering::Relaxed), 0);

    let empty = matrix::Matrix::empty(0, 3);
    assert!(
        Problem::new(&empty, Gaussian::new(&[]), Ridge::new(0.1))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    let array = ArrayBuilder::new(vec![0, 3], vec![2, 2], DataType::Float64, 0.0_f64)
        .build(store.clone(), "/empty")
        .unwrap();
    let empty = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    assert!(matches!(
        fit_buffered(
            &Problem::new(&empty, Gaussian::new(&[]), Ridge::new(0.1)),
            &ProximalGradient::new(),
            [2, 2],
            &mut [0.0; 4]
        ),
        Err(FitError::InvalidInput { .. })
    ));
    assert_eq!(store.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn nonfinite_entries_are_rejected_during_buffered_validation() {
    let array = ArrayBuilder::new(vec![2, 2], vec![2, 2], DataType::Float64, 0.0_f64)
        .build(Arc::new(MemoryStore::new()), "/invalid")
        .unwrap();
    array
        .store_chunk_elements(&[0, 0], &[1.0, 2.0, f64::NAN, 3.0])
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    let error = fit_buffered(
        &Problem::new(&matrix, Gaussian::new(&[1.0, 2.0]), Ridge::new(0.1))
            .normalize(Normalization::None),
        &ProximalGradient::new(),
        [2, 2],
        &mut [0.0; 4],
    )
    .unwrap_err();
    assert!(
        matches!(error, FitError::InvalidInput { message } if message.contains("row 1, column 0"))
    );
}

#[test]
fn zero_column_designs_fit_an_intercept_without_storage_reads() {
    let (_, store) = fixture();
    let array = ArrayBuilder::new(vec![3, 0], vec![2, 2], DataType::Float64, 0.0_f64)
        .build(store.clone(), "/zero-columns")
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    let fit = fit_buffered(
        &Problem::new(&matrix, Gaussian::new(&[2.0; 3]), Ridge::new(0.1)),
        &ProximalGradient::new(),
        [2, 2],
        &mut [f64::NAN; 4],
    )
    .unwrap();
    assert!(fit.coefficients().is_empty());
    close(fit.intercept(), 2.0);
    assert_eq!(store.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn a_failed_backtracking_read_never_reaches_loss_evaluation() {
    use shrinkage::{Datafit, SmoothDatafit};
    use std::cell::Cell;
    struct CountingLoss<'a> {
        gaussian: Gaussian<'a>,
        evaluations: &'a Cell<usize>,
    }
    impl Datafit for CountingLoss<'_> {
        type Error = FitError;
        fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
            self.evaluations.set(self.evaluations.get() + 1);
            self.gaussian.value(predictor)
        }
    }
    impl SmoothDatafit for CountingLoss<'_> {
        fn nobs(&self) -> usize {
            self.gaussian.nobs()
        }
        fn fits_intercept(&self) -> bool {
            self.gaussian.fits_intercept()
        }
        fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
            self.gaussian.gradient(predictor, output)
        }
    }
    let (matrix, store) = fixture();
    let evaluations = Cell::new(0);
    let loss = CountingLoss {
        gaussian: Gaussian::new(&RESPONSE),
        evaluations: &evaluations,
    };
    let problem =
        Problem::new(&matrix, loss, Ridge::new(0.2)).normalize(Normalization::Standardize);
    // An oversized initial step rejects the first trial; fail inside the second.
    store.reset(37);
    let error = fit_buffered(
        &problem,
        &ProximalGradient::new().initial_step(100.0),
        [4, 3],
        &mut [999.0; 12],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        FitError::Backend {
            operation: "computing a forward design product",
            ..
        }
    ));
    assert_eq!(store.reads.load(Ordering::Relaxed), 38);
    assert_eq!(evaluations.get(), 2);
}

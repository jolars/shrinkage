//! Experimental buffered input for proximal gradient.
//!
//! Uses the published LazyMatrix block-reader capability with caller-owned
//! storage buffers. The reader and adapter interfaces remain provisional.

use std::cell::RefCell;

use lazymatrix::{
    ColumnStats, DenseBlock, MatTransposeVecInto, MatVecInto, MatrixErrorType, MatrixShape,
    NormalizationStats, ReadBlock,
};

use crate::error::invalid;
use crate::solver::proximal::iterate;
use crate::{FitError, Problem, ProximalFit, ProximalGradient, ProximalPenalty, SmoothDatafit};

/// A borrowed design with one caller-owned block buffer for sequential scans.
///
/// Raw products use packed blocks; [`lazymatrix::LazyMatrix`] applies training
/// normalization. Statistics retain the backend's combined scan hook. The
/// adapter does not provide borrowed columns or rows, cache chunks, or require
/// a full-width row buffer. Product outputs may be partial after an error.
///
/// The buffer remains borrowed until this adapter is dropped. Internal mutable
/// borrowing supports the shared-reference operator traits; this adapter is
/// intended for synchronous use and is not `Sync`. The design must remain
/// unchanged across validation, normalization, and fitting.
pub struct BufferedDesign<'a, M: ?Sized> {
    matrix: &'a M,
    block_shape: [usize; 2],
    buffer: RefCell<&'a mut [f64]>,
}

impl<'a, M: ReadBlock<f64> + ?Sized> BufferedDesign<'a, M> {
    /// Borrow a design and initialized buffer for blocks of at most `block_shape`.
    ///
    /// Storage chunk and codec workspace can exceed the caller's buffer size.
    ///
    /// # Panics
    /// Panics for a zero block dimension, dimension overflow, or a buffer shorter
    /// than the product of the requested block dimensions.
    pub fn new(matrix: &'a M, block_shape: [usize; 2], buffer: &'a mut [f64]) -> Self {
        assert!(
            block_shape.iter().all(|&n| n > 0),
            "block dimensions must be positive"
        );
        let capacity = block_shape[0]
            .checked_mul(block_shape[1])
            .expect("block dimensions overflow");
        assert!(buffer.len() >= capacity, "block buffer too short");
        Self {
            matrix,
            block_shape,
            buffer: RefCell::new(buffer),
        }
    }

    fn scan(
        &self,
        mut visit: impl FnMut(usize, usize, DenseBlock<'_, f64>) -> bool,
    ) -> Result<(), M::Error> {
        let mut buffer = self.buffer.borrow_mut();
        for row in (0..self.nrows()).step_by(self.block_shape[0]) {
            let row_end = row.saturating_add(self.block_shape[0]).min(self.nrows());
            for col in (0..self.ncols()).step_by(self.block_shape[1]) {
                let col_end = col.saturating_add(self.block_shape[1]).min(self.ncols());
                let block = self
                    .matrix
                    .read_block(row..row_end, col..col_end, &mut buffer)?;
                assert_eq!(
                    (block.nrows(), block.ncols()),
                    (row_end - row, col_end - col),
                    "reader returned invalid block dimensions"
                );
                // An error must prevent the consumer from observing any partial block.
                if !visit(row, col, block) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), FitError<M::Error>> {
        let mut invalid_entry = None;
        self.scan(|row, col, block| {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    if !block.get(i, j).is_finite() {
                        invalid_entry = Some((row + i, col + j));
                        return false;
                    }
                }
            }
            true
        })
        .map_err(|source| FitError::Backend {
            operation: "validating buffered matrix entries",
            source,
        })?;
        if let Some((row, col)) = invalid_entry {
            return Err(invalid(format!(
                "invalid matrix entry at row {row}, column {col}: expected a finite value"
            )));
        }
        Ok(())
    }
}

impl<M: MatrixShape + ?Sized> MatrixShape for BufferedDesign<'_, M> {
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }
    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }
}

impl<M: MatrixErrorType + ?Sized> MatrixErrorType for BufferedDesign<'_, M> {
    type Error = M::Error;
}

impl<M: ReadBlock<f64> + ?Sized> MatVecInto<Vec<f64>> for BufferedDesign<'_, M> {
    fn matvec_into(&self, input: &Vec<f64>, output: &mut Vec<f64>) -> Result<(), Self::Error> {
        assert_eq!(input.len(), self.ncols(), "matvec_into: dimension mismatch");
        assert_eq!(
            output.len(),
            self.nrows(),
            "matvec_into: output dimension mismatch"
        );
        output.fill(0.0);
        self.scan(|row, col, block| {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    output[row + i] += block.get(i, j) * input[col + j];
                }
            }
            true
        })
    }
}

impl<M: ReadBlock<f64> + ?Sized> MatTransposeVecInto<Vec<f64>> for BufferedDesign<'_, M> {
    fn mat_transpose_vec_into(
        &self,
        input: &Vec<f64>,
        output: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            input.len(),
            self.nrows(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.ncols(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        output.fill(0.0);
        self.scan(|row, col, block| {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    output[col + j] += block.get(i, j) * input[row + i];
                }
            }
            true
        })
    }
}

macro_rules! delegate_stats {
    ($($name:ident $(($arg:ident))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: &[f64])?) -> Result<Vec<f64>, Self::Error> {
            self.matrix.$name($($arg)?)
        })*
    };
}

impl<M: ColumnStats<f64> + ?Sized> ColumnStats<f64> for BufferedDesign<'_, M> {
    fn normalization_stats(
        &self,
        spec: lazymatrix::Normalization,
    ) -> Result<NormalizationStats<f64>, Self::Error> {
        self.matrix.normalization_stats(spec)
    }
    delegate_stats!(
        col_means,
        col_sds,
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l1_centered(centers),
        col_l2_centered(centers),
        col_maxabs_centered(centers)
    );
}

/// Fit with existing proximal-gradient iteration over caller-buffered blocks.
///
/// Finite-entry validation and product passes reuse `buffer`. Normalization
/// statistics use the underlying backend and may allocate their own workspace.
/// Partial products and failed blocks are discarded; a retry can reuse the same
/// buffer. The returned fit retains original-scale coefficients and training
/// preprocessing. Its regular `predict` method still requires borrowed columns.
///
/// # Errors
/// Preserves backend errors during validation, statistics, and products, and
/// forwards the regular proximal solver's input and numerical errors.
///
/// # Panics
/// Panics for invalid block dimensions or insufficient buffer length, as in
/// [`BufferedDesign::new`].
pub fn fit_buffered<M, D, P>(
    problem: &Problem<&M, D, P>,
    options: &ProximalGradient<Vec<f64>>,
    block_shape: [usize; 2],
    buffer: &mut [f64],
) -> Result<ProximalFit, FitError<M::Error>>
where
    M: ReadBlock<f64> + ColumnStats<f64> + ?Sized,
    D: SmoothDatafit,
    P: ProximalPenalty,
{
    let threshold = options.validate()?;
    if problem.datafit().nobs() != problem.design().nrows() {
        return Err(invalid(
            "datafit observation count must match training rows",
        ));
    }
    if problem.design().nrows() == 0 {
        return Err(invalid("training data must have at least one observation"));
    }
    let design = BufferedDesign::new(*problem.design(), block_shape, buffer);
    design.validate()?;
    let prepared = problem
        .prepare_normalization(problem.datafit().fits_intercept())?
        .map_data(|_| design);
    let solution = iterate(
        &prepared.matrix,
        problem.datafit(),
        problem.penalty(),
        options,
        threshold,
    )?;
    ProximalFit::from_solution(solution, prepared.into_preprocessing())
}

//! Coarse design operations for the shared proximal iteration.

use lazymatrix::{MatTransposeVecInto, MatVecInto, MatrixShape};

use crate::error::invalid;
use crate::problem::Prepared;
use crate::{FitError, Preprocessing, ProximalVector};

/// A prepared CPU design oracle for proximal gradient.
///
/// Products use the optimization-scale design. Each call operates on complete
/// slices so runtime dispatch stays outside scalar loops. Concrete adapters
/// retain LazyMatrix normalization, native product buffers, and column views.
/// This solver capability does not replace LazyMatrix's matrix traits.
pub trait ProximalDesign {
    /// Backend failure type.
    type Error;

    /// Number of observations.
    fn nrows(&self) -> usize;

    /// Number of coefficients.
    fn ncols(&self) -> usize;

    /// Write the normalized forward product into reusable output storage.
    ///
    /// # Errors
    /// Reject mismatched dimensions and preserve backend failures. Output is
    /// unspecified after an error and must not be consumed.
    fn forward(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), FitError<Self::Error>>;

    /// Write the normalized transposed product into reusable output storage.
    ///
    /// # Errors
    /// Reject mismatched dimensions and preserve backend failures. Output is
    /// unspecified after an error and must not be consumed.
    fn transpose(&mut self, input: &[f64], output: &mut [f64])
    -> Result<(), FitError<Self::Error>>;

    /// Consume the prepared design and retain its fitted normalization vectors.
    fn into_preprocessing(self: Box<Self>) -> Preprocessing;
}

pub(crate) struct NativeDesign<M, V> {
    prepared: Prepared<M>,
    parameters: V,
    predictor: V,
    derivative: V,
    gradient: V,
}

impl<M: MatrixShape, V: ProximalVector> NativeDesign<M, V> {
    pub(crate) fn new(prepared: Prepared<M>) -> Self {
        let n = prepared.matrix.nrows();
        let p = prepared.matrix.ncols();
        Self {
            prepared,
            parameters: V::zeros(p),
            predictor: V::zeros(n),
            derivative: V::zeros(n),
            gradient: V::zeros(p),
        }
    }

    pub(crate) fn into_preprocessing(self) -> Preprocessing {
        self.prepared.into_preprocessing()
    }
}

impl<M, V> ProximalDesign for NativeDesign<M, V>
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    type Error = M::Error;

    fn nrows(&self) -> usize {
        self.prepared.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.prepared.matrix.ncols()
    }

    fn forward(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        if input.len() != self.ncols() || output.len() != self.nrows() {
            return Err(invalid(
                "forward product dimensions must match the prepared design",
            ));
        }
        for (j, &value) in input.iter().enumerate() {
            self.parameters.set(j, value);
        }
        self.prepared
            .matrix
            .matvec_into(&self.parameters, &mut self.predictor)
            .map_err(|source| FitError::Backend {
                operation: "computing a forward design product",
                source,
            })?;
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.predictor.get(i);
        }
        Ok(())
    }

    fn transpose(
        &mut self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        if input.len() != self.nrows() || output.len() != self.ncols() {
            return Err(invalid(
                "transposed product dimensions must match the prepared design",
            ));
        }
        for (i, &value) in input.iter().enumerate() {
            self.derivative.set(i, value);
        }
        self.prepared
            .matrix
            .mat_transpose_vec_into(&self.derivative, &mut self.gradient)
            .map_err(|source| FitError::Backend {
                operation: "computing a transposed design product",
                source,
            })?;
        for (j, value) in output.iter_mut().enumerate() {
            *value = self.gradient.get(j);
        }
        Ok(())
    }

    fn into_preprocessing(self: Box<Self>) -> Preprocessing {
        (*self).into_preprocessing()
    }
}

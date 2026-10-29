//! Coarse design operations for the shared proximal iteration.

use lazymatrix::{
    LazyColumn, LazyMatrix, MatTransposeVecInto, MatVecInto, MatrixShape, RawColumn, RawColumns,
};

use crate::error::{finite, invalid};
use crate::problem::Prepared;
use crate::screening::ScreeningDesign;
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

    /// Borrow optional normalized-column operations for safe screening.
    ///
    /// Product-only adapters can retain their existing implementation. The
    /// solver screens only when the design exposes this capability.
    fn screening_design(&mut self) -> Option<&mut dyn ScreeningDesign<Error = Self::Error>> {
        None
    }

    /// Consume the prepared design and retain its fitted normalization vectors.
    fn into_preprocessing(self: Box<Self>) -> Preprocessing;
}

pub(crate) struct NativeDesign<M, V> {
    prepared: Prepared<M>,
    parameters: V,
    predictor: V,
    derivative: V,
    gradient: V,
    columns: Option<ColumnOperations<M>>,
}

type SelectedProduct<M> = fn(&LazyMatrix<M>, &[f64], &[usize], &mut [f64]) -> Result<(), FitError>;

struct ColumnOperations<M> {
    norms: fn(&LazyMatrix<M>, &mut [f64]) -> Result<(), FitError>,
    error_scales: fn(&LazyMatrix<M>, &mut [f64]) -> Result<(), FitError>,
    forward: SelectedProduct<M>,
    transpose: SelectedProduct<M>,
}

impl<M: MatrixShape, V: ProximalVector> NativeDesign<M, V> {
    pub(crate) fn matrix(&self) -> &lazymatrix::LazyMatrix<M> {
        &self.prepared.matrix
    }

    pub(crate) fn new(prepared: Prepared<M>) -> Self {
        let n = prepared.matrix.nrows();
        let p = prepared.matrix.ncols();
        Self {
            prepared,
            parameters: V::zeros(p),
            predictor: V::zeros(n),
            derivative: V::zeros(n),
            gradient: V::zeros(p),
            columns: None,
        }
    }

    pub(crate) fn into_preprocessing(self) -> Preprocessing {
        self.prepared.into_preprocessing()
    }
}

impl<M: RawColumns<f64>, V: ProximalVector> NativeDesign<M, V> {
    pub(crate) fn new_screenable(prepared: Prepared<M>) -> Self {
        let mut design = Self::new(prepared);
        design.columns = Some(ColumnOperations {
            norms: column_norms::<M>,
            error_scales: column_error_scales::<M>,
            forward: forward_active::<M>,
            transpose: transpose_active::<M>,
        });
        design
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

    fn screening_design(&mut self) -> Option<&mut dyn ScreeningDesign<Error = Self::Error>> {
        self.columns.as_ref()?;
        Some(self)
    }
}

impl<M, V> ScreeningDesign for NativeDesign<M, V>
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    type Error = M::Error;

    fn column_norms(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        let columns = self
            .columns
            .as_ref()
            .ok_or_else(|| invalid("screening requires normalized column access"))?;
        (columns.norms)(&self.prepared.matrix, output)
            .map_err(|error| FitError::from_component(error, 0))
    }

    fn column_error_scales(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        let columns = self
            .columns
            .as_ref()
            .ok_or_else(|| invalid("screening requires normalized column access"))?;
        (columns.error_scales)(&self.prepared.matrix, output)
            .map_err(|error| FitError::from_component(error, 0))
    }

    fn forward_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        let columns = self
            .columns
            .as_ref()
            .ok_or_else(|| invalid("screening requires normalized column access"))?;
        (columns.forward)(&self.prepared.matrix, input, active, output)
            .map_err(|error| FitError::from_component(error, 0))
    }

    fn transpose_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        let columns = self
            .columns
            .as_ref()
            .ok_or_else(|| invalid("screening requires normalized column access"))?;
        (columns.transpose)(&self.prepared.matrix, input, active, output)
            .map_err(|error| FitError::from_component(error, 0))
    }
}

fn validate_active(active: &[usize], ncols: usize) -> Result<(), FitError> {
    if active.last().is_some_and(|&j| j >= ncols)
        || active.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "active feature indices must be increasing and within the design",
        ));
    }
    Ok(())
}

fn column_norms<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    output: &mut [f64],
) -> Result<(), FitError> {
    if output.len() != matrix.ncols() {
        return Err(invalid(
            "column norm dimensions must match the prepared design",
        ));
    }
    for (j, value) in output.iter_mut().enumerate() {
        let (norm, _) = column_bounds(&matrix.column(j));
        *value = finite(norm, "computing screening column norms", 0)?;
    }
    Ok(())
}

fn normalized_deviation(raw: f64, center: f64, scale: f64) -> f64 {
    let difference = raw - center;
    if difference.is_finite() {
        divide_bound(difference, scale)
    } else {
        raw / scale - center / scale
    }
}

fn divide_bound(value: f64, scale: f64) -> f64 {
    let quotient = value / scale;
    // A positive safe radius must still see columns below the subnormal range.
    if quotient == 0.0 && value != 0.0 {
        f64::from_bits(1).copysign(value * scale.signum())
    } else {
        quotient
    }
}

fn column_bounds<C: RawColumn<f64>>(column: &LazyColumn<C, f64>) -> (f64, f64) {
    let mut raw_norm = 0.0_f64;
    let mut logical_norm = 0.0_f64;
    let mut raw_abs_sum = 0.0;
    let mut previous = None;
    let mut canonical = true;
    column.raw().for_each_stored(|row, raw| {
        let scaled = divide_bound(raw, column.scale());
        raw_norm = raw_norm.hypot(scaled);
        raw_abs_sum += scaled.abs();
        // Squaring first can lose an otherwise representable screening norm.
        logical_norm =
            logical_norm.hypot(normalized_deviation(raw, column.center(), column.scale()));
        canonical &= previous.is_none_or(|last| last < row);
        previous = Some(row);
    });
    let background = divide_bound(column.center(), column.scale()).abs();
    let center_norm = (column.len() as f64).sqrt() * background;
    let error_scale = if canonical {
        let implicit_count = column.len() - column.raw().stored_len();
        if implicit_count > 0 {
            logical_norm = logical_norm.hypot(background * (implicit_count as f64).sqrt());
        }
        raw_norm + center_norm
    } else {
        // Duplicate entries add before normalization. The triangle inequality
        // bounds their logical column without gathering or copying entries.
        logical_norm = raw_abs_sum + center_norm;
        logical_norm
    };
    (logical_norm, error_scale.max(logical_norm))
}

fn column_error_scales<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    output: &mut [f64],
) -> Result<(), FitError> {
    if output.len() != matrix.ncols() {
        return Err(invalid(
            "column error scale dimensions must match the prepared design",
        ));
    }
    for (j, value) in output.iter_mut().enumerate() {
        // Native products subtract the centering term after summing raw data.
        // Their rounding error depends on both terms, even for a zero column.
        let (_, error_scale) = column_bounds(&matrix.column(j));
        *value = finite(error_scale, "computing screening column error scales", 0)?;
    }
    Ok(())
}

fn forward_active<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    input: &[f64],
    active: &[usize],
    output: &mut [f64],
) -> Result<(), FitError> {
    if input.len() != matrix.ncols() || output.len() != matrix.nrows() {
        return Err(invalid(
            "forward product dimensions must match the prepared design",
        ));
    }
    validate_active(active, matrix.ncols())?;
    output.fill(0.0);
    for &j in active {
        if input[j] != 0.0 {
            matrix.column(j).scaled_add_to(input[j], output);
        }
    }
    for &value in output.iter() {
        finite(value, "computing a selected forward design product", 0)?;
    }
    Ok(())
}

fn transpose_active<M: RawColumns<f64>>(
    matrix: &LazyMatrix<M>,
    input: &[f64],
    active: &[usize],
    output: &mut [f64],
) -> Result<(), FitError> {
    if input.len() != matrix.nrows() || output.len() != matrix.ncols() {
        return Err(invalid(
            "transposed product dimensions must match the prepared design",
        ));
    }
    validate_active(active, matrix.ncols())?;
    let sum = input.iter().sum();
    output.fill(0.0);
    for &j in active {
        output[j] = finite(
            matrix.column(j).dot_with_sum(input, sum),
            "computing a selected transposed design product",
            0,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    mod matrix {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/matrix.rs"
        ));
    }

    use super::*;
    use crate::{Gaussian, L1, MatrixDesign, Normalization, Problem, RuntimeDesign};

    fn fixture() -> matrix::Matrix {
        matrix::Matrix::from_rows(&[
            &[0.0, 2.0, 4.0],
            &[2.0, 0.0, 4.0],
            &[4.0, 3.0, 4.0],
            &[6.0, 1.0, 4.0],
        ])
    }

    fn problem(matrix: &matrix::Matrix) -> Problem<&matrix::Matrix, Gaussian<'static>, L1> {
        Problem::new(matrix, Gaussian::new(&[0.0; 4]), L1::new(0.1))
            .normalize(Normalization::None)
            .with_centers(vec![1.0, 1.5, 4.0])
            .with_scales(vec![2.0, 0.5, 1.0])
    }

    #[test]
    fn selected_products_preserve_normalization_and_skip_excluded_columns() {
        let matrix = fixture();
        let problem = problem(&matrix);
        let mut design =
            NativeDesign::<_, Vec<f64>>::new_screenable(problem.prepare(false).unwrap());
        let mut norms = [0.0; 3];
        design.column_norms(&mut norms).unwrap();
        for (actual, expected) in norms.into_iter().zip([3.0, 20.0_f64.sqrt(), 0.0]) {
            assert!((actual - expected).abs() <= 1e-15 * expected);
        }
        let mut error_scales = [0.0; 3];
        design.column_error_scales(&mut error_scales).unwrap();
        assert_eq!(error_scales[2], 16.0);
        for (scale, norm) in error_scales.into_iter().zip(norms) {
            assert!(scale >= norm);
        }

        matrix.visits.set(0);
        let mut predictor = [f64::NAN; 4];
        design
            .forward_active(&[2.0, 999.0, -3.0], &[0], &mut predictor)
            .unwrap();
        assert_eq!(predictor, [-1.0, 1.0, 3.0, 5.0]);
        assert_eq!(matrix.visits.get(), 3);

        matrix.visits.set(0);
        let mut gradient = [f64::NAN; 3];
        design
            .transpose_active(&[1.0, 2.0, -1.0, 0.5], &[0, 2], &mut gradient)
            .unwrap();
        assert_eq!(gradient, [0.25, 0.0, 0.0]);
        assert_eq!(matrix.visits.get(), 7);

        let mut full = [0.0; 4];
        design.forward(&[2.0, 0.0, 0.0], &mut full).unwrap();
        assert_eq!(full, predictor);
        let mut full = [0.0; 3];
        design.transpose(&[1.0, 2.0, -1.0, 0.5], &mut full).unwrap();
        assert_eq!(full, [0.25, -8.5, 0.0]);
    }

    #[test]
    fn empty_selection_does_no_column_work_and_validates_indices() {
        let matrix = fixture();
        let problem = problem(&matrix);
        let mut design =
            NativeDesign::<_, Vec<f64>>::new_screenable(problem.prepare(false).unwrap());
        matrix.visits.set(0);
        let mut output = [f64::NAN; 4];
        design.forward_active(&[1.0; 3], &[], &mut output).unwrap();
        assert_eq!(output, [0.0; 4]);
        let mut output = [f64::NAN; 3];
        design
            .transpose_active(&[1.0; 4], &[], &mut output)
            .unwrap();
        assert_eq!(output, [0.0; 3]);
        assert_eq!(matrix.visits.get(), 0);
        for active in [&[0, 0][..], &[2, 0][..], &[3][..]] {
            assert!(
                design
                    .forward_active(&[1.0; 3], active, &mut [0.0; 4])
                    .is_err()
            );
            assert!(
                design
                    .transpose_active(&[1.0; 4], active, &mut [0.0; 3])
                    .is_err()
            );
        }
        assert!(design.forward_active(&[], &[0], &mut [0.0; 4]).is_err());
        assert!(design.transpose_active(&[], &[0], &mut [0.0; 3]).is_err());
        assert!(design.column_norms(&mut []).is_err());
        assert!(design.column_error_scales(&mut []).is_err());

        let empty = matrix::Matrix::empty(4, 0);
        let problem = Problem::new(&empty, Gaussian::new(&[0.0; 4]), L1::new(0.1))
            .normalize(Normalization::None);
        let mut design =
            NativeDesign::<_, Vec<f64>>::new_screenable(problem.prepare(false).unwrap());
        design.column_norms(&mut []).unwrap();
        design.column_error_scales(&mut []).unwrap();
        design.forward_active(&[], &[], &mut [0.0; 4]).unwrap();
        design.transpose_active(&[1.0; 4], &[], &mut []).unwrap();
    }

    #[test]
    fn native_screening_is_optional_and_runtime_adapter_preserves_it() {
        let matrix = fixture();
        let problem = problem(&matrix);
        let mut product_only = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
        assert!(product_only.screening_design().is_none());
        let mut native =
            NativeDesign::<_, Vec<f64>>::new_screenable(problem.prepare(false).unwrap());
        assert!(native.screening_design().is_some());

        let runtime = MatrixDesign::<_, Vec<f64>>::new(&matrix);
        let mut erased = runtime
            .prepare(
                Normalization::None,
                Some(&[1.0, 1.5, 4.0]),
                Some(&[2.0, 0.5, 1.0]),
                false,
            )
            .unwrap();
        let columns = erased.screening_design().unwrap();
        let mut norms = [0.0; 3];
        columns.column_error_scales(&mut norms).unwrap();
        assert_eq!(norms[2], 16.0);
        matrix.visits.set(0);
        let mut output = [0.0; 4];
        columns
            .forward_active(&[2.0, 999.0, -3.0], &[0], &mut output)
            .unwrap();
        assert_eq!(output, [-1.0, 1.0, 3.0, 5.0]);
        assert_eq!(matrix.visits.get(), 3);
    }

    #[test]
    fn column_norms_avoid_intermediate_underflow_and_overflow() {
        let matrix = matrix::Matrix::from_rows(&[&[1e-200, 1e200], &[2e-200, 2e200]]);
        let matrix = LazyMatrix::from_parts(&matrix, None, None);
        let mut norms = [0.0; 2];
        column_norms(&matrix, &mut norms).unwrap();
        for (actual, expected) in norms
            .into_iter()
            .zip([5.0_f64.sqrt() * 1e-200, 5.0_f64.sqrt() * 1e200])
        {
            assert!((actual / expected - 1.0).abs() < 1e-15);
        }

        let matrix = matrix::Matrix::from_rows(&[&[1e308, 1e150], &[1e308, 1e150]]);
        let matrix =
            LazyMatrix::from_parts(&matrix, Some(vec![-1e308, 1e150]), Some(vec![1e308, 1.0]));
        column_norms(&matrix, &mut norms).unwrap();
        assert_eq!(norms, [8.0_f64.sqrt(), 0.0]);
        let mut errors = [0.0; 2];
        column_error_scales(&matrix, &mut errors).unwrap();
        assert!(errors[0] >= norms[0]);
        assert!((errors[1] / (2.0 * 2.0_f64.sqrt() * 1e150) - 1.0).abs() < 1e-15);

        let matrix = matrix::Matrix::from_rows(&[&[1e-100], &[1e-100]]);
        let matrix = LazyMatrix::from_parts(&matrix, None, Some(vec![1e308]));
        column_norms(&matrix, &mut norms[..1]).unwrap();
        assert!(norms[0] > 0.0);
    }

    #[test]
    fn noncanonical_columns_receive_conservative_norm_bounds() {
        struct Matrix(Vec<(usize, f64)>);
        impl MatrixShape for Matrix {
            fn nrows(&self) -> usize {
                4
            }
            fn ncols(&self) -> usize {
                1
            }
        }
        struct Column<'a>(&'a [(usize, f64)]);
        impl RawColumn<f64> for Column<'_> {
            fn len(&self) -> usize {
                4
            }
            fn stored_len(&self) -> usize {
                self.0.len()
            }
            fn for_each_stored(&self, mut visit: impl FnMut(usize, f64)) {
                for &(row, value) in self.0 {
                    visit(row, value);
                }
            }
        }
        impl RawColumns<f64> for Matrix {
            type Column<'a> = Column<'a>;
            fn raw_column(&self, _: usize) -> Self::Column<'_> {
                Column(&self.0)
            }
        }

        for entries in [
            vec![(0, 2.0), (0, 2.0), (2, -1.0)],
            vec![(2, -1.0), (0, 4.0)],
        ] {
            let matrix = LazyMatrix::from_parts(Matrix(entries), Some(vec![0.25]), Some(vec![2.0]));
            let mut norm = [0.0];
            let mut errors = [0.0];
            column_norms(&matrix, &mut norm).unwrap();
            column_error_scales(&matrix, &mut errors).unwrap();
            let mut expanded = [0.0; 4];
            forward_active(&matrix, &[1.0], &[0], &mut expanded).unwrap();
            let actual = expanded.into_iter().fold(0.0_f64, f64::hypot);
            assert!(norm[0] >= actual);
            assert!(errors[0] >= actual);
            assert_eq!(norm[0], 2.75);
        }
    }
}

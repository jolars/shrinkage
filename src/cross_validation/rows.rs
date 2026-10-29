use lazymatrix::{ColumnStats, RawColumns};

use crate::FitError;

/// Select raw observations before fitting training-fold preprocessing.
///
/// Selected designs retain column order and use the same backend error type.
/// Implementations must neither estimate nor apply normalization. The built-in
/// dense and CSC integrations copy only selected entries into owned storage,
/// then delegate statistics and products to LazyMatrix's backend integrations.
/// Custom backends can supply their own raw selection implementation.
pub trait RowSubset: RawColumns<f64> + ColumnStats<f64> {
    /// Selected raw design used by the fold's ordinary path solver.
    type Subset: RawColumns<f64> + ColumnStats<f64, Error = Self::Error>;

    /// Select strictly increasing original row indices without changing columns.
    ///
    /// # Errors
    /// Reject repeated, unsorted, or out-of-bounds indices and backend failures.
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError<Self::Error>>;
}

impl<M: RowSubset + ?Sized> RowSubset for &M {
    type Subset = M::Subset;
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError<Self::Error>> {
        (**self).select_rows(rows)
    }
}

#[cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17",
    feature = "sprs_v0_11"
))]
fn validate<E>(rows: &[usize], nrows: usize) -> Result<(), FitError<E>> {
    if rows.last().is_some_and(|&row| row >= nrows)
        || rows.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(crate::error::invalid(
            "selected rows must be strictly increasing and in bounds",
        ));
    }
    Ok(())
}

#[cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17"
))]
fn dense<M, O>(matrix: &M, rows: &[usize]) -> Result<O, FitError<M::Error>>
where
    M: lazymatrix::MatrixRead<f64> + lazymatrix::MatrixErrorType,
    O: lazymatrix::MatrixOwned<f64>,
{
    validate(rows, matrix.nrows())?;
    let mut selected = O::zeros(rows.len(), matrix.ncols());
    for j in 0..matrix.ncols() {
        for (i, &row) in rows.iter().enumerate() {
            selected.set(i, j, matrix.get(row, j));
        }
    }
    Ok(selected)
}

#[cfg(feature = "faer_v0_24")]
impl RowSubset for faer::Mat<f64> {
    type Subset = Self;
    fn select_rows(&self, rows: &[usize]) -> Result<Self, FitError> {
        dense(self, rows)
    }
}

#[cfg(feature = "faer_v0_24")]
impl RowSubset for faer::MatRef<'_, f64> {
    type Subset = faer::Mat<f64>;
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError> {
        dense(self, rows)
    }
}

#[cfg(feature = "nalgebra_v0_34")]
impl<R, C, S> RowSubset for nalgebra::Matrix<f64, R, C, S>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<f64, R, C>,
{
    type Subset = nalgebra::DMatrix<f64>;
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError> {
        dense(self, rows)
    }
}

#[cfg(feature = "ndarray_v0_17")]
impl<S: ndarray::Data<Elem = f64>> RowSubset for ndarray::ArrayBase<S, ndarray::Ix2> {
    type Subset = ndarray::Array2<f64>;
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError> {
        dense(self, rows)
    }
}

#[cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "sprs_v0_11"
))]
struct SelectedCsc {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

#[cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "sprs_v0_11"
))]
fn sparse<M: RawColumns<f64> + lazymatrix::MatrixErrorType>(
    matrix: &M,
    rows: &[usize],
) -> Result<SelectedCsc, FitError<M::Error>> {
    use lazymatrix::RawColumn;
    validate(rows, matrix.nrows())?;
    let mut offsets = Vec::with_capacity(matrix.ncols() + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for j in 0..matrix.ncols() {
        offsets.push(indices.len());
        matrix.raw_column(j).for_each_stored(|row, value| {
            if let Ok(index) = rows.binary_search(&row) {
                indices.push(index);
                values.push(value);
            }
        });
    }
    offsets.push(indices.len());
    Ok(SelectedCsc {
        offsets,
        indices,
        values,
    })
}

#[cfg(feature = "faer_v0_24")]
impl RowSubset for faer::sparse::SparseColMat<usize, f64> {
    type Subset = Self;
    fn select_rows(&self, rows: &[usize]) -> Result<Self, FitError> {
        faer_sparse(self, rows)
    }
}

#[cfg(feature = "faer_v0_24")]
fn faer_sparse<
    M: RawColumns<f64> + lazymatrix::MatrixErrorType<Error = std::convert::Infallible>,
>(
    matrix: &M,
    rows: &[usize],
) -> Result<faer::sparse::SparseColMat<usize, f64>, FitError> {
    let SelectedCsc {
        offsets,
        indices,
        values,
    } = sparse(matrix, rows)?;
    let mut triplets = Vec::with_capacity(values.len());
    for j in 0..matrix.ncols() {
        for k in offsets[j]..offsets[j + 1] {
            triplets.push(faer::sparse::Triplet::new(indices[k], j, values[k]));
        }
    }
    faer::sparse::SparseColMat::try_new_from_triplets(rows.len(), matrix.ncols(), &triplets)
        .map_err(|error| crate::error::invalid(format!("selecting sparse rows failed: {error:?}")))
}

#[cfg(feature = "nalgebra_v0_34")]
impl RowSubset for nalgebra_sparse::CscMatrix<f64> {
    type Subset = Self;
    fn select_rows(&self, rows: &[usize]) -> Result<Self, FitError> {
        let SelectedCsc {
            offsets,
            indices,
            values,
        } = sparse(self, rows)?;
        Self::try_from_csc_data(rows.len(), self.ncols(), offsets, indices, values).map_err(
            |error| crate::error::invalid(format!("selecting sparse rows failed: {error}")),
        )
    }
}

#[cfg(feature = "sprs_v0_11")]
impl<M> RowSubset for lazymatrix::SprsCsc<M>
where
    Self: RawColumns<f64> + ColumnStats<f64, Error = std::convert::Infallible>,
{
    type Subset = lazymatrix::SprsCsc<sprs::CsMat<f64>>;
    fn select_rows(&self, rows: &[usize]) -> Result<Self::Subset, FitError> {
        use lazymatrix::MatrixShape;
        let SelectedCsc {
            offsets,
            indices,
            values,
        } = sparse(self, rows)?;
        let matrix = sprs::CsMat::new_csc((rows.len(), self.ncols()), offsets, indices, values);
        Ok(lazymatrix::SprsCsc::try_new(matrix).unwrap())
    }
}

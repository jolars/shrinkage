use std::cell::Cell;
use std::convert::Infallible;

use shrinkage::lazymatrix::{ColumnStats, MatrixErrorType, MatrixShape, RawColumn, RawColumns};

/// Small CSC oracle that also works without a matrix backend feature.
pub struct Matrix {
    nrows: usize,
    columns: Vec<Vec<(usize, f64)>>,
    pub visits: Cell<usize>,
}

impl Matrix {
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let ncols = rows.first().map_or(0, |row| row.len());
        Self {
            nrows: rows.len(),
            columns: (0..ncols)
                .map(|j| {
                    rows.iter()
                        .enumerate()
                        .filter_map(|(i, row)| (row[j] != 0.0).then_some((i, row[j])))
                        .collect()
                })
                .collect(),
            visits: Cell::new(0),
        }
    }

    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            columns: vec![Vec::new(); ncols],
            visits: Cell::new(0),
        }
    }
}

pub struct Column<'a> {
    matrix: &'a Matrix,
    index: usize,
}

impl RawColumn<f64> for Column<'_> {
    fn len(&self) -> usize {
        self.matrix.nrows
    }

    fn stored_len(&self) -> usize {
        self.matrix.columns[self.index].len()
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, f64)) {
        for &(i, value) in &self.matrix.columns[self.index] {
            self.matrix.visits.set(self.matrix.visits.get() + 1);
            f(i, value);
        }
    }
}

impl MatrixShape for Matrix {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.columns.len()
    }
}

impl RawColumns<f64> for Matrix {
    type Column<'a> = Column<'a>;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        Column {
            matrix: self,
            index: j,
        }
    }
}

macro_rules! unused_stats {
    ($($name:ident $(($arg:ident))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: &[f64])?) -> Result<Vec<f64>, Self::Error> {
            $(let _ = $arg;)?
            panic!("Gaussian lasso does not request this statistic")
        })*
    };
}

impl MatrixErrorType for Matrix {
    type Error = Infallible;
}

impl ColumnStats<f64> for Matrix {
    fn col_means(&self) -> Result<Vec<f64>, Self::Error> {
        Ok(self
            .columns
            .iter()
            .map(|column| column.iter().map(|&(_, value)| value).sum::<f64>() / self.nrows as f64)
            .collect())
    }

    fn col_sds(&self) -> Result<Vec<f64>, Self::Error> {
        Ok(self
            .columns
            .iter()
            .zip(self.col_means()?)
            .map(|(column, mean)| {
                let stored = column
                    .iter()
                    .map(|&(_, value)| (value - mean).powi(2))
                    .sum::<f64>();
                ((stored + (self.nrows - column.len()) as f64 * mean * mean) / self.nrows as f64)
                    .sqrt()
            })
            .collect())
    }

    unused_stats!(
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l2_centered(centers),
        col_l1_centered(centers),
        col_maxabs_centered(centers)
    );
}

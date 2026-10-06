use std::cell::Cell;

use lazymatrix::MatrixShape;

use super::*;

struct Sparse {
    nrows: usize,
    columns: Vec<Vec<(usize, f64)>>,
    visits: Cell<usize>,
}

struct Column<'a> {
    matrix: &'a Sparse,
    j: usize,
}

impl MatrixShape for Sparse {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
}

impl RawColumns<f64> for Sparse {
    type Column<'a> = Column<'a>;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        Column { matrix: self, j }
    }
}

impl RawColumn<f64> for Column<'_> {
    fn len(&self) -> usize {
        self.matrix.nrows
    }
    fn stored_len(&self) -> usize {
        self.matrix.columns[self.j].len()
    }
    fn for_each_stored(&self, mut f: impl FnMut(usize, f64)) {
        for &(i, value) in &self.matrix.columns[self.j] {
            self.matrix.visits.set(self.matrix.visits.get() + 1);
            f(i, value);
        }
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn coordinate_update_and_cached_dot_visit_only_stored_entries() {
    for nrows in [10, 100_000] {
        let raw = Sparse {
            nrows,
            columns: vec![vec![(1, 2.0), (5, -3.0), (8, 0.0)]],
            visits: Cell::new(0),
        };
        let matrix = LazyMatrix::from_parts(&raw, Some(vec![0.25]), Some(vec![2.0]));
        let column = matrix.column(0);
        let summary = ColumnSummary::new(&column).unwrap();
        let mut residual = Residual::new(&vec![1.0; nrows], 0.0).unwrap();
        raw.visits.set(0);
        residual.subtract_column(&column, &summary, 0.5, 1).unwrap();
        assert_eq!(raw.visits.get(), 3);
        close(residual.base[0], 1.0);
        close(residual.base[1], 0.5);
        close(residual.base[5], 1.75);
        close(residual.offset, 0.0625);
        raw.visits.set(0);
        let dot = residual.dot(&column, &summary, 1).unwrap();
        assert_eq!(raw.visits.get(), 3);
        let expected: f64 = (0..nrows)
            .map(|i| {
                let raw = match i {
                    1 => 2.0,
                    5 => -3.0,
                    _ => 0.0,
                };
                let x = (raw - 0.25) / 2.0;
                x * (1.0 - 0.5 * x)
            })
            .sum();
        close(dot, expected);
    }
}

#[test]
fn repeated_updates_and_refresh_match_materialized_residuals() {
    let raw = Sparse {
        nrows: 5,
        columns: vec![vec![(0, 2.0), (3, -1.0)], vec![(1, 4.0), (4, 3.0)]],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, Some(vec![0.2, 1.4]), Some(vec![0.8, 2.0]));
    let summaries: Vec<_> = (0..2)
        .map(|j| ColumnSummary::new(&matrix.column(j)).unwrap())
        .collect();
    let explicit = [
        [2.25, -0.7],
        [-0.25, 1.3],
        [-0.25, -0.7],
        [-1.5, -0.7],
        [-0.25, 0.8],
    ];
    let y = [2.0, -1.0, 4.0, 0.0, 3.0];
    let mut coefficients = [0.0; 2];
    let mut intercept = 0.5;
    let mut residual = Residual::new(&y, intercept).unwrap();
    for iteration in 1..=250 {
        let j = iteration % 2;
        let delta = if iteration % 3 == 0 { -0.023 } else { 0.017 };
        coefficients[j] += delta;
        residual
            .subtract_column(&matrix.column(j), &summaries[j], delta, iteration)
            .unwrap();
        intercept += 0.003;
        residual.offset -= 0.003;
        if iteration % REFRESH_INTERVAL == 0 {
            residual
                .refresh(&matrix, &y, &coefficients, intercept, &summaries, iteration)
                .unwrap();
        }
        let reference: Vec<_> = explicit
            .iter()
            .zip(y)
            .map(|(row, y)| y - intercept - row[0] * coefficients[0] - row[1] * coefficients[1])
            .collect();
        for (&base, &expected) in residual.base.iter().zip(&reference) {
            close(base + residual.offset, expected);
        }
        close(residual.base_sum, residual.base.iter().sum());
        close(
            residual.mean(iteration).unwrap(),
            reference.iter().sum::<f64>() / 5.0,
        );
        close(
            residual.loss(iteration).unwrap(),
            reference.iter().map(|r| r * r).sum::<f64>() / 10.0,
        );
        for j in 0..2 {
            close(
                residual
                    .dot(&matrix.column(j), &summaries[j], iteration)
                    .unwrap(),
                explicit
                    .iter()
                    .zip(&reference)
                    .map(|(row, r)| row[j] * r)
                    .sum(),
            );
        }
    }
    // A refresh must discard accumulated cache drift, including drift in the
    // scalar offset, rather than only resumming an already stale residual.
    residual.base_sum += 1.0;
    residual.offset += 0.1;
    residual.base[0] += 0.2;
    residual
        .refresh(&matrix, &y, &coefficients, intercept, &summaries, 250)
        .unwrap();
    for (i, row) in explicit.iter().enumerate() {
        close(
            residual.base[i] + residual.offset,
            y[i] - intercept - row[0] * coefficients[0] - row[1] * coefficients[1],
        );
    }
}

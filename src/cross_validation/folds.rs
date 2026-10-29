use crate::FitError;
use crate::error::invalid;

/// Validated, zero-based held-out fold identifiers in original row order.
///
/// Every identifier from zero through `nfolds - 1` must occur. At least two
/// nonempty folds ensure that every row is held out once and trained on in
/// every other fold. Seeded constructors use a fixed SplitMix64 shuffle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldAssignments {
    assignments: Vec<usize>,
    sizes: Vec<usize>,
}

impl FoldAssignments {
    /// Validate user-supplied fold identifiers without reordering rows.
    ///
    /// # Errors
    /// Reject empty input, fewer than two folds, gaps, or out-of-range labels.
    pub fn new(assignments: Vec<usize>) -> Result<Self, FitError> {
        let maximum = assignments.iter().copied().max().unwrap_or(0);
        if maximum == 0 || maximum >= assignments.len() {
            return Err(invalid(
                "fold labels must cover at least two nonempty folds, numbered consecutively from zero",
            ));
        }
        let mut sizes = vec![0; maximum + 1];
        for &fold in &assignments {
            sizes[fold] += 1;
        }
        if sizes.contains(&0) {
            return Err(invalid("every fold label must have held-out observations"));
        }
        Ok(Self { assignments, sizes })
    }

    /// Shuffle observations into folds whose sizes differ by at most one.
    ///
    /// The same arguments produce the same assignments across platforms.
    ///
    /// # Errors
    /// Require `2 <= nfolds <= nobs`.
    pub fn kfold(nobs: usize, nfolds: usize, seed: u64) -> Result<Self, FitError> {
        validate_count(nobs, nfolds)?;
        let mut rows = (0..nobs).collect::<Vec<_>>();
        Shuffle(seed).shuffle(&mut rows);
        let mut assignments = vec![0; nobs];
        for (position, row) in rows.into_iter().enumerate() {
            assignments[row] = position % nfolds;
        }
        Self::new(assignments)
    }

    /// Shuffle binary responses separately, balancing each class across folds.
    ///
    /// Total fold sizes also differ by at most one. Fractional responses need
    /// user-supplied or ordinary shuffled folds. A rare class may still leave a
    /// training fold with a boundary response; fitting validates that separately.
    ///
    /// # Errors
    /// Require binary responses and `2 <= nfolds <= response.len()`.
    pub fn stratified_binomial(
        response: &[f64],
        nfolds: usize,
        seed: u64,
    ) -> Result<Self, FitError> {
        validate_count(response.len(), nfolds)?;
        let mut classes = [Vec::new(), Vec::new()];
        for (row, &y) in response.iter().enumerate() {
            if y != 0.0 && y != 1.0 {
                return Err(invalid(
                    "binomial stratification requires responses equal to zero or one",
                ));
            }
            classes[usize::from(y == 1.0)].push(row);
        }
        let mut shuffle = Shuffle(seed);
        let mut assignments = vec![0; response.len()];
        let mut position = 0;
        for rows in &mut classes {
            shuffle.shuffle(rows);
            for &row in rows.iter() {
                assignments[row] = position % nfolds;
                position += 1;
            }
        }
        Self::new(assignments)
    }

    /// Held-out fold for each observation in original row order.
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }

    /// Number of nonempty validation folds.
    pub fn nfolds(&self) -> usize {
        self.sizes.len()
    }

    /// Validation observation counts in fold order.
    pub fn fold_sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub(super) fn rows(&self, fold: usize) -> (Vec<usize>, Vec<usize>) {
        (0..self.assignments.len()).partition(|&row| self.assignments[row] != fold)
    }
}

fn validate_count(nobs: usize, nfolds: usize) -> Result<(), FitError> {
    if nfolds < 2 || nfolds > nobs {
        return Err(invalid(
            "fold count must lie between two and the observation count",
        ));
    }
    Ok(())
}

struct Shuffle(u64);

impl Shuffle {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn shuffle(&mut self, rows: &mut [usize]) {
        for end in (1..rows.len()).rev() {
            let bound = end as u64 + 1;
            let threshold = bound.wrapping_neg() % bound;
            let position = loop {
                let value = self.next();
                // Rejection keeps modulo reduction unbiased for every bound.
                if value >= threshold {
                    break (value % bound) as usize;
                }
            };
            rows.swap(end, position);
        }
    }
}

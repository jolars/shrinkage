include!("matrix.rs");

impl shrinkage::RowSubset for Matrix {
    type Subset = Self;

    fn select_rows(&self, rows: &[usize]) -> Result<Self, shrinkage::FitError> {
        if rows.last().is_some_and(|&row| row >= self.nrows)
            || rows.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(shrinkage::FitError::InvalidInput {
                message: "invalid rows".into(),
            });
        }
        Ok(Self {
            nrows: rows.len(),
            columns: self
                .columns
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .filter_map(|&(row, value)| {
                            rows.binary_search(&row).ok().map(|i| (i, value))
                        })
                        .collect()
                })
                .collect(),
            visits: Cell::new(0),
        })
    }
}

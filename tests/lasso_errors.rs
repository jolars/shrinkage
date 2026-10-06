//! Backend error propagation and preprocessing validation before construction.

use std::cell::Cell;
use std::error::Error;
use std::fmt;

use shrinkage::lazymatrix::{
    ColumnStats, MatrixErrorType, MatrixShape, Normalization as MatrixNormalization,
    NormalizationStats, RawColumn, RawColumns,
};
use shrinkage::{Centering, Lasso, LassoError, Normalization, Scaling};

#[derive(Debug, PartialEq)]
struct ReadFailure;

impl fmt::Display for ReadFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected statistics failure")
    }
}
impl Error for ReadFailure {}

struct Source {
    values: [f64; 2],
    calls: Cell<usize>,
    statistics: Result<NormalizationStats<f64>, ReadFailure>,
}

impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        2
    }
    fn ncols(&self) -> usize {
        1
    }
}

struct Column<'a>(&'a [f64; 2]);
impl RawColumn<f64> for Column<'_> {
    fn len(&self) -> usize {
        2
    }
    fn stored_len(&self) -> usize {
        2
    }
    fn for_each_stored(&self, mut f: impl FnMut(usize, f64)) {
        for (i, &value) in self.0.iter().enumerate() {
            f(i, value);
        }
    }
}
impl RawColumns<f64> for Source {
    type Column<'a> = Column<'a>;
    fn raw_column(&self, _: usize) -> Self::Column<'_> {
        Column(&self.values)
    }
}
impl MatrixErrorType for Source {
    type Error = ReadFailure;
}

macro_rules! unused_stats {
    ($($name:ident $(($arg:ident))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: &[f64])?) -> Result<Vec<f64>, Self::Error> {
            $(let _ = $arg;)?
            panic!("fitting must use the combined normalization_stats hook")
        })*
    };
}

impl ColumnStats<f64> for Source {
    fn normalization_stats(
        &self,
        _: MatrixNormalization,
    ) -> Result<NormalizationStats<f64>, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        self.statistics.as_ref().cloned().map_err(|_| ReadFailure)
    }
    fn col_l2_centered(&self, centers: &[f64]) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![
            self.values
                .iter()
                .map(|value| (value - centers[0]).powi(2))
                .sum::<f64>()
                .sqrt(),
        ])
    }
    fn col_l1_centered(&self, centers: &[f64]) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![
            self.values
                .iter()
                .map(|value| (value - centers[0]).abs())
                .sum(),
        ])
    }
    fn col_maxabs_centered(&self, centers: &[f64]) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![
            self.values
                .iter()
                .map(|value| (value - centers[0]).abs())
                .fold(0.0, f64::max),
        ])
    }
    unused_stats!(
        col_means, col_sds, col_mins, col_ranges, col_maxabs, col_l1, col_l2
    );
}

#[test]
fn supplied_centers_set_the_basis_for_computed_norm_scales() {
    for (scale, expected) in [
        (Scaling::L1, 1.0),
        (Scaling::L2, 0.5_f64.sqrt()),
        (Scaling::MaxAbs, 0.5),
    ] {
        let source = Source {
            values: [1.0, 2.0],
            calls: Cell::new(0),
            statistics: Err(ReadFailure),
        };
        let fit = Lasso::new(0.0)
            .fit_intercept(false)
            .normalize(Normalization::Custom {
                center: Centering::None,
                scale,
            })
            .with_centers(vec![1.5])
            .fit(&source, &[-0.5, 0.5])
            .unwrap();
        assert_eq!(source.calls.get(), 0);
        assert_eq!(fit.preprocessing().centers(), Some([1.5].as_slice()));
        assert!((fit.preprocessing().scales().unwrap()[0] - expected).abs() < 1e-12);
        assert!((fit.coefficients()[0] - 1.0).abs() < 1e-12);
        assert!((fit.intercept() + 1.5).abs() < 1e-12);
    }
    let constant = Source {
        values: [1.0, 1.0],
        calls: Cell::new(0),
        statistics: Err(ReadFailure),
    };
    let fit = Lasso::new(0.0)
        .fit_intercept(false)
        .normalize(Normalization::L2)
        .with_centers(vec![1.0])
        .fit(&constant, &[1.0, 2.0])
        .unwrap();
    assert_eq!(fit.preprocessing().scales(), Some([1.0].as_slice()));
    assert_eq!(fit.coefficients(), &[0.0]);
}

#[test]
fn preprocessing_failure_retains_source_and_context() {
    let source = Source {
        values: [1.0, 2.0],
        calls: Cell::new(0),
        statistics: Err(ReadFailure),
    };
    let error = Lasso::new(0.1).fit(&source, &[1.0, 2.0]).unwrap_err();
    assert_eq!(source.calls.get(), 1);
    assert!(
        error
            .source()
            .unwrap()
            .downcast_ref::<ReadFailure>()
            .is_some()
    );
    assert!(matches!(
        error,
        LassoError::Backend {
            operation: "computing training normalization statistics",
            source: ReadFailure
        }
    ));
    for normalization in [
        Normalization::None,
        Normalization::Custom {
            center: Centering::None,
            scale: Scaling::None,
        },
    ] {
        let fit = Lasso::new(0.1)
            .normalize(normalization)
            .fit(&source, &[1.0, 2.0])
            .unwrap();
        assert!(fit.objective().is_finite());
    }
    assert_eq!(source.calls.get(), 1);
}

#[test]
fn centering_and_scaling_can_be_requested_independently() {
    for (normalization, statistics) in [
        (Normalization::Center, (Some(vec![1.5]), None)),
        (Normalization::MaxAbs, (None, Some(vec![2.0]))),
    ] {
        let source = Source {
            values: [1.0, 2.0],
            calls: Cell::new(0),
            statistics: Ok(statistics.clone()),
        };
        let fit = Lasso::new(0.1)
            .normalize(normalization)
            .fit_intercept(false)
            .fit(&source, &[1.0, 2.0])
            .unwrap();
        assert_eq!(source.calls.get(), 1);
        assert_eq!(fit.preprocessing().centers(), statistics.0.as_deref());
        assert_eq!(fit.preprocessing().scales(), statistics.1.as_deref());
    }
}

#[test]
fn nonfinite_data_is_rejected_before_requesting_statistics() {
    let source = Source {
        values: [f64::NAN, 2.0],
        calls: Cell::new(0),
        statistics: Err(ReadFailure),
    };
    assert!(matches!(
        Lasso::new(0.1).fit(&source, &[1.0, 2.0]),
        Err(LassoError::InvalidInput { .. })
    ));
    assert_eq!(source.calls.get(), 0);
}

#[test]
fn invalid_backend_statistics_cannot_panic_in_lazy_construction() {
    let cases = [
        (None, Some(vec![1.0])),
        (Some(vec![]), Some(vec![1.0])),
        (Some(vec![1.5]), None),
        (Some(vec![1.5]), Some(vec![1.0, 2.0])),
        (Some(vec![1.5]), Some(vec![-1.0])),
        (Some(vec![f64::NAN]), Some(vec![1.0])),
        (Some(vec![1.5]), Some(vec![f64::INFINITY])),
    ];
    for statistics in cases {
        let source = Source {
            values: [1.0, 2.0],
            calls: Cell::new(0),
            statistics: Ok(statistics),
        };
        assert!(Lasso::new(0.1).fit(&source, &[1.0, 2.0]).is_err());
    }
}

#[test]
fn invalid_supplied_vectors_are_rejected_before_backend_statistics() {
    let cases = [
        (Some(vec![]), None),
        (Some(vec![f64::NAN]), None),
        (Some(vec![f64::INFINITY]), None),
        (None, Some(vec![])),
        (None, Some(vec![0.0])),
        (None, Some(vec![-0.0])),
        (None, Some(vec![-1.0])),
        (None, Some(vec![f64::NAN])),
        (None, Some(vec![f64::INFINITY])),
    ];
    for (centers, scales) in cases {
        let source = Source {
            values: [1.0, 2.0],
            calls: Cell::new(0),
            statistics: Err(ReadFailure),
        };
        let mut model = Lasso::new(0.1);
        if let Some(centers) = centers {
            model = model.with_centers(centers);
        }
        if let Some(scales) = scales {
            model = model.with_scales(scales);
        }
        assert!(matches!(
            model.fit(&source, &[1.0, 2.0]),
            Err(LassoError::InvalidInput { .. })
        ));
        assert_eq!(source.calls.get(), 0);
    }
}

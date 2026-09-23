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
    unused_stats!(
        col_means,
        col_sds,
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

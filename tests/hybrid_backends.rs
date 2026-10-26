//! Dense native products and lazy preprocessing agree across enabled backends.

#![cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17"
))]

#[path = "common/matrix.rs"]
mod matrix;

#[cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17"
))]
mod dense {
    use shrinkage::{ClusteredLasso, Gaussian, Hybrid, HybridFit, Problem, Termination};

    pub const ROWS: [[f64; 3]; 5] = [
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
        [1.0, -1.0, 1.0],
        [3.0, 0.5, -1.0],
    ];
    pub const Y: [f64; 5] = [1.0, -2.0, 3.0, 0.7, 2.0];

    pub fn compare(fit: HybridFit) {
        let x = super::matrix::Matrix::from_rows(
            &ROWS.iter().map(|r| r.as_slice()).collect::<Vec<_>>(),
        );
        let reference = Problem::new(&x, Gaussian::new(&Y), ClusteredLasso::new(0.03, 0.02))
            .fit_with(&Hybrid::<Vec<f64>>::new().tolerance(1e-10))
            .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert!((fit.objective() - reference.objective()).abs() < 1e-10);
        for (&actual, &expected) in fit.coefficients().iter().zip(reference.coefficients()) {
            assert!((actual - expected).abs() < 1e-8);
        }
        assert!((fit.intercept() - reference.intercept()).abs() < 1e-8);
        assert!(
            fit.predict(&super::matrix::Matrix::empty(0, 3))
                .unwrap()
                .is_empty()
        );
    }
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn faer_dense() {
    use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem};
    let x = faer::Mat::from_fn(5, 3, |i, j| dense::ROWS[i][j]);
    dense::compare(
        Problem::new(
            &x,
            Gaussian::new(&dense::Y),
            ClusteredLasso::new(0.03, 0.02),
        )
        .fit_with(&Hybrid::<faer::Col<f64>>::new().tolerance(1e-10))
        .unwrap(),
    );
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn nalgebra_dense() {
    use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem};
    let x = nalgebra::DMatrix::from_fn(5, 3, |i, j| dense::ROWS[i][j]);
    dense::compare(
        Problem::new(
            &x,
            Gaussian::new(&dense::Y),
            ClusteredLasso::new(0.03, 0.02),
        )
        .fit_with(&Hybrid::<nalgebra::DVector<f64>>::new().tolerance(1e-10))
        .unwrap(),
    );
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_strided_view() {
    use shrinkage::{ClusteredLasso, Gaussian, Hybrid, Problem};
    let storage = ndarray::Array2::from_shape_fn((10, 6), |(i, j)| dense::ROWS[i / 2][j / 2]);
    let rows = storage.slice_axis(ndarray::Axis(0), ndarray::Slice::new(0, None, 2));
    let x = rows.slice_axis(ndarray::Axis(1), ndarray::Slice::new(0, None, 2));
    dense::compare(
        Problem::new(
            &x,
            Gaussian::new(&dense::Y),
            ClusteredLasso::new(0.03, 0.02),
        )
        .fit_with(&Hybrid::<ndarray::Array1<f64>>::new().tolerance(1e-10))
        .unwrap(),
    );
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn every_normalization_preset_agrees_with_proximal_gradient() {
    use shrinkage::{
        ClusteredLasso, Gaussian, Hybrid, Normalization, Problem, ProximalGradient, Termination,
    };
    let x = faer::Mat::from_fn(5, 3, |i, j| dense::ROWS[i][j]);
    for free_intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Center,
            Normalization::Standardize,
            Normalization::MinMax,
            Normalization::MaxAbs,
            Normalization::L1,
            Normalization::L2,
        ] {
            let problem = Problem::new(
                &x,
                Gaussian::new(&dense::Y).fit_intercept(free_intercept),
                ClusteredLasso::new(0.03, 0.02),
            )
            .normalize(normalization);
            let hybrid = problem
                .fit_with(&Hybrid::<faer::Col<f64>>::new().tolerance(1e-9))
                .unwrap();
            let pg = problem
                .fit_with(&ProximalGradient::<faer::Col<f64>>::new().tolerance(1e-9))
                .unwrap();
            assert_eq!(hybrid.termination(), Termination::Converged);
            assert_eq!(pg.termination(), Termination::Converged);
            assert!((hybrid.objective() - pg.objective()).abs() < 1e-9);
            for (actual, expected) in hybrid
                .predict(&x)
                .unwrap()
                .iter()
                .zip(pg.predict(&x).unwrap())
            {
                assert!((actual - expected).abs() < 1e-7);
            }
        }
    }
}

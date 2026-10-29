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

    pub fn verify_paths<M, V>(x: &M)
    where
        M: shrinkage::lazymatrix::RawColumns<f64>
            + shrinkage::lazymatrix::ColumnStats<f64>
            + shrinkage::lazymatrix::MatVecInto<V>
            + shrinkage::lazymatrix::MatTransposeVecInto<V>,
        V: shrinkage::ProximalVector,
    {
        use shrinkage::{Binomial, ChainFusedLasso, PathOptions, Slope};
        let options = PathOptions::new()
            .nlambda(3)
            .lambda_min_ratio(0.1)
            .early_stopping(false);
        let solver = Hybrid::<V>::new().tolerance(1e-9).max_iterations(100_000);
        macro_rules! check {
            ($datafit:expr, $penalty:expr) => {{
                let datafit = $datafit;
                let path = Problem::new(x, datafit, $penalty)
                    .fit_regularization_path_with(&solver, &options)
                    .unwrap();
                for (penalty, fit) in path.iter() {
                    let cold = Problem::new(x, datafit, penalty.clone())
                        .fit_with(&solver)
                        .unwrap();
                    assert_eq!(fit.termination(), Termination::Converged);
                    assert!((fit.objective() - cold.objective()).abs() < 1e-9);
                    for (a, b) in fit.predict(x).unwrap().iter().zip(cold.predict(x).unwrap()) {
                        assert!((a - b).abs() < 1e-6);
                    }
                }
            }};
        }
        check!(Gaussian::new(&Y), ClusteredLasso::new(1.0, 0.2));
        check!(
            Gaussian::new(&Y),
            ChainFusedLasso::new(1.0, 0.2).with_weights(vec![0.0, 2.0])
        );
        check!(Gaussian::new(&Y), Slope::new(vec![1.0, 0.5, 0.2]));
        let y = [0.1, 0.8, 0.3, 0.7, 0.9];
        check!(Binomial::new(&y), ClusteredLasso::new(1.0, 0.2));
        check!(
            Binomial::new(&y),
            ChainFusedLasso::new(1.0, 0.2).with_weights(vec![0.0, 2.0])
        );
        check!(Binomial::new(&y), Slope::new(vec![1.0, 0.5, 0.2]));
    }

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

    pub fn verify_chain<M, V>(x: &M)
    where
        M: shrinkage::lazymatrix::RawColumns<f64>
            + shrinkage::lazymatrix::ColumnStats<f64>
            + shrinkage::lazymatrix::MatVecInto<V>
            + shrinkage::lazymatrix::MatTransposeVecInto<V>,
        V: shrinkage::ProximalVector,
    {
        use shrinkage::{ChainFusedLasso, Normalization, ProximalGradient};
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
                for weights in [vec![0.0, 2.0], vec![0.5, 2.0]] {
                    let problem = Problem::new(
                        x,
                        Gaussian::new(&Y).fit_intercept(free_intercept),
                        ChainFusedLasso::new(0.03, 0.02).with_weights(weights),
                    )
                    .normalize(normalization);
                    let hybrid = problem
                        .fit_with(&Hybrid::<V>::new().tolerance(1e-9))
                        .unwrap();
                    let pg = problem
                        .fit_with(&ProximalGradient::<V>::new().tolerance(1e-9))
                        .unwrap();
                    assert_eq!(hybrid.termination(), Termination::Converged);
                    assert_eq!(pg.termination(), Termination::Converged);
                    assert!((hybrid.objective() - pg.objective()).abs() < 1e-9);
                    for (&actual, expected) in hybrid
                        .predict(x)
                        .unwrap()
                        .iter()
                        .zip(pg.predict(x).unwrap())
                    {
                        assert!((actual - expected).abs() < 1e-7);
                    }
                }
            }
        }
    }

    pub fn verify_slope<M, V>(x: &M)
    where
        M: shrinkage::lazymatrix::RawColumns<f64>
            + shrinkage::lazymatrix::ColumnStats<f64>
            + shrinkage::lazymatrix::MatVecInto<V>
            + shrinkage::lazymatrix::MatTransposeVecInto<V>,
        V: shrinkage::ProximalVector,
    {
        use shrinkage::{Centering, Normalization, ProximalGradient, Scaling, Slope};
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
                Normalization::Custom {
                    center: Centering::Mean,
                    scale: Scaling::L2,
                },
            ] {
                for weights in [vec![0.03, 0.01, 0.0], vec![0.02; 3], vec![0.0; 3]] {
                    let problem = Problem::new(
                        x,
                        Gaussian::new(&Y).fit_intercept(free_intercept),
                        Slope::new(weights),
                    )
                    .normalize(normalization);
                    let hybrid = problem
                        .fit_with(&Hybrid::<V>::new().tolerance(1e-9))
                        .unwrap();
                    let pg = problem
                        .fit_with(&ProximalGradient::<V>::new().tolerance(1e-9))
                        .unwrap();
                    assert_eq!(hybrid.termination(), Termination::Converged);
                    assert_eq!(pg.termination(), Termination::Converged);
                    assert!((hybrid.objective() - pg.objective()).abs() < 1e-9);
                    for (&actual, expected) in hybrid
                        .predict(x)
                        .unwrap()
                        .iter()
                        .zip(pg.predict(x).unwrap())
                    {
                        assert!((actual - expected).abs() < 1e-7);
                    }
                    for (j, &label) in hybrid.cluster_labels().iter().enumerate() {
                        let scale = hybrid
                            .preprocessing()
                            .scales()
                            .map_or(1.0, |scales| scales[j]);
                        assert!(
                            (hybrid.coefficients()[j] * scale
                                - hybrid.coefficient_signs()[j] * hybrid.cluster_values()[label])
                                .abs()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }

    pub fn verify_logistic<M, V>(x: &M)
    where
        M: shrinkage::lazymatrix::RawColumns<f64>
            + shrinkage::lazymatrix::ColumnStats<f64>
            + shrinkage::lazymatrix::MatVecInto<V>
            + shrinkage::lazymatrix::MatTransposeVecInto<V>,
        V: shrinkage::ProximalVector,
    {
        use shrinkage::{Binomial, ChainFusedLasso, Normalization, ProximalGradient, Slope};
        let y = [0.0, 0.8, 1.0, 0.2, 0.9];
        for intercept in [false, true] {
            for normalization in [
                Normalization::None,
                Normalization::Standardize,
                Normalization::MinMax,
            ] {
                macro_rules! compare {
                    ($penalty:expr) => {{
                        let problem =
                            Problem::new(x, Binomial::new(&y).fit_intercept(intercept), $penalty)
                                .normalize(normalization);
                        let fit = problem
                            .fit_with(&Hybrid::<V>::new().tolerance(1e-9))
                            .unwrap();
                        let reference = problem
                            .fit_with(
                                &ProximalGradient::<V>::new()
                                    .tolerance(1e-10)
                                    .max_iterations(100_000),
                            )
                            .unwrap();
                        assert_eq!(fit.termination(), Termination::Converged);
                        assert_eq!(reference.termination(), Termination::Converged);
                        assert!((fit.objective() - reference.objective()).abs() < 1e-9);
                        for (a, b) in fit
                            .predict(x)
                            .unwrap()
                            .into_iter()
                            .zip(reference.predict(x).unwrap())
                        {
                            assert!((a - b).abs() < 2e-6);
                        }
                    }};
                }
                compare!(ClusteredLasso::new(0.03, 0.02));
                compare!(ChainFusedLasso::new(0.03, 0.02).with_weights(vec![0.0, 2.0]));
                compare!(Slope::new(vec![0.04, 0.03, 0.0]));
            }
        }
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
    dense::verify_chain::<_, faer::Col<f64>>(&x);
    dense::verify_slope::<_, faer::Col<f64>>(&x);
    dense::verify_logistic::<_, faer::Col<f64>>(&x);
    dense::verify_paths::<_, faer::Col<f64>>(&x);
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
    dense::verify_chain::<_, nalgebra::DVector<f64>>(&x);
    dense::verify_slope::<_, nalgebra::DVector<f64>>(&x);
    dense::verify_logistic::<_, nalgebra::DVector<f64>>(&x);
    dense::verify_paths::<_, nalgebra::DVector<f64>>(&x);
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
    dense::verify_chain::<_, ndarray::Array1<f64>>(&x);
    dense::verify_slope::<_, ndarray::Array1<f64>>(&x);
    dense::verify_logistic::<_, ndarray::Array1<f64>>(&x);
    dense::verify_paths::<_, ndarray::Array1<f64>>(&x);
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

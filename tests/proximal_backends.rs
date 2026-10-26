//! Native product buffers, sparse designs, and strided backend views.

#![cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17",
    feature = "sprs_v0_11"
))]

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns};
use shrinkage::{
    AxisPenalty, Centering, CoefficientLayout, DisjointGroups, ElasticNet, Gaussian, GroupLasso,
    MatrixDesign, MultivariateGaussian, Normalization, PenaltyAxis, Problem, ProximalGradient,
    ProximalVector, RuntimeMultivariateProblem, RuntimeProblem, Scaling, Termination,
    UncoveredPolicy,
};
use shrinkage::{ErasedPathPenalty, PathOptions, RuntimePathProblem};

const ROWS: [[f64; 3]; 5] = [
    [0.0, 1.0, 5.0],
    [2.0, 0.0, 5.0],
    [4.0, 3.0, 5.0],
    [6.0, 1.0, 5.0],
    [1.0, 0.0, 5.0],
];
const Y: [f64; 5] = [1.0, 2.0, -1.0, 4.0, 2.0];

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7 * (1.0 + b.abs()), "{a} != {b}");
}

fn verify<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: std::error::Error + 'static,
    V: ProximalVector + 'static,
{
    verify_groups::<M, V>(design);
    verify_multivariate::<M, V>(design);
    verify_paths::<M, V>(design);
    verify_chain::<M, V>(design);
    let rows: Vec<_> = ROWS.iter().map(|row| row.as_slice()).collect();
    let reference = matrix::Matrix::from_rows(&rows);
    for intercept in [false, true] {
        for normalization in [
            Normalization::Auto,
            Normalization::None,
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
            for (l1, l2) in [(0.1, 0.0), (0.0, 0.3), (0.1, 0.3)] {
                let datafit = Gaussian::new(&Y).fit_intercept(intercept);
                let penalty = ElasticNet::new(l1, l2);
                let actual = Problem::new(design, datafit, penalty)
                    .normalize(normalization)
                    .fit_with(
                        &ProximalGradient::<V>::new()
                            .tolerance(1e-9)
                            .max_iterations(200_000),
                    )
                    .unwrap();
                let path = ProximalGradient::<V>::new()
                    .tolerance(1e-9)
                    .max_iterations(200_000)
                    .solve_path(
                        &Problem::new(design, datafit, penalty).normalize(normalization),
                        &[ElasticNet::new(l1 + 0.1, l2 + 0.1), penalty],
                    )
                    .unwrap();
                let last = &path[1];
                assert_eq!(last.termination(), actual.termination());
                for (&a, &b) in last.coefficients().iter().zip(actual.coefficients()) {
                    close(a, b);
                }
                close(last.intercept(), actual.intercept());
                close(last.objective(), actual.objective());
                let runtime = RuntimeProblem::new(
                    Box::new(MatrixDesign::<_, V>::new(design)),
                    Box::new(datafit),
                    Box::new(penalty),
                )
                .normalize(normalization)
                .fit_with(
                    &ProximalGradient::new()
                        .tolerance(1e-9)
                        .max_iterations(200_000),
                )
                .unwrap();
                assert_eq!(runtime.coefficients(), actual.coefficients());
                assert_eq!(runtime.intercept(), actual.intercept());
                assert_eq!(runtime.objective(), actual.objective());
                assert_eq!(runtime.termination(), actual.termination());
                assert_eq!(runtime.iterations(), actual.iterations());
                assert_eq!(runtime.stopping_criterion(), actual.stopping_criterion());
                assert_eq!(runtime.stopping_threshold(), actual.stopping_threshold());
                assert_eq!(runtime.stopping_value(), actual.stopping_value());
                assert_eq!(runtime.step_size(), actual.step_size());
                assert_eq!(
                    runtime.preprocessing().centers(),
                    actual.preprocessing().centers()
                );
                assert_eq!(
                    runtime.preprocessing().scales(),
                    actual.preprocessing().scales()
                );
                assert_eq!(
                    runtime.preprocessing().centering(),
                    actual.preprocessing().centering()
                );
                assert_eq!(
                    runtime.preprocessing().scaling(),
                    actual.preprocessing().scaling()
                );
                assert_eq!(
                    runtime.preprocessing().centers_were_supplied(),
                    actual.preprocessing().centers_were_supplied()
                );
                assert_eq!(
                    runtime.preprocessing().scales_were_supplied(),
                    actual.preprocessing().scales_were_supplied()
                );
                assert_eq!(
                    runtime.predict(design).unwrap(),
                    actual.predict(design).unwrap()
                );
                assert_eq!(
                    actual.termination(),
                    Termination::Converged,
                    "intercept={intercept}, normalization={normalization:?}, penalty=({l1}, {l2}), fit={actual:?}"
                );
                assert!(
                    actual
                        .predict(&matrix::Matrix::empty(0, 3))
                        .unwrap()
                        .is_empty()
                );
                // The independent CSC fixture implements only means and standard
                // deviations; every policy still compares typed and runtime fits.
                if matches!(
                    normalization,
                    Normalization::Auto
                        | Normalization::None
                        | Normalization::Center
                        | Normalization::Standardize
                ) {
                    let expected = Problem::new(&reference, datafit, penalty)
                        .normalize(normalization)
                        .fit_with(
                            &ProximalGradient::<Vec<f64>>::new()
                                .tolerance(1e-9)
                                .max_iterations(200_000),
                        )
                        .unwrap();
                    assert_eq!(expected.termination(), Termination::Converged);
                    close(actual.objective(), expected.objective());
                    close(actual.intercept(), expected.intercept());
                    for (a, b) in actual.coefficients().iter().zip(expected.coefficients()) {
                        close(*a, *b);
                    }
                    for (a, b) in actual
                        .predict(design)
                        .unwrap()
                        .iter()
                        .zip(expected.predict(&reference).unwrap())
                    {
                        close(*a, b);
                    }
                }
            }
        }
    }
}

fn verify_chain<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: std::error::Error + 'static,
    V: ProximalVector + 'static,
{
    use shrinkage::ChainFusedLasso;
    let reference =
        matrix::Matrix::from_rows(&ROWS.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
    let penalty = ChainFusedLasso::new(0.03, 0.07).with_weights(vec![0.0, 2.0]);
    for free_intercept in [false, true] {
        let datafit = Gaussian::new(&Y).fit_intercept(free_intercept);
        let fit = Problem::new(design, datafit, penalty.clone())
            .normalize(Normalization::None)
            .fit_with(
                &ProximalGradient::<V>::new()
                    .tolerance(1e-9)
                    .max_iterations(200_000),
            )
            .unwrap();
        let expected = Problem::new(&reference, datafit, penalty.clone())
            .normalize(Normalization::None)
            .fit_with(
                &ProximalGradient::<Vec<f64>>::new()
                    .tolerance(1e-9)
                    .max_iterations(200_000),
            )
            .unwrap();
        let runtime = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, V>::new(design)),
            Box::new(datafit),
            Box::new(penalty.clone()),
        )
        .normalize(Normalization::None)
        .fit_with(
            &ProximalGradient::new()
                .tolerance(1e-9)
                .max_iterations(200_000),
        )
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(expected.termination(), Termination::Converged);
        assert_eq!(runtime.termination(), Termination::Converged);
        close(fit.objective(), expected.objective());
        close(runtime.objective(), expected.objective());
        for (&actual, &expected) in fit.coefficients().iter().zip(expected.coefficients()) {
            close(actual, expected);
        }
    }
}

fn verify_groups<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: std::error::Error + 'static,
    V: ProximalVector + 'static,
{
    let rows: Vec<_> = ROWS.iter().map(|row| row.as_slice()).collect();
    let reference = matrix::Matrix::from_rows(&rows);
    for intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Standardize,
        ] {
            for weight in [0.0, 1.5] {
                let groups = DisjointGroups::new(
                    3,
                    vec![vec![1, 0]],
                    vec![weight],
                    UncoveredPolicy::Unpenalized,
                )
                .unwrap();
                let penalty = GroupLasso::new(0.1, groups);
                let datafit = Gaussian::new(&Y).fit_intercept(intercept);
                let actual = Problem::new(design, datafit, penalty.clone())
                    .normalize(normalization)
                    .fit_with(&ProximalGradient::<V>::new().tolerance(1e-9))
                    .unwrap();
                let runtime = RuntimeProblem::new(
                    Box::new(MatrixDesign::<_, V>::new(design)),
                    Box::new(datafit),
                    Box::new(penalty.clone()),
                )
                .normalize(normalization)
                .fit_with(&ProximalGradient::new().tolerance(1e-9))
                .unwrap();
                let expected = Problem::new(&reference, datafit, penalty)
                    .normalize(normalization)
                    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-9))
                    .unwrap();
                for fit in [&actual, &runtime, &expected] {
                    assert_eq!(fit.termination(), Termination::Converged);
                }
                assert_eq!(actual.coefficients(), runtime.coefficients());
                assert_eq!(actual.intercept(), runtime.intercept());
                assert_eq!(actual.iterations(), runtime.iterations());
                close(actual.objective(), expected.objective());
                for (&a, b) in actual
                    .predict(design)
                    .unwrap()
                    .iter()
                    .zip(expected.predict(&reference).unwrap())
                {
                    close(a, b);
                }
            }
        }
    }
}

fn verify_paths<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: std::error::Error + 'static,
    V: ProximalVector + 'static,
{
    let rows = ROWS.iter().map(|row| row.as_slice()).collect::<Vec<_>>();
    let reference = matrix::Matrix::from_rows(&rows);
    let options = PathOptions::new().nlambda(3).lambda_min_ratio(0.1);
    let penalty = ElasticNet::from_alpha(0.0, 0.5);
    for intercept in [false, true] {
        for normalization in [Normalization::None, Normalization::Standardize] {
            let datafit = Gaussian::new(&Y).fit_intercept(intercept);
            let typed = Problem::new(design, datafit, penalty)
                .normalize(normalization)
                .fit_regularization_path_with(
                    &ProximalGradient::<V>::new().tolerance(1e-9),
                    &options,
                )
                .unwrap();
            let runtime = RuntimePathProblem::new(
                Box::new(MatrixDesign::<_, V>::new(design)),
                Box::new(datafit),
                Box::new(ErasedPathPenalty::new(penalty)),
            )
            .normalize(normalization)
            .fit_regularization_path_with(&ProximalGradient::new().tolerance(1e-9), &options)
            .unwrap();
            let expected = Problem::new(&reference, datafit, penalty)
                .normalize(normalization)
                .fit_regularization_path_with(&ProximalGradient::new().tolerance(1e-9), &options)
                .unwrap();
            for ((a, b), c) in typed.fits().iter().zip(runtime.fits()).zip(expected.fits()) {
                assert_eq!(a.termination(), Termination::Converged);
                assert_eq!(a.coefficients(), b.coefficients());
                assert_eq!(a.intercept(), b.intercept());
                close(a.objective(), c.objective());
                for (&a, &c) in a.coefficients().iter().zip(c.coefficients()) {
                    close(a, c);
                }
            }
            for (&a, &c) in typed
                .lambdas()
                .unwrap()
                .iter()
                .zip(expected.lambdas().unwrap())
            {
                close(a, c);
            }
        }
    }
}

fn verify_multivariate<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: std::error::Error + 'static,
    V: ProximalVector + 'static,
{
    let rows: Vec<_> = ROWS.iter().map(|row| row.as_slice()).collect();
    let reference = matrix::Matrix::from_rows(&rows);
    let y = [1.0, 2.0, -1.0, 4.0, 2.0, -3.0, 2.0, 0.5, 1.0, 2.0];
    let layout = CoefficientLayout::new(3, 2).unwrap();
    for intercept in [false, true] {
        for normalization in [Normalization::None, Normalization::Standardize] {
            for axis in [
                PenaltyAxis::All,
                PenaltyAxis::Features,
                PenaltyAxis::Responses,
            ] {
                let datafit = MultivariateGaussian::new(&y, 5, 2).fit_intercept(intercept);
                let penalty = ElasticNet::new(0.1, 0.3);
                let actual = Problem::new(design, datafit, AxisPenalty::new(layout, axis, penalty))
                    .normalize(normalization)
                    .fit_with(
                        &ProximalGradient::<V>::new()
                            .tolerance(1e-9)
                            .max_iterations(200_000),
                    )
                    .unwrap();
                let expected =
                    Problem::new(&reference, datafit, AxisPenalty::new(layout, axis, penalty))
                        .normalize(normalization)
                        .fit_with(
                            &ProximalGradient::<Vec<f64>>::new()
                                .tolerance(1e-9)
                                .max_iterations(200_000),
                        )
                        .unwrap();
                let runtime = RuntimeMultivariateProblem::new(
                    Box::new(MatrixDesign::<_, V>::new(design)),
                    datafit,
                    AxisPenalty::new(layout, axis, Box::new(penalty)),
                )
                .normalize(normalization)
                .fit_with(
                    &ProximalGradient::new()
                        .tolerance(1e-9)
                        .max_iterations(200_000),
                )
                .unwrap();
                assert_eq!(actual.termination(), Termination::Converged);
                assert_eq!(expected.termination(), Termination::Converged);
                assert_eq!(actual.layout(), layout);
                close(actual.objective(), expected.objective());
                for (&a, &b) in actual.coefficients().iter().zip(expected.coefficients()) {
                    close(a, b);
                }
                for (&a, &b) in actual.intercepts().iter().zip(expected.intercepts()) {
                    close(a, b);
                }
                for (&a, b) in actual
                    .predict(design)
                    .unwrap()
                    .iter()
                    .zip(expected.predict(&reference).unwrap())
                {
                    close(a, b);
                }
                assert_eq!(runtime.coefficients(), actual.coefficients());
                assert_eq!(runtime.intercepts(), actual.intercepts());
                assert_eq!(runtime.objective(), actual.objective());
                assert_eq!(runtime.termination(), actual.termination());
                assert_eq!(runtime.iterations(), actual.iterations());
                assert_eq!(runtime.stopping_value(), actual.stopping_value());
                assert_eq!(runtime.step_size(), actual.step_size());
                assert_eq!(
                    runtime.preprocessing().centers(),
                    actual.preprocessing().centers()
                );
                assert_eq!(
                    runtime.preprocessing().scales(),
                    actual.preprocessing().scales()
                );
            }
        }
    }
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn faer_dense_view_and_csc_use_native_columns() {
    use faer::{
        Col, Mat,
        sparse::{SparseColMat, Triplet},
    };
    let dense = Mat::from_fn(5, 3, |i, j| ROWS[i][j]);
    verify::<_, Col<f64>>(&dense);
    verify::<_, Col<f64>>(&dense.as_ref());
    let triplets: Vec<_> = (0..3)
        .flat_map(|j| {
            (0..5)
                .filter_map(move |i| (ROWS[i][j] != 0.0).then_some(Triplet::new(i, j, ROWS[i][j])))
        })
        .collect();
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(5, 3, &triplets).unwrap();
    verify::<_, Col<f64>>(&sparse);
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn nalgebra_dense_view_and_csc_use_native_vectors() {
    use nalgebra::{DMatrix, DVector};
    use nalgebra_sparse::CscMatrix;
    let dense = DMatrix::from_fn(5, 3, |i, j| ROWS[i][j]);
    verify::<_, DVector<f64>>(&dense);
    verify::<_, DVector<f64>>(&dense.view((0, 0), (5, 3)));
    let offsets = vec![0, 4, 7, 12];
    let indices = vec![1, 2, 3, 4, 0, 2, 3, 0, 1, 2, 3, 4];
    let values = vec![2.0, 4.0, 6.0, 1.0, 1.0, 3.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let sparse = CscMatrix::try_from_csc_data(5, 3, offsets, indices, values).unwrap();
    verify::<_, DVector<f64>>(&sparse);
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_row_major_column_major_and_strided_views_use_arrays() {
    use ndarray::{Array1, Array2, Axis, ShapeBuilder, Slice};
    let row_major = Array2::from_shape_fn((5, 3), |(i, j)| ROWS[i][j]);
    let column_major = Array2::from_shape_fn((5, 3).f(), |(i, j)| ROWS[i][j]);
    let padded = Array2::from_shape_fn((10, 6), |(i, j)| ROWS[i / 2][j / 2]);
    verify::<_, Array1<f64>>(&row_major);
    verify::<_, Array1<f64>>(&column_major);
    let rows = padded.slice_axis(Axis(0), Slice::new(0, None, 2));
    verify::<_, Array1<f64>>(&rows.slice_axis(Axis(1), Slice::new(0, None, 2)));
}

#[cfg(feature = "sprs_v0_11")]
#[test]
fn sprs_owned_and_borrowed_csc_use_vectors() {
    use shrinkage::lazymatrix::SprsCsc;
    let sparse = sprs::CsMat::new_csc(
        (5, 3),
        vec![0, 4, 7, 12],
        vec![1, 2, 3, 4, 0, 2, 3, 0, 1, 2, 3, 4],
        vec![2.0, 4.0, 6.0, 1.0, 1.0, 3.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    );
    verify::<_, Vec<f64>>(&SprsCsc::try_new(sparse.view()).unwrap());
    verify::<_, Vec<f64>>(&SprsCsc::try_new(sparse).unwrap());
}

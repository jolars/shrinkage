//! Dual feasibility and objective checks for Gaussian lasso.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    CoordinateDescent, Gaussian, L1, Lasso, Normalization, Problem, StoppingCriterion, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-8 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn dual_certificate_is_feasible_and_uses_averaged_loss() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let y = [1.0, 5.0, 9.0];
    for intercept in [false, true] {
        for normalization in [Normalization::None, Normalization::Auto] {
            let fit = Lasso::new(0.5)
                .fit_intercept(intercept)
                .normalize(normalization)
                .fit(&x, &y)
                .unwrap();
            let u = fit.dual_certificate();
            let center = fit.preprocessing().centers().map_or(0.0, |c| c[0]);
            let scale = fit.preprocessing().scales().map_or(1.0, |s| s[0]);
            let correlation: f64 = [0.0, 2.0, 4.0]
                .iter()
                .zip(u)
                .map(|(&x, &u)| (x - center) / scale * u)
                .sum();
            assert!(correlation.abs() <= 0.5 + 1e-12);
            if intercept {
                close(u.iter().sum(), 0.0);
            }
            let dual = y.iter().zip(u).map(|(&y, &u)| y * u).sum::<f64>()
                - 1.5 * u.iter().map(|u| u * u).sum::<f64>();
            close(fit.dual_objective(), dual);
            close(fit.duality_gap(), fit.objective() - dual);
            assert!(fit.duality_gap() >= 0.0);
            assert_eq!(fit.termination(), Termination::Converged);
        }
    }
}

#[test]
fn analytical_optimum_has_matching_primal_and_dual_objectives() {
    let x = Matrix::from_rows(&[&[0.0], &[2.0], &[4.0]]);
    let fit = Lasso::new(0.5)
        .normalize(Normalization::None)
        .tolerance(1e-12)
        .fit(&x, &[1.0, 5.0, 9.0])
        .unwrap();
    close(fit.objective(), 0.953125);
    close(fit.dual_objective(), 0.953125);
    assert!(fit.duality_gap() < fit.stopping_threshold());
}

#[test]
fn supplied_normalization_and_constant_columns_keep_dual_feasible() {
    let rows = [[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]];
    let x = Matrix::from_rows(&rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>());
    let fit = Lasso::new(0.5)
        .fit_intercept(false)
        .with_centers(vec![1.0, 3.0])
        .with_scales(vec![2.0, 4.0])
        .fit(&x, &[1.0, 5.0, 9.0])
        .unwrap();
    let u = fit.dual_certificate();
    for j in 0..2 {
        let center = fit.preprocessing().centers().unwrap()[j];
        let scale = fit.preprocessing().scales().unwrap()[j];
        let correlation: f64 = rows
            .iter()
            .zip(u)
            .map(|(row, &u)| (row[j] - center) / scale * u)
            .sum();
        assert!(correlation.abs() <= 0.5 + 1e-12);
    }
    assert!(fit.duality_gap() >= 0.0);
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn dual_feasibility_holds_across_normalization_policies() {
    use shrinkage::{Centering, Scaling};

    let x = faer::Mat::from_fn(4, 2, |i, j| {
        [[0.0, 3.0], [2.0, 0.0], [4.0, 5.0], [6.0, 1.0]][i][j]
    });
    let y = [1.0, 5.0, 9.0, 3.0];
    for intercept in [false, true] {
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
                center: Centering::Min,
                scale: Scaling::L2,
            },
        ] {
            let fit = Lasso::new(0.5)
                .fit_intercept(intercept)
                .normalize(normalization)
                .fit(&x, &y)
                .unwrap();
            for j in 0..2 {
                let center = fit.preprocessing().centers().map_or(0.0, |c| c[j]);
                let scale = fit.preprocessing().scales().map_or(1.0, |s| s[j]);
                let correlation: f64 = fit
                    .dual_certificate()
                    .iter()
                    .enumerate()
                    .map(|(i, &u)| (x[(i, j)] - center) / scale * u)
                    .sum();
                assert!(correlation.abs() <= 0.5 + 1e-12);
            }
            if intercept {
                close(fit.dual_certificate().iter().sum(), 0.0);
            }
        }
    }
}

#[test]
fn default_gap_uses_fixed_reference_and_handles_zero_reference_loss() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0, 3.0, 5.0];
    let fit = Lasso::new(0.1).fit(&x, &y).unwrap();
    assert_eq!(
        fit.stopping_criterion(),
        StoppingCriterion::duality_gap(1e-6)
    );
    close(fit.reference_loss(), 4.0 / 3.0);
    close(fit.stopping_threshold(), 1e-6 * fit.reference_loss());
    close(fit.stopping_value(), fit.duality_gap());

    let exact = Lasso::new(0.1).fit(&x, &[2.0, 2.0, 2.0]).unwrap();
    close(exact.reference_loss(), 0.0);
    close(exact.duality_gap(), 0.0);
    assert_eq!(exact.termination(), Termination::Converged);
}

#[test]
fn zero_penalty_still_has_a_feasible_dual_certificate() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0, 3.0, 4.0];
    let fit = Lasso::new(0.0).fit(&x, &y).unwrap();
    let u = fit.dual_certificate();
    let center = fit.preprocessing().centers().unwrap()[0];
    let scale = fit.preprocessing().scales().unwrap()[0];
    close(u.iter().sum(), 0.0);
    close(
        [0.0, 1.0, 2.0]
            .iter()
            .zip(u)
            .map(|(&x, &u)| (x - center) / scale * u)
            .sum(),
        0.0,
    );
    assert!(fit.duality_gap() >= 0.0);
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(
        fit.stopping_criterion(),
        StoppingCriterion::kkt_violation(1e-6)
    );

    let empty = Lasso::new(0.0).fit(&Matrix::empty(3, 0), &y).unwrap();
    close(empty.duality_gap(), 0.0);
    close(empty.dual_objective(), empty.objective());
}

#[test]
fn absolute_gap_and_legacy_kkt_tolerances_keep_their_meanings() {
    let x = Matrix::from_rows(&[&[1.0, 1.0], &[2.0, 0.0], &[3.0, 2.0]]);
    let y = [1.0, 2.0, 4.0];
    let gap_rule = StoppingCriterion::DualityGap {
        absolute: 1e-12,
        relative: 1e-10,
    };
    let gap_fit = Lasso::new(0.1)
        .terminate_on(gap_rule)
        .max_iterations(1)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(gap_fit.stopping_criterion(), gap_rule);
    close(
        gap_fit.stopping_threshold(),
        1e-12 + 1e-10 * gap_fit.reference_loss(),
    );
    close(gap_fit.stopping_value(), gap_fit.duality_gap());
    assert_eq!(gap_fit.termination(), Termination::IterationLimit);

    let kkt_fit = Lasso::new(0.1)
        .terminate_on(gap_rule)
        .tolerance(1e-8)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        kkt_fit.stopping_criterion(),
        StoppingCriterion::kkt_violation(1e-8)
    );
    close(kkt_fit.stopping_value(), kkt_fit.kkt_violation());
    close(kkt_fit.stopping_threshold(), 1e-8);

    let typed = Problem::new(&x, Gaussian::new(&y), L1::new(0.1))
        .fit_with(&CoordinateDescent::new())
        .unwrap();
    assert_eq!(
        typed.stopping_criterion(),
        StoppingCriterion::duality_gap(1e-6)
    );
}

#[test]
fn invalid_gap_tolerances_are_rejected_before_fitting() {
    let x = Matrix::from_rows(&[&[1.0], &[2.0]]);
    let y = [1.0, 2.0];
    for (absolute, relative) in [
        (0.0, 0.0),
        (-1.0, 1e-6),
        (0.0, f64::NAN),
        (f64::INFINITY, 0.0),
    ] {
        assert!(
            Lasso::new(0.1)
                .terminate_on(StoppingCriterion::DualityGap { absolute, relative })
                .fit(&x, &y)
                .is_err()
        );
    }
}

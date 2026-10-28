use shrinkage_widget::{Config, Loss, Method, Regularizer, scene, trace, trace_from};

#[test]
fn all_supported_compositions_have_matching_native_solutions_and_real_objectives() {
    for loss in [Loss::Gaussian, Loss::Binomial] {
        for penalty in [
            Regularizer::L1,
            Regularizer::Ridge,
            Regularizer::ElasticNet,
            Regularizer::Slope,
            Regularizer::ClusteredLasso,
        ] {
            let config = Config {
                loss,
                penalty,
                strength: 0.05,
                ..Config::default()
            };
            let scene = scene(&config).unwrap();
            assert!(
                scene.reference.converged,
                "{config:?}: {}",
                scene.reference.stopping_value
            );
            for i in (0..scene.loss.len()).step_by(97) {
                assert_eq!(scene.objective[i], scene.loss[i] + scene.penalty[i]);
            }
            for method in [
                Method::Coordinate,
                Method::Proximal,
                Method::Irls,
                Method::Hybrid,
            ] {
                if !config.supports(method) {
                    assert!(trace(&config, method).is_err());
                    continue;
                }
                let trace = trace(&config, method).unwrap();
                assert!(trace.fit.converged, "{config:?} {method:?}");
                assert!((trace.fit.objective - scene.reference.objective).abs() < 1e-8);
                assert_eq!(trace.points.first().unwrap().coefficients, [0., 0.]);
                assert_eq!(
                    trace.points.last().unwrap().coefficients,
                    trace.fit.coefficients
                );
                assert!(
                    (trace.points.last().unwrap().objective - trace.fit.objective).abs() < 1e-13
                );
                for pair in trace
                    .points
                    .iter()
                    .filter(|p| !p.inner)
                    .collect::<Vec<_>>()
                    .windows(2)
                {
                    assert!(pair[1].objective <= pair[0].objective + 1e-12);
                }
                for start in [[-2., 1.5], [2.5, -2.5]] {
                    let initialized = trace_from(&config, method, start).unwrap();
                    assert_eq!(initialized.points[0].coefficients, start);
                    assert!(initialized.fit.converged, "{config:?} {method:?} {start:?}");
                    assert!((initialized.fit.objective - scene.reference.objective).abs() < 1e-8);
                    for pair in initialized
                        .points
                        .iter()
                        .filter(|p| !p.inner)
                        .collect::<Vec<_>>()
                        .windows(2)
                    {
                        assert!(pair[1].objective <= pair[0].objective + 1e-12);
                    }
                }
            }
        }
    }
}

#[test]
fn orthogonal_gaussian_lasso_and_ridge_match_analytical_solutions() {
    let config = Config {
        correlation: 0.,
        strength: 0.3,
        ..Config::default()
    };
    let fit = trace(&config, Method::Coordinate).unwrap().fit;
    assert!((fit.coefficients[0] - 1.7).abs() < 1e-12);
    assert!((fit.coefficients[1] + 0.7).abs() < 1e-12);
    let config = Config {
        penalty: Regularizer::Ridge,
        ..config
    };
    let fit = trace(&config, Method::Proximal).unwrap().fit;
    assert!((fit.coefficients[0] - 2. / 1.3).abs() < 1e-6);
    assert!((fit.coefficients[1] + 1. / 1.3).abs() < 1e-6);
}

#[test]
fn degenerate_penalties_and_extreme_supported_geometries_stay_finite() {
    for loss in [Loss::Gaussian, Loss::Binomial] {
        for correlation in [-0.9, 0.9] {
            for penalty in [
                Regularizer::L1,
                Regularizer::Slope,
                Regularizer::ClusteredLasso,
            ] {
                for strength in [0., 0.1] {
                    let config = Config {
                        loss,
                        penalty,
                        correlation,
                        strength,
                        mixing: 0.,
                        rank_ratio: 0.,
                        ..Config::default()
                    };
                    let scene = scene(&config).unwrap();
                    assert!(
                        scene.reference.converged,
                        "{config:?}: {}",
                        scene.reference.stopping_value
                    );
                    assert!(scene.objective.iter().all(|v| v.is_finite()));
                    if strength == 0. || penalty == Regularizer::ClusteredLasso {
                        assert!(scene.penalty_clipped);
                    }
                    if strength == 0. {
                        assert!(scene.penalty.iter().all(|p| *p == 0.));
                    }
                }
            }
        }
    }
}

#[test]
fn binomial_proportions_have_the_selected_unpenalized_minimum() {
    let config = Config {
        loss: Loss::Binomial,
        strength: 0.,
        ..Config::default()
    };
    let (x, y) = config.observations();
    assert!(y.iter().all(|v| *v > 0. && *v < 1.));
    for j in 0..2 {
        assert!((0..x.nrows()).map(|i| x[(i, j)]).sum::<f64>().abs() < 1e-13);
    }
    let fit = scene(&config).unwrap().reference;
    for j in 0..2 {
        assert!((fit.coefficients[j] - config.signal[j]).abs() < 1e-7);
    }
}

#[test]
fn reference_solutions_pass_an_independent_gradient_and_proximal_check() {
    use shrinkage::{ClusteredLasso, ElasticNet, L1, Penalty, ProximalPenalty, Ridge, Slope};
    for loss in [Loss::Gaussian, Loss::Binomial] {
        for penalty in [
            Regularizer::L1,
            Regularizer::Ridge,
            Regularizer::ElasticNet,
            Regularizer::Slope,
            Regularizer::ClusteredLasso,
        ] {
            let c = Config {
                loss,
                penalty,
                strength: 0.05,
                ..Config::default()
            };
            let scene = scene(&c).unwrap();
            let fit = &scene.reference;
            let (x, y) = c.observations();
            let b = fit.coefficients;
            let mut gradient = [0.; 2];
            for i in 0..x.nrows() {
                let eta = x[(i, 0)] * b[0] + x[(i, 1)] * b[1];
                let mean = match loss {
                    Loss::Gaussian => eta,
                    Loss::Binomial => 1. / (1. + (-eta).exp()),
                };
                for j in 0..2 {
                    gradient[j] += x[(i, j)] * (mean - y[i]) / x.nrows() as f64;
                }
            }
            let p: Box<dyn ProximalPenalty> = match penalty {
                Regularizer::L1 => Box::new(L1::new(c.strength)),
                Regularizer::Ridge => Box::new(Ridge::new(c.strength)),
                Regularizer::ElasticNet => Box::new(ElasticNet::new(
                    c.strength * c.mixing,
                    c.strength * (1. - c.mixing),
                )),
                Regularizer::Slope => {
                    Box::new(Slope::new(vec![c.strength, c.strength * c.rank_ratio]))
                }
                Regularizer::ClusteredLasso => Box::new(ClusteredLasso::new(
                    c.strength * c.mixing,
                    c.strength * (1. - c.mixing),
                )),
            };
            let mut z = [0.; 2];
            p.prox(&[b[0] - gradient[0], b[1] - gradient[1]], 1., &mut z)
                .unwrap();
            for j in 0..2 {
                assert!((b[j] - z[j]).abs() < 2e-9, "{c:?}");
            }
            assert_eq!(scene.penalty_budget, p.value(&b).unwrap());
            assert!(
                (scene.reference.objective - scene.loss_at_optimum - scene.penalty_budget).abs()
                    < 1e-13
            );
        }
    }
}

#[test]
fn zero_budget_regions_preserve_the_exact_zero_set() {
    use shrinkage_widget::ZeroSet;
    let config = Config {
        correlation: 0.,
        strength: 2.,
        ..Config::default()
    };
    let point = scene(&config).unwrap();
    assert_eq!(point.penalty_budget, 0.);
    assert_eq!(point.penalty_zero_set, ZeroSet::Origin);
    assert!(!point.penalty_clipped);
    let line = scene(&Config {
        penalty: Regularizer::ClusteredLasso,
        mixing: 0.,
        ..config.clone()
    })
    .unwrap();
    assert_eq!(line.penalty_budget, 0.);
    assert_eq!(line.penalty_zero_set, ZeroSet::Diagonal);
    assert!(line.penalty_clipped);
    let plane = scene(&Config {
        strength: 0.,
        ..config
    })
    .unwrap();
    assert_eq!(plane.penalty_budget, 0.);
    assert_eq!(plane.penalty_zero_set, ZeroSet::Plane);
    assert!(plane.penalty_clipped);
}

#[test]
fn binomial_traces_exclude_surrogate_updates() {
    for loss in [Loss::Gaussian, Loss::Binomial] {
        for (penalty, method) in [
            (Regularizer::L1, Method::Irls),
            (Regularizer::Slope, Method::Hybrid),
            (Regularizer::ClusteredLasso, Method::Hybrid),
        ] {
            let config = Config {
                loss,
                penalty,
                strength: 0.05,
                ..Config::default()
            };
            let trace = trace_from(&config, method, [-2., 1.5]).unwrap();
            let scene = scene(&config).unwrap();
            let inner: Vec<_> = trace.points.iter().filter(|p| p.inner).collect();
            if loss == Loss::Binomial {
                assert!(inner.is_empty(), "{config:?} {method:?}");
                for pair in trace.points.windows(2) {
                    assert!(pair[1].objective <= pair[0].objective + 1e-12);
                }
            } else {
                assert!(!inner.is_empty(), "{config:?} {method:?}");
            }
            assert!(
                inner
                    .iter()
                    .all(|p| p.iteration > 0 && p.inner_iteration.unwrap() > 0)
            );
            assert!(inner.iter().all(|p| p.objective.is_finite()));
            assert!((trace.fit.objective - scene.reference.objective).abs() < 1e-8);
        }
    }
}

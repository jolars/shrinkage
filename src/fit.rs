//! Owned fitted parameters, training transformations, and diagnostics.

mod hybrid;
mod irls;
mod lasso;
mod multivariate;
mod proximal;
pub use hybrid::HybridFit;
pub use irls::IrlsFit;
pub use lasso::LassoFit;
pub use multivariate::MultivariateFit;
pub use proximal::ProximalFit;

use crate::error::{finite, invalid};
use crate::problem::validate_matrix;
use crate::{Centering, LassoError, Scaling};
use lazymatrix::{RawColumn, RawColumns};

/// Training normalization retained by a fitted model.
#[derive(Clone, Debug)]
pub struct Preprocessing {
    pub(crate) spec: lazymatrix::Normalization,
    pub(crate) centers: Option<Vec<f64>>,
    pub(crate) scales: Option<Vec<f64>>,
    pub(crate) supplied_centers: bool,
    pub(crate) supplied_scales: bool,
}

impl Preprocessing {
    /// The configured centering rule, with automatic choices resolved.
    /// A supplied center vector overrides this rule.
    pub fn centering(&self) -> Centering {
        self.spec.center
    }

    /// The configured scaling rule, with automatic choices resolved.
    /// A supplied scale vector overrides this rule.
    pub fn scaling(&self) -> Scaling {
        self.spec.scale
    }

    /// Training-column centers, or `None` when centering was disabled.
    pub fn centers(&self) -> Option<&[f64]> {
        self.centers.as_deref()
    }

    /// Training-column scales with zeros replaced by one, or `None` when disabled.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Whether the centers were supplied rather than computed during fitting.
    pub fn centers_were_supplied(&self) -> bool {
        self.supplied_centers
    }

    /// Whether the scales were supplied rather than computed during fitting.
    pub fn scales_were_supplied(&self) -> bool {
        self.supplied_scales
    }
}

/// Why a finite fit stopped. An iteration limit does not certify optimality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The selected check passed using freshly evaluated model state.
    Converged,
    /// The iteration budget was exhausted before meeting the selected threshold.
    IterationLimit,
}

pub(crate) fn back_transform<E>(
    mut coefficients: Vec<f64>,
    mut intercept: f64,
    preprocessing: &Preprocessing,
    iterations: usize,
) -> Result<(Vec<f64>, f64), LassoError<E>> {
    back_transform_in_place(&mut coefficients, &mut intercept, preprocessing, iterations)?;
    Ok((coefficients, intercept))
}

pub(crate) fn back_transform_in_place<E>(
    coefficients: &mut [f64],
    intercept: &mut f64,
    preprocessing: &Preprocessing,
    iterations: usize,
) -> Result<(), LassoError<E>> {
    if let Some(scales) = &preprocessing.scales {
        for (coefficient, scale) in coefficients.iter_mut().zip(scales) {
            *coefficient = finite(
                *coefficient / scale,
                "back-transforming coefficients",
                iterations,
            )?;
        }
    }
    if let Some(centers) = &preprocessing.centers {
        for (coefficient, center) in coefficients.iter().zip(centers) {
            *intercept = finite(
                *intercept - coefficient * center,
                "back-transforming intercept",
                iterations,
            )?;
        }
    }
    Ok(())
}

fn predict<M: RawColumns<f64> + ?Sized>(
    coefficients: &[f64],
    intercept: f64,
    iterations: usize,
    x: &M,
) -> Result<Vec<f64>, LassoError> {
    if x.ncols() != coefficients.len() {
        return Err(invalid(format!(
            "prediction design has {} columns; expected {}",
            x.ncols(),
            coefficients.len()
        )));
    }
    validate_matrix(x)?;
    let mut predictions = vec![intercept; x.nrows()];
    for (j, &coefficient) in coefficients.iter().enumerate() {
        if coefficient != 0.0 {
            x.raw_column(j)
                .affine_add_to(coefficient, 0.0, &mut predictions);
        }
    }
    for &value in &predictions {
        finite(value, "predicting", iterations)?;
    }
    Ok(predictions)
}

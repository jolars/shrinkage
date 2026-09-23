//! Regularized statistical models with lazy matrix normalization.
//!
//! [`Lasso`] fits Gaussian lasso models using dense or sparse CSC columns.
//! Fits return original-scale parameters, reusable training preprocessing,
//! and explicit convergence diagnostics. See [`Lasso::tolerance`] for the
//! stopping criterion and [`LassoFit::termination`] before using a fit.
//!
//! No matrix backend is enabled by default. Enable `faer` or `nalgebra` for
//! LazyMatrix's dense and sparse CSC integrations.

pub mod lasso;
pub mod normalization;

pub use lasso::{Lasso, LassoError, LassoFit, Preprocessing, Termination};
pub use normalization::{Centering, Normalization, Scaling};

/// Matrix capabilities, borrowed columns, and lazy normalization.
pub use lazymatrix;

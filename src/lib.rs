//! Regularized statistical models with lazy matrix normalization.
//!
//! [`Lasso`] fits Gaussian lasso models using dense or sparse CSC columns.
//! Fits return original-scale parameters, reusable training preprocessing,
//! and explicit convergence diagnostics. See [`Lasso::tolerance`] for the
//! stopping criterion and [`LassoFit::termination`] before using a fit.
//!
//! No matrix backend is enabled by default. Versioned features select
//! LazyMatrix's integrations with faer 0.24 (`faer_v0_24`), nalgebra 0.34 and
//! nalgebra-sparse 0.11 (`nalgebra_v0_34`), ndarray 0.17 (`ndarray_v0_17`), or
//! sprs 0.11 (`sprs_v0_11`). The short names `faer`, `nalgebra`, `ndarray`, and
//! `sprs` are aliases for these features. Match your direct matrix dependency
//! to the selected release line.
//!
//! LazyMatrix 0.3.0 implements traits only for the newest enabled release of
//! each backend. Another dependency enabling a newer adapter on the same
//! LazyMatrix package can remove support for older matrix types.
//!
//! With `ndarray_v0_17` or its `ndarray` alias, fit directly against an array or
//! borrowed view:
//!
//! ```
//! # #[cfg(feature = "ndarray_v0_17")] {
//! use ndarray::array;
//! use shrinkage::Lasso;
//!
//! let x = array![[0.0], [1.0], [2.0]];
//! let fit = Lasso::new(0.1).fit(&x.view(), &[1.0, 3.0, 5.0])?;
//! let predictions = fit.predict(&x)?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! With `sprs_v0_11` or its `sprs` alias, wrap CSC storage in
//! `lazymatrix::SprsCsc`. Borrowing a view checks its orientation without copying
//! its entries:
//!
//! ```
//! # #[cfg(feature = "sprs_v0_11")] {
//! use shrinkage::{Lasso, lazymatrix::SprsCsc};
//! use sprs::CsMat;
//!
//! let x = CsMat::new_csc((3, 1), vec![0, 2], vec![1, 2], vec![1.0, 2.0]);
//! let columns = SprsCsc::try_new(x.view()).unwrap();
//! let fit = Lasso::new(0.1).fit(&columns, &[1.0, 3.0, 5.0])?;
//! let predictions = fit.predict(&columns)?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! For CSR input, explicitly convert with `to_csc()` before wrapping. ndarray
//! supports row-major, column-major, and strided arrays; column-major storage
//! keeps each column contiguous for coordinate descent.

pub mod lasso;
pub mod normalization;

pub use lasso::{Lasso, LassoError, LassoFit, Preprocessing, Termination};
pub use normalization::{Centering, Normalization, Scaling};

/// Matrix capabilities, borrowed columns, and lazy normalization.
pub use lazymatrix;

//! Infrastructure for composable regularized statistical models.
//!
//! The bootstrap exposes the matrix dependency used by future solvers. Model
//! fitting is not implemented yet; the repository's `DESIGN.md` specifies the
//! statistical conventions and implementation sequence.
//!
//! No matrix backend is enabled by default. Enable `faer` or `nalgebra` for
//! LazyMatrix's dense and sparse CSC integrations.

/// Matrix capabilities, borrowed columns, and lazy normalization.
pub use lazymatrix;

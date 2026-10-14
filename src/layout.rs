//! Explicit feature and response axes for CPU coefficient matrices.

use crate::FitError;
use crate::error::invalid;

/// A feature-by-response coefficient matrix stored in response-major order.
///
/// Each response owns a contiguous vector of feature coefficients. Entry
/// `(feature, response)` has index `response * nfeatures + feature`. Intercepts
/// are separate and never appear in this layout. This CPU layout is provisional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientLayout {
    nfeatures: usize,
    nresponses: usize,
    len: usize,
}

impl CoefficientLayout {
    /// Specify both axes, allowing zero features but requiring a response.
    ///
    /// # Errors
    /// Reject zero responses or a coefficient count that overflows `usize`.
    pub fn new(nfeatures: usize, nresponses: usize) -> Result<Self, FitError> {
        if nresponses == 0 {
            return Err(invalid(
                "a coefficient layout must have at least one response",
            ));
        }
        let len = nfeatures
            .checked_mul(nresponses)
            .ok_or_else(|| invalid("coefficient dimensions overflow usize"))?;
        Ok(Self {
            nfeatures,
            nresponses,
            len,
        })
    }

    /// Number of training features.
    pub fn nfeatures(self) -> usize {
        self.nfeatures
    }

    /// Number of responses.
    pub fn nresponses(self) -> usize {
        self.nresponses
    }

    /// Number of coefficient entries, excluding intercepts.
    pub fn len(self) -> usize {
        self.len
    }

    /// Whether there are no feature coefficients.
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Flat index of a feature and response, or `None` for an invalid axis index.
    pub fn index(self, feature: usize, response: usize) -> Option<usize> {
        (feature < self.nfeatures && response < self.nresponses)
            .then(|| response * self.nfeatures + feature)
    }
}

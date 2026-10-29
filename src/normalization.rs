//! Column normalization presets and independent centering and scaling choices.

pub use lazymatrix::{Centering, QuantileRange, Scaling};

/// Training-column normalization applied before fitting.
///
/// Penalties act on normalized coefficients. Fitted coefficients and predictions
/// use the original input scale. Except for [`Self::Auto`], choices are independent
/// of whether an intercept is fitted. Explicit centering without a fitted
/// intercept can therefore induce a fixed original-scale intercept.
///
/// Norm and maximum-absolute-value scales use the columns after any requested
/// centering. Standard deviations and ranges are invariant to centering.
/// Computed zero scales are replaced with one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Normalization {
    /// Divide by population standard deviations, centering by column means only
    /// when fitting an intercept. This is the default fitting policy.
    #[default]
    Auto,
    /// Use the raw design without centering or scaling.
    None,
    /// Subtract column means without scaling.
    Center,
    /// Subtract column means and divide by population standard deviations.
    Standardize,
    /// Subtract column minima and divide by column ranges.
    MinMax,
    /// Divide by column maximum absolute values without centering.
    MaxAbs,
    /// Divide by column L1 norms without centering.
    L1,
    /// Divide by column L2 norms without centering.
    L2,
    /// Choose centering and scaling independently using LazyMatrix's options.
    ///
    /// Use `Scaling::Sd { df: 1 }` for sample standard deviations, or combine
    /// [`Centering::Median`] with [`Scaling::QuantileRange`] for robust scaling.
    Custom {
        /// How to center each column.
        center: Centering,
        /// How to scale each column after centering.
        scale: Scaling,
    },
}

impl Normalization {
    pub(crate) fn specification(self, fit_intercept: bool) -> lazymatrix::Normalization {
        use lazymatrix::Normalization as Spec;

        match self {
            Self::Auto => Spec::new(
                if fit_intercept {
                    Centering::Mean
                } else {
                    Centering::None
                },
                Scaling::Sd { df: 0 },
            ),
            Self::None => Spec::default(),
            Self::Center => Spec::new(Centering::Mean, Scaling::None),
            Self::Standardize => Spec::standardize(),
            Self::MinMax => Spec::min_max(),
            Self::MaxAbs => Spec::new(Centering::None, Scaling::MaxAbs),
            Self::L1 => Spec::new(Centering::None, Scaling::L1),
            Self::L2 => Spec::new(Centering::None, Scaling::L2),
            Self::Custom { center, scale } => Spec::new(center, scale),
        }
    }
}

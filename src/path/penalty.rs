use crate::error::{finite, invalid};
use crate::{
    AxisPenalty, ElasticNet, FitError, L1, Penalty, PenaltyAxis, PenaltyMaximum, ProximalPenalty,
    Ridge, Slope,
};

/// A penalty shape that supplies thresholds and complete penalties along a path.
///
/// L1 and ridge replace their scalar strength. Elastic net retains its mixing
/// parameter; SLOPE multiplies its supplied base rank weights. Implementations
/// must use the same coefficient scale as their proximal map.
pub trait PathPenalty: ProximalPenalty {
    /// Complete penalty produced at a scalar strength.
    type AtStrength: ProximalPenalty;
    /// Smallest strength that makes zero optimal for the supplied null gradient.
    ///
    /// # Errors
    /// Reject invalid shapes, dimensions, gradients, or nonfinite arithmetic.
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError>;
    /// Finite starting anchor for automatic sequences.
    ///
    /// # Errors
    /// Reject shapes without a usable finite anchor. Ridge overrides this
    /// method with the L1 threshold divided by `0.001`.
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        self.maximum_penalty(gradient)?.finite().ok_or_else(|| {
            invalid("penalty has no finite automatic path anchor; supply lambda_start")
        })
    }
    /// Construct a complete penalty at a finite, nonnegative scalar strength.
    ///
    /// # Errors
    /// Reject invalid shape parameters, strengths, or arithmetic.
    fn at_strength(&self, lambda: f64) -> Result<Self::AtStrength, FitError>;
}

fn max_abs(gradient: &[f64]) -> Result<f64, FitError> {
    if gradient.iter().any(|v| !v.is_finite()) {
        return Err(invalid("null coefficient gradient must be finite"));
    }
    Ok(gradient.iter().fold(0.0_f64, |max, v| max.max(v.abs())))
}

impl PathPenalty for L1 {
    type AtStrength = Self;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.validate()?;
        Ok(PenaltyMaximum::Finite(max_abs(gradient)?))
    }
    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        self.validate()?;
        let penalty = Self::new(lambda);
        penalty.validate()?;
        Ok(penalty)
    }
}

impl PathPenalty for Ridge {
    type AtStrength = Self;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        L1::new(self.lambda()).validate()?;
        Ok(if max_abs(gradient)? == 0.0 {
            PenaltyMaximum::Finite(0.0)
        } else {
            PenaltyMaximum::Unbounded
        })
    }
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        self.maximum_penalty(gradient)?;
        Ok(finite(
            max_abs(gradient)? / 0.001,
            "computing the ridge path anchor",
            0,
        )?)
    }
    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        L1::new(self.lambda()).validate()?;
        L1::new(lambda).validate()?;
        Ok(Self::new(lambda))
    }
}

impl PathPenalty for ElasticNet {
    type AtStrength = Self;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.value(&[])?;
        let alpha = self.alpha().ok_or_else(|| invalid("automatic elastic-net paths require an identifiable mixing parameter; use ElasticNet::from_alpha"))?;
        let maximum = max_abs(gradient)?;
        if alpha == 0.0 {
            return Ok(if maximum == 0.0 {
                PenaltyMaximum::Finite(0.0)
            } else {
                PenaltyMaximum::Unbounded
            });
        }
        Ok(PenaltyMaximum::Finite(finite(
            maximum / alpha,
            "computing the elastic-net maximum penalty",
            0,
        )?))
    }
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        match self.maximum_penalty(gradient)? {
            PenaltyMaximum::Finite(value) => Ok(value),
            PenaltyMaximum::Unbounded => Ok(finite(
                max_abs(gradient)? / 0.001,
                "computing the ridge path anchor",
                0,
            )?),
        }
    }
    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        self.value(&[])?;
        L1::new(lambda).validate()?;
        let alpha = self.alpha().ok_or_else(|| invalid("automatic elastic-net paths require a mixing parameter; use ElasticNet::from_alpha"))?;
        let penalty = Self::from_alpha(lambda, alpha);
        penalty.value(&[])?;
        Ok(penalty)
    }
}

impl PathPenalty for Slope {
    type AtStrength = Self;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.value(&vec![0.0; gradient.len()])?;
        max_abs(gradient)?;
        let mut order = gradient.iter().map(|v| v.abs()).collect::<Vec<_>>();
        order.sort_unstable_by(|a, b| b.total_cmp(a));
        let mut weight_sum = 0.0;
        let mut gradient_sum = 0.0;
        let mut maximum = 0.0_f64;
        for (magnitude, &weight) in order.iter().zip(self.weights()) {
            gradient_sum = finite(gradient_sum + magnitude, "summing sorted null gradients", 0)?;
            weight_sum = finite(weight_sum + weight, "summing SLOPE rank weights", 0)?;
            if weight_sum == 0.0 {
                if gradient_sum != 0.0 {
                    return Ok(PenaltyMaximum::Unbounded);
                }
            } else {
                maximum = maximum.max(finite(
                    gradient_sum / weight_sum,
                    "computing the SLOPE maximum penalty",
                    0,
                )?);
            }
        }
        Ok(PenaltyMaximum::Finite(maximum))
    }
    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        self.value(&vec![0.0; self.weights().len()])?;
        L1::new(lambda).validate()?;
        let weights = self
            .weights()
            .iter()
            .map(|weight| {
                finite(lambda * weight, "scaling SLOPE rank weights", 0).map_err(FitError::from)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self::new(weights))
    }
}

impl<P: PathPenalty> PathPenalty for AxisPenalty<P> {
    type AtStrength = AxisPenalty<P::AtStrength>;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.value(&vec![0.0; gradient.len()])?;
        max_abs(gradient)?;
        let groups = axis_groups(self, gradient);
        let mut maximum = 0.0_f64;
        for group in groups {
            match self.penalty().maximum_penalty(&group)? {
                PenaltyMaximum::Finite(value) => maximum = maximum.max(value),
                PenaltyMaximum::Unbounded => return Ok(PenaltyMaximum::Unbounded),
            }
        }
        Ok(PenaltyMaximum::Finite(maximum))
    }
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        self.maximum_penalty(gradient)?;
        let mut anchor = 0.0_f64;
        for group in axis_groups(self, gradient) {
            anchor = anchor.max(self.penalty().path_anchor(&group)?);
        }
        Ok(anchor)
    }
    fn at_strength(&self, lambda: f64) -> Result<Self::AtStrength, FitError> {
        Ok(AxisPenalty::new(
            self.layout(),
            self.axis(),
            self.penalty().at_strength(lambda)?,
        ))
    }
}

fn axis_groups<P>(penalty: &AxisPenalty<P>, gradient: &[f64]) -> Vec<Vec<f64>> {
    let layout = penalty.layout();
    if layout.is_empty() && penalty.axis() == PenaltyAxis::Responses {
        return Vec::new();
    }
    if penalty.axis() == PenaltyAxis::All || layout.is_empty() {
        return vec![gradient.to_vec()];
    }
    match penalty.axis() {
        PenaltyAxis::All => unreachable!(),
        PenaltyAxis::Features => gradient
            .chunks_exact(layout.nfeatures())
            .map(<[f64]>::to_vec)
            .collect(),
        PenaltyAxis::Responses => (0..layout.nfeatures())
            .map(|feature| {
                (0..layout.nresponses())
                    .map(|response| gradient[response * layout.nfeatures() + feature])
                    .collect()
            })
            .collect(),
    }
}

impl<P: PathPenalty + ?Sized> PathPenalty for Box<P> {
    type AtStrength = P::AtStrength;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        (**self).maximum_penalty(gradient)
    }
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        (**self).path_anchor(gradient)
    }
    fn at_strength(&self, lambda: f64) -> Result<Self::AtStrength, FitError> {
        (**self).at_strength(lambda)
    }
}

/// Adapt a typed penalty shape to independently selected runtime path factories.
/// Numerical dispatch remains at whole proximal-operation boundaries.
pub struct ErasedPathPenalty<P>(P);
impl<P> ErasedPathPenalty<P> {
    /// Retain the penalty's automatic path capabilities before boxing it.
    pub fn new(penalty: P) -> Self {
        Self(penalty)
    }
}
impl<P: ProximalPenalty> Penalty for ErasedPathPenalty<P> {
    type Error = FitError;
    fn value(&self, input: &[f64]) -> Result<f64, FitError> {
        self.0.value(input)
    }
    fn validate_input(&self, input: &[f64]) -> Result<(), FitError> {
        self.0.validate_input(input)
    }
}
impl<P: ProximalPenalty> ProximalPenalty for ErasedPathPenalty<P> {
    fn gap_safe(&self) -> Option<&dyn crate::GapSafePenalty> {
        self.0.gap_safe()
    }

    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        self.0.prox(input, step, output)
    }
}
impl<P: PathPenalty> PathPenalty for ErasedPathPenalty<P>
where
    P::AtStrength: 'static,
{
    type AtStrength = Box<dyn ProximalPenalty>;
    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.0.maximum_penalty(gradient)
    }
    fn path_anchor(&self, gradient: &[f64]) -> Result<f64, FitError> {
        self.0.path_anchor(gradient)
    }
    fn at_strength(&self, lambda: f64) -> Result<Self::AtStrength, FitError> {
        Ok(Box::new(self.0.at_strength(lambda)?))
    }
}

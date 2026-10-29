use crate::error::{finite, invalid};
use crate::{
    AxisPenalty, ChainFusedLasso, ClusteredLasso, ElasticNet, FitError, GapSafePenalty, L1,
    Penalty, PenaltyAxis, PenaltyMaximum, ProximalPenalty, Ridge, Slope,
};

/// A penalty shape that supplies thresholds and complete penalties along a path.
///
/// L1 and ridge replace their scalar strength. Elastic net retains its mixing
/// parameter; SLOPE multiplies its supplied base rank weights. Clustered and
/// chain fused lasso scale both supplied strengths, preserving their ratio.
/// Implementations
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

impl PathPenalty for ClusteredLasso {
    type AtStrength = Self;

    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.validate_input(gradient)?;
        max_abs(gradient)?;
        let mut maximum = 0.0_f64;
        for sign in [-1.0, 1.0] {
            let result = self.sorted_bound_indices(
                gradient.len(),
                |j| sign * gradient[j],
                |order| -> Result<PenaltyMaximum, FitError> {
                    let mut sum = 0.0;
                    for (rank, &j) in order.iter().enumerate() {
                        sum = finite(
                            sum + sign * gradient[j],
                            "summing clustered null gradients",
                            0,
                        )?;
                        let count = (rank + 1) as f64;
                        let outside = gradient.len() - rank - 1;
                        // Complete-graph cuts have k(p-k) fusion edges. Both signs
                        // are needed because fusion acts on signed differences.
                        let fusion = if outside == 0 {
                            0.0
                        } else {
                            self.fusion() * count * outside as f64
                        };
                        let capacity = finite(
                            self.l1() * count + fusion,
                            "computing clustered path capacities",
                            0,
                        )?;
                        if capacity == 0.0 {
                            if sum > 0.0 {
                                return Ok(PenaltyMaximum::Unbounded);
                            }
                        } else {
                            maximum = maximum.max(finite(
                                sum / capacity,
                                "computing the clustered maximum penalty",
                                0,
                            )?);
                        }
                    }
                    Ok(PenaltyMaximum::Finite(maximum))
                },
            )?;
            if result == PenaltyMaximum::Unbounded {
                return Ok(result);
            }
        }
        Ok(PenaltyMaximum::Finite(maximum))
    }

    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        self.validate_input(&[])?;
        L1::new(lambda).validate()?;
        Ok(Self::new(
            finite(lambda * self.l1(), "scaling clustered L1 strength", 0)?,
            finite(
                lambda * self.fusion(),
                "scaling clustered fusion strength",
                0,
            )?,
        ))
    }
}

impl PathPenalty for ChainFusedLasso {
    type AtStrength = Self;

    fn maximum_penalty(&self, gradient: &[f64]) -> Result<PenaltyMaximum, FitError> {
        self.validate_input(gradient)?;
        if max_abs(gradient)? == 0.0 {
            return Ok(PenaltyMaximum::Finite(0.0));
        }
        if self.l1() == 0.0 {
            let (mut sum, mut maximum) = (0.0, 0.0_f64);
            for (j, &g) in gradient.iter().enumerate() {
                sum = finite(sum + g, "summing pure-fusion null gradients", 0)?;
                let edge = if j + 1 == gradient.len() {
                    0.0
                } else {
                    finite(
                        self.fusion() * self.weight(j),
                        "computing pure-fusion edge capacity",
                        0,
                    )?
                };
                if edge == 0.0 {
                    if sum != 0.0 {
                        return Ok(PenaltyMaximum::Unbounded);
                    }
                    sum = 0.0;
                } else {
                    maximum = maximum.max(finite(
                        sum.abs() / edge,
                        "computing the pure-fusion maximum penalty",
                        0,
                    )?);
                }
            }
            return Ok(PenaltyMaximum::Finite(maximum));
        }
        let mut scaled = vec![0.0; gradient.len()];
        let errors = vec![0.0; gradient.len()];
        let mut feasible = |lambda: f64| -> Result<bool, FitError> {
            for (target, &g) in scaled.iter_mut().zip(gradient) {
                *target = g / lambda;
                if !target.is_finite() {
                    return Ok(false);
                }
            }
            self.dual_feasible(&scaled, &errors)
        };
        // Searching the complete dual polytope costs linear work per trial and
        // avoids enumerating every chain interval or allocating scaled terms.
        let mut upper = 1.0;
        let mut lower;
        if feasible(upper)? {
            loop {
                lower = upper * 0.5;
                if lower == 0.0 || !feasible(lower)? {
                    break;
                }
                upper = lower;
            }
        } else {
            loop {
                lower = upper;
                upper = (upper * 2.0).min(f64::MAX);
                if feasible(upper)? {
                    break;
                }
                if upper == f64::MAX {
                    return Err(FitError::NumericalFailure {
                        operation: "bracketing the chain maximum penalty",
                        iteration: 0,
                    });
                }
            }
        }
        for _ in 0..64 {
            let middle = lower + (upper - lower) * 0.5;
            if middle == lower || middle == upper {
                break;
            }
            if feasible(middle)? {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        Ok(PenaltyMaximum::Finite(upper))
    }

    fn at_strength(&self, lambda: f64) -> Result<Self, FitError> {
        self.validate_input(&vec![0.0; self.weights().map_or(0, |w| w.len() + 1)])?;
        L1::new(lambda).validate()?;
        let penalty = Self::new(
            finite(lambda * self.l1(), "scaling chain L1 strength", 0)?,
            finite(lambda * self.fusion(), "scaling chain fusion strength", 0)?,
        );
        Ok(if let Some(weights) = self.weights() {
            penalty.with_weights(weights.to_vec())
        } else {
            penalty
        })
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

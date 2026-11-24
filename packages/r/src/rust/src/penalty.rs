use std::rc::Rc;

use extendr_api::{Result, prelude::*};
use shrinkage::{
    AxisPenalty, ChainFusedLasso, ClusteredLasso, CoefficientLayout, DisjointGroups, ElasticNet,
    FitError, GapSafePenalty, GroupLasso, L1, Masked, PathPenalty, Penalty, PenaltyAxis,
    PenaltyMaximum, ProximalPenalty, RestrictedProximalPenalty, Ridge, Scaled, Slope,
    UncoveredPolicy,
};

use crate::result::PenaltyRecord;
use crate::{error, field};

pub(crate) fn string(list: &List, key: &str) -> Result<String> {
    field(list, key)?.try_into()
}

fn number(list: &List, key: &str) -> Result<f64> {
    field(list, key)?
        .as_real()
        .ok_or_else(|| error(format!("{key} must be a number")))
}

fn weights(list: &List, key: &str) -> Result<Vec<f64>> {
    let value = field(list, key)?;
    value
        .as_real_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| error(format!("{key} must contain numeric values")))
}

pub(crate) fn l1(p: &List) -> Result<L1> {
    Ok(L1::new(number(p, "lambda")?))
}

pub(crate) fn ridge(p: &List) -> Result<Ridge> {
    Ok(Ridge::new(number(p, "lambda")?))
}

pub(crate) fn elastic_net(p: &List) -> Result<ElasticNet> {
    Ok(ElasticNet::new(number(p, "l1")?, number(p, "l2")?))
}

pub(crate) fn clustered_lasso(p: &List) -> Result<ClusteredLasso> {
    Ok(ClusteredLasso::new(number(p, "l1")?, number(p, "fusion")?))
}

pub(crate) fn chain_fused_lasso(p: &List) -> Result<ChainFusedLasso> {
    let penalty = ChainFusedLasso::new(number(p, "l1")?, number(p, "fusion")?);
    if field(p, "weights")?.is_null() {
        Ok(penalty)
    } else {
        Ok(penalty.with_weights(weights(p, "weights")?))
    }
}

pub(crate) fn slope(p: &List) -> Result<Slope> {
    Ok(Slope::new(weights(p, "weights")?))
}

trait PathShape {
    fn maximum_penalty(&self, gradient: &[f64]) -> std::result::Result<PenaltyMaximum, FitError>;
    fn path_anchor(&self, gradient: &[f64]) -> std::result::Result<f64, FitError>;
    fn at_strength(&self, lambda: f64) -> std::result::Result<Proximal, FitError>;
}

impl<P> PathShape for P
where
    P: PathPenalty + PenaltyRecord + 'static,
    P::AtStrength: PenaltyRecord + 'static,
{
    fn maximum_penalty(&self, gradient: &[f64]) -> std::result::Result<PenaltyMaximum, FitError> {
        PathPenalty::maximum_penalty(self, gradient)
    }

    fn path_anchor(&self, gradient: &[f64]) -> std::result::Result<f64, FitError> {
        PathPenalty::path_anchor(self, gradient)
    }

    fn at_strength(&self, lambda: f64) -> std::result::Result<Proximal, FitError> {
        let penalty = PathPenalty::at_strength(self, lambda)?;
        let spec = penalty.record();
        let lambda = penalty.lambda();
        Ok(Proximal {
            lambda,
            ..Proximal::fixed(penalty, spec)
        })
    }
}

/// A complete proximal penalty that can be cloned into training folds.
#[derive(Clone)]
pub(crate) struct Proximal {
    inner: Rc<dyn ProximalPenalty>,
    spec: List,
    lambda: Option<f64>,
    path: Option<Rc<dyn PathShape>>,
}

impl Proximal {
    fn fixed(penalty: impl ProximalPenalty + 'static, spec: List) -> Self {
        Self {
            inner: Rc::new(penalty),
            spec,
            lambda: None,
            path: None,
        }
    }

    fn scalable<P>(penalty: P) -> Self
    where
        P: PathPenalty + PenaltyRecord + 'static,
        P::AtStrength: PenaltyRecord + 'static,
    {
        let spec = penalty.record();
        let lambda = penalty.lambda();
        let penalty = Rc::new(penalty);
        Self {
            inner: penalty.clone(),
            spec,
            lambda,
            path: Some(penalty),
        }
    }

    fn path_shape(&self) -> std::result::Result<&dyn PathShape, FitError> {
        self.path.as_deref().ok_or_else(|| FitError::InvalidInput {
            message: "this penalty requires explicit path candidates".into(),
        })
    }

    pub(crate) fn parse(spec: &List, ncoefficients: usize) -> Result<Self> {
        Ok(match string(spec, "name")?.as_str() {
            "l1" => Self::scalable(l1(spec)?),
            "ridge" => Self::scalable(ridge(spec)?),
            "elastic_net" => Self::scalable(elastic_net(spec)?),
            "clustered_lasso" => Self::scalable(clustered_lasso(spec)?),
            "chain_fused_lasso" => Self::scalable(chain_fused_lasso(spec)?),
            "slope" => Self::scalable(slope(spec)?),
            "group_lasso" => {
                let groups = List::try_from(field(spec, "groups")?)?
                    .values()
                    .map(|group| {
                        group
                            .as_integer_slice()
                            .ok_or_else(|| error("groups must contain integer indices"))?
                            .iter()
                            .map(|&index| {
                                usize::try_from(index)
                                    .ok()
                                    .and_then(|index| index.checked_sub(1))
                                    .ok_or_else(|| error("group indices must be positive"))
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let uncovered = match string(spec, "uncovered")?.as_str() {
                    "reject" => UncoveredPolicy::Reject,
                    "unpenalized" => UncoveredPolicy::Unpenalized,
                    _ => return Err(error("unknown group coverage policy")),
                };
                let groups = DisjointGroups::new(
                    ncoefficients,
                    groups,
                    weights(spec, "weights")?,
                    uncovered,
                )
                .map_err(error)?;
                Self::fixed(
                    GroupLasso::new(number(spec, "lambda")?, groups),
                    spec.clone(),
                )
            }
            "scaled" => {
                let inner = List::try_from(field(spec, "penalty")?)?;
                let penalty = Self::parse(&inner, ncoefficients)?;
                Self::fixed(Scaled::new(number(spec, "scale")?, penalty), spec.clone())
            }
            "masked" => {
                let unpenalized = field(spec, "unpenalized")?;
                let mask: Vec<bool> = unpenalized.try_into()?;
                let inner = List::try_from(field(spec, "penalty")?)?;
                let penalty: Box<dyn RestrictedProximalPenalty> =
                    match string(&inner, "name")?.as_str() {
                        "l1" => Box::new(l1(&inner)?),
                        "ridge" => Box::new(ridge(&inner)?),
                        "elastic_net" => Box::new(elastic_net(&inner)?),
                        _ => {
                            return Err(error(
                                "masking requires an L1, ridge, or elastic-net penalty",
                            ));
                        }
                    };
                Self::fixed(Masked::new(mask, penalty), spec.clone())
            }
            "axis" => return Err(error("an axis penalty requires a multivariate loss")),
            name => return Err(error(format!("unknown penalty '{name}'"))),
        })
    }
}

impl PenaltyRecord for Proximal {
    fn record(&self) -> List {
        self.spec.clone()
    }

    fn lambda(&self) -> Option<f64> {
        self.lambda
    }
}

impl Penalty for Proximal {
    type Error = FitError;

    fn value(&self, beta: &[f64]) -> std::result::Result<f64, FitError> {
        self.inner.value(beta)
    }
}

impl ProximalPenalty for Proximal {
    fn gap_safe(&self) -> Option<&dyn GapSafePenalty> {
        self.inner.gap_safe()
    }

    fn prox(
        &self,
        input: &[f64],
        step: f64,
        output: &mut [f64],
    ) -> std::result::Result<(), FitError> {
        self.inner.prox(input, step, output)
    }
}

impl PathPenalty for Proximal {
    type AtStrength = Self;

    fn maximum_penalty(&self, gradient: &[f64]) -> std::result::Result<PenaltyMaximum, FitError> {
        self.path_shape()?.maximum_penalty(gradient)
    }

    fn path_anchor(&self, gradient: &[f64]) -> std::result::Result<f64, FitError> {
        self.path_shape()?.path_anchor(gradient)
    }

    fn at_strength(&self, lambda: f64) -> std::result::Result<Self, FitError> {
        self.path_shape()?.at_strength(lambda)
    }
}

pub(crate) fn axis(spec: &List, layout: CoefficientLayout) -> Result<AxisPenalty<Proximal>> {
    let (axis, inner) = if string(spec, "name")? == "axis" {
        let axis = match string(spec, "axis")?.as_str() {
            "all" => PenaltyAxis::All,
            "features" => PenaltyAxis::Features,
            "responses" => PenaltyAxis::Responses,
            _ => return Err(error("unknown penalty axis")),
        };
        (axis, List::try_from(field(spec, "penalty")?)?)
    } else {
        (PenaltyAxis::Features, spec.clone())
    };
    let dimension = match axis {
        PenaltyAxis::All => layout.len(),
        PenaltyAxis::Features => layout.nfeatures(),
        PenaltyAxis::Responses => layout.nresponses(),
    };
    Ok(AxisPenalty::new(
        layout,
        axis,
        Proximal::parse(&inner, dimension)?,
    ))
}

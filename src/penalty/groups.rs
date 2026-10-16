//! Disjoint coefficient groups with explicit weights and coverage rules.

use crate::FitError;
use crate::error::invalid;

/// How disjoint group definitions handle coefficients outside every group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UncoveredPolicy {
    /// Require every coefficient to belong to exactly one group.
    Reject,
    /// Leave coefficients outside the groups unpenalized.
    Unpenalized,
}

/// Validated disjoint groups of optimization-scale coefficient indices.
///
/// Groups and their members retain the supplied order. Indices address a vector
/// of `ncoefficients` entries, excluding intercepts. For a flattened coefficient
/// matrix, use [`crate::CoefficientLayout::index`] to respect its response-major
/// layout. Each coefficient belongs to at most one group.
///
/// Supply one finite, nonnegative weight per group. Weights are multipliers of
/// group penalty values, stored without adjustment for group size or count.
/// A zero weight still counts as coverage and must obey all membership rules.
/// The coverage policy explicitly requires full coverage or leaves uncovered
/// coefficients unpenalized. All definitions are checked at construction and
/// exposed through immutable slices.
///
/// ```
/// use shrinkage::{DisjointGroups, UncoveredPolicy};
///
/// let groups = DisjointGroups::new(
///     5,
///     vec![vec![3, 1], vec![0]],
///     vec![0.5, 2.0],
///     UncoveredPolicy::Unpenalized,
/// )?;
/// assert_eq!(groups.groups(), [vec![3, 1], vec![0]]);
/// assert_eq!(groups.weights(), [0.5, 2.0]);
/// assert_eq!(groups.uncovered(), [2, 4]);
/// # Ok::<(), shrinkage::FitError>(())
/// ```
#[derive(Clone, Debug)]
pub struct DisjointGroups {
    ncoefficients: usize,
    groups: Vec<Vec<usize>>,
    weights: Vec<f64>,
    uncovered_policy: UncoveredPolicy,
    uncovered: Vec<usize>,
}

impl DisjointGroups {
    /// Validate and own weighted groups over a fixed coefficient vector.
    ///
    /// Individual groups must be nonempty. An empty collection is valid for
    /// zero coefficients under either policy, or for any coefficient count
    /// under [`UncoveredPolicy::Unpenalized`].
    ///
    /// # Errors
    /// Reject a weight count different from the group count, nonfinite or
    /// negative weights, empty groups, repeated indices within a group,
    /// overlapping groups (including duplicate groups), or indices outside
    /// `0..ncoefficients`. [`UncoveredPolicy::Reject`] also rejects any
    /// coefficient omitted from the groups. Errors identify group and
    /// coefficient indices using zero-based numbering when applicable.
    pub fn new(
        ncoefficients: usize,
        groups: Vec<Vec<usize>>,
        weights: Vec<f64>,
        uncovered_policy: UncoveredPolicy,
    ) -> Result<Self, FitError> {
        if weights.len() != groups.len() {
            return Err(invalid("group weights must have one entry per group"));
        }
        let mut owners = vec![None; ncoefficients];
        for (group_index, (group, &weight)) in groups.iter().zip(&weights).enumerate() {
            if group.is_empty() {
                return Err(invalid(format!("group {group_index} must not be empty")));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(invalid(format!(
                    "weight for group {group_index} must be finite and nonnegative"
                )));
            }
            for &coefficient in group {
                if coefficient >= ncoefficients {
                    return Err(invalid(format!(
                        "coefficient {coefficient} in group {group_index} is out of range for {ncoefficients} coefficients"
                    )));
                }
                if let Some(previous) = owners[coefficient].replace(group_index) {
                    return Err(invalid(if previous == group_index {
                        format!(
                            "coefficient {coefficient} appears more than once in group {group_index}"
                        )
                    } else {
                        format!(
                            "coefficient {coefficient} is shared by groups {previous} and {group_index}"
                        )
                    }));
                }
            }
        }
        let mut uncovered = Vec::new();
        for (coefficient, owner) in owners.iter().enumerate() {
            if owner.is_none() {
                if uncovered_policy == UncoveredPolicy::Reject {
                    return Err(invalid(format!(
                        "coefficient {coefficient} is not covered by any group"
                    )));
                }
                uncovered.push(coefficient);
            }
        }
        Ok(Self {
            ncoefficients,
            groups,
            weights,
            uncovered_policy,
            uncovered,
        })
    }

    /// Coefficient vector length, excluding intercepts.
    pub fn ncoefficients(&self) -> usize {
        self.ncoefficients
    }

    /// Disjoint index vectors in their supplied group and member order.
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// One finite, nonnegative weight per group, in group order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Policy governing coefficients outside every group.
    pub fn uncovered_policy(&self) -> UncoveredPolicy {
        self.uncovered_policy
    }

    /// Uncovered coefficient indices in increasing order.
    ///
    /// Always empty under [`UncoveredPolicy::Reject`]. Under
    /// [`UncoveredPolicy::Unpenalized`], these coefficients remain unpenalized.
    pub fn uncovered(&self) -> &[usize] {
        &self.uncovered
    }
}

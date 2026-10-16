//! Validated disjoint membership, explicit weights, and coverage policies.

use shrinkage::{CoefficientLayout, DisjointGroups, FitError, UncoveredPolicy};

fn invalid(result: Result<DisjointGroups, FitError>, fragments: &[&str]) {
    let FitError::InvalidInput { message } = result.unwrap_err() else {
        panic!("expected an invalid-input error");
    };
    for fragment in fragments {
        assert!(message.contains(fragment), "{message:?} lacks {fragment:?}");
    }
}

#[test]
fn definitions_preserve_group_order_member_order_and_explicit_weights() {
    let members = vec![vec![4, 0, 2], vec![5], vec![1, 3]];
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for weights in [
            vec![0.5, 3.0, 2.0],
            vec![0.0, f64::MAX, 1.0],
            vec![-0.0, f64::MIN_POSITIVE, f64::from_bits(1)],
        ] {
            let groups = DisjointGroups::new(6, members.clone(), weights.clone(), policy).unwrap();
            assert_eq!(groups.ncoefficients(), 6);
            assert_eq!(groups.groups(), members);
            assert_eq!(groups.weights(), weights);
            assert!(groups.uncovered().is_empty());
            assert_eq!(groups.uncovered_policy(), policy);
            let cloned = groups.clone();
            drop(groups);
            assert_eq!(cloned.groups(), members);
            assert_eq!(cloned.weights(), weights);
        }
    }
}

#[test]
fn uncovered_coefficients_follow_the_explicit_policy() {
    let members = vec![vec![4, 1], vec![2]];
    let weights = vec![2.0, 0.0];
    invalid(
        DisjointGroups::new(6, members.clone(), weights.clone(), UncoveredPolicy::Reject),
        &["coefficient 0", "not covered"],
    );
    let groups = DisjointGroups::new(
        6,
        members.clone(),
        weights.clone(),
        UncoveredPolicy::Unpenalized,
    )
    .unwrap();
    assert_eq!(groups.ncoefficients(), 6);
    assert_eq!(groups.groups(), members);
    assert_eq!(groups.weights(), weights);
    assert_eq!(groups.uncovered_policy(), UncoveredPolicy::Unpenalized);
    assert_eq!(groups.uncovered(), [0, 3, 5]);

    for omitted in 0..6 {
        let members: Vec<_> = (0..6).rev().filter(|&index| index != omitted).collect();
        invalid(
            DisjointGroups::new(6, vec![members.clone()], vec![0.0], UncoveredPolicy::Reject),
            &[&format!("coefficient {omitted}"), "not covered"],
        );
        let groups =
            DisjointGroups::new(6, vec![members], vec![0.0], UncoveredPolicy::Unpenalized).unwrap();
        assert_eq!(groups.uncovered(), [omitted]);
    }
}

#[test]
fn empty_collections_and_zero_coefficient_dimensions_follow_coverage_policy() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        let groups = DisjointGroups::new(0, vec![], vec![], policy).unwrap();
        assert_eq!(groups.ncoefficients(), 0);
        assert!(groups.groups().is_empty());
        assert!(groups.weights().is_empty());
        assert!(groups.uncovered().is_empty());
        assert_eq!(groups.uncovered_policy(), policy);
    }
    let groups = DisjointGroups::new(3, vec![], vec![], UncoveredPolicy::Unpenalized).unwrap();
    assert_eq!(groups.ncoefficients(), 3);
    assert!(groups.groups().is_empty());
    assert!(groups.weights().is_empty());
    assert_eq!(groups.uncovered(), [0, 1, 2]);
    invalid(
        DisjointGroups::new(3, vec![], vec![], UncoveredPolicy::Reject),
        &["coefficient 0", "not covered"],
    );
}

#[test]
fn weights_require_one_finite_nonnegative_value_per_group() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for weights in [vec![], vec![1.0], vec![1.0, 2.0, 3.0]] {
            invalid(
                DisjointGroups::new(2, vec![vec![0], vec![1]], weights, policy),
                &["one", "per group"],
            );
        }
        invalid(
            DisjointGroups::new(0, vec![], vec![1.0], policy),
            &["one", "per group"],
        );
        for weight in [
            -1.0,
            -f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for weights in [vec![weight, 1.0], vec![1.0, weight]] {
                invalid(
                    DisjointGroups::new(2, vec![vec![0], vec![1]], weights, policy),
                    &["weight", "finite", "nonnegative"],
                );
            }
        }
    }
}

#[test]
fn individual_empty_groups_are_rejected_even_at_zero_weight() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for (ncoefficients, members, weights, group_index) in [
            (0, vec![vec![]], vec![1.0], 0),
            (1, vec![vec![0], vec![]], vec![1.0, 0.0], 1),
            (1, vec![vec![], vec![0]], vec![0.0, 1.0], 0),
        ] {
            invalid(
                DisjointGroups::new(ncoefficients, members, weights, policy),
                &[&format!("group {group_index}"), "empty"],
            );
        }
    }
}

#[test]
fn repeated_members_and_duplicate_groups_are_rejected() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for weight in [0.0, 1.0] {
            for members in [vec![vec![0, 0]], vec![vec![1, 0, 1]]] {
                invalid(
                    DisjointGroups::new(2, members, vec![weight], policy),
                    &["coefficient", "group 0", "more than once"],
                );
            }
            for members in [vec![vec![0, 1], vec![0, 1]], vec![vec![0, 1], vec![1, 0]]] {
                invalid(
                    DisjointGroups::new(2, members, vec![weight; 2], policy),
                    &["coefficient", "groups 0 and 1"],
                );
            }
        }
    }
}

#[test]
fn partially_overlapping_and_nonadjacent_groups_are_rejected() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for weight in [0.0, 1.0] {
            invalid(
                DisjointGroups::new(
                    4,
                    vec![vec![2, 0], vec![1], vec![3, 2]],
                    vec![1.0, 1.0, weight],
                    policy,
                ),
                &["coefficient 2", "groups 0 and 2"],
            );
        }
    }
}

#[test]
fn out_of_range_members_are_rejected_even_at_zero_weight() {
    for policy in [UncoveredPolicy::Reject, UncoveredPolicy::Unpenalized] {
        for weight in [0.0, 1.0] {
            for (ncoefficients, index) in [(0, 0), (2, 2), (2, usize::MAX)] {
                invalid(
                    DisjointGroups::new(ncoefficients, vec![vec![index]], vec![weight], policy),
                    &[&format!("coefficient {index}"), "group 0", "out of range"],
                );
            }
            invalid(
                DisjointGroups::new(2, vec![vec![0, 1], vec![2]], vec![1.0, weight], policy),
                &["coefficient 2", "group 1", "out of range"],
            );
        }
    }
}

#[test]
fn multivariate_members_use_the_explicit_coefficient_layout() {
    let layout = CoefficientLayout::new(2, 3).unwrap();
    let members: Vec<Vec<_>> = (0..layout.nfeatures())
        .map(|feature| {
            (0..layout.nresponses())
                .map(|response| layout.index(feature, response).unwrap())
                .collect()
        })
        .collect();
    let groups = DisjointGroups::new(
        layout.len(),
        members,
        vec![1.0, 2.0],
        UncoveredPolicy::Reject,
    )
    .unwrap();
    assert_eq!(groups.ncoefficients(), 6);
    assert_eq!(groups.groups(), [vec![0, 2, 4], vec![1, 3, 5]]);
    assert!(groups.uncovered().is_empty());
}

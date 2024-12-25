/// Represents a set of imbalanced nodes, with negative and positive imbalances.
#[derive(Debug)]
pub(crate) struct ImbalancedNodeSet {
    pub(crate) negative: Vec<usize>,
    pub(crate) positive: Vec<usize>,
}

impl ImbalancedNodeSet {
    /// Checks if the set of imbalanced nodes is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.negative.is_empty() && self.positive.is_empty()
    }
}

#[test]
fn test_imbalanced_node_set_is_empty() {
    let empty_set = ImbalancedNodeSet {
        negative: Vec::new(),
        positive: Vec::new(),
    };
    assert!(empty_set.is_empty());

    let non_empty_set = ImbalancedNodeSet {
        negative: vec![1],
        positive: Vec::new(),
    };
    assert!(!non_empty_set.is_empty());
}

use ndarray::ArrayView2;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres_min, prelude::Matrix};

use crate::graph::ImbalancedNodeSet;

#[derive(Debug)]
pub(super) struct Matching {
    pub(super) from: usize,
    pub(super) to: usize,
}

pub(super) fn best_match(
    imbalanced_nodes: &ImbalancedNodeSet,
    shortest_distance_between_nodes: ArrayView2<f64>,
) -> Vec<Matching> {
    let mut matching = Vec::new();
    let weights = shortest_distances_between_imbalanced_nodes(
        imbalanced_nodes,
        shortest_distance_between_nodes,
    );
    let (_, best_match) = kuhn_munkres_min(&weights);
    for (from, to) in imbalanced_nodes.negative_difference_nodes.iter().zip(
        best_match
            .iter()
            .map(|x| imbalanced_nodes.positive_difference_nodes[*x]),
    ) {
        matching.push(Matching { from: *from, to });
    }
    matching
}

fn shortest_distances_between_imbalanced_nodes(
    imbalanced_nodes: &ImbalancedNodeSet,
    shortest_distance_between_nodes: ArrayView2<f64>,
) -> Matrix<OrderedFloat<f64>> {
    Matrix::from_fn(
        imbalanced_nodes.negative_difference_nodes.len(),
        imbalanced_nodes.positive_difference_nodes.len(),
        |(i, j)| {
            let from = imbalanced_nodes.negative_difference_nodes[i];
            let to = imbalanced_nodes.positive_difference_nodes[j];
            OrderedFloat(shortest_distance_between_nodes[(from, to)])
        },
    )
}

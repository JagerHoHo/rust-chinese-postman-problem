use std::collections::HashMap;

use ndarray::ArrayView2;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres_min, prelude::Matrix};

use crate::graph::ImbalancedNodeSet;

pub(super) fn best_match(
    imbalanced_nodes: &ImbalancedNodeSet,
    shortest_distance_between_nodes: ArrayView2<f64>,
) -> HashMap<usize, usize> {
    let weights = shortest_distances_between_imbalanced_nodes(
        imbalanced_nodes,
        shortest_distance_between_nodes,
    );
    let (_, best_match) = kuhn_munkres_min(&weights);
    let mut matching = HashMap::new();
    for (from, to) in imbalanced_nodes
        .negative_difference_nodes
        .iter()
        .zip(best_match.iter())
    {
        matching.insert(*from, imbalanced_nodes.positive_difference_nodes[*to]);
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
            let node = imbalanced_nodes.negative_difference_nodes[i];
            let other_node = imbalanced_nodes.positive_difference_nodes[j];
            OrderedFloat(shortest_distance_between_nodes[(node, other_node)])
        },
    )
}

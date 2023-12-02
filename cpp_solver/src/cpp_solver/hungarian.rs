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
    let weights = shortest_distances_between_imbalanced_nodes(
        imbalanced_nodes,
        shortest_distance_between_nodes,
    );
    let (_, best_match) = kuhn_munkres_min(&weights);
    imbalanced_nodes
        .negative
        .iter()
        .zip(best_match.iter().map(|&x| imbalanced_nodes.positive[x]))
        .map(|(&from, to)| Matching { from, to })
        .collect()
}

fn shortest_distances_between_imbalanced_nodes(
    imbalanced_nodes: &ImbalancedNodeSet,
    shortest_distance_between_nodes: ArrayView2<f64>,
) -> Matrix<OrderedFloat<f64>> {
    Matrix::from_fn(
        imbalanced_nodes.negative.len(),
        imbalanced_nodes.positive.len(),
        |(i, j)| {
            let from = imbalanced_nodes.negative[i];
            let to = imbalanced_nodes.positive[j];
            OrderedFloat(shortest_distance_between_nodes[(from, to)])
        },
    )
}

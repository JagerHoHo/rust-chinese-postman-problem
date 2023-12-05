use ndarray::ArrayView2;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres_min, prelude::Matrix};

use crate::graph::ImbalancedNodeSet;

/// Represents a matching between two nodes.
#[derive(Debug)]
pub(super) struct Matching {
    pub(super) from: usize,
    pub(super) to: usize,
}

/// Finds the best match between imbalanced nodes based on the shortest distance between them.
///
/// # Arguments
///
/// * `imbalanced_nodes` - The set of imbalanced nodes.
/// * `shortest_distance_between_nodes` - The shortest distance between nodes represented as a 2D array.
///
/// # Returns
///
/// A vector of `Matching` structs representing the best match between imbalanced nodes.
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

/// Calculates the shortest distances between imbalanced nodes based on the shortest distance between all nodes.
///
/// # Arguments
///
/// * `imbalanced_nodes` - The set of imbalanced nodes.
/// * `shortest_distance_between_nodes` - The shortest distance between nodes represented as a 2D array.
///
/// # Returns
///
/// A matrix representing the shortest distances between imbalanced nodes.
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

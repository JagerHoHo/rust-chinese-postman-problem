use ndarray::{Array2, ArrayView2};
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres_min, prelude::Matrix};

use crate::graph::ImbalancedNodeSet;

#[derive(Debug)]
pub(crate) struct Matching {
    pub(crate) from: usize,
    pub(crate) to: usize,
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
    let weights = Matrix::from_rows(
        weights
            .view()
            .rows()
            .into_iter()
            .map(|row| row.into_iter().copied().collect::<Vec<_>>()),
    )
    .unwrap();
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
) -> Array2<OrderedFloat<f64>> {
    Array2::from_shape_fn(
        (
            imbalanced_nodes.negative_difference_nodes.len(),
            imbalanced_nodes.positive_difference_nodes.len(),
        ),
        |(i, j)| {
            let from = imbalanced_nodes.negative_difference_nodes[i];
            let to = imbalanced_nodes.positive_difference_nodes[j];
            OrderedFloat(shortest_distance_between_nodes[(from, to)])
        },
    )
}

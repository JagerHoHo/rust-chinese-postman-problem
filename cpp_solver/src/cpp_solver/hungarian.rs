use std::collections::{BinaryHeap, HashMap, VecDeque};

use ndarray::{Array2, ArrayView2, Axis};
use ndarray_stats::QuantileExt;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres_min, prelude::Matrix};

use crate::graph::ImbalancedNodeSet;

#[derive(PartialEq, Eq, Debug)]
struct ShortestFirst {
    row: usize,
    row_min: OrderedFloat<f64>,
}

impl PartialOrd for ShortestFirst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShortestFirst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.row_min.cmp(&self.row_min)
    }
}

pub(super) fn best_match(
    imbalanced_nodes: &ImbalancedNodeSet,
    shortest_distance_between_nodes: ArrayView2<f64>,
) -> HashMap<usize, usize> {
    let mut matching = HashMap::new();
    let mut weights = shortest_distances_between_imbalanced_nodes(
        imbalanced_nodes,
        shortest_distance_between_nodes,
    );
    let mut neg_nodes = VecDeque::from_iter(&imbalanced_nodes.negative_difference_nodes);
    if weights.nrows() > weights.ncols() {
        greedy_matching(
            &mut weights,
            &mut matching,
            imbalanced_nodes,
            &mut neg_nodes,
        );
    }
    let weights = Matrix::from_rows(
        weights
            .view()
            .rows()
            .into_iter()
            .map(|row| row.into_iter().copied().collect::<Vec<_>>()),
    )
    .unwrap();
    let (_, best_match) = kuhn_munkres_min(&weights);
    for (from, to) in neg_nodes.iter().zip(best_match.iter()) {
        matching.insert(**from, imbalanced_nodes.positive_difference_nodes[*to]);
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

fn greedy_matching(
    weights: &mut Array2<OrderedFloat<f64>>,
    matching: &mut HashMap<usize, usize>,
    imbalanced_nodes: &ImbalancedNodeSet,
    neg_nodes: &mut VecDeque<&usize>,
) {
    let in_out_diff = weights.nrows() - weights.ncols();
    let mut heap = BinaryHeap::new();
    for (i, row) in weights.rows().into_iter().enumerate() {
        let row_min = *row.min().unwrap();
        heap.push(ShortestFirst { row: i, row_min });
    }
    let mut del_rows = Vec::new();
    for _ in 0..in_out_diff {
        let ShortestFirst { mut row, .. } = heap.pop().unwrap();
        for del_row in &del_rows {
            if row > *del_row {
                row -= 1;
            }
        }
        del_rows.push(row);
        matching.insert(
            imbalanced_nodes.negative_difference_nodes[row],
            imbalanced_nodes.positive_difference_nodes[weights.row(row).argmin().unwrap()],
        );
        neg_nodes.remove(row);
        weights.remove_index(Axis(0), row);
    }
}

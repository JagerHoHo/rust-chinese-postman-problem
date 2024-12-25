mod edge;
mod graph_builder;
mod imbalanced_nodeset;
use edge::Edge;

pub use graph_builder::GraphBuilder;
pub(crate) use imbalanced_nodeset::ImbalancedNodeSet;

use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

/// Represents a graph, with weight matrix, out degrees, edge count, and node labels.
pub struct Graph {
    weight_matrix: Array2<f64>,
    node_labels: Vec<String>,
    edge_counts: HashMap<(usize, usize), usize>,
    out_degrees: Array1<usize>,
}

impl Graph {
    /// Constructs a new Graph from a weight matrix.
    pub fn new(weight_matrix: Array2<f64>, node_labels: Vec<String>) -> Self {
        let out_degrees = weight_matrix
            .rows()
            .into_iter()
            .map(|row| row.iter().filter(|&&x| x != f64::INFINITY).count())
            .collect();
        let edge_counts = Self::compute_edge_counts(&weight_matrix);
        Self {
            weight_matrix,
            node_labels,
            edge_counts,
            out_degrees,
        }
    }

    pub fn from_weight_matrix(
        weight_matrix: Array2<f64>,
        node_labels: Option<Vec<String>>,
    ) -> Self {
        let out_degrees = weight_matrix
            .rows()
            .into_iter()
            .map(|row| row.iter().filter(|&&x| x != f64::INFINITY).count())
            .collect();
        let edge_counts = Self::compute_edge_counts(&weight_matrix);

        // If no labels are provided, generate default numeric labels
        let labels = node_labels
            .unwrap_or_else(|| (0..weight_matrix.nrows()).map(|i| i.to_string()).collect());

        Self {
            weight_matrix,
            node_labels: labels,
            edge_counts,
            out_degrees,
        }
    }

    /// Computes edge counts from a weight matrix.
    fn compute_edge_counts(weight_matrix: &Array2<f64>) -> HashMap<(usize, usize), usize> {
        let mut counts = HashMap::new();
        for (i, row) in weight_matrix.rows().into_iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                if weight != f64::INFINITY {
                    *counts.entry((i, j)).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Adds an edge to the graph with a weight.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.weight_matrix[[from, to]] = weight;
        self.out_degrees[from] += 1;
        *self.edge_counts.entry((from, to)).or_insert(0) += 1;
    }

    /// Returns the in-degrees of the nodes.
    pub fn in_degrees(&self) -> Array1<usize> {
        self.weight_matrix
            .t()
            .rows()
            .into_iter()
            .map(|row| row.iter().filter(|&&x| x != f64::INFINITY).count())
            .collect()
    }

    /// Returns the out-degrees of the nodes.
    pub fn out_degrees(&self) -> Array1<usize> {
        self.out_degrees.clone()
    }

    /// Retrieves the edge set in a sparse representation.
    pub fn edge_set(&self) -> Vec<Vec<usize>> {
        let mut edge_set = vec![Vec::new(); self.weight_matrix.nrows()];
        for (&(from, to), &count) in &self.edge_counts {
            edge_set[from].extend(vec![to; count]);
        }
        edge_set
    }

    /// Returns the set of imbalanced nodes in the graph.
    pub(crate) fn imbalanced_nodes(&self) -> ImbalancedNodeSet {
        let mut negative_difference_nodes = Vec::new();
        let mut positive_difference_nodes = Vec::new();

        for (node, (row, col)) in self
            .weight_matrix
            .rows()
            .into_iter()
            .zip(self.weight_matrix.columns())
            .enumerate()
        {
            match Graph::out_in_diff(&row, &col) {
                x if x > 0 => positive_difference_nodes.extend(vec![node; x as usize]),
                x if x < 0 => negative_difference_nodes.extend(vec![node; (-x) as usize]),
                _ => (),
            }
        }
        ImbalancedNodeSet {
            negative: negative_difference_nodes,
            positive: positive_difference_nodes,
        }
    }

    /// Calculates the out-in degree difference of a node.
    fn out_in_diff(row: &ArrayView1<f64>, col: &ArrayView1<f64>) -> isize {
        let out_degree = row.iter().filter(|&&x| x != f64::INFINITY).count();
        let in_degree = col.iter().filter(|&&x| x != f64::INFINITY).count();
        out_degree as isize - in_degree as isize
    }

    /// Relabels the nodes in the graph with the given labels.
    pub fn relabel(&mut self, node_labels: Option<Vec<String>>) {
        self.node_labels = node_labels.unwrap_or_else(|| {
            (0..self.weight_matrix.nrows())
                .map(|i| i.to_string())
                .collect()
        });
    }

    /// Returns the weight matrix (for debugging or advanced usage).
    pub fn weight_matrix(&self) -> &Array2<f64> {
        &self.weight_matrix
    }

    /// Returns the node labels (for debugging or advanced usage).
    pub fn node_labels(&self) -> &[String] {
        &self.node_labels
    }
}

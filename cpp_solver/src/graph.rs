use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView1};

/// Represents a set of imbalanced nodes, with negative and positive imbalances.
#[derive(Debug)]
pub(crate) struct ImbalancedNodeSet {
    pub(crate) negative: Vec<usize>,
    pub(crate) positive: Vec<usize>,
}

/// Represents an edge in the graph, with a source node, target node, and weight.
struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

/// Builder for constructing a graph.
pub struct GraphBuilder {
    edges: Vec<Edge>,
    max_node: usize,
    node_labels: HashMap<String, usize>,
    used_labels: HashSet<String>,
}

/// Represents a graph, with weight matrix, out degrees, edge count, and node labels.
pub struct Graph {
    pub(crate) weight_matrix: Array2<f64>,
    pub(crate) node_labels: Vec<String>,
    edge_counts: HashMap<(usize, usize), usize>, // More efficient edge storage
    out_degrees: Array1<usize>,
}

impl ImbalancedNodeSet {
    /// Checks if the set of imbalanced nodes is empty.
    pub(crate) fn empty(&self) -> bool {
        self.negative.is_empty() && self.positive.is_empty()
    }
}

impl GraphBuilder {
    /// Creates a new instance of `GraphBuilder`.
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            max_node: 0,
            node_labels: HashMap::new(),
            used_labels: HashSet::new(),
        }
    }

    /// Adds an edge to the graph using numeric indices.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> &mut Self {
        self.max_node = self.max_node.max(from).max(to);
        self.edges.push(Edge { from, to, weight });
        self
    }

    /// Adds an edge to the graph using labeled nodes.
    pub fn add_labeled_edge(&mut self, from_label: &str, to_label: &str, weight: f64) -> &mut Self {
        let from = self.get_or_insert_label(from_label);
        let to = self.get_or_insert_label(to_label);
        self.add_edge(from, to, weight)
    }

    /// Builds the graph from the added edges.
    pub fn build(self) -> Graph {
        let n_nodes = if self.max_node > 0 {
            self.max_node + 1
        } else {
            0
        };

        // Create a weight matrix initialized to infinity
        let mut weight_matrix = Array2::from_elem((n_nodes, n_nodes), f64::INFINITY);

        // Populate the weight matrix with edges
        for Edge { from, to, weight } in self.edges {
            weight_matrix[[from, to]] = weight;
        }

        // Convert node labels map to a sorted vector
        let mut node_labels = vec![String::new(); self.node_labels.len()];
        for (label, &index) in &self.node_labels {
            node_labels[index] = label.clone();
        }
        let node_labels = if !node_labels.is_empty() {
            Some(node_labels)
        } else {
            None
        };

        // Build the graph using from_weight_matrix
        Graph::from_weight_matrix(weight_matrix, node_labels)
    }

    /// Retrieves or inserts a label into the `node_labels` map.
    fn get_or_insert_label(&mut self, label: &str) -> usize {
        if let Some(&index) = self.node_labels.get(label) {
            index
        } else {
            let index = self.node_labels.len();
            self.node_labels.insert(label.to_string(), index);
            self.used_labels.insert(label.to_string());
            index
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
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
}

/// Test that an empty graph is correctly initialized.
#[test]
fn test_empty_graph() {
    let builder = GraphBuilder::new();
    let graph = builder.build();
    let weight_matrix = &graph.weight_matrix;

    // Ensure the matrix dimensions are zero or all entries are `f64::INFINITY`
    assert!(
        weight_matrix.is_empty(),
        "Weight matrix should be empty or all entries should be `f64::INFINITY`"
    );

    // Check that no node labels are present
    assert!(
        graph.node_labels.is_empty(),
        "Node labels should be empty for an empty graph"
    );
}

/// Test adding a labeled edge to the graph and verifying the matrix and labels.
#[test]
fn test_add_labeled_edge() {
    let mut builder = GraphBuilder::new();
    builder.add_labeled_edge("A", "B", 5.0);
    let graph = builder.build();
    assert_eq!(graph.weight_matrix[[0, 1]], 5.0);
    assert_eq!(graph.node_labels[0], "A");
    assert_eq!(graph.node_labels[1], "B");
}

/// Test that out-in degree difference calculations work correctly for balanced graphs.
#[test]
fn test_out_in_diff() {
    let mut builder = GraphBuilder::new();
    builder.add_edge(0, 1, 1.0).add_edge(1, 0, 1.0);
    let graph = builder.build();
    let imbalanced_nodes = graph.imbalanced_nodes();
    assert!(imbalanced_nodes.empty());
}

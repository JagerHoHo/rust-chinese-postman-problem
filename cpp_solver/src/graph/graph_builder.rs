use ndarray::Array2;

use super::Edge;
use super::Graph;
use std::collections::{HashMap, HashSet};
/// Builder for constructing a graph.
pub struct GraphBuilder {
    edges: Vec<Edge>,
    max_node: usize,
    node_labels: HashMap<String, usize>,
    used_labels: HashSet<String>,
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
    assert!(imbalanced_nodes.is_empty());
}

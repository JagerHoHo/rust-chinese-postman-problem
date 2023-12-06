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
    out_degrees: Array1<usize>,
    edge_count: Array2<usize>,
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

    /// Adds an edge to the graph builder.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> &mut Self {
        self.max_node = self.max_node.max(from).max(to);
        self.edges.push(Edge { from, to, weight });
        self
    }

    /// Adds a labeled edge to the graph builder.
    pub fn add_labeled_edge(&mut self, from_label: &str, to_label: &str, weight: f64) -> &mut Self {
        if !self.used_labels.contains(from_label) {
            self.node_labels
                .insert(from_label.to_string(), self.used_labels.len());
            self.used_labels.insert(from_label.to_string());
        }
        if !self.used_labels.contains(to_label) {
            self.node_labels
                .insert(to_label.to_string(), self.used_labels.len());
            self.used_labels.insert(to_label.to_string());
        }
        let from = self.node_labels[from_label];
        let to = self.node_labels[to_label];
        self.add_edge(from, to, weight)
    }

    /// Builds the graph from the added edges.
    pub fn build(self) -> Graph {
        let mut weight_matrix =
            Array2::from_elem((self.max_node + 1, self.max_node + 1), f64::INFINITY);
        for edge in self.edges {
            weight_matrix[[edge.from, edge.to]] = edge.weight;
        }
        let mut node_labels = Vec::new();
        node_labels.resize(self.node_labels.len(), String::new());
        for (label, node) in self.node_labels {
            node_labels[node] = label;
        }
        let mut graph = Graph::from_weight_matrix(weight_matrix);
        graph.relabel(if node_labels.is_empty() {
            None
        } else {
            Some(node_labels)
        });
        graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates a graph from a weight matrix.
    pub fn from_weight_matrix(weight_matrix: Array2<f64>) -> Self {
        Self {
            out_degrees: weight_matrix
                .rows()
                .into_iter()
                .map(|row| row.iter().filter(|x| **x != f64::INFINITY).count())
                .collect(),
            edge_count: weight_matrix.mapv(|x| if x != f64::INFINITY { 1 } else { 0 }),
            node_labels: Vec::new(),
            weight_matrix,
        }
    }

    /// Relabels the nodes in the graph with the given labels.
    pub(crate) fn relabel(&mut self, node_labels: Option<Vec<String>>) {
        self.node_labels = match node_labels {
            Some(node_labels) => node_labels,
            None => Vec::from_iter((0..self.weight_matrix.nrows()).map(|x| x.to_string())),
        }
    }

    /// Returns the out degrees of the nodes in the graph.
    pub(crate) fn out_degrees(&self) -> Array1<usize> {
        self.out_degrees.clone()
    }

    /// Adds an edge to the graph.
    pub(crate) fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        println!(
            "Edge added from {} to {}",
            self.node_labels[from], self.node_labels[to]
        );
        self.weight_matrix[[from, to]] = weight;
        self.out_degrees[from] += 1;
        self.edge_count[[from, to]] += 1;
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
                x if x > 0 => {
                    println!(
                        "Node {} has a positive difference of {}",
                        self.node_labels[node], x
                    );
                    for _ in 0..x {
                        positive_difference_nodes.push(node)
                    }
                }
                x if x < 0 => {
                    println!(
                        "Node {} has a negative difference of {}",
                        self.node_labels[node], x
                    );
                    for _ in 0..-x {
                        negative_difference_nodes.push(node)
                    }
                }
                _ => (),
            }
        }
        ImbalancedNodeSet {
            negative: negative_difference_nodes,
            positive: positive_difference_nodes,
        }
    }

    /// Returns the edge set of the graph.
    pub(crate) fn edge_set(&self) -> Vec<Vec<usize>> {
        let mut edge_set = Vec::new();
        edge_set.resize(self.weight_matrix.nrows(), Vec::new());
        for (from, row) in self.weight_matrix.rows().into_iter().enumerate() {
            for (to, weight) in row.iter().enumerate() {
                if *weight != f64::INFINITY {
                    for _ in 0..self.edge_count[(from, to)] {
                        edge_set[from].push(to);
                    }
                }
            }
        }
        edge_set
    }

    /// Returns the out in degree difference of a node.
    fn out_in_diff(row: &ArrayView1<f64>, col: &ArrayView1<f64>) -> isize {
        let out_degree = row.iter().filter(|x| **x != f64::INFINITY).count();
        let in_degree = col.iter().filter(|x| **x != f64::INFINITY).count();
        out_degree as isize - in_degree as isize
    }
}

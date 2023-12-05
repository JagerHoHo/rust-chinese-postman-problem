/// Module for the Floyd-Warshall algorithm implementation.
mod floyd_warshall;

/// Module for the Hierholzer algorithm implementation.
mod hierholzer;

/// Module for the Hungarian algorithm implementation.
mod hungarian;

use ndarray::Array2;

use crate::{cpp_solver::hungarian::Matching, graph::Graph};
use floyd_warshall::FloydWarshallRunner;
use hierholzer::HierholzerRunner;
use std::{collections::VecDeque, fmt};

/// Represents a path in the graph.
pub struct Path {
    pub path: VecDeque<usize>,
    pub cost: f64,
    labels: Vec<String>,
}

/// Solver for the Chinese Postman Problem.
pub struct CppSolver {
    graph: Graph,
    floyd_warshall: FloydWarshallRunner,
    hierholzer: HierholzerRunner,
}

impl CppSolver {
    /// Creates a new instance of the solver.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph to solve the problem on.
    pub fn new(graph: Graph) -> Self {
        Self {
            floyd_warshall: FloydWarshallRunner::new(graph.weight_matrix.clone()),
            hierholzer: HierholzerRunner::new(),
            graph,
        }
    }

    /// Solves the Chinese Postman Problem and returns the optimal path.
    ///
    /// # Returns
    ///
    /// An `Option` containing the optimal path if the graph is solvable, or `None` otherwise.
    pub fn solve(&mut self) -> Option<Path> {
        if !self.solvable() {
            println!("The graph is not solvable");
            return None;
        }
        println!("The graph is solvable");
        self.balance_node();
        self.hierholzer.run(&self.graph);
        Some(Path::new(
            self.hierholzer.path(),
            &self.graph.weight_matrix,
            &self.graph.node_labels,
        ))
    }

    /// Balances the imbalanced nodes in the graph using the Hungarian algorithm.
    fn balance_node(&mut self) {
        let imbalanced_nodes = self.graph.imbalanced_nodes();
        if imbalanced_nodes.empty() {
            println!("The graph is balanced");
            return;
        }
        println!("Imbalanced nodes found, using the Hungarian algorithm to find the best match.");
        let shortest_distance_between_nodes = self.floyd_warshall.shortest_distances();
        let imbalanced_nodes_best_match =
            hungarian::best_match(&imbalanced_nodes, shortest_distance_between_nodes);
        println!("Best match found");
        for matching in &imbalanced_nodes_best_match {
            println!(
                "A connection will be added from {} to {}",
                self.graph.node_labels[matching.from], self.graph.node_labels[matching.to]
            );
        }
        println!("Adding edges to the graph according to the best match.");
        for Matching { from, to } in imbalanced_nodes_best_match {
            println!(
                "Connecting {} and {} ",
                self.graph.node_labels[from], self.graph.node_labels[to]
            );
            let shortest_path = self.floyd_warshall.shortest_path_between(from, to);
            for (i, node) in shortest_path.iter().enumerate().skip(1) {
                let from = shortest_path[i - 1];
                let to = *node;
                self.graph
                    .add_edge(from, to, self.graph.weight_matrix[(from, to)]);
            }
        }
    }

    /// Checks if the graph is solvable.
    ///
    /// # Returns
    ///
    /// `true` if the graph is solvable, `false` otherwise.
    fn solvable(&self) -> bool {
        let connected = self.floyd_warshall.graph_is_strongly_connected();
        let has_no_negative_cycle = self.floyd_warshall.graph_has_no_negative_cycle();
        println!(
            "The graph is {} strongly connected",
            if connected { "not" } else { "" }
        );
        println!(
            "The graph has {} negative cycle",
            if has_no_negative_cycle { "no" } else { "" }
        );
        connected && has_no_negative_cycle
    }
}

impl Path {
    /// Creates a new instance of `Path`.
    ///
    /// # Arguments
    ///
    /// * `path` - The path as a sequence of node indices.
    /// * `weight_matrix` - The weight matrix of the graph.
    /// * `labels` - The labels of the nodes in the graph.
    pub(crate) fn new(
        path: VecDeque<usize>,
        weight_matrix: &Array2<f64>,
        labels: &[String],
    ) -> Self {
        let cost = path
            .iter()
            .zip(path.iter().skip(1))
            .map(|(from, to)| weight_matrix[(*from, *to)])
            .sum();
        Self {
            path,
            cost,
            labels: labels.to_vec(),
        }
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = &self.labels[self.path[0]];
        let path = self
            .path
            .iter()
            .skip(1)
            .fold(path.to_string(), |path, &next| {
                path + "->" + &self.labels[next]
            });
        write!(f, "Path: {}, Cost: {}", path, self.cost)
    }
}

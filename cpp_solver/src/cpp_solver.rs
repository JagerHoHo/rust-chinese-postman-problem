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
            floyd_warshall: FloydWarshallRunner::new(graph.weight_matrix().clone()),
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
            println!("The graph is not solvable.");
            return None;
        }
        println!("The graph is solvable. Proceeding with the solution.");

        self.balance_node();
        self.hierholzer.run(&self.graph);

        Some(Path::new(
            self.hierholzer.path(),
            self.graph.weight_matrix(),
            self.graph.node_labels(),
        ))
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
            "The graph is {}strongly connected.",
            if connected { "" } else { "not " }
        );
        println!(
            "The graph has {} negative cycles.",
            if has_no_negative_cycle { "no" } else { "" }
        );

        connected && has_no_negative_cycle
    }

    /// Balances the imbalanced nodes in the graph using the Hungarian algorithm.
    fn balance_node(&mut self) {
        let imbalanced_nodes = self.graph.imbalanced_nodes();
        if imbalanced_nodes.is_empty() {
            println!("The graph is already balanced.");
            return;
        }

        println!("Balancing imbalanced nodes using the Hungarian algorithm.");
        let shortest_distance_between_nodes = self.floyd_warshall.shortest_distances();
        let best_match = hungarian::best_match(&imbalanced_nodes, shortest_distance_between_nodes);

        for Matching { from, to } in best_match {
            let path = self.floyd_warshall.shortest_path_between(from, to);

            for (i, &node) in path.iter().enumerate().skip(1) {
                let prev = path[i - 1];
                self.graph
                    .add_edge(prev, node, self.graph.weight_matrix()[[prev, node]]);
            }
        }
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

/// Test that the solver correctly identifies an unsolvable graph.
#[test]
fn test_solver_unsolvable_graph() {
    use crate::GraphBuilder;
    let mut builder = GraphBuilder::new();
    builder.add_edge(0, 1, 10.0).add_edge(1, 2, -20.0);
    let graph = builder.build();
    let mut solver = CppSolver::new(graph);
    assert!(solver.solve().is_none());
}

/// Test that the solver correctly solves a simple, balanced graph.
#[test]
fn test_solver_simple_graph() {
    use crate::GraphBuilder;
    let mut builder = GraphBuilder::new();
    builder.add_edge(0, 1, 1.0).add_edge(1, 0, 1.0);
    let graph = builder.build();
    let mut solver = CppSolver::new(graph);
    let solution = solver.solve();
    assert!(solution.is_some());
    assert_eq!(solution.unwrap().cost, 2.0);
}

/// Test that a Path calculates its cost correctly.
#[test]
fn test_path_cost() {
    use ndarray::array;
    let weight_matrix = array![
        [0.0, 1.0, f64::INFINITY],
        [f64::INFINITY, 0.0, 2.0],
        [f64::INFINITY, f64::INFINITY, 0.0]
    ];
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let path = Path::new(vec![0, 1, 2].into_iter().collect(), &weight_matrix, &labels);
    assert_eq!(
        path.cost, 3.0,
        "The cost of the path should be the sum of the edge weights"
    );
}

/// Test that a Path formats its display correctly.
#[test]
fn test_path_display() {
    use ndarray::array;
    let weight_matrix = array![
        [0.0, 1.0, f64::INFINITY],
        [f64::INFINITY, 0.0, 2.0],
        [f64::INFINITY, f64::INFINITY, 0.0]
    ];
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let path = Path::new(vec![0, 1, 2].into_iter().collect(), &weight_matrix, &labels);
    assert_eq!(
        path.to_string(),
        "Path: A->B->C, Cost: 3",
        "The path display should match the expected format"
    );
}

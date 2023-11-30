mod floyd_warshall;
mod hierholzer;
mod hungarian;

use ndarray::Array2;

use crate::{cpp_solver::hungarian::Matching, graph::Graph};
use floyd_warshall::FloydWarshallRunner;
use hierholzer::HierholzerRunner;
use std::{collections::VecDeque, fmt};

pub struct Path {
    pub path: VecDeque<usize>,
    pub cost: f64,
    labels: Vec<String>,
}

pub struct CppSolver {
    graph: Graph,
    floyd_warshall: FloydWarshallRunner,
    hierholzer: HierholzerRunner,
}

impl CppSolver {
    pub fn new(graph: Graph) -> Self {
        Self {
            floyd_warshall: FloydWarshallRunner::new(graph.weight_matrix.clone()),
            hierholzer: HierholzerRunner::new(),
            graph,
        }
    }

    pub fn solve(&mut self) -> Option<Path> {
        if !self.solvable() {
            println!("Graph is not solvable");
            return None;
        }
        println!("Graph is solvable");
        self.balance_node();
        self.hierholzer.run(&self.graph);
        Some(Path::new(
            self.hierholzer.path(),
            &self.graph.weight_matrix,
            &self.graph.node_labels,
        ))
    }

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
        for mathching in &imbalanced_nodes_best_match {
            println!(
                "Add connection from {} to {}",
                self.graph.node_labels[mathching.from], self.graph.node_labels[mathching.to]
            );
        }
        println!("Adding edges to the graph according to the best match.");
        for Matching { from, to } in imbalanced_nodes_best_match {
            for (i, node) in self
                .floyd_warshall
                .shortest_path_between(from, to)
                .iter()
                .enumerate()
            {
                if i == 0 {
                    continue;
                }
                let from = self.floyd_warshall.shortest_path_between(from, to)[i - 1];
                let to = *node;
                self.graph
                    .add_edge(from, to, self.graph.weight_matrix[(from, to)]);
            }
        }
    }

    fn solvable(&self) -> bool {
        self.floyd_warshall.graph_is_strongly_connected()
            && !self.floyd_warshall.graph_has_negative_cycle()
    }
}

impl Path {
    pub fn new(path: VecDeque<usize>, weight_matrix: &Array2<f64>, labels: &[String]) -> Self {
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
        let path = self
            .path
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == self.path.len() - 1 {
                    self.labels[*x].to_owned()
                } else {
                    format!("{}->", self.labels[*x])
                }
            })
            .collect::<String>();
        write!(f, "Path: {}, Cost: {}", path, self.cost)
    }
}

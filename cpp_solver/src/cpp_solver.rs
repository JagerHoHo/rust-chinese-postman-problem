use ndarray::Array2;

use crate::graph::Graph;
mod floyd_warshall;
use floyd_warshall::FloydWarshallRunner;
mod hungarian;
use std::{collections::VecDeque, fmt};
mod hierholzer;
use hierholzer::HierholzerRunner;

pub struct Path {
    pub path: VecDeque<usize>,
    pub cost: f64,
    labels: Vec<String>,
}

pub struct CppSolver {
    graph: Graph,
    floyd_warshall: FloydWarshallRunner,
    hierholzer: HierholzerRunner,
    labels: Vec<String>,
}

impl CppSolver {
    pub fn new(graph: Graph) -> Self {
        Self {
            floyd_warshall: FloydWarshallRunner::new(graph.weight_matrix.clone()),
            hierholzer: HierholzerRunner::new(),
            graph,
            labels: Vec::new(),
        }
    }

    pub fn set_labels(&mut self, labels: Vec<String>) {
        if labels.len() != self.graph.weight_matrix.nrows() {
            panic!("Number of labels must match number of nodes");
        }
        self.labels = labels;
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
            &self.labels,
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
        for (from, to) in imbalanced_nodes_best_match {
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
        let mut labels = Vec::from(labels);
        if labels.is_empty() {
            let mut current_sequence = String::from("a");
            for _ in 0..weight_matrix.nrows() {
                labels.push(current_sequence.clone());
                current_sequence = Self::increment_sequence(&current_sequence);
            }
        }

        let cost = path
            .iter()
            .zip(path.iter().skip(1))
            .map(|(from, to)| weight_matrix[(*from, *to)])
            .sum();
        Self { path, cost, labels }
    }

    fn increment_sequence(s: &str) -> String {
        let mut chars: Vec<char> = s.chars().collect();
        let mut carry = 1;
        for c in chars.iter_mut().rev() {
            if carry == 0 {
                break;
            }
            let current_value = (*c as u8 - b'a') + carry;
            *c = ((current_value % 26) + b'a') as char;
            carry = current_value / 26;
        }
        if carry > 0 {
            chars.insert(0, ((carry % 26) + b'a') as char);
        }
        chars.iter().collect()
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
        write!(f, "path: {}, cost: {}", path, self.cost)
    }
}

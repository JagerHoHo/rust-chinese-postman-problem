use ndarray::{Array2, ArrayView2};

/// Represents a runner for the Floyd-Warshall algorithm.
pub struct FloydWarshallRunner {
    n_nodes: usize,
    shortest_distances: Array2<f64>,
    next: Array2<Option<usize>>,
    have_negative_cycle: bool,
}

impl FloydWarshallRunner {
    /// Initializes the Floyd-Warshall runner.
    pub fn new(weight_matrix: Array2<f64>) -> Self {
        let n_nodes = weight_matrix.nrows();
        let next = Array2::from_shape_fn((n_nodes, n_nodes), |(i, j)| {
            if weight_matrix[(i, j)] < f64::INFINITY {
                Some(j)
            } else {
                None
            }
        });

        let mut runner = Self {
            n_nodes,
            shortest_distances: weight_matrix,
            next,
            have_negative_cycle: false,
        };

        runner.find_shortest_distances();
        runner.detect_negative_cycles();
        runner
    }

    /// Optimized calculation of shortest distances using Floyd-Warshall.
    fn find_shortest_distances(&mut self) {
        for k in 0..self.n_nodes {
            for i in 0..self.n_nodes {
                if self.shortest_distances[(i, k)] == f64::INFINITY {
                    continue; // Skip unreachable intermediates
                }

                for j in 0..self.n_nodes {
                    if self.shortest_distances[(k, j)] == f64::INFINITY {
                        continue; // Skip unreachable destinations
                    }

                    let new_dist =
                        self.shortest_distances[(i, k)] + self.shortest_distances[(k, j)];
                    if new_dist < self.shortest_distances[(i, j)] {
                        self.shortest_distances[(i, j)] = new_dist;
                        self.next[(i, j)] = self.next[(i, k)];
                    }
                }
            }
        }
    }

    /// Detects negative cycles in the graph.
    fn detect_negative_cycles(&mut self) {
        for i in 0..self.n_nodes {
            if self.shortest_distances[(i, i)] < 0.0 {
                self.have_negative_cycle = true;
                return;
            }
        }
    }

    /// Retrieves the shortest path between two nodes.
    pub fn shortest_path_between(&self, start: usize, end: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current_node = Some(start);

        while let Some(node) = current_node {
            path.push(node);
            if node == end {
                break;
            }
            current_node = self.next[(node, end)];
        }

        if path.last() == Some(&end) {
            path
        } else {
            Vec::new() // Return empty path if no path exists
        }
    }

    /// Returns a view of the shortest distances matrix.
    pub fn shortest_distances(&self) -> ArrayView2<f64> {
        self.shortest_distances.view()
    }

    /// Checks if the graph has no negative cycle.
    pub fn graph_has_no_negative_cycle(&self) -> bool {
        !self.have_negative_cycle
    }

    /// Checks if the graph is strongly connected.
    pub fn graph_is_strongly_connected(&self) -> bool {
        self.shortest_distances.iter().all(|&x| x != f64::INFINITY)
    }
}

/// Test the shortest path between two nodes in a small graph.
#[test]
fn test_shortest_path_between() {
    let weight_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            0.0,
            1.0,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
            1.0,
            1.0,
            f64::INFINITY,
            0.0,
        ],
    )
    .unwrap();
    let runner = FloydWarshallRunner::new(weight_matrix);
    let path = runner.shortest_path_between(0, 2);
    assert_eq!(path, vec![0, 1, 2]);
}

/// Test that the algorithm detects negative cycles in the graph.
#[test]
fn test_graph_has_no_negative_cycle() {
    let weight_matrix = Array2::from_shape_vec((2, 2), vec![0.0, -1.0, -1.0, 0.0]).unwrap();
    let runner = FloydWarshallRunner::new(weight_matrix);
    assert!(!runner.graph_has_no_negative_cycle());
}

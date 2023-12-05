use ndarray::{Array2, ArrayView2};

pub(super) struct FloydWarshallRunner {
    n_nodes: usize,
    shortest_distances: Array2<f64>,
    next: Array2<usize>,
    have_negative_cycle: bool,
}

impl FloydWarshallRunner {
    pub(super) fn new(weight_matrix: Array2<f64>) -> Self {
        let n_nodes = weight_matrix.nrows();
        let next = Array2::<usize>::from_shape_fn((n_nodes, n_nodes), |(i, j)| {
            if weight_matrix[(i, j)] != f64::INFINITY {
                return j;
            }
            0
        });
        let mut runner = Self {
            n_nodes,
            shortest_distances: weight_matrix,
            next,
            have_negative_cycle: false,
        };
        runner.find_shortest_distances();
        runner.find_negative_cycle();
        runner
    }

    pub(super) fn shortest_path_between(&self, start: usize, end: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current_node = start;
        while current_node != end {
            path.push(current_node);
            current_node = self.next[(current_node, end)];
        }
        path.push(end);
        path
    }

    pub(super) fn shortest_distances(&self) -> ArrayView2<f64> {
        self.shortest_distances.view()
    }

    pub(super) fn graph_has_no_negative_cycle(&self) -> bool {
        !self.have_negative_cycle
    }

    pub(super) fn graph_is_strongly_connected(&self) -> bool {
        self.shortest_distances.iter().all(|x| *x != f64::INFINITY)
    }

    fn find_shortest_distances(&mut self) {
        for k in 0..self.n_nodes {
            for i in 0..self.n_nodes {
                for j in 0..self.n_nodes {
                    let dist_with_mid_point_k =
                        self.shortest_distances[(i, k)] + self.shortest_distances[(k, j)];
                    if dist_with_mid_point_k < self.shortest_distances[(i, j)] {
                        self.shortest_distances[(i, j)] = dist_with_mid_point_k;
                        self.next[(i, j)] = self.next[(i, k)];
                    }
                }
            }
        }
    }

    fn find_negative_cycle(&mut self) {
        for k in 0..self.n_nodes {
            for i in 0..self.n_nodes {
                for j in 0..self.n_nodes {
                    let dist_with_mid_point_k =
                        self.shortest_distances[(i, k)] + self.shortest_distances[(k, j)];
                    if dist_with_mid_point_k < self.shortest_distances[(i, j)] {
                        self.have_negative_cycle = true;
                        self.shortest_distances[(i, j)] = f64::NEG_INFINITY;
                        self.next[(i, j)] = usize::MAX;
                    }
                }
            }
        }
    }
}

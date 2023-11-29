use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug)]
pub(crate) struct ImbalancedNodeSet {
    pub(crate) negative_difference_nodes: Vec<usize>,
    pub(crate) positive_difference_nodes: Vec<usize>,
}

struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

pub struct GraphBuilder {
    edges: Vec<Edge>,
    max_node: usize,
}

pub struct Graph {
    pub(super) weight_matrix: Array2<f64>,
    pub(super) out_degrees: Array1<usize>,
    pub(super) edge_count: Array2<usize>,
}

impl ImbalancedNodeSet {
    pub(super) fn empty(&self) -> bool {
        self.negative_difference_nodes.is_empty() && self.positive_difference_nodes.is_empty()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            max_node: 0,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> &mut Self {
        self.max_node = self.max_node.max(from).max(to);
        self.edges.push(Edge { from, to, weight });
        self
    }

    pub fn build(self) -> Graph {
        let mut weight_matrix =
            Array2::from_elem((self.max_node + 1, self.max_node + 1), f64::INFINITY);
        for edge in self.edges {
            weight_matrix[[edge.from, edge.to]] = edge.weight;
        }
        Graph::from_weight_matrix(weight_matrix)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn from_weight_matrix(weight_matrix: Array2<f64>) -> Self {
        Self {
            out_degrees: weight_matrix
                .rows()
                .into_iter()
                .map(|row| row.iter().filter(|x| **x != f64::INFINITY).count())
                .collect(),
            edge_count: weight_matrix.mapv(|x| if x != f64::INFINITY { 1 } else { 0 }),
            weight_matrix,
        }
    }

    pub(crate) fn out_degrees(&self) -> Array1<usize> {
        self.out_degrees.clone()
    }

    pub(crate) fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        println!("Edge added from {} to {}", from, to);
        self.weight_matrix[[from, to]] = weight;
        self.out_degrees[from] += 1;
        self.edge_count[[from, to]] += 1;
    }

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
                x if x > 0 => positive_difference_nodes.push(node),
                x if x < 0 => negative_difference_nodes.push(node),
                _ => (),
            }
        }
        ImbalancedNodeSet {
            negative_difference_nodes,
            positive_difference_nodes,
        }
    }

    pub(super) fn edge_set(&self) -> Vec<Vec<usize>> {
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

    fn out_in_diff(row: &ArrayView1<f64>, col: &ArrayView1<f64>) -> isize {
        let out_degree = row.iter().filter(|x| **x != f64::INFINITY).count();
        let in_degree = col.iter().filter(|x| **x != f64::INFINITY).count();
        out_degree as isize - in_degree as isize
    }
}

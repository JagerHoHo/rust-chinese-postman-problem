use std::collections::VecDeque;

use ndarray::Array1;

use super::Graph;

pub(super) struct HierholzerRunner {
    path: VecDeque<usize>,
}

impl HierholzerRunner {
    pub(super) fn new() -> Self {
        Self {
            path: VecDeque::new(),
        }
    }

    pub(super) fn run(&mut self, graph: &Graph) {
        let mut out_degrees = graph.out_degrees();
        let mut edge_set = graph.edge_set();
        self.dfs(0, &mut edge_set, &mut out_degrees);
    }

    pub(super) fn path(&self) -> VecDeque<usize> {
        self.path.clone()
    }

    fn dfs(
        &mut self,
        node: usize,
        edge_set: &mut Vec<Vec<usize>>,
        out_degrees: &mut Array1<usize>,
    ) {
        while out_degrees[node] != 0 {
            out_degrees[node] -= 1;
            let next_edge = edge_set[node][out_degrees[node]];
            self.dfs(next_edge, edge_set, out_degrees);
        }
        self.path.push_front(node);
    }
}

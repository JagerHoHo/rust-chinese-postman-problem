use std::collections::VecDeque;

use super::Graph;

pub(super) struct HierholzerRunner {
    path: VecDeque<usize>,
}

impl HierholzerRunner {
    /// Creates a new instance of `HierholzerRunner`.
    pub fn new() -> Self {
        Self {
            path: VecDeque::new(),
        }
    }

    /// Runs the algorithm to find an Eulerian path or circuit.
    ///
    /// # Arguments
    ///
    /// * `graph` - A reference to the graph. The graph must be Eulerian.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the path is found, or an error message if the graph is not Eulerian.
    pub fn run(&mut self, graph: &Graph) {
        if !Self::is_eulerian(graph) {
            panic!("The graph is not Eulerian.");
        }

        let mut edge_set = graph.edge_set(); // Clone edge set
        let mut out_degrees = graph.out_degrees().to_vec(); // Clone out-degrees

        self.find_path(0, &mut edge_set, &mut out_degrees);
    }

    /// Validates if a graph is Eulerian.
    fn is_eulerian(graph: &Graph) -> bool {
        graph
            .out_degrees()
            .iter()
            .zip(graph.edge_set().iter())
            .all(|(out, edges)| edges.len() == *out)
    }

    /// Retrieves the Eulerian path or circuit.
    pub fn path(&self) -> VecDeque<usize> {
        self.path.clone()
    }

    /// Finds the Eulerian path or circuit using an iterative DFS approach.
    fn find_path(
        &mut self,
        start_node: usize,
        edge_set: &mut [Vec<usize>],
        out_degrees: &mut [usize],
    ) {
        let mut stack = Vec::new();
        stack.push(start_node);

        while let Some(node) = stack.last() {
            if out_degrees[*node] > 0 {
                out_degrees[*node] -= 1;
                let next_node = edge_set[*node].pop().unwrap();
                stack.push(next_node);
            } else {
                self.path.push_front(stack.pop().unwrap());
            }
        }
    }
}

/// Test that Hierholzer's algorithm finds a simple cycle correctly.
#[test]
fn test_hierholzer_simple_cycle() {
    use crate::GraphBuilder;
    let mut builder = GraphBuilder::new();
    builder.add_edge(0, 1, 1.0).add_edge(1, 0, 1.0);
    let graph = builder.build();
    let mut runner = HierholzerRunner::new();
    runner.run(&graph);
    assert_eq!(
        runner.path().iter().cloned().collect::<Vec<_>>(),
        vec![0, 1, 0]
    );
}

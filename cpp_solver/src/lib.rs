mod cpp_solver;
mod graph;
pub use cpp_solver::CppSolver;
pub use graph::Graph;
pub use graph::GraphBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use graph::GraphBuilder;

    #[test]
    fn test_circle() {
        let mut graph_builder = GraphBuilder::new();
        for i in 0..5 {
            graph_builder.add_edge(i, (i + 1) % 5, 1.);
        }
        let graph = graph_builder.build();
        let mut solver = CppSolver::new(graph);
        match solver.solve() {
            Some(path) => {
                println!("{}", path);
            }
            None => panic!("No solution found"),
        }
    }

    #[test]
    fn test_standard() {
        let mut graph_builder = GraphBuilder::new();
        graph_builder
            .add_edge(0, 2, 20.)
            .add_edge(0, 1, 10.)
            .add_edge(1, 4, 10.)
            .add_edge(1, 3, 50.)
            .add_edge(2, 4, 33.)
            .add_edge(2, 3, 20.)
            .add_edge(3, 4, 5.)
            .add_edge(3, 5, 12.)
            .add_edge(4, 0, 12.)
            .add_edge(4, 5, 1.)
            .add_edge(5, 2, 22.);
        let graph = graph_builder.build();
        let mut solver = CppSolver::new(graph);
        match solver.solve() {
            Some(path) => {
                println!("{}", path);
            }
            None => panic!("No solution found"),
        }
    }

    #[test]
    fn test_label() {
        let mut graph_builder = GraphBuilder::new();
        graph_builder
            .add_labeled_edge("a", "c", 20.)
            .add_labeled_edge("a", "b", 10.)
            .add_labeled_edge("b", "e", 10.)
            .add_labeled_edge("b", "d", 50.)
            .add_labeled_edge("c", "e", 33.)
            .add_labeled_edge("c", "d", 20.)
            .add_labeled_edge("d", "e", 5.)
            .add_labeled_edge("d", "f", 12.)
            .add_labeled_edge("e", "a", 12.)
            .add_labeled_edge("e", "f", 1.)
            .add_labeled_edge("f", "c", 22.);

        let graph = graph_builder.build();
        let mut solver = CppSolver::new(graph);
        match solver.solve() {
            Some(path) => {
                println!("{}", path);
            }
            None => panic!("No solution found"),
        }
    }
}

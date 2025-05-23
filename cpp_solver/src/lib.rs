mod cpp_solver;
mod graph;
pub use cpp_solver::CppSolver;
pub use graph::Graph;
pub use graph::GraphBuilder;

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn check_path(graph_builder: GraphBuilder, cost: f64) {
        let graph = graph_builder.build();
        let mut solver = CppSolver::new(graph);
        match solver.solve() {
            Some(path) => {
                assert_eq!(path.cost, cost)
            }
            None => panic!("No solution found"),
        }
    }

    #[test]
    fn test_circle() {
        let mut graph_builder = GraphBuilder::new();
        for i in 0..5 {
            graph_builder.add_edge(i, (i + 1) % 5, 1.);
        }
        check_path(graph_builder, 5.);
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
        check_path(graph_builder, 276.);
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
        check_path(graph_builder, 276.);
    }

    #[test]
    fn test_odd_in_out_diff() {
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
            .add_labeled_edge("f", "c", 22.)
            .add_labeled_edge("g", "c", 88.)
            .add_labeled_edge("a", "g", 18.);
        check_path(graph_builder, 419.);
    }

    #[test]
    fn test_non_one_in_out_diff() {
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
            .add_labeled_edge("f", "c", 22.)
            .add_labeled_edge("g", "f", 2.)
            .add_labeled_edge("b", "g", 67.);
        check_path(graph_builder, 414.);
    }
}

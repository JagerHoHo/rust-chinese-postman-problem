use cpp_solver::CppSolver;
use cpp_solver::GraphBuilder;

fn main() {
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
    let graph = graph_builder.build();
    let mut solver = CppSolver::new(graph);
    match solver.solve() {
        Some(path) => {
            println!("{}", path);
        }
        None => panic!("No solution found"),
    }
}

use cpp_solver::CppSolver;
use cpp_solver::GraphBuilder;
fn main() {
    let mut graph_builder = GraphBuilder::new();
    graph_builder.add_edge(0, 2, 20.);
    graph_builder.add_edge(0, 1, 10.);

    graph_builder.add_edge(1, 4, 10.);
    graph_builder.add_edge(1, 3, 50.);

    graph_builder.add_edge(2, 4, 33.);
    graph_builder.add_edge(2, 3, 20.);

    graph_builder.add_edge(3, 4, 5.);
    graph_builder.add_edge(3, 5, 12.);

    graph_builder.add_edge(4, 0, 12.);
    graph_builder.add_edge(4, 5, 1.);

    graph_builder.add_edge(5, 2, 22.);
    let graph = graph_builder.build();
    let mut solver = CppSolver::new(graph);
    match solver.solve() {
        Some(path) => {
            println!("{}", path);
        }
        None => panic!("No solution found"),
    }
}

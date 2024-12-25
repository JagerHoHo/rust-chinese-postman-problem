/// Represents an edge in the graph, with a source node, target node, and weight.
pub(super) struct Edge {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) weight: f64,
}

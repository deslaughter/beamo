use crate::node::Node;

pub struct Model {
    nodes: Vec<Node>,
}

impl Model {
    pub fn add_node(&mut self, mut node: Node) -> usize {
        let id = self.nodes.len();
        node.id = id;
        self.nodes.push(node);
        id
    }
}

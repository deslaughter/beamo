use faer::{Col, Mat};

pub struct RigidPlatform {
    /// Gravity (xyz)
    pub gravity: [f64; 3],
    /// Platform node position and orientation (xyz,q)
    pub node_position: Col<f64>,
    /// Platform node ID
    pub node_id: usize,
    /// Platform mass matrix, `[6x6]`
    pub mass_matrix: Mat<f64>,
    /// Mooring lines
    pub mooring_lines: Vec<MooringLine>,
}

pub struct MooringLine {
    /// Stiffness of mooring line
    pub stiffness: f64,
    /// Upstretched mooring line length
    pub unstretched_length: f64,
    /// Fairlead connection point position (xyz)
    pub fairlead_node_position: Col<f64>,
    /// Fairlead node ID
    pub fairlead_node_id: usize,
    /// Anchor point position (xyz)
    pub anchor_node_position: Col<f64>,
    /// Anchor node ID
    pub anchor_node_id: usize,
}

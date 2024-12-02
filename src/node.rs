use itertools::Itertools;

pub struct Node {
    /// Node identifier
    pub id: usize,
    /// Position along beam element length [0-1]
    pub s: f64,
    /// Initial position
    pub x: [f64; 7],
    /// Initial displacement
    pub u: [f64; 7],
    /// Initial velocity
    pub v: [f64; 6],
    /// Initial acceleration
    pub vd: [f64; 6],
    /// Packed active degrees of freedom
    pub active_dofs: ActiveDOFs,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum ActiveDOFs {
    None,
    Translation,
    Rotation,
    All,
}

impl ActiveDOFs {
    pub fn add_dofs(&mut self, dofs: ActiveDOFs) {
        *self = match self {
            ActiveDOFs::None => dofs,
            ActiveDOFs::All => ActiveDOFs::All,
            ActiveDOFs::Rotation => match dofs {
                ActiveDOFs::None => ActiveDOFs::Rotation,
                ActiveDOFs::Rotation => ActiveDOFs::Rotation,
                ActiveDOFs::Translation => ActiveDOFs::All,
                ActiveDOFs::All => ActiveDOFs::All,
            },
            ActiveDOFs::Translation => match dofs {
                ActiveDOFs::None => ActiveDOFs::Translation,
                ActiveDOFs::Rotation => ActiveDOFs::All,
                ActiveDOFs::Translation => ActiveDOFs::Translation,
                ActiveDOFs::All => ActiveDOFs::All,
            },
        };
    }
    pub fn n_dofs(self) -> usize {
        match self {
            ActiveDOFs::None => 0,
            ActiveDOFs::Translation => 3,
            ActiveDOFs::Rotation => 3,
            ActiveDOFs::All => 6,
        }
    }
}

pub struct NodeDOFs {
    pub first_dof_index: usize,
    pub n_dofs: usize,
    pub active: ActiveDOFs,
}

#[repr(usize)]
pub enum Direction {
    X = 0,
    Y = 1,
    Z = 2,
    RX = 3,
    RY = 4,
    RZ = 5,
}

pub struct NodeFreedomMap {
    pub total_dofs: usize,
    pub node_dofs: Vec<NodeDOFs>,
}

impl NodeFreedomMap {
    pub fn new(nodes: &[Node]) -> Self {
        let mut first_dof = 0;

        let mut nfm = Self {
            node_dofs: nodes
                .iter()
                .map(|n| {
                    let ndofs = NodeDOFs {
                        first_dof_index: first_dof,
                        n_dofs: match n.active_dofs {
                            ActiveDOFs::None => 0,
                            ActiveDOFs::Translation => 3,
                            ActiveDOFs::Rotation => 3,
                            ActiveDOFs::All => 6,
                        },
                        active: n.active_dofs,
                    };

                    // Increment next DOF index for node
                    first_dof += ndofs.n_dofs;

                    ndofs
                })
                .collect_vec(),
            total_dofs: 0,
        };

        nfm.total_dofs = first_dof;

        nfm
    }

    /// Get DOF number for node and direction
    pub fn get_dof(&self, node_id: usize, dir: Direction) -> Option<usize> {
        if node_id >= self.node_dofs.len() {
            return None;
        }
        let node_dof = &self.node_dofs[node_id];
        match node_dof.active {
            ActiveDOFs::None => None,
            ActiveDOFs::Translation => match dir {
                Direction::X => Some(node_dof.first_dof_index),
                Direction::Y => Some(node_dof.first_dof_index + 1),
                Direction::Z => Some(node_dof.first_dof_index + 2),
                _ => None,
            },
            ActiveDOFs::Rotation => match dir {
                Direction::RX => Some(node_dof.first_dof_index),
                Direction::RY => Some(node_dof.first_dof_index + 1),
                Direction::RZ => Some(node_dof.first_dof_index + 2),
                _ => None,
            },
            ActiveDOFs::All => match dir {
                Direction::X => Some(node_dof.first_dof_index),
                Direction::Y => Some(node_dof.first_dof_index + 1),
                Direction::Z => Some(node_dof.first_dof_index + 2),
                Direction::RX => Some(node_dof.first_dof_index + 3),
                Direction::RY => Some(node_dof.first_dof_index + 4),
                Direction::RZ => Some(node_dof.first_dof_index + 5),
            },
        }
    }
}

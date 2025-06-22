use faer::prelude::*;
use itertools::Itertools;

use crate::util::{quat_compose_alloc, quat_from_rotation_vector_alloc, quat_rotate_vector_alloc};

#[derive(Debug)]
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

impl Node {
    pub fn translate(&mut self, xyz: [f64; 3]) -> &mut Self {
        self.x[0] += xyz[0];
        self.x[1] += xyz[1];
        self.x[2] += xyz[2];
        self
    }

    pub fn rotate(&mut self, rv: ColRef<f64>, rotation_center: ColRef<f64>) -> &mut Self {
        let dq = quat_from_rotation_vector_alloc(rv);
        let dx = quat_rotate_vector_alloc(
            dq.as_ref(),
            col![
                self.x[0] - rotation_center[0],
                self.x[1] - rotation_center[1],
                self.x[2] - rotation_center[2]
            ]
            .as_ref(),
        );

        let q = col![self.x[3], self.x[4], self.x[5], self.x[6]];
        let q_new = quat_compose_alloc(dq.as_ref(), q.as_ref());

        self.x[0] = rotation_center[0] + dx[0];
        self.x[1] = rotation_center[1] + dx[1];
        self.x[2] = rotation_center[2] + dx[2];
        self.x[3] = q_new[0];
        self.x[4] = q_new[1];
        self.x[5] = q_new[2];
        self.x[6] = q_new[3];
        self
    }
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
    pub n_system_dofs: usize,
    pub n_constraint_dofs: usize,
    pub node_dofs: Vec<NodeDOFs>,
}

impl NodeFreedomMap {
    pub fn n_dofs(&self) -> usize {
        self.n_system_dofs + self.n_constraint_dofs
    }
    pub fn new(nodes: &[Node]) -> Self {
        let mut first_dof = 0;

        let node_dofs = nodes
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
            .collect_vec();

        let mut nfm = Self {
            node_dofs,
            n_system_dofs: 0,
            n_constraint_dofs: 0,
        };

        nfm.n_system_dofs = first_dof;

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

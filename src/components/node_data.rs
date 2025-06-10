use faer::prelude::*;

use crate::state::State;

pub struct NodeData {
    pub id: usize,              // Node identifier in model
    pub position: Col<f64>,     // Absolute position of node in global coordinates
    pub displacement: Col<f64>, // Displacement from reference position
    pub velocity: Col<f64>,     // Velocity of node in global coordinates
    pub acceleration: Col<f64>, // Acceleration of node in global coordinates
    pub loads: Col<f64>,        // Point loads/moment applied to node in global coordinates
}

impl NodeData {
    pub fn new(id: usize) -> Self {
        NodeData {
            id,
            position: Col::zeros(7),
            displacement: Col::zeros(7),
            velocity: Col::zeros(6),
            acceleration: Col::zeros(6),
            loads: Col::zeros(6),
        }
    }

    pub fn get_motion(&mut self, state: &State) {
        self.position.copy_from(&state.x.col(self.id));
        self.displacement.copy_from(&state.u.col(self.id));
        self.velocity.copy_from(&state.v.col(self.id));
        self.acceleration.copy_from(&state.a.col(self.id));
    }

    pub fn set_loads(&self, state: &mut State) {
        state.fx.col_mut(self.id).copy_from(&self.loads);
    }
}

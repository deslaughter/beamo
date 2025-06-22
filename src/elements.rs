pub mod beam_qps;
pub mod beams;
pub mod kernels;
pub mod masses;
pub mod springs;

use crate::{node::NodeFreedomMap, state::State};
use beams::Beams;

use faer::{ColMut, MatMut};
use masses::Masses;
use springs::Springs;

pub struct Elements {
    pub beams: Beams,
    pub masses: Masses,
    pub springs: Springs,
}

impl Elements {
    pub fn new(beams: Beams, masses: Masses, springs: Springs) -> Self {
        Self {
            beams,
            masses,
            springs,
        }
    }
    pub fn assemble_system(
        &mut self,
        state: &State,
        h: f64,
        mut r: ColMut<f64>, // Residual
    ) {
        // Add beams to system
        self.beams.calculate_system(state, h);
        self.beams.assemble_system(r.as_mut());

        // Add mass elements to system
        self.masses.assemble_system(state, r.as_mut());

        // Add spring elements to system
        self.springs.assemble_system(state, r.as_mut());
    }
}

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
    pub fn assemble_system(
        &mut self,
        state: &State,
        nfm: &NodeFreedomMap,
        mut m: MatMut<f64>, // Mass
        mut g: MatMut<f64>, // Damping
        mut k: MatMut<f64>, // Stiffness
        mut r: ColMut<f64>, // Residual
    ) {
        // Add beams to system
        self.beams.calculate_system(state);
        self.beams
            .assemble_system(nfm, m.as_mut(), g.as_mut(), k.as_mut(), r.as_mut());

        // Add mass elements to system
        self.masses
            .assemble_system(nfm, state, m.as_mut(), g.as_mut(), k.as_mut(), r.as_mut());

        // Add spring elements to system
        self.springs
            .assemble_system(nfm, state, k.as_mut(), r.as_mut());
    }
}

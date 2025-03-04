use faer::{Col, Mat};
use itertools::Itertools;

use crate::constraints::{ConstraintInput, ConstraintKind, Constraints};
use crate::elements::beams::{BeamElement, BeamSection, Beams, Damping};
use crate::elements::masses::{MassElement, Masses};
use crate::elements::springs::{SpringElement, Springs};
use crate::elements::Elements;
use crate::node::{ActiveDOFs, Node, NodeFreedomMap};
use crate::quadrature::Quadrature;
use crate::solver::{Solver, StepParameters};
use crate::state::State;

pub struct Model {
    gravity: [f64; 3],
    h: f64,
    rho_inf: f64,
    max_iter: usize,
    is_static: bool,
    solver_abs_tol: f64,
    solver_rel_tol: f64,
    pub nodes: Vec<Node>,
    pub beam_elements: Vec<BeamElement>,
    pub mass_elements: Vec<MassElement>,
    pub spring_elements: Vec<SpringElement>,
    constraints: Vec<ConstraintInput>,
}

impl Model {
    /// Creates and initializes a model
    pub fn new() -> Model {
        Model {
            gravity: [0., 0., 0.],
            h: 0.01,
            rho_inf: 1.,
            max_iter: 6,
            is_static: false,
            solver_abs_tol: 1e-5,
            solver_rel_tol: 1e-3,
            nodes: vec![],
            beam_elements: vec![],
            mass_elements: vec![],
            spring_elements: vec![],
            constraints: vec![],
        }
    }

    pub fn create_node_freedom_map(&self) -> NodeFreedomMap {
        NodeFreedomMap::new(&self.nodes)
    }

    /// Set the gravity acceleration in each direction
    pub fn set_gravity(&mut self, x: f64, y: f64, z: f64) {
        self.gravity[0] = x;
        self.gravity[1] = y;
        self.gravity[2] = z;
    }

    pub fn set_time_step(&mut self, h: f64) {
        self.h = h;
    }

    pub fn set_rho_inf(&mut self, rho_inf: f64) {
        self.rho_inf = rho_inf;
    }

    pub fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    pub fn set_dynamic_solve(&mut self) {
        self.is_static = false;
    }

    pub fn set_static_solve(&mut self) {
        self.is_static = true;
    }

    /// Create solver
    pub fn create_solver(&self) -> Solver {
        let nfm = self.create_node_freedom_map();
        let constraints = Constraints::new(&self.constraints, &nfm);
        let elements = self.create_elements();
        let step_parameters = StepParameters::new(
            self.h,
            self.rho_inf,
            self.solver_abs_tol,
            self.solver_rel_tol,
            self.max_iter,
            self.is_static,
        );
        Solver::new(step_parameters, nfm, elements, constraints)
    }

    /// Create elements
    pub fn create_elements(&self) -> Elements {
        Elements {
            beams: self.create_beams(),
            masses: self.create_masses(),
            springs: self.create_springs(),
        }
    }

    pub fn create_masses(&self) -> Masses {
        Masses::new(&self.mass_elements, &self.gravity, &self.nodes)
    }

    /// Creates and returns a node builder for adding a new node to the model
    pub fn add_node(&mut self) -> NodeBuilder {
        self.nodes.push(Node {
            id: self.nodes.len(),
            s: 0.,
            x: [0., 0., 0., 1., 0., 0., 0.],
            u: [0., 0., 0., 1., 0., 0., 0.],
            v: [0., 0., 0., 0., 0., 0.],
            vd: [0., 0., 0., 0., 0., 0.],
            active_dofs: ActiveDOFs::None,
        });

        // Return builder
        NodeBuilder {
            node: self.nodes.last_mut().unwrap(),
            built: false,
        }
    }

    /// Creates and returns state object
    pub fn create_state(&self) -> State {
        // calculate number of quadrature points
        let mut nqp = 0;

        self.beam_elements
            .iter()
            .for_each(|elem| nqp += elem.quadrature.points.len());

        let mut n_prony = 1;

        if self.beam_elements.len() > 0 {
            match &self.beam_elements[0].damping {
                Damping::Viscoelastic(_, tau_i) => n_prony = tau_i.nrows(),
                _ => (), // already set to n_prony = 1
            }
        }

        State::new(&self.nodes, nqp, n_prony)
    }

    /// Add rigid constraint
    pub fn add_rigid_constraint(&mut self, base_node_id: usize, target_node_id: usize) -> usize {
        let base_node = &self.nodes[base_node_id];
        let target_node = &self.nodes[target_node_id];
        self.constraints.push(ConstraintInput {
            id: self.constraints.len(),
            kind: ConstraintKind::Rigid,
            node_id_base: base_node_id,
            node_id_target: target_node_id,
            x0: Col::<f64>::from_fn(3, |i| target_node.x[i] - base_node.x[i]),
            vec: Col::zeros(3),
        });
        self.constraints.last().unwrap().id
    }

    /// Add prescribed constraint
    pub fn add_prescribed_constraint(&mut self, target_node_id: usize) -> usize {
        self.constraints.push(ConstraintInput {
            id: self.constraints.len(),
            kind: ConstraintKind::Prescribed,
            node_id_base: 0,
            node_id_target: target_node_id,
            x0: Col::<f64>::zeros(3),
            vec: Col::zeros(3),
        });
        self.constraints.last().unwrap().id
    }

    pub fn add_beam_element(
        &mut self,
        node_ids: &[usize],
        quadrature: &Quadrature,
        sections: &[BeamSection],
        damping: Damping,
    ) -> usize {
        let id = self.beam_elements.len();
        self.beam_elements.push(BeamElement {
            id,
            node_ids: node_ids.to_vec(),
            quadrature: quadrature.clone(),
            sections: sections
                .iter()
                .map(|s| BeamSection {
                    s: s.s,
                    m_star: s.m_star.clone(),
                    c_star: s.c_star.clone(),
                })
                .collect_vec(),
            damping,
        });
        id
    }

    pub fn add_mass_element(&mut self, node_id: usize, mass_matrix: Mat<f64>) -> usize {
        let id = self.mass_elements.len();
        self.mass_elements.push(MassElement {
            id,
            node_id,
            m: mass_matrix,
        });
        id
    }

    pub fn create_beams(&self) -> Beams {
        Beams::new(&self.beam_elements, &self.gravity, &self.nodes)
    }

    pub fn create_springs(&self) -> Springs {
        Springs::new(&self.spring_elements, &self.nodes)
    }

    pub fn add_spring_element(
        &mut self,
        node_1_id: usize,
        node_2_id: usize,
        stiffness: f64,
        undeformed_length: Option<f64>,
    ) -> usize {
        let id = self.spring_elements.len();
        self.spring_elements.push(SpringElement {
            id,
            node_ids: [node_1_id, node_2_id],
            stiffness,
            undeformed_length,
        });
        id
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_solver_tolerance(&mut self, x_tol: f64, phi_tol: f64) {
        self.solver_abs_tol = x_tol;
        self.solver_rel_tol = phi_tol;
    }
}

//------------------------------------------------------------------------------
// Degrees of freedom
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Builder
//------------------------------------------------------------------------------

pub struct NodeBuilder<'a> {
    node: &'a mut Node,
    built: bool,
}

impl<'a> NodeBuilder<'a> {
    /// Sets position of node within beam element
    pub fn element_location(self, s: f64) -> Self {
        self.node.s = s;
        self.node.active_dofs = ActiveDOFs::All;
        self
    }

    /// Sets initial position
    pub fn position(self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.x[3] = w;
        self.node.x[4] = i;
        self.node.x[5] = j;
        self.node.x[6] = k;
        self.node.active_dofs = ActiveDOFs::All;
        self
    }

    /// Sets initial displacement
    pub fn displacement(self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.u[3] = w;
        self.node.u[4] = i;
        self.node.u[5] = j;
        self.node.u[6] = k;
        self.node.active_dofs = ActiveDOFs::All;
        self
    }

    /// Sets initial velocity
    pub fn velocity(self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.v[3] = i;
        self.node.v[4] = j;
        self.node.v[5] = k;
        self.node.active_dofs = ActiveDOFs::All;
        self
    }

    /// Sets initial acceleration
    pub fn acceleration(self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.vd[3] = i;
        self.node.vd[4] = j;
        self.node.vd[5] = k;
        self.node.active_dofs = ActiveDOFs::All;
        self
    }

    /// Sets initial position
    pub fn position_xyz(self, x: f64, y: f64, z: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Translation);
        self
    }

    /// Sets initial orientation from quaternion
    pub fn orientation(self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.x[3] = w;
        self.node.x[4] = x;
        self.node.x[5] = y;
        self.node.x[6] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Rotation);
        self
    }

    /// Sets translational displacement
    pub fn translation_displacement(self, x: f64, y: f64, z: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Translation);
        self
    }

    /// Sets angular displacement with quaternion
    pub fn angular_displacement(self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.u[3] = w;
        self.node.u[4] = x;
        self.node.u[5] = y;
        self.node.u[6] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Rotation);
        self
    }

    /// Sets translational velocity
    pub fn translation_velocity(self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Translation);
        self
    }

    /// Sets angular velocity
    pub fn angular_velocity(self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[3] = x;
        self.node.v[4] = y;
        self.node.v[5] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Rotation);
        self
    }

    /// Sets translational acceleration
    pub fn translation_acceleration(self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Translation);
        self
    }

    /// Sets angular acceleration
    pub fn angular_acceleration(self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[3] = x;
        self.node.vd[4] = y;
        self.node.vd[5] = z;
        self.node.active_dofs.add_dofs(ActiveDOFs::Rotation);
        self
    }

    pub fn build(mut self) -> usize {
        self.built = true;
        self.node.id
    }
}

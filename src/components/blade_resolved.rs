use std::vec;

use itertools::Itertools;

use crate::{node::Node, state::State};

pub struct BladeResolvedCoupling {
    bodies: Vec<Body>,
}

impl BladeResolvedCoupling {
    /// Create a new BladeResolvedCoupling component
    pub fn new(input: &[BodyInput], nodes: &[Node]) -> Self {
        Self {
            bodies: input.iter().map(|elem| Body::new(elem, nodes)).collect(),
        }
    }
}

pub struct BodyInput {
    pub id: usize,                 // Element ID
    pub beam_node_ids: Vec<usize>, // Node IDs for the beam element
}

pub struct Body {
    pub id: usize,              // Body ID
    pub node_ids: Vec<usize>,   // Beam node IDs for this aero element
    pub node_xi: Vec<f64>,      // Beam node locations on reference axis `[n_nodes]`
    pub node_xr: Vec<[f64; 7]>, // Beam node reference position `[n_nodes][7]`
    pub node_u: Vec<[f64; 7]>,  // Beam node displacements `[n_nodes][7]`
    pub node_v: Vec<[f64; 6]>,  // Beam node velocities `[n_nodes][6]`
    pub node_f: Vec<[f64; 6]>,  // Beam node forces `[n_nodes][6]`
}

impl Body {
    pub fn new(input: &BodyInput, nodes: &[Node]) -> Self {
        Self {
            id: input.id,
            node_ids: input.beam_node_ids.clone(),
            node_xi: input
                .beam_node_ids
                .iter()
                .map(|&id| 2. * nodes[id].s - 1.)
                .collect(),
            node_xr: input.beam_node_ids.iter().map(|&id| nodes[id].xr).collect(),
            node_u: input.beam_node_ids.iter().map(|&id| nodes[id].u).collect(),
            node_v: input.beam_node_ids.iter().map(|&id| nodes[id].v).collect(),
            node_f: vec![[0.0; 6]; input.beam_node_ids.len()],
        }
    }

    // Update node displacement and velocity values from state
    pub fn update_nodes(&mut self, state: &State) {
        self.node_ids.iter().enumerate().for_each(|(i, &node_id)| {
            self.node_u[i]
                .copy_from_slice(state.u.col(node_id).try_as_col_major().unwrap().as_slice());
            self.node_v[i]
                .copy_from_slice(state.v.col(node_id).try_as_col_major().unwrap().as_slice());
        });
    }

    /// Return weights to interpolate node values to a point on the structural reference axis.
    /// The point is given as a normalized value between 0 and 1.
    pub fn point_interpolation_weights(&self, s: f64) -> Vec<f64> {
        let xi = 2. * s - 1.; // Convert to normalized reference axis [-1, 1]
        lagrange_polynomial(xi, &self.node_xi)
    }

    /// Calculate vector of reference position and orientation at the point
    /// corresponding to the given interpolation weights. Position [x, y, z]
    /// is represented by the first three components, while rotational displacement
    /// is represented by the last four components which is a quaternion in [w, i, j, k] format.
    pub fn get_reference_position(&self, weights: &[f64]) -> [f64; 7] {
        interpolate_position_displacement(weights, &self.node_xr)
    }

    /// Calculate vector of current position and orientation at the point
    /// corresponding to the given interpolation weights. Position [x, y, z]
    /// is represented by the first three components, while rotational displacement
    /// is represented by the last four components which is a quaternion in [w, i, j, k] format.
    pub fn get_current_position(&self, weights: &[f64]) -> [f64; 7] {
        // Get reference position
        let xr = interpolate_position_displacement(weights, &self.node_xr);

        // Get displacement from reference
        let u = interpolate_position_displacement(weights, &self.node_u);

        // Compose rotational displacement and reference orientation
        let uxr = quat_compose([u[0], u[1], u[2], u[3]], [xr[0], xr[1], xr[2], xr[3]]);

        // Return vector of current position and orientation
        [
            xr[0] + u[0],
            xr[1] + u[1],
            xr[2] + u[2],
            uxr[0],
            uxr[1],
            uxr[2],
            uxr[3],
        ]
    }

    /// Calculate vector of translational and rotational displacement at the point
    /// corresponding to the given interpolation weights. Translation displacement [x, y, z]
    /// is represented by the first three components, while rotational displacement
    /// is represented by the last four components which is a quaternion in [w, i, j, k] format.
    pub fn get_displacement(&self, weights: &[f64]) -> [f64; 7] {
        interpolate_position_displacement(weights, &self.node_u)
    }

    /// Calculate vector of translational and rotational velocity at the point
    /// corresponding to the given interpolation weights. Translational velocity [x, y, z]
    /// is represented by the first three components, while rotational velocity
    /// is represented by the last three components [omega_x, omega_y, omega_z].
    pub fn get_velocity(&self, weights: &[f64]) -> [f64; 6] {
        interpolate_velocity(weights, &self.node_v)
    }

    /// Set forces on nodes
    pub fn set_node_forces(&mut self, node_forces: &[[f64; 6]]) {
        self.node_f.copy_from_slice(node_forces);
    }
}

pub fn distribute_point_force_to_nodes(weights: &[f64], point_force: [f64; 6]) -> Vec<[f64; 6]> {
    weights
        .iter()
        .map(|&weight| {
            [
                weight * point_force[0],
                weight * point_force[1],
                weight * point_force[2],
                weight * point_force[3],
                weight * point_force[4],
                weight * point_force[5],
            ]
        })
        .collect_vec()
}

fn interpolate_position_displacement(weights: &[f64], node_data: &[[f64; 7]]) -> [f64; 7] {
    let mut u = [0.0; 7];
    weights
        .iter()
        .zip(node_data.iter())
        .for_each(|(&weight, node_u)| {
            (0..7).for_each(|i| {
                u[i] += weight * node_u[i];
            });
        });
    // Normalize quaternion so it has unit length
    let m = (u[3] * u[3] + u[4] * u[4] + u[5] * u[5] + u[6] * u[6]).sqrt();
    u[3] /= m;
    u[4] /= m;
    u[5] /= m;
    u[6] /= m;
    u
}

fn interpolate_velocity(weights: &[f64], node_data: &[[f64; 6]]) -> [f64; 6] {
    let mut v = [0.0; 6];
    weights
        .iter()
        .zip(node_data.iter())
        .for_each(|(&weight, node_v)| {
            (0..6).for_each(|i| {
                v[i] += weight * node_v[i];
            });
        });
    v
}

fn quat_compose(q1: [f64; 4], q2: [f64; 4]) -> [f64; 4] {
    [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ]
}

pub fn lagrange_polynomial(x: f64, xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(j, &xj)| {
            xs.iter()
                .enumerate()
                .filter(|(m, _)| *m != j)
                .map(|(_, &xm)| (x - xm) / (xj - xm))
                .product()
        })
        .collect()
}

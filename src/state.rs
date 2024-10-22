use std::rc::Rc;

use itertools::Itertools;
use ndarray::{arr2, Array2, Array3, Zip};

use crate::{node::Node, quaternion::Quaternion};

pub struct State {
    num_system_nodes: usize, // Number of system nodes
    x0: Array2<f64>,         // [n_nodes][7]    Initial global position/rotation
    x: Array2<f64>,          // [n_nodes][7]    Current global position/rotation
    q_delta: Array2<f64>,    // [n_nodes][6]    Displacement increment
    q_prev: Array2<f64>,     // [n_nodes][7]    Previous state
    q: Array2<f64>,          // [n_nodes][7]    Current state
    v: Array2<f64>,          // [n_nodes][6]    Velocity
    vd: Array2<f64>,         // [n_nodes][6]    Acceleration
    a: Array2<f64>,          // [n_nodes][6]    Algorithmic acceleration
    tangent: Array3<f64>,    // [n_nodes][6][6] Algorithmic acceleration
}

impl State {
    pub fn new(nodes: Vec<Rc<Node>>) -> Self {
        let num_system_nodes = nodes.len();
        let mut state = Self {
            num_system_nodes,
            x0: arr2(&nodes.iter().map(|n| n.x).collect_vec()),
            x: Array2::zeros((num_system_nodes, 7)),
            q_delta: Array2::zeros((num_system_nodes, 6)),
            q_prev: Array2::zeros((num_system_nodes, 7)),
            q: arr2(&nodes.iter().map(|n| n.u).collect_vec()),
            v: arr2(&nodes.iter().map(|n| n.v).collect_vec()),
            vd: arr2(&nodes.iter().map(|n| n.vd).collect_vec()),
            a: Array2::zeros((num_system_nodes, 6)),
            tangent: Array3::zeros((num_system_nodes, 6, 6)),
        };
        state.q_prev.assign(&state.q);
        state
    }

    pub fn calculate_x(mut self) {}
    pub fn calculate_q(mut self, h: f64) {
        Zip::from(self.q.rows_mut())
            .and(self.q_delta.rows())
            .and(self.q_prev.rows())
            .for_each(|mut q, q_delta, q_prev| {
                let rot =
                    Quaternion::from_axis_angle(&[q_delta[3] * h, q_delta[4] * h, q_delta[5] * h])
                        .compose(&Quaternion::from_vec(&[
                            q_prev[3], q_prev[4], q_prev[5], q_prev[6],
                        ]))
                        .as_vec();
                q[0] = q_prev[0] + q_delta[0] * h;
                q[1] = q_prev[1] + q_delta[1] * h;
                q[2] = q_prev[2] + q_delta[2] * h;
            });
    }
}

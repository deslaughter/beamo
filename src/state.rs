use std::rc::Rc;

use crate::quaternion::Quat;
use faer::{unzipped, zipped, Col, Mat};
use itertools::izip;

use crate::node::Node;

pub struct State {
    num_system_nodes: usize, // Number of system nodes
    x0: Mat<f64>,            // [7][n_nodes]    Initial global position/rotation
    x: Mat<f64>,             // [7][n_nodes]    Current global position/rotation
    q_delta: Mat<f64>,       // [6][n_nodes]    Displacement increment
    q_prev: Mat<f64>,        // [7][n_nodes]    Previous state
    u: Mat<f64>,             // [7][n_nodes]    Current state
    v: Mat<f64>,             // [6][n_nodes]    Velocity
    vd: Mat<f64>,            // [6][n_nodes]    Acceleration
    a: Mat<f64>,             // [6][n_nodes]    Algorithmic acceleration
    tangent: Mat<f64>,       // [6][6][n_nodes] Tangent matrices
}

impl State {
    pub fn new(nodes: Vec<Rc<Node>>) -> Self {
        let num_system_nodes = nodes.len();
        let mut state = Self {
            num_system_nodes,
            x0: Mat::from_fn(7, num_system_nodes, |i, j| nodes[j].x[i]),
            x: Mat::zeros(7, num_system_nodes),
            q_delta: Mat::zeros(6, num_system_nodes),
            q_prev: Mat::from_fn(7, num_system_nodes, |i, j| nodes[j].u[i]),
            u: Mat::zeros(7, num_system_nodes),
            v: Mat::from_fn(6, num_system_nodes, |i, j| nodes[j].v[i]),
            vd: Mat::from_fn(6, num_system_nodes, |i, j| nodes[j].vd[i]),
            a: Mat::zeros(6, num_system_nodes),
            tangent: Mat::zeros(6 * 6, num_system_nodes),
        };
        state.calculate_q();
        state.calculate_x();
        state
    }

    pub fn calculate_q(&mut self) {
        // Get translation parts of matrices
        let mut q_t = self.u.subrows_mut(0, 3);
        let q_prev_t = self.q_prev.subrows(0, 3);
        let q_delta_t = self.q_delta.subrows(0, 3);

        // Calculate change in position
        zipped!(&mut q_t, &q_prev_t, &q_delta_t).for_each(|unzipped!(mut q, q_prev, q_delta)| {
            *q = *q_prev + *q_delta;
        });

        // Get rotation parts of matrices
        let r = self.u.subrows_mut(3, 4);
        let r_prev = self.q_prev.subrows(3, 4);
        let rv_delta = self.q_delta.subrows(3, 3);

        // Calculate change in rotation
        let mut q_delta = Col::<f64>::zeros(4);
        izip!(r_prev.col_iter(), rv_delta.col_iter(), r.col_iter_mut()).for_each(
            |(q_prev, rv_delta, mut q)| {
                q_delta.as_mut().quat_from_rotation_vector(rv_delta);
                q.quat_compose(q_delta.as_ref(), q_prev);
            },
        );
    }

    pub fn calculate_x(&mut self) {
        let mut x_t = self.x.subrows_mut(0, 3);
        let x0_t = self.x0.subrows(0, 3);
        let q_t = self.u.subrows(0, 3);

        // Calculate current position
        zipped!(&mut x_t, &x0_t, &q_t).for_each(|unzipped!(mut x, x0, q)| {
            *x = *x0 + *q;
        });

        let x_r = self.x.subrows_mut(3, 4);
        let x0_r = self.x0.subrows(3, 4);
        let q_r = self.u.subrows(3, 4);

        // Calculate current rotation
        izip!(x0_r.col_iter(), q_r.col_iter(), x_r.col_iter_mut())
            .for_each(|(x0, q, mut x)| x.quat_compose(q, x0));
    }
}

#[cfg(test)]
mod tests {

    use std::f64::consts::PI;

    use faer::{assert_matrix_eq, col, mat};

    use super::*;

    fn create_state() -> State {
        let mut q1: Col<f64> = Col::zeros(4);
        let mut q2: Col<f64> = Col::zeros(4);
        q1.as_mut()
            .quat_from_axis_angle(90. * PI / 180., col![1., 0., 0.].as_ref());
        q2.as_mut()
            .quat_from_axis_angle(45. * PI / 180., col![0., 1., 0.].as_ref());
        State::new(vec![
            Rc::new(Node {
                id: 0,
                x: [3., 5., 7., q1[0], q1[1], q1[2], q1[3]],
                u: [4., 6., 8., q1[0], q1[1], q1[2], q1[3]],
                v: [1., 2., 3., 4., 5., 6.],
                vd: [7., 8., 9., 10., 11., 12.],
            }),
            Rc::new(Node {
                id: 1,
                x: [2., -3., -5., q1[0], q1[1], q1[2], q1[3]],
                u: [1., 1., 6., q2[0], q2[1], q2[2], q2[3]],
                v: [-1., -2., -3., -4., -5., -6.],
                vd: [-7., -8., -9., -10., -11., -12.],
            }),
        ])
    }

    #[test]
    fn test_state_num_system_nodes() {
        let state = create_state();
        assert_eq!(state.num_system_nodes, 2);
    }

    #[test]
    fn test_state_x0() {
        let state = create_state();
        assert_matrix_eq!(
            state.x0,
            mat![
                [3.0000000000000000, 2.0000000000000000],
                [5.0000000000000000, -3.0000000000000000],
                [7.0000000000000000, -5.0000000000000000],
                [0.7071067811865476, 0.7071067811865476],
                [0.7071067811865475, 0.7071067811865475],
                [0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000],
            ],
            comp = float
        );
    }

    #[test]
    fn test_state_u() {
        let state = create_state();
        assert_matrix_eq!(
            state.u,
            mat![
                [4.0000000000000000, 1.0000000000000000],
                [6.0000000000000000, 1.0000000000000000],
                [8.0000000000000000, 6.0000000000000000],
                [0.7071067811865476, 0.9238795325112867],
                [0.7071067811865475, 0.0000000000000000],
                [0.0000000000000000, 0.3826834323650898],
                [0.0000000000000000, 0.0000000000000000],
            ],
            comp = float
        );
    }

    #[test]
    fn test_state_x() {
        let state = create_state();
        assert_matrix_eq!(
            state.x,
            mat![
                [7.000000000000000, 3.000000000000000],
                [11.000000000000000, -2.000000000000000],
                [15.000000000000000, 1.000000000000000],
                [0.000000000000000, 0.6532814824381883],
                [1.000000000000000, 0.6532814824381882],
                [0.000000000000000, 0.2705980500730985],
                [0.000000000000000, -0.27059805007309845],
            ],
            comp = float
        );
    }

    #[test]
    fn test_state_v() {
        let state = create_state();
        assert_matrix_eq!(
            state.v,
            mat![
                [1., -1.],
                [2., -2.],
                [3., -3.],
                [4., -4.],
                [5., -5.],
                [6., -6.],
            ],
            comp = float
        );
    }

    #[test]
    fn test_state_vd() {
        let state = create_state();
        assert_matrix_eq!(
            state.vd,
            mat![
                [7., -7.],
                [8., -8.],
                [9., -9.],
                [10., -10.],
                [11., -11.],
                [12., -12.],
            ],
            comp = float
        );
    }
}

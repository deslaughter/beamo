use std::rc::Rc;

use crate::quaternion::Quat;
use faer::{mat, unzipped, zipped, Mat, Row, RowMut, RowRef};

use crate::node::Node;

pub struct State {
    num_system_nodes: usize, // Number of system nodes
    x0: Mat<f64>,            // [n_nodes][7]    Initial global position/rotation
    x: Mat<f64>,             // [n_nodes][7]    Current global position/rotation
    q_delta: Mat<f64>,       // [n_nodes][6]    Displacement increment
    q_prev: Mat<f64>,        // [n_nodes][7]    Previous state
    u: Mat<f64>,             // [n_nodes][7]    Current state
    v: Mat<f64>,             // [n_nodes][6]    Velocity
    vd: Mat<f64>,            // [n_nodes][6]    Acceleration
    a: Mat<f64>,             // [n_nodes][6]    Algorithmic acceleration
    tangent: Mat<f64>,       // [n_nodes][6][6] Tangent matrices
}

impl State {
    pub fn new(nodes: Vec<Rc<Node>>) -> Self {
        let num_system_nodes = nodes.len();
        let mut state = Self {
            num_system_nodes,
            x0: Mat::from_fn(num_system_nodes, 7, |i, j| nodes[i].x[j]),
            x: Mat::zeros(num_system_nodes, 7),
            q_delta: Mat::zeros(num_system_nodes, 6),
            q_prev: Mat::from_fn(num_system_nodes, 7, |i, j| nodes[i].u[j]),
            u: Mat::zeros(num_system_nodes, 7),
            v: Mat::from_fn(num_system_nodes, 6, |i, j| nodes[i].v[j]),
            vd: Mat::from_fn(num_system_nodes, 6, |i, j| nodes[i].vd[j]),
            a: Mat::zeros(num_system_nodes, 6),
            tangent: Mat::zeros(num_system_nodes, 6 * 6),
        };
        state.calculate_q();
        state.calculate_x();
        state
    }

    pub fn calculate_q(&mut self) {
        // Get translation parts of matrices
        let mut q_t = self.u.subcols_mut(0, 3);
        let q_prev_t = self.q_prev.subcols(0, 3);
        let q_delta_t = self.q_delta.subcols(0, 3);

        // Calculate change in position
        zipped!(&mut q_t, &q_prev_t, &q_delta_t).for_each(|unzipped!(mut q, q_prev, q_delta)| {
            *q = *q_prev + *q_delta;
        });

        // Get rotation parts of matrices
        let r = self.u.subcols_mut(3, 4);
        let r_prev = self.q_prev.subcols(3, 4);
        let rv_delta = self.q_delta.subcols(3, 3);

        // Calculate change in rotation
        let mut q_delta = Row::<f64>::zeros(4);
        r_prev
            .row_iter()
            .zip(rv_delta.row_iter())
            .zip(r.row_iter_mut())
            .for_each(|((q_prev, rv_delta), q)| {
                q_delta.as_mut().quat_from_rotation_vector(rv_delta);
                q.quat_compose(q_delta.as_ref(), q_prev);
            });
    }

    pub fn calculate_x(&mut self) {
        let mut x_t = self.x.subcols_mut(0, 3);
        let x0_t = self.x0.subcols(0, 3);
        let q_t = self.u.subcols(0, 3);

        // Calculate current position
        zipped!(&mut x_t, &x0_t, &q_t).for_each(|unzipped!(mut x, x0, q)| {
            *x = *x0 + *q;
        });

        let x_r = self.x.subcols_mut(3, 4);
        let x0_r = self.x0.subcols(3, 4);
        let q_r = self.u.subcols(3, 4);

        // Calculate current rotation
        x0_r.row_iter()
            .zip(q_r.row_iter())
            .zip(x_r.row_iter_mut())
            .for_each(|((x0, q), x)| x.quat_compose(q, x0));
    }
}

#[cfg(test)]
mod tests {

    use std::f64::consts::PI;

    use faer::row;

    use super::*;

    #[test]
    fn test_me() {
        let mut q1: Row<f64> = Row::zeros(4);
        let mut q2: Row<f64> = Row::zeros(4);
        q1.as_mut()
            .quat_from_axis_angle(90. * PI / 180., row![1., 0., 0.].as_ref());
        q2.as_mut()
            .quat_from_axis_angle(45. * PI / 180., row![0., 1., 0.].as_ref());
        let state = State::new(vec![
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
        ]);

        assert_eq!(state.num_system_nodes, 2);

        faer::assert_matrix_eq!(
            state.x0,
            mat![
                [3., 5., 7., 0.7071067811865476, 0.7071067811865475, 0.0, 0.0],
                [
                    2.,
                    -3.,
                    -5.,
                    0.7071067811865476,
                    0.7071067811865475,
                    0.0,
                    0.0
                ]
            ],
            comp = float
        );

        faer::assert_matrix_eq!(
            state.u,
            mat![
                [4., 6., 8., 0.7071067811865476, 0.7071067811865475, 0.0, 0.0],
                [1., 1., 6., 0.9238795325112867, 0.0, 0.3826834323650898, 0.0]
            ],
            comp = float
        );

        faer::assert_matrix_eq!(
            state.x,
            mat![
                [7., 11., 15., 0., 1., 0., 0.],
                [
                    3.,
                    -2.,
                    1.,
                    0.6532814824381883,
                    0.6532814824381882,
                    0.2705980500730985,
                    -0.27059805007309845
                ]
            ],
            comp = float
        );

        faer::assert_matrix_eq!(
            state.v,
            mat![[1., 2., 3., 4., 5., 6.], [-1., -2., -3., -4., -5., -6.]],
            comp = float
        );

        faer::assert_matrix_eq!(
            state.vd,
            mat![
                [7., 8., 9., 10., 11., 12.],
                [-7., -8., -9., -10., -11., -12.]
            ],
            comp = float
        );
    }
}

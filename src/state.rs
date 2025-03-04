use crate::node::Node;
use crate::util::Quat;
use faer::{unzipped, zipped, Col, Mat, MatRef};
use itertools::izip;

#[derive(Debug, Clone)]
pub struct State {
    /// Number of nodes
    pub n_nodes: usize,
    /// Initial global position/rotation `[7][n_nodes]`
    pub x0: Mat<f64>,
    /// Current global position/rotation `[7][n_nodes]`
    pub x: Mat<f64>,
    /// Displacement increment `[6][n_nodes]`
    pub u_delta: Mat<f64>,
    /// Previous state `[7][n_nodes]`
    pub u_prev: Mat<f64>,
    /// Current state `[7][n_nodes]`
    pub u: Mat<f64>,
    /// Velocity `[6][n_nodes]`
    pub v: Mat<f64>,
    /// Acceleration `[6][n_nodes]`
    pub vd: Mat<f64>,
    /// Algorithmic acceleration `[6][n_nodes]`
    pub a: Mat<f64>,
    /// Viscoelastic history states `[6*n_terms][n_quadrature]`
    pub visco_hist: Mat<f64>,
    /// Viscoelastic history states contributions from
    /// time n in the step to time n+1 `[6][n_quadrature]`
    /// In the sectional coordinates
    pub strain_dot_n: Mat<f64>,
}

impl State {
    pub fn new(nodes: &[Node], nqp: usize, n_prony: usize) -> Self {
        let n_nodes = nodes.len();
        let mut state = Self {
            n_nodes,
            x0: Mat::from_fn(7, n_nodes, |i, j| nodes[j].x[i]),
            x: Mat::from_fn(7, n_nodes, |i, j| nodes[j].x[i]),
            u_delta: Mat::zeros(6, n_nodes),
            u_prev: Mat::from_fn(7, n_nodes, |i, j| nodes[j].u[i]),
            u: Mat::zeros(7, n_nodes),
            v: Mat::from_fn(6, n_nodes, |i, j| nodes[j].v[i]),
            vd: Mat::from_fn(6, n_nodes, |i, j| nodes[j].vd[i]),
            a: Mat::from_fn(6, n_nodes, |i, j| nodes[j].vd[i]),
            visco_hist: Mat::zeros(6 * n_prony, nqp),
            strain_dot_n: Mat::zeros(6, nqp),
        };
        state.calc_displacement(0.);
        state.calculate_x();
        state
    }

    /// Calculate current displacement from previous displacement and displacement increment
    pub fn calc_displacement(&mut self, h: f64) {
        // Get translation parts of matrices
        let mut q_t = self.u.subrows_mut(0, 3);
        let q_prev_t = self.u_prev.subrows(0, 3);
        let q_delta_t = self.u_delta.subrows(0, 3);

        // Calculate change in position
        zipped!(&mut q_t, &q_prev_t, &q_delta_t).for_each(|unzipped!(q, q_prev, q_delta)| {
            *q = *q_prev + *q_delta * h;
        });

        // Get rotation parts of matrices
        let r = self.u.subrows_mut(3, 4);
        let r_prev = self.u_prev.subrows(3, 4);
        let rv_delta = self.u_delta.subrows(3, 3);

        // Calculate change in rotation
        let mut q_delta = Col::<f64>::zeros(4);
        let mut h_rv_delta = Col::<f64>::zeros(3);
        izip!(r_prev.col_iter(), rv_delta.col_iter(), r.col_iter_mut()).for_each(
            |(q_prev, rv_delta, mut q)| {
                zipped!(&mut h_rv_delta, &rv_delta)
                    .for_each(|unzipped!(h_rv_delta, rv_delta)| *h_rv_delta = *rv_delta * h);
                q_delta
                    .as_mut()
                    .quat_from_rotation_vector(h_rv_delta.as_ref());
                q.quat_compose(q_delta.as_ref(), q_prev); // This should be reverse of reference paper.
            },
        );
    }

    /// Calculate current position from initial position and displacement
    pub fn calculate_x(&mut self) {
        let mut x_t = self.x.subrows_mut(0, 3);
        let x0_t = self.x0.subrows(0, 3);
        let q_t = self.u.subrows(0, 3);

        // Calculate current position
        zipped!(&mut x_t, &x0_t, &q_t).for_each(|unzipped!(x, x0, q)| {
            *x = *x0 + *q;
        });

        let x_r = self.x.subrows_mut(3, 4);
        let x0_r = self.x0.subrows(3, 4);
        let q_r = self.u.subrows(3, 4);

        // Calculate current rotation
        izip!(x0_r.col_iter(), q_r.col_iter(), x_r.col_iter_mut())
            .for_each(|(x0, q, mut x)| x.quat_compose(q, x0));
    }

    /// Calculate state prediction at end of next time step
    pub fn predict_next_state(
        &mut self,
        h: f64,
        beta: f64,
        gamma: f64,
        alpha_m: f64,
        alpha_f: f64,
    ) {
        self.u_prev.copy_from(&self.u);
        zipped!(&mut self.u_delta, &mut self.v, &mut self.vd, &mut self.a).for_each(
            |unzipped!(q_delta, v, vd, a)| {
                let (v_p, vd_p, a_p) = (*v, *vd, *a);
                *vd = 0.;
                *a = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
                *v = v_p + h * (1. - gamma) * a_p + gamma * h * *a;
                *q_delta = v_p + (0.5 - beta) * h * a_p + beta * h * *a;
            },
        );

        self.calc_displacement(h);
        self.calculate_x();
    }

    /// Update state dynamic prediction from iteration increment
    pub fn update_dynamic_prediction(
        &mut self,
        h: f64,
        beta_prime: f64,
        gamma_prime: f64,
        x_delta: MatRef<f64>,
    ) {
        zipped!(&mut self.u_delta, &mut self.v, &mut self.vd, &x_delta).for_each(
            |unzipped!(q_delta, v, vd, x_delta)| {
                *q_delta += *x_delta / h;
                *v += gamma_prime * *x_delta;
                *vd += beta_prime * *x_delta;
            },
        );

        self.calc_displacement(h);
        self.calculate_x();
    }

    /// Update state static prediction from iteration increment
    pub fn update_static_prediction(&mut self, h: f64, x_delta: MatRef<f64>) {
        zipped!(&mut self.u_delta, &x_delta).for_each(|unzipped!(q_delta, x_delta)| {
            *q_delta += *x_delta / h;
        });

        self.calc_displacement(h);
        self.calculate_x();
    }

    /// Calculate algorithmic acceleration for next step
    pub fn update_algorithmic_acceleration(&mut self, alpha_m: f64, alpha_f: f64) {
        self.a += (1. - alpha_f) / (1. - alpha_m) * &self.vd;
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::Model;
    use faer::{assert_matrix_eq, col, mat};
    use std::f64::consts::PI;

    fn create_state() -> State {
        let mut q1: Col<f64> = Col::zeros(4);
        let mut q2: Col<f64> = Col::zeros(4);
        q1.as_mut()
            .quat_from_axis_angle(90. * PI / 180., col![1., 0., 0.].as_ref());
        q2.as_mut()
            .quat_from_axis_angle(45. * PI / 180., col![0., 1., 0.].as_ref());
        let mut model = Model::new();
        model
            .add_node()
            .position(3., 5., 7., q1[0], q1[1], q1[2], q1[3])
            .displacement(4., 6., 8., q1[0], q1[1], q1[2], q1[3])
            .velocity(1., 2., 3., 4., 5., 6.)
            .acceleration(7., 8., 9., 10., 11., 12.)
            .build();
        model
            .add_node()
            .position(2., -3., -5., q1[0], q1[1], q1[2], q1[3])
            .displacement(1., 1., 6., q2[0], q2[1], q2[2], q2[3])
            .velocity(-1., -2., -3., -4., -5., -6.)
            .acceleration(-7., -8., -9., -10., -11., -12.)
            .build();
        model.create_state()
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
                [7., 11., 15., 0., 1., 0., 0.],
                [
                    3.,
                    -2.,
                    1.,
                    0.6532814824381883,
                    0.6532814824381882,
                    0.2705980500730985,
                    -0.27059805007309845,
                ]
            ]
            .transpose(),
            comp = float
        );
    }

    #[test]
    fn test_state_v() {
        let state = create_state();
        assert_matrix_eq!(
            state.v,
            mat![[1., 2., 3., 4., 5., 6.], [-1., -2., -3., -4., -5., -6.]].transpose(),
            comp = float
        );
    }

    #[test]
    fn test_state_vd() {
        let state = create_state();
        assert_matrix_eq!(
            state.vd,
            mat![
                [7., 8., 9., 10., 11., 12.],
                [-7., -8., -9., -10., -11., -12.],
            ]
            .transpose(),
            comp = float
        );
    }
}

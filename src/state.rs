use crate::{
    node::Node,
    util::{quat_compose, quat_from_rotation_vector},
};
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
    /// Previous step displacement `[7][n_nodes]`
    pub u_prev: Mat<f64>,
    /// Displacement `[7][n_nodes]`
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
        state.calc_step_end(0.);
        state
    }

    /// Calculate current displacement from previous displacement and displacement increment
    pub fn calc_step_end(&mut self, h: f64) {
        // Get translation parts of matrices
        let mut u_np1 = self.u.subrows_mut(0, 3);
        let u_n = self.u_prev.subrows(0, 3);
        let u_delta_n = self.u_delta.subrows(0, 3);
        let mut x_np1 = self.x.subrows_mut(0, 3);
        let x0 = self.x0.subrows(0, 3);

        // Calculate total displacement at end-of-step
        zipped!(&mut u_np1, &u_n, &u_delta_n).for_each(|unzipped!(u_np1, u_n, u_delta_n)| {
            *u_np1 = *u_n + *u_delta_n * h;
        });

        // Calculate absolute rotation at end-of-step
        zipped!(&mut x_np1, &x0, &u_np1).for_each(|unzipped!(x, x0, u)| {
            *x = *x0 + *u;
        });

        // Get rotation parts of matrices
        let r_np1 = self.u.subrows_mut(3, 4);
        let r_n = self.u_prev.subrows(3, 4);
        let ur_delta = self.u_delta.subrows(3, 3);
        let rr0_np1 = self.x.subrows_mut(3, 4);
        let r0 = self.x0.subrows(3, 4);

        // Calculate change in rotation
        let mut q_delta_np1 = Col::<f64>::zeros(4);
        let mut r_delta = Col::<f64>::zeros(3);
        izip!(
            r_n.col_iter(),
            r0.col_iter(),
            ur_delta.col_iter(),
            r_np1.col_iter_mut(),
            rr0_np1.col_iter_mut(),
        )
        .for_each(|(r_n, r0, ur_delta, mut r_np1, rr0_np1)| {
            // Get delta rotation vector
            r_delta.copy_from(&ur_delta * h);

            // Get delta rotation as quaternion
            quat_from_rotation_vector(r_delta.as_ref(), q_delta_np1.as_mut());

            // Compose rotation displacement at start-of-step with delta rotation to get end-of-step rotation displacement
            quat_compose(q_delta_np1.as_ref(), r_n, r_np1.as_mut());

            // Compose end-of-step rotation displacement with initial rotation to get end-of-step absolute rotation
            quat_compose(r_np1.as_ref(), r0, rr0_np1);
        });
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
            |unzipped!(u_delta, v, vd, a)| {
                let (v_p, vd_p, a_p) = (*v, *vd, *a);
                *vd = 0.;
                *a = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
                *v = v_p + h * (1. - gamma) * a_p + gamma * h * *a;
                *u_delta = v_p + (0.5 - beta) * h * a_p + beta * h * *a;
            },
        );

        self.calc_step_end(h);
    }

    /// Update state dynamic prediction from iteration increment
    pub fn update_dynamic_prediction(
        &mut self,
        h: f64,
        beta_prime: f64,
        gamma_prime: f64,
        x_delta: MatRef<f64>,
    ) {
        // Add incremental change in translation/rotation position
        zipped!(&x_delta, &mut self.u_delta).for_each(|unzipped!(x_delta, u_delta)| {
            *u_delta += *x_delta / h;
        });

        // Add incremental change in translational velocity/acceleration
        zipped!(
            &x_delta.subrows(0, 3),
            &mut self.v.subrows_mut(0, 3),
            &mut self.vd.subrows_mut(0, 3)
        )
        .for_each(|unzipped!(x_delta, v, vd)| {
            *v += gamma_prime * *x_delta;
            *vd += beta_prime * *x_delta;
        });

        // Add incremental change in rotational velocity/acceleration
        zipped!(
            &x_delta.subrows(3, 3),
            &mut self.v.subrows_mut(3, 3),
            &mut self.vd.subrows_mut(3, 3)
        )
        .for_each(|unzipped!(x_delta, v, vd)| {
            *v += gamma_prime * *x_delta;
            *vd += beta_prime * *x_delta;
        });

        self.calc_step_end(h);
    }

    /// Update state static prediction from iteration increment
    pub fn update_static_prediction(&mut self, h: f64, x_delta: MatRef<f64>) {
        zipped!(&mut self.u_delta, &x_delta).for_each(|unzipped!(q_delta, x_delta)| {
            *q_delta += *x_delta / h;
        });

        self.calc_step_end(h);
    }

    /// Calculate algorithmic acceleration for next step
    pub fn update_algorithmic_acceleration(&mut self, alpha_m: f64, alpha_f: f64) {
        self.a += (1. - alpha_f) / (1. - alpha_m) * &self.vd;
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{model::Model, util::quat_from_axis_angle};
    use faer::{assert_matrix_eq, col, mat};
    use std::f64::consts::PI;

    fn create_state() -> State {
        let mut q1: Col<f64> = Col::zeros(4);
        let mut q2: Col<f64> = Col::zeros(4);
        quat_from_axis_angle(90. * PI / 180., col![1., 0., 0.].as_ref(), q1.as_mut());
        quat_from_axis_angle(45. * PI / 180., col![0., 1., 0.].as_ref(), q2.as_mut());
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

use std::ops::Div;

use faer::{linalg::matmul::matmul, solvers::SpSolver, unzipped, zipped, Col, Mat, Scale};

use crate::{
    beams::{Beams, ColAsMatRef},
    beams_qp::vec_tilde,
    model::Node,
    state::State,
};

pub struct Constraint {}

pub struct StepParameters {
    h: f64, // time step
    alpha_f: f64,
    alpha_m: f64,
    beta: f64,
    gamma: f64,
    beta_prime: f64,
    gamma_prime: f64,
    max_iter: usize,
    conditioner: f64,
    convergence_atol: f64,
    convergence_rtol: f64,
}

impl StepParameters {
    pub fn new(h: f64, rho_inf: f64, max_iter: usize) -> Self {
        let alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f = rho_inf / (rho_inf + 1.);
        let gamma = 0.5 + alpha_f - alpha_m;
        let beta = 0.25 * (gamma + 0.5) * (gamma + 0.5);
        Self {
            max_iter,
            h,
            alpha_m,
            alpha_f,
            gamma,
            beta,
            gamma_prime: gamma / (h * beta),
            beta_prime: (1. - alpha_m) / (h * h * beta * (1. - alpha_f)),
            conditioner: beta * h * h,
            convergence_atol: 1e-7,
            convergence_rtol: 1e-5,
        }
    }
}

pub struct Solver {
    pub p: StepParameters,
    pub n_nodes: usize,
    pub n_system_dofs: usize, //
    pub n_lambda_dofs: usize, //
    pub n_dofs: usize,        //
    pub kt: Mat<f64>,         // Kt
    pub ct: Mat<f64>,         // Ct
    pub m: Mat<f64>,          // M
    pub t: Mat<f64>,          // T
    pub st: Mat<f64>,         // St
    pub x: Col<f64>,          // x solution vector
    pub x_delta: Mat<f64>,    //
    pub fx: Mat<f64>,         // nodal forces
    pub r: Col<f64>,          // R residual vector
    pub b: Mat<f64>,          // B constraint gradient matrix
    pub lambda: Col<f64>,     //
}

pub struct StepResults {
    pub err: f64,
    pub iter: usize,
    pub converged: bool,
}

impl Solver {
    pub fn new(
        step_parameters: StepParameters,
        nodes: &[Node],
        _constraints: &[Constraint],
    ) -> Self {
        let n_nodes = nodes.len();
        let n_system_dofs = 6 * n_nodes;
        let n_constraint_dofs = 0;
        let n_dofs = n_system_dofs + n_constraint_dofs;

        Solver {
            p: step_parameters,
            n_nodes,
            n_system_dofs,
            n_lambda_dofs: n_constraint_dofs,
            n_dofs,
            kt: Mat::zeros(n_system_dofs, n_system_dofs),
            ct: Mat::zeros(n_system_dofs, n_system_dofs),
            m: Mat::zeros(n_system_dofs, n_system_dofs),
            t: Mat::zeros(n_system_dofs, n_system_dofs),
            fx: Mat::zeros(6, n_nodes),
            st: Mat::zeros(n_dofs, n_dofs),
            x: Col::zeros(n_dofs),
            x_delta: Mat::zeros(6, n_nodes),
            r: Col::zeros(n_dofs),
            b: Mat::zeros(n_constraint_dofs, n_system_dofs),
            lambda: Col::zeros(n_constraint_dofs),
        }
    }

    pub fn step(&mut self, state: &mut State, beams: &mut Beams) -> StepResults {
        state.predict_next_state(
            self.p.h,
            self.p.beta,
            self.p.gamma,
            self.p.alpha_m,
            self.p.alpha_f,
        );

        let mut res = StepResults {
            err: 1000.,
            iter: 1,
            converged: false,
        };

        while res.err > 1. {
            // Reset matrices
            self.m.fill(0.);
            self.kt.fill(0.);
            self.ct.fill(0.);
            self.b.fill(0.);
            self.t
                .copy_from(Mat::<f64>::identity(self.n_system_dofs, self.n_system_dofs));
            self.r.fill(0.);
            self.st.fill(0.);

            // Add beams to system
            beams.calculate_system(state);
            beams.assemble_system(
                self.m.as_mut(),
                self.ct.as_mut(),
                self.kt.as_mut(),
                self.r.as_mut(),
            );

            // Subtract direct nodal loads
            self.fx.col_iter().enumerate().for_each(|(i, f)| {
                let mut r = self.r.subrows_mut(6 * i, 6);
                r -= f;
            });

            // Calculate tangent matrix
            let mut mt = Mat::<f64>::zeros(3, 3);
            state
                .u_delta
                .subrows(3, 3)
                .col_iter()
                .enumerate()
                .for_each(|(i, r_delta)| {
                    let rv = r_delta * self.p.h;
                    let phi = rv.norm_l2();
                    if phi > 1e-16 {
                        let (phi_s, phi_c) = phi.sin_cos();
                        vec_tilde(rv.as_ref(), mt.as_mut());
                        self.t.submatrix_mut(i * 6 + 3, i * 6 + 3, 3, 3).copy_from(
                            Mat::<f64>::identity(3, 3)
                                + &mt * &mt * Scale((1. - phi_s / phi) / (phi * phi))
                                + &mt * Scale((phi_c - 1.) / (phi * phi)),
                        );
                    }
                });

            // Assemble system matrix
            let mut st_sys = self
                .st
                .submatrix_mut(0, 0, self.n_system_dofs, self.n_system_dofs);
            zipped!(&mut st_sys, &self.m, &self.ct).for_each(|unzipped!(mut st, m, ct)| {
                *st = *m * self.p.beta_prime + *ct * self.p.gamma_prime
            });
            matmul(
                st_sys,
                self.kt.as_ref(),
                self.t.as_ref(),
                Some(1.),
                1.,
                faer::Parallelism::None,
            );

            // Make copies of St and R for solving system
            let mut st_c = self.st.clone();
            let mut r_c = self.r.clone();

            // Condition residual
            zipped!(&mut r_c.subrows_mut(0, self.n_system_dofs))
                .for_each(|unzipped!(mut v)| *v *= self.p.conditioner);

            // Condition system
            zipped!(&mut st_c.subrows_mut(0, self.n_system_dofs))
                .for_each(|unzipped!(mut v)| *v *= self.p.conditioner);
            zipped!(&mut st_c.subcols_mut(self.n_system_dofs, self.n_lambda_dofs))
                .for_each(|unzipped!(mut v)| *v /= self.p.conditioner);

            let reduced_dofs = self.n_system_dofs - 6;

            // Solve system
            let lu = st_c
                .submatrix(6, 6, reduced_dofs, reduced_dofs)
                .partial_piv_lu();
            let rhs = r_c.subrows(6, reduced_dofs);
            let x = lu.solve(&rhs);
            self.x.subrows_mut(6, reduced_dofs).copy_from(&x);

            // De-condition solution vector
            zipped!(&mut self.x.subrows_mut(self.n_system_dofs, self.n_lambda_dofs))
                .for_each(|unzipped!(mut v)| *v /= self.p.conditioner);

            // Negate solution vector
            zipped!(&mut self.x).for_each(|unzipped!(mut v)| *v *= -1.);

            // Reshape solution vector to match state
            self.x_delta.copy_from(
                self.x
                    .subrows(0, self.n_system_dofs)
                    .as_mat_ref(6, self.n_nodes),
            );

            // Calculate convergence error
            res.err = zipped!(&self.x_delta, &state.u_delta)
                .map(|unzipped!(xd, ud)| {
                    *xd / ((*ud * self.p.convergence_rtol).abs() + self.p.convergence_atol)
                })
                .norm_l2()
                .div((self.n_system_dofs as f64).sqrt());

            // Update state prediction
            state.update_prediction(
                self.p.h,
                self.p.beta_prime,
                self.p.gamma_prime,
                self.x_delta.as_ref(),
            );

            // Iteration limit reached return not converged
            if res.iter >= self.p.max_iter {
                return res;
            }

            // Increment iteration count
            res.iter += 1;
        }

        // Converged, update algorithmic acceleration
        state.update_algorithmic_acceleration(self.p.alpha_m, self.p.alpha_f);
        res.converged = true;
        res
    }
}

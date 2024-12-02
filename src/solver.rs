use std::ops::Div;

use faer::{linalg::matmul::matmul, solvers::SpSolver, unzipped, zipped, Col, Mat, Parallelism};
use itertools::izip;

use crate::{
    constraints::Constraints,
    elements::Elements,
    node::{ActiveDOFs, NodeFreedomMap},
    state::State,
    util::vec_tilde,
};

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
    pub nfm: NodeFreedomMap,
    pub elements: Elements,
    pub constraints: Constraints,
    pub n_system: usize,   //
    pub n_lambda: usize,   //
    pub n_dofs: usize,     //
    pub kt: Mat<f64>,      // Kt
    pub ct: Mat<f64>,      // Ct
    pub m: Mat<f64>,       // M
    pub t: Mat<f64>,       // T
    pub st: Mat<f64>,      // St
    pub x: Col<f64>,       // x solution vector
    pub fx: Col<f64>,      // nodal forces
    pub r: Col<f64>,       // R residual vector
    pub phi: Col<f64>,     // R residual vector
    pub b: Mat<f64>,       // B constraint gradient matrix
    pub lambda: Col<f64>,  //
    pub x_delta: Mat<f64>, //
    pub rhs: Col<f64>,     // Right hand side
}

#[derive(Debug)]
pub struct StepResults {
    pub err: f64,
    pub iter: usize,
    pub converged: bool,
}

impl Solver {
    pub fn new(
        step_parameters: StepParameters,
        nfm: NodeFreedomMap,
        elements: Elements,
        constraints: Constraints,
    ) -> Self {
        let n_system_dofs = nfm.total_dofs;
        let n_constraint_dofs = constraints.n_dofs;
        let n_dofs = n_system_dofs + n_constraint_dofs;
        let n_nodes = nfm.node_dofs.len();
        Solver {
            p: step_parameters,
            nfm,
            elements,
            constraints,
            n_system: n_system_dofs,
            n_lambda: n_constraint_dofs,
            n_dofs,
            kt: Mat::zeros(n_system_dofs, n_system_dofs),
            ct: Mat::zeros(n_system_dofs, n_system_dofs),
            m: Mat::zeros(n_system_dofs, n_system_dofs),
            t: Mat::zeros(n_system_dofs, n_system_dofs),
            fx: Col::zeros(n_system_dofs),
            st: Mat::zeros(n_dofs, n_dofs),
            r: Col::zeros(n_system_dofs),
            b: Mat::zeros(n_constraint_dofs, n_system_dofs),
            phi: Col::zeros(n_constraint_dofs),
            lambda: Col::zeros(n_constraint_dofs),
            x_delta: Mat::zeros(6, n_nodes),
            x: Col::zeros(n_dofs),
            rhs: Col::zeros(n_dofs),
        }
    }

    pub fn step(&mut self, state: &mut State) -> StepResults {
        state.predict_next_state(
            self.p.h,
            self.p.beta,
            self.p.gamma,
            self.p.alpha_m,
            self.p.alpha_f,
        );

        // Create step results
        let mut res = StepResults {
            err: 1000.,
            iter: 1,
            converged: false,
        };

        // Loop until converged or max iteration limit reached
        while res.err > 1. {
            //------------------------------------------------------------------
            // Build System
            //------------------------------------------------------------------

            // Reset matrices
            self.m.fill_zero();
            self.kt.fill_zero();
            self.ct.fill_zero();
            self.b.fill_zero();
            self.t.fill_zero();
            self.t.diagonal_mut().column_vector_mut().fill(1.);
            self.r.fill_zero();
            self.lambda.fill_zero();

            // Subtract direct nodal loads
            zipped!(&mut self.r, &self.fx).for_each(|unzipped!(mut r, fx)| *r -= *fx);

            // Add elements to system
            self.elements.assemble_system(
                state,
                &self.nfm,
                self.m.as_mut(),
                self.ct.as_mut(),
                self.kt.as_mut(),
                self.r.as_mut(),
            );

            // Calculate constraints
            self.constraints.assemble_constraints(
                &self.nfm,
                state,
                self.phi.as_mut(),
                self.b.as_mut(),
            );

            // Calculate tangent matrix
            self.populate_tangent_matrix(state);

            // Assemble system matrix
            let mut st_11 = self.st.submatrix_mut(0, 0, self.n_system, self.n_system);
            zipped!(&mut st_11, &self.m, &self.ct).for_each(|unzipped!(mut st, m, ct)| {
                *st = *m * self.p.beta_prime + *ct * self.p.gamma_prime
            });
            matmul(
                st_11,
                self.kt.as_ref(),
                self.t.as_ref(),
                Some(1.),
                1.,
                faer::Parallelism::None,
            );

            // Assemble constraints
            let st_21 = self
                .st
                .submatrix_mut(self.n_system, 0, self.n_lambda, self.n_system);
            matmul(st_21, &self.b, &self.t, None, 1., faer::Parallelism::None);
            let mut st_12 = self
                .st
                .submatrix_mut(0, self.n_system, self.n_system, self.n_lambda);
            zipped!(&mut st_12, &self.b.transpose()).for_each(|unzipped!(mut st, bt)| *st = *bt);
            self.st
                .submatrix_mut(self.n_system, self.n_system, self.n_lambda, self.n_lambda)
                .fill_zero();

            matmul(
                self.r.subrows_mut(0, self.n_system),
                self.b.transpose(),
                self.lambda.as_ref(),
                Some(1.),
                1.,
                faer::Parallelism::None,
            );

            //------------------------------------------------------------------
            // Solve System
            //------------------------------------------------------------------

            // Make copies of St and R for solving system
            let mut st_c = self.st.clone();
            self.rhs.subrows_mut(0, self.n_system).copy_from(&self.r);
            self.rhs
                .subrows_mut(self.n_system, self.n_lambda)
                .copy_from(&self.phi);

            // Condition residual
            zipped!(&mut self.rhs.subrows_mut(0, self.n_system))
                .for_each(|unzipped!(mut v)| *v *= self.p.conditioner);

            // Condition system
            zipped!(&mut st_c.subrows_mut(0, self.n_system))
                .for_each(|unzipped!(mut v)| *v *= self.p.conditioner);
            zipped!(&mut st_c.subcols_mut(self.n_system, self.n_lambda))
                .for_each(|unzipped!(mut v)| *v /= self.p.conditioner);

            // Solve system
            let lu = st_c.partial_piv_lu();
            let x = lu.solve(&self.rhs);
            self.x.copy_from(&x);

            // De-condition solution vector
            zipped!(&mut self.x.subrows_mut(self.n_system, self.n_lambda))
                .for_each(|unzipped!(mut v)| *v /= self.p.conditioner);

            // Negate solution vector
            zipped!(&mut self.x).for_each(|unzipped!(mut x)| *x *= -1.);

            //------------------------------------------------------------------
            // Update State & lambda
            //------------------------------------------------------------------

            // Convert solution vector to match state node layout
            self.nfm
                .node_dofs
                .iter()
                .enumerate()
                .for_each(|(node_id, dofs)| {
                    let mut node_xd = self.x_delta.col_mut(node_id);
                    let xd = self.x.subrows(dofs.first_dof_index, dofs.n_dofs);
                    match dofs.active {
                        ActiveDOFs::None => unreachable!(),
                        ActiveDOFs::Translation => node_xd.subrows_mut(0, 3).copy_from(xd),
                        ActiveDOFs::Rotation => node_xd.subrows_mut(3, 3).copy_from(xd),
                        ActiveDOFs::All => node_xd.copy_from(xd),
                    };
                });

            // Calculate convergence error
            res.err = zipped!(&self.x_delta, &state.u_delta)
                .map(|unzipped!(xd, ud)| {
                    *xd / ((*ud * self.p.convergence_rtol).abs() + self.p.convergence_atol)
                })
                .norm_l2()
                .div((self.n_system as f64).sqrt());

            // Update state prediction
            state.update_prediction(
                self.p.h,
                self.p.beta_prime,
                self.p.gamma_prime,
                self.x_delta.as_ref(),
            );

            // Update lambda
            zipped!(
                &mut self.lambda,
                &self.x.subrows(self.n_system, self.n_lambda)
            )
            .for_each(|unzipped!(mut lambda, dl)| *lambda -= *dl);

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

    fn populate_tangent_matrix(&mut self, state: &State) {
        let mut rv = Col::<f64>::zeros(3);
        let mut mt = Mat::<f64>::zeros(3, 3);
        let mut tan = Mat::<f64>::zeros(3, 3);
        izip!(
            state.u_delta.subrows(3, 3).col_iter(),
            self.nfm.node_dofs.iter()
        )
        .for_each(|(r_delta, dofs)| {
            match dofs.active {
                ActiveDOFs::All | ActiveDOFs::Rotation => {
                    // Multiply r_delta by h
                    zipped!(&mut rv, &r_delta)
                        .for_each(|unzipped!(mut rv, r_delta)| *rv = self.p.h * *r_delta);

                    // Get angle, return if angle is basically zero
                    let phi = rv.norm_l2();
                    if phi < 1e-16 {
                        return;
                    }

                    // Get row/column index
                    let i = match dofs.active {
                        ActiveDOFs::Rotation => dofs.first_dof_index,
                        ActiveDOFs::All => dofs.first_dof_index + 3,
                        _ => unreachable!(),
                    };

                    // Construct tangent matrix
                    let (phi_s, phi_c) = phi.sin_cos();
                    vec_tilde(rv.as_ref(), mt.as_mut());
                    let a = (1. - phi_s / phi) / (phi * phi);
                    let b = (phi_c - 1.) / (phi * phi);

                    // Construct tangent matrix
                    tan.fill_zero();
                    tan.diagonal_mut().column_vector_mut().fill(1.);
                    matmul(tan.as_mut(), &mt, &mt, Some(1.), a, Parallelism::None);
                    zipped!(&mut tan, &mt).for_each(|unzipped!(mut t, mt)| *t += *mt * b);

                    self.t.submatrix_mut(i, i, 3, 3).copy_from(&tan);
                }
                _ => {}
            };
        });
    }
}

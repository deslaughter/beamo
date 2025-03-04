use crate::util::Quat;
use std::ops::Div;

use faer::{
    linalg::matmul::matmul, linalg::solvers::SpSolver, unzipped, zipped, Col, ColMut, ColRef, Mat,
    MatMut, Parallelism,
};
use itertools::izip;

use crate::{
    constraints::Constraints,
    elements::Elements,
    node::{ActiveDOFs, NodeFreedomMap},
    state::State,
    util::{quat_as_matrix, vec_tilde},
};

pub struct StepParameters {
    pub h: f64, // time step
    pub alpha_f: f64,
    pub alpha_m: f64,
    pub beta: f64,
    pub gamma: f64,
    pub beta_prime: f64,
    pub gamma_prime: f64,
    pub max_iter: usize,
    pub conditioner: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub is_static: bool,
}

impl StepParameters {
    pub fn new(
        h: f64,
        rho_inf: f64,
        atol: f64,
        rtol: f64,
        max_iter: usize,
        is_static: bool,
    ) -> Self {
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
            abs_tol: atol,
            rel_tol: rtol,
            is_static,
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
        // Update strain_dot from previous step before
        // predicting (which overrides velocities)
        self.elements.beams.calculate_strain_dot(state);

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
            iter: 0,
            converged: false,
        };

        // Initialize lambda to zero
        self.lambda.fill_zero();

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

            // Subtract direct nodal loads
            zipped!(&mut self.r, &self.fx).for_each(|unzipped!(r, fx)| *r -= *fx);

            // Add elements to system
            self.elements.assemble_system(
                state,
                &self.nfm,
                self.p.h,
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
            if self.p.is_static {
                matmul(
                    st_11,
                    self.kt.as_ref(),
                    self.t.as_ref(),
                    None,
                    1.,
                    faer::Parallelism::None,
                );
            } else {
                zipped!(&mut st_11, &self.m, &self.ct).for_each(|unzipped!(st, m, ct)| {
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
            }

            // Assemble constraints
            let st_21 = self
                .st
                .submatrix_mut(self.n_system, 0, self.n_lambda, self.n_system);
            matmul(st_21, &self.b, &self.t, None, 1., faer::Parallelism::None);
            let mut st_12 = self
                .st
                .submatrix_mut(0, self.n_system, self.n_system, self.n_lambda);
            st_12.copy_from(self.b.transpose());
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

            // println!("rhs = {:?}", self.rhs);
            // println!("st = {:?}", self.st.submatrix(0, 0, 6, 6));

            // Condition residual
            zipped!(&mut self.rhs.subrows_mut(0, self.n_system))
                .for_each(|unzipped!(v)| *v *= self.p.conditioner);

            // Condition system
            zipped!(&mut st_c.subrows_mut(0, self.n_system))
                .for_each(|unzipped!(v)| *v *= self.p.conditioner);
            zipped!(&mut st_c.subcols_mut(self.n_system, self.n_lambda))
                .for_each(|unzipped!(v)| *v /= self.p.conditioner);

            // println!("rhs = {:?}", self.rhs);
            // println!("st_c = {:?}", st_c.submatrix(0, 0, 6, 6));

            // Solve system
            let lu = st_c.partial_piv_lu();
            let x = lu.solve(&self.rhs);
            self.x.copy_from(&x);

            // println!("x = {:?}", self.x);

            // De-condition solution vector
            zipped!(&mut self.x.subrows_mut(self.n_system, self.n_lambda))
                .for_each(|unzipped!(v)| *v /= self.p.conditioner);

            // Negate solution vector
            zipped!(&mut self.x).for_each(|unzipped!(x)| *x *= -1.);

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

            //------------------------------------------------------------------
            // Calculate convergence error (https://doi.org/10.1115/1.4033441)
            //------------------------------------------------------------------

            let sys_sum_err_squared = zipped!(&self.x_delta, &state.u_delta)
                .map(|unzipped!(pi, xi)| {
                    (*pi / (self.p.abs_tol + (*xi * self.p.h * self.p.rel_tol).abs())).powi(2)
                })
                .as_ref()
                .sum();

            let const_sum_err_squared =
                zipped!(&self.x.subrows(self.n_system, self.n_lambda), &self.lambda)
                    .map(|unzipped!(pi, xi)| {
                        (*pi / (self.p.abs_tol + (*xi * self.p.rel_tol).abs())).powi(2)
                    })
                    .sum();

            res.err = ((sys_sum_err_squared + const_sum_err_squared) / (self.n_dofs as f64)).sqrt();

            //------------------------------------------------------------------
            // Update state and lambda predictions
            //------------------------------------------------------------------

            // Update state prediction
            if self.p.is_static {
                state.update_static_prediction(self.p.h, self.x_delta.as_ref());
            } else {
                state.update_dynamic_prediction(
                    self.p.h,
                    self.p.beta_prime,
                    self.p.gamma_prime,
                    self.x_delta.as_ref(),
                );
            }

            // Update lambda
            zipped!(
                &mut self.lambda,
                &self.x.subrows(self.n_system, self.n_lambda)
            )
            .for_each(|unzipped!(lambda, dl)| *lambda += *dl);

            // Iteration limit reached return not converged
            if res.iter >= self.p.max_iter {
                return res;
            }

            // println!("Error: {} (iter {})", res.err, res.iter);

            // Increment iteration count
            res.iter += 1;
        }

        // Converged, update algorithmic acceleration
        state.update_algorithmic_acceleration(self.p.alpha_m, self.p.alpha_f);
        self.elements
            .beams
            .update_viscoelastic_history(state, self.p.h);
        res.converged = true;
        res
    }

    // Function to calculate the residual and gradient for a solver step
    // Does not actually do an update
    // state does get modified
    pub fn step_res_grad(
        &mut self,
        state: &mut State,
        xd: ColRef<f64>,
        mut res_vec: ColMut<f64>,
        mut dres_mat: MatMut<f64>,
    ) -> StepResults {
        //------------------------------------------------------------------
        // Setup Solution Point like step
        //------------------------------------------------------------------

        // May need to calculate the displacements before proceeding with strain calculation.
        println!("TODO: Not sure if this is needed here. If needed, add to solver.step as well.");
        state.calc_displacement(self.p.h);

        // Update strain_dot from previous step before
        // predicting (which overrides velocities)
        self.elements.beams.calculate_strain_dot(state);

        state.predict_next_state(
            self.p.h,
            self.p.beta,
            self.p.gamma,
            self.p.alpha_m,
            self.p.alpha_f,
        );

        //------------------------------------------------------------------
        // Perturb Solution Point (Same as update in self.step)
        //------------------------------------------------------------------

        self.x.copy_from(&xd);

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

        state.update_dynamic_prediction(
            self.p.h,
            self.p.beta_prime,
            self.p.gamma_prime,
            self.x_delta.as_ref(),
        );

        // Initialize lambda to zero
        self.lambda.fill_zero();

        // Update lambda
        zipped!(
            &mut self.lambda,
            &self.x.subrows(self.n_system, self.n_lambda)
        )
        .for_each(|unzipped!(lambda, dl)| *lambda += *dl);

        //------------------------------------------------------------------
        // Residual Evaluation + Gradient (Same as one loop in self.step)
        //------------------------------------------------------------------

        // Create step results
        let res = StepResults {
            err: 1000.,
            iter: 1,
            converged: false,
        };

        // Loop until converged or max iteration limit reached
        //while res.iter <= 1 {
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

        // Subtract direct nodal loads - not needed for gradient checking
        // zipped!(&mut self.r, &self.fx).for_each(|unzipped!(r, fx)| *r -= *fx);

        // Add elements to system
        self.elements.assemble_system(
            state,
            &self.nfm,
            self.p.h,
            self.m.as_mut(),
            self.ct.as_mut(),
            self.kt.as_mut(),
            self.r.as_mut(),
        );

        // Calculate constraints
        self.constraints
            .assemble_constraints(&self.nfm, state, self.phi.as_mut(), self.b.as_mut());

        // Calculate tangent matrix
        self.populate_tangent_matrix(state);

        // Assemble system matrix
        let mut st_11 = self.st.submatrix_mut(0, 0, self.n_system, self.n_system);
        zipped!(&mut st_11, &self.m, &self.ct).for_each(|unzipped!(st, m, ct)| {
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
        zipped!(&mut st_12, &self.b.transpose()).for_each(|unzipped!(st, bt)| *st = *bt);
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
        // Solve System (just save data from before solve)
        //------------------------------------------------------------------

        // Make copies of St and R for solving system
        let st_c = self.st.clone();
        self.rhs.subrows_mut(0, self.n_system).copy_from(&self.r);
        self.rhs
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&self.phi);

        res_vec.copy_from(self.rhs.clone());
        dres_mat.copy_from(st_c.clone());

        /*
        // Condition residual
        zipped!(&mut self.rhs.subrows_mut(0, self.n_system))
            .for_each(|unzipped!(v)| *v *= self.p.conditioner);

        // Condition system
        zipped!(&mut st_c.subrows_mut(0, self.n_system))
            .for_each(|unzipped!(v)| *v *= self.p.conditioner);
        zipped!(&mut st_c.subcols_mut(self.n_system, self.n_lambda))
            .for_each(|unzipped!(v)| *v /= self.p.conditioner);

        // Solve system
        let lu = st_c.partial_piv_lu();
        let x = lu.solve(&self.rhs);
        self.x.copy_from(&x);

        // De-condition solution vector
        zipped!(&mut self.x.subrows_mut(self.n_system, self.n_lambda))
            .for_each(|unzipped!(v)| *v /= self.p.conditioner);

        // Negate solution vector
        zipped!(&mut self.x).for_each(|unzipped!(x)| *x *= -1.);

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
            .map(|unzipped!(xd, ud)| *xd / ((*ud * self.p.phi_tol).abs() + self.p.x_tol))
            .norm_l2()
            .div((self.n_system as f64).sqrt());

        // Calculate convergence error
        let x_err = self.x.subrows(0, self.n_system).norm_l2() / (self.n_system as f64);
        let phi_err = if self.n_lambda > 0 {
            self.x.subrows(self.n_system, self.n_lambda).norm_l2() / (self.n_lambda as f64)
        } else {
            0.
        };

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
        .for_each(|unzipped!(lambda, dl)| *lambda += *dl);

        // Iteration limit reached return not converged
        if x_err < self.p.x_tol && phi_err < self.p.phi_tol {
            // Converged, update algorithmic acceleration
            state.update_algorithmic_acceleration(self.p.alpha_m, self.p.alpha_f);

            // Update Prony series states as appropriate.
            self.elements.beams.update_viscoelastic_history(state, self.p.h);

            res.converged = true;
            return res;
        }
        // if res.err < 1. {
        //     // Converged, update algorithmic acceleration
        //     state.update_algorithmic_acceleration(self.p.alpha_m, self.p.alpha_f);
        //     res.converged = true;
        //     return res;
        // }


        println!("Error: {} (iter {}, tol {})", x_err, res.iter, self.p.x_tol);

        // Increment iteration count
        res.iter += 1;

        */
        //}

        res
    }

    fn populate_tangent_matrix(&mut self, state: &State) {
        let mut rv = Col::<f64>::zeros(3);
        let mut mt = Mat::<f64>::zeros(3, 3);
        let mut tan = Mat::<f64>::zeros(3, 3);
        izip!(
            state.u_delta.subrows(3, 3).col_iter(),
            state.x.subrows(3, 4).col_iter(),
            self.nfm.node_dofs.iter()
        )
        .for_each(|(r_delta, quat, dofs)| {
            match dofs.active {
                ActiveDOFs::All | ActiveDOFs::Rotation => {
                    // Multiply r_delta by h
                    zipped!(&mut rv, &r_delta)
                        .for_each(|unzipped!(rv, r_delta)| *rv = self.p.h * *r_delta);

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
                    zipped!(&mut tan, &mt).for_each(|unzipped!(t, mt)| *t += *mt * b);

                    // // ------- Rotation from x[4:] for time n+1
                    // // This does not work correctly, here for testing
                    // let mut r_mat = Mat::<f64>::zeros(3,3);
                    // quat_as_matrix(quat, r_mat.as_mut());
                    // self.t.submatrix_mut(i, i, 3, 3).copy_from(r_mat * &tan);

                    // // ------- Rotation from from just the increment
                    // // This gives the same error values in gradient as using T transpose.
                    // let mut q_delta = Col::<f64>::zeros(4);
                    // let mut r_mat = Mat::<f64>::zeros(3,3);
                    // println!("Using R*T");
                    // q_delta
                    //     .as_mut()
                    //     .quat_from_rotation_vector((r_delta * self.p.h).as_ref());
                    // quat_as_matrix(q_delta.as_ref(), r_mat.as_mut());
                    // self.t.submatrix_mut(i, i, 3, 3).copy_from(r_mat * &tan);
                    // println!("norm(RT - T^T)/norm(T): {:}", (r_mat * &tan - &tan.transpose()).norm_l2()/&tan.norm_l2());

                    // ------- Transpose or no Tranpose cases (use a single line)
                    self.t.submatrix_mut(i, i, 3, 3).copy_from(&tan.transpose());
                    // self.t.submatrix_mut(i, i, 3, 3).copy_from(&tan); // Original code, this is wrong.
                }
                _ => {}
            };
        });
    }
}

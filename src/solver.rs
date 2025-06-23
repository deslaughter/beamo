use std::usize;

use crate::{
    constraints::Constraints,
    elements::Elements,
    node::{ActiveDOFs, NodeFreedomMap},
    state::State,
    util::vec_tilde,
};
use faer::sparse;
use faer::sparse::linalg::matmul::{
    sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    get_global_parallelism,
};
use faer::{linalg::matmul::matmul, prelude::*, Accum};
use itertools::{izip, Itertools};

const ENABLE_VISCOELASTIC: bool = false;

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
    pub n_system: usize,                 //
    pub n_lambda: usize,                 //
    pub n_dofs: usize,                   //
    pub t_sp: SparseColMat<usize, f64>,  // T sparse
    pub t_offsets: Vec<usize>,           // T offsets
    pub x: Col<f64>,                     // x solution vector
    pub r: Col<f64>,                     // R residual vector
    pub lambda: Col<f64>,                //
    pub x_delta: Mat<f64>,               //
    pub rhs: Col<f64>,                   // Right hand side
    pub st_sp: SparseColMat<usize, f64>, // St sparse
    tmp_sp: SparseColMat<usize, f64>,    // Temporary sparse matrix for St
    lu_sym: sparse::linalg::solvers::SymbolicLu<usize>,
    mminfo: sparse::linalg::matmul::SparseMatMulInfo,
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
        let n_system_dofs = nfm.n_system_dofs;
        let n_constraint_dofs = constraints.n_rows;
        let n_dofs = n_system_dofs + n_constraint_dofs;
        let n_nodes = nfm.node_dofs.len();
        let mut t_offsets: Vec<usize> = vec![];
        let mut offset = 0;
        nfm.node_dofs
            .iter()
            .for_each(|node_dof| match node_dof.active {
                ActiveDOFs::All => {
                    t_offsets.push(offset + 3);
                    offset += 12;
                }
                ActiveDOFs::Rotation => {
                    t_offsets.push(offset);
                    offset += 9;
                }
                ActiveDOFs::Translation => {
                    t_offsets.push(usize::MAX);
                    offset += 3;
                }
                ActiveDOFs::None => t_offsets.push(usize::MAX),
            });

        //----------------------------------------------------------------------
        // Create T sparse matrix
        //----------------------------------------------------------------------

        let t_triplets = nfm
            .node_dofs
            .iter()
            .flat_map(|node_dof| {
                let i = node_dof.first_dof_index;
                match node_dof.active {
                    ActiveDOFs::All => (i..i + 3)
                        .map(|j| sparse::Triplet::new(j, j, 1.))
                        .chain(
                            (i + 3..i + 6)
                                .cartesian_product(i + 3..i + 6)
                                .map(|(k, j)| sparse::Triplet::new(j, k, 9.)),
                        )
                        .collect_vec(),
                    ActiveDOFs::Translation => (i..i + 3)
                        .map(|j| sparse::Triplet::new(j, j, 1.))
                        .collect_vec(),
                    ActiveDOFs::Rotation => (i..i + 3)
                        .cartesian_product(i..i + 3)
                        .map(|(k, j)| sparse::Triplet::new(j, k, 9.))
                        .collect_vec(),
                    ActiveDOFs::None => vec![],
                }
            })
            .chain((nfm.n_system_dofs..nfm.n_dofs()).map(|i| sparse::Triplet::new(i, i, 1.)))
            .collect_vec();

        let t_sp = SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &t_triplets).unwrap();

        //----------------------------------------------------------------------
        // Compute sparsity pattern for St
        //----------------------------------------------------------------------

        let (st_sym, _) =
            sparse::SymbolicSparseColMat::try_new_from_indices(n_dofs, n_dofs, &[]).unwrap();

        let st_sym = [
            elements.beams.k_sp.symbolic(),
            elements.masses.k_sp.symbolic(),
            elements.springs.k_sp.symbolic(),
            constraints.b_sp.symbolic(),
            constraints
                .b_sp
                .symbolic()
                .transpose()
                .to_col_major()
                .unwrap()
                .as_ref(),
        ]
        .into_iter()
        .fold(st_sym, |acc, sp| {
            sparse::ops::union_symbolic(acc.as_ref(), sp.as_ref()).unwrap()
        });
        let st_sym = constraints.constraints.iter().fold(st_sym, |acc, c| {
            sparse::ops::union_symbolic(acc.as_ref(), c.k_sp.symbolic()).unwrap()
        });

        let data = vec![0.; st_sym.compute_nnz()];
        let tmp_sp = SparseColMat::<usize, f64>::new(st_sym, data);

        //----------------------------------------------------------------------
        // Create symbolic LU factorization if not already done
        //----------------------------------------------------------------------

        // Calculate matrix multiplication information for St*T
        let (_, mminfo) =
            sparse_sparse_matmul_symbolic(tmp_sp.symbolic(), t_sp.symbolic()).unwrap();

        //----------------------------------------------------------------------
        // Create symbolic LU factorization if not already done
        //----------------------------------------------------------------------

        let lu_sym = sparse::linalg::solvers::SymbolicLu::try_new(tmp_sp.symbolic()).unwrap();

        //----------------------------------------------------------------------
        // Populate structure
        //----------------------------------------------------------------------

        Solver {
            p: step_parameters,
            nfm,
            elements,
            constraints,
            n_system: n_system_dofs,
            n_lambda: n_constraint_dofs,
            n_dofs,
            t_sp,
            t_offsets,
            r: Col::zeros(n_system_dofs),
            lambda: Col::zeros(n_constraint_dofs),
            x_delta: Mat::zeros(6, n_nodes),
            x: Col::zeros(n_dofs),
            rhs: Col::zeros(n_dofs),
            st_sp: tmp_sp.clone(),
            tmp_sp,
            lu_sym,
            mminfo,
        }
    }

    pub fn step(&mut self, state: &mut State) -> StepResults {
        // Update strain_dot from previous step before
        // predicting (which overrides velocities)
        if ENABLE_VISCOELASTIC {
            self.elements.beams.calculate_strain_dot(state);
        }

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
        self.lambda.fill(0.);

        // Loop until converged or max iteration limit reached
        while res.err > 1. {
            // Reset the matrices
            self.reset_matrices();

            // Add external loads to residual
            self.add_external_loads_to_residual(state);

            // Add elements to system
            self.elements
                .assemble_system(state, self.p.h, self.r.as_mut());

            // Calculate constraints
            self.constraints
                .assemble_constraints(state, self.lambda.as_ref());

            self.populate_tangent_matrix(state);

            self.build_system();

            // Solve System
            self.solve_system();

            // Convert solution vector to match state node layout
            self.update_x_delta();

            // Calculate convergence error
            res.err = self.calculate_convergence_error(&state);

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
            self.update_lambda();

            // Iteration limit reached return not converged
            if res.iter >= self.p.max_iter {
                return res;
            }

            // Increment iteration count
            res.iter += 1;
        }

        // Converged, update algorithmic acceleration
        state.update_algorithmic_acceleration(self.p.alpha_m, self.p.alpha_f);

        if ENABLE_VISCOELASTIC {
            // Update Prony series states as appropriate.
            self.elements
                .beams
                .update_viscoelastic_history(state, self.p.h);
        }

        res.converged = true;
        res
    }

    fn reset_matrices(&mut self) {
        self.r.fill(0.);
        self.st_sp.val_mut().iter_mut().for_each(|v| *v = 0.);
        self.tmp_sp.val_mut().iter_mut().for_each(|v| *v = 0.);
    }

    fn add_external_loads_to_residual(&mut self, state: &State) {
        self.nfm
            .node_dofs
            .iter()
            .enumerate()
            .for_each(|(node_id, dofs)| {
                let mut r = self.r.subrows_mut(dofs.first_dof_index, dofs.n_dofs);
                match dofs.active {
                    ActiveDOFs::Translation => r -= state.fx.col(node_id).subrows(0, 3),
                    ActiveDOFs::Rotation => r -= state.fx.col(node_id).subrows(3, 3),
                    ActiveDOFs::All => r -= state.fx.col(node_id),
                    ActiveDOFs::None => unreachable!(),
                };
            });
    }

    fn build_system(&mut self) {
        let par = get_global_parallelism();

        // Calculate transpose of B
        let b_sp_t = self.constraints.b_sp.transpose().to_col_major().unwrap();

        // Add stiffness matrices to temporary matrix
        sparse::ops::add_assign(self.tmp_sp.rb_mut(), self.elements.beams.k_sp.as_ref());
        sparse::ops::add_assign(self.tmp_sp.rb_mut(), self.elements.masses.k_sp.as_ref());
        sparse::ops::add_assign(self.tmp_sp.rb_mut(), self.elements.springs.k_sp.as_ref());
        self.constraints
            .constraints
            .iter()
            .for_each(|c| sparse::ops::add_assign(self.tmp_sp.rb_mut(), c.k_sp.as_ref()));

        // Apply conditioning
        self.tmp_sp.val_mut().iter_mut().for_each(|v| {
            *v *= self.p.conditioner;
        });

        //----------------------------------------------------------------------
        // Add B to St tmp matrix
        //----------------------------------------------------------------------

        sparse::ops::add_assign(self.tmp_sp.rb_mut(), self.constraints.b_sp.as_ref());

        //----------------------------------------------------------------------
        // Add K * T and B * T to St matrix
        //----------------------------------------------------------------------

        // Create buffer for st_sp sparse matrix multiply
        let mut st_buffer = MemBuffer::try_new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
            self.st_sp.symbolic(),
            par,
        ))
        .unwrap();

        sparse_sparse_matmul_numeric(
            self.st_sp.rb_mut(),
            Accum::Replace,
            self.tmp_sp.as_ref(),
            self.t_sp.as_ref(),
            1.,
            &self.mminfo,
            par,
            MemStack::new(&mut st_buffer),
        );

        //----------------------------------------------------------------------
        // Add B^T to St matrix
        //----------------------------------------------------------------------

        sparse::ops::add_assign(self.st_sp.rb_mut(), b_sp_t.as_ref());

        //----------------------------------------------------------------------
        // Add beta_prime * M + gamma_prime * G to St matrix
        //----------------------------------------------------------------------

        if !self.p.is_static {
            let bpc = self.p.beta_prime * self.p.conditioner;
            let gpc = self.p.gamma_prime * self.p.conditioner;
            sparse::ops::binary_op_assign_into(
                self.st_sp.rb_mut(),
                self.elements.beams.m_sp.as_ref(),
                |st, m| match m {
                    Some(m) => *st += *m * bpc,
                    None => {}
                },
            );
            sparse::ops::binary_op_assign_into(
                self.st_sp.rb_mut(),
                self.elements.beams.g_sp.as_ref(),
                |st, g| match g {
                    Some(g) => *st += *g * gpc,
                    None => {}
                },
            );
            sparse::ops::binary_op_assign_into(
                self.st_sp.rb_mut(),
                self.elements.masses.m_sp.as_ref(),
                |st, m| match m {
                    Some(m) => *st += *m * bpc,
                    None => {}
                },
            );
            sparse::ops::binary_op_assign_into(
                self.st_sp.rb_mut(),
                self.elements.masses.g_sp.as_ref(),
                |st, g| match g {
                    Some(g) => *st += *g * gpc,
                    None => {}
                },
            );
        }

        // Add B^T * lambda to the system residual
        let mut lambda = Col::zeros(self.n_dofs);
        lambda
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&self.lambda);
        self.r += (b_sp_t * &lambda).subrows(0, self.n_system);
    }

    fn solve_system(&mut self) {
        // Condition system residual and populate right-hand side
        zip!(&mut self.rhs.subrows_mut(0, self.n_system), &self.r)
            .for_each(|unzip!(rhs, r)| *rhs = *r * self.p.conditioner);

        // Copy constraint residual to right-hand side
        self.rhs
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&self.constraints.phi);

        // Solve system
        let lu = sparse::linalg::solvers::Lu::try_new_with_symbolic(
            self.lu_sym.clone(),
            self.st_sp.as_ref(),
        )
        .unwrap();

        // Solve the system in place
        lu.solve_in_place(self.rhs.as_mut());

        // Copy rhs to solution vector and apply negation
        zip!(&mut self.x, &self.rhs).for_each(|unzip!(x, rhs)| *x = -*rhs);

        // Remove conditioning from solution vector
        zip!(&mut self.x.subrows_mut(self.n_system, self.n_lambda))
            .for_each(|unzip!(v)| *v /= self.p.conditioner);
    }

    // Function to update the x_delta matrix based on the current x vector
    fn update_x_delta(&mut self) {
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
    }

    // Calculate convergence error (https://doi.org/10.1115/1.4033441)
    fn calculate_convergence_error(&self, state: &State) -> f64 {
        let sys_sum_err_squared = zip!(&self.x_delta, &state.u_delta)
            .map(|unzip!(pi, xi)| {
                (*pi / (self.p.abs_tol + (*xi * self.p.h * self.p.rel_tol).abs())).powi(2)
            })
            .as_ref()
            .sum();

        let const_sum_err_squared =
            zip!(&self.x.subrows(self.n_system, self.n_lambda), &self.lambda)
                .map(|unzip!(pi, xi)| {
                    (*pi / (self.p.abs_tol + (*xi * self.p.rel_tol).abs())).powi(2)
                })
                .sum();

        let sum_err_squared = sys_sum_err_squared + const_sum_err_squared;
        return sum_err_squared.sqrt() / (self.n_dofs as f64).sqrt();
    }

    fn update_lambda(&mut self) {
        zip!(
            &mut self.lambda,
            &self.x.subrows(self.n_system, self.n_lambda)
        )
        .for_each(|unzip!(lambda, dl)| *lambda += *dl);
    }

    fn populate_tangent_matrix(&mut self, state: &State) {
        let mut rv = Col::<f64>::zeros(3);
        let mut mt = Mat::<f64>::zeros(3, 3);
        let mut tan = Mat::<f64>::zeros(3, 3);
        let vals = self.t_sp.val_mut();
        izip!(
            state.u_delta.subrows(3, 3).col_iter(),
            self.nfm.node_dofs.iter(),
            self.t_offsets.iter()
        )
        .for_each(|(r_delta, dofs, &t_offset)| {
            match dofs.active {
                ActiveDOFs::All | ActiveDOFs::Rotation => {
                    // Multiply r_delta by h
                    zip!(&mut rv, &r_delta)
                        .for_each(|unzip!(rv, r_delta)| *rv = self.p.h * *r_delta);

                    // Get angle
                    let phi = rv.norm_l2();

                    // If angle is effectively zero, set tangent matrix to identity and return
                    if phi < 1e-12 {
                        vals[t_offset..t_offset + 9]
                            .copy_from_slice(&[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
                        return;
                    }

                    // Construct tangent matrix
                    let (phi_s, phi_c) = phi.sin_cos();
                    vec_tilde(rv.as_ref(), mt.as_mut());
                    let a = (1. - phi_s / phi) / (phi * phi);
                    let b = (phi_c - 1.) / (phi * phi);

                    // Construct tangent matrix
                    tan.fill(0.);
                    tan.diagonal_mut().column_vector_mut().fill(1.);
                    matmul(tan.as_mut(), Accum::Add, &mt, &mt, a, Par::Seq);
                    zip!(&mut tan, &mt).for_each(|unzip!(t, mt)| *t += *mt * b);

                    // Transpose tan during copy to since vals is column major
                    vals[t_offset..t_offset + 9].copy_from_slice(&[
                        tan[(0, 0)],
                        tan[(0, 1)],
                        tan[(0, 2)],
                        tan[(1, 0)],
                        tan[(1, 1)],
                        tan[(1, 2)],
                        tan[(2, 0)],
                        tan[(2, 1)],
                        tan[(2, 2)],
                    ]);
                }
                _ => {}
            };
        });
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
        state.calc_step_end(self.p.h);

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
        self.lambda.fill(0.);

        // Update lambda
        zip!(
            &mut self.lambda,
            &self.x.subrows(self.n_system, self.n_lambda)
        )
        .for_each(|unzip!(lambda, dl)| *lambda += *dl);

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

        self.reset_matrices();
        // self.add_external_loads_to_residual(state);

        // Add elements to system
        self.elements
            .assemble_system(state, self.p.h, self.r.as_mut());

        // Calculate constraints
        self.constraints
            .assemble_constraints(state, self.lambda.as_ref());

        // Calculate tangent matrix
        self.populate_tangent_matrix(state);

        self.build_system();

        //------------------------------------------------------------------
        // Solve System (just save data from before solve)
        //------------------------------------------------------------------

        // Make copies of St and R for solving system
        self.rhs.subrows_mut(0, self.n_system).copy_from(&self.r);
        self.rhs
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&self.constraints.phi);

        res_vec.copy_from(self.rhs.clone());
        // dres_mat.copy_from(self.st.clone());

        /*
        // Condition residual
        zip!(&mut self.rhs.subrows_mut(0, self.n_system))
            .for_each(|unzip!(v)| *v *= self.p.conditioner);

        // Condition system
        zip!(&mut st_c.subrows_mut(0, self.n_system))
            .for_each(|unzip!(v)| *v *= self.p.conditioner);
        zip!(&mut st_c.subcols_mut(self.n_system, self.n_lambda))
            .for_each(|unzip!(v)| *v /= self.p.conditioner);

        // Solve system
        let lu = st_c.partial_piv_lu();
        let x = lu.solve(&self.rhs);
        self.x.copy_from(&x);

        // De-condition solution vector
        zip!(&mut self.x.subrows_mut(self.n_system, self.n_lambda))
            .for_each(|unzip!(v)| *v /= self.p.conditioner);

        // Negate solution vector
        zip!(&mut self.x).for_each(|unzip!(x)| *x *= -1.);

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
        res.err = zip!(&self.x_delta, &state.u_delta)
            .map(|unzip!(xd, ud)| *xd / ((*ud * self.p.phi_tol).abs() + self.p.x_tol))
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
        zip!(
            &mut self.lambda,
            &self.x.subrows(self.n_system, self.n_lambda)
        )
        .for_each(|unzip!(lambda, dl)| *lambda += *dl);

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
}

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
use std::ops::DivAssign;

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
    matmul_info: sparse::linalg::matmul::SparseMatMulInfo,
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
        // Create symbolic LU factorization
        //----------------------------------------------------------------------

        // Calculate matrix multiplication information for St*T
        let (_, matmul_info) =
            sparse_sparse_matmul_symbolic(tmp_sp.symbolic(), t_sp.symbolic()).unwrap();

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
            matmul_info,
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
            self.constraints.assemble(state, self.lambda.as_ref());

            self.populate_tangent_matrix(state);

            self.build_system();

            // let (err_sys, err_con) = self.calculate_convergence_error_separate();
            // if err_sys < self.p.abs_tol && err_con < self.p.rel_tol {
            //     res.err = (err_sys.powi(2) + err_con.powi(2)).sqrt();
            //     break;
            // }

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

    pub fn constraint_loads(&self, constraint_id: usize) -> Col<f64> {
        let c = &self.constraints.constraints[constraint_id];
        self.lambda.subrows(c.first_row_index, c.n_rows).to_owned()
    }

    // Linearization based on s11071-020-06069-5
    pub fn linearize(&mut self, state: &State) -> Mat<f64> {
        //----------------------------------------------------------------------
        // Get base state without acceleration
        //----------------------------------------------------------------------

        // Create base state for perturbation (no acceleration)
        let mut base_state = state.clone();
        base_state.calc_step_end(self.p.h);
        base_state.u_prev.copy_from(&base_state.u);
        base_state.u_delta.fill(0.);
        base_state.vd.fill(0.);

        self.elements
            .assemble_system(&base_state, 1., self.r.as_mut());
        self.constraints.assemble(&base_state, self.lambda.as_ref());

        //----------------------------------------------------------------------
        // Build A0 matrix
        //----------------------------------------------------------------------

        // Clear sparse matrix
        self.st_sp.val_mut().iter_mut().for_each(|v| *v = 0.);

        // Add mass matrices to St matrix
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.beams.m_sp.as_ref());
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.masses.m_sp.as_ref());

        // Add B to St matrix
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.constraints.b_sp.as_ref());

        // Add B^T to St matrix
        let b_sp_t = self.constraints.b_sp.transpose().to_col_major().unwrap();
        sparse::ops::add_assign(self.st_sp.rb_mut(), b_sp_t.as_ref());

        // Convert to dense matrix
        let a0 = self.st_sp.to_dense();

        //----------------------------------------------------------------------
        // Get dQdx matrix (K)
        //----------------------------------------------------------------------

        self.st_sp.val_mut().iter_mut().for_each(|v| *v = 0.);
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.beams.k_sp.as_ref());
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.masses.k_sp.as_ref());
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.springs.k_sp.as_ref());
        let dq_dx = -self
            .st_sp
            .to_dense()
            .submatrix(0, 0, self.n_system, self.n_system);

        //----------------------------------------------------------------------
        // Get dQdx_dot matrix (G)
        //----------------------------------------------------------------------

        self.st_sp.val_mut().iter_mut().for_each(|v| *v = 0.);
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.beams.g_sp.as_ref());
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.masses.g_sp.as_ref());
        let dq_dx_dot = -self
            .st_sp
            .to_dense()
            .submatrix(0, 0, self.n_system, self.n_system);

        //----------------------------------------------------------------------
        // Get vector of accelerations from state
        //----------------------------------------------------------------------

        let (mut acc, lambda0) = self.calc_acceleration(&state);
        acc.resize_with(self.n_dofs, |_| 0.);

        // Create lambda vector that can be multiplied by full sparse matrix
        let mut lambda = Col::zeros(self.n_dofs);
        lambda
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&lambda0);

        //----------------------------------------------------------------------
        // Finite difference initialization
        //----------------------------------------------------------------------

        // Position perturbation size (divided by time step as it is multiplied by h during application)
        let x_perturbation = 1e-6;

        // Velocity perturbation size
        let v_perturbation = 1e-6;

        // Create a vector of node identifiers and active dof indices
        let perturb_node_indices = self
            .nfm
            .node_dofs
            .iter()
            .enumerate()
            .flat_map(|(node_id, dofs)| {
                dofs.node_dof_indices()
                    .map(|node_dof_index| (node_id, node_dof_index))
                    .collect_vec()
            })
            .collect_vec();

        //----------------------------------------------------------------------
        // Position finite difference
        //----------------------------------------------------------------------

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Matrices to hold partial derivatives wrt position
        let mut dmx_ddot_dx = Mat::<f64>::zeros(self.n_system, self.n_system);
        let mut dbt_lambda_dx = Mat::<f64>::zeros(self.n_system, self.n_system);
        let mut dbx_ddot_dx = Mat::<f64>::zeros(self.n_lambda, self.n_system);

        // Loop through each node and dof
        perturb_node_indices
            .iter()
            .enumerate()
            .for_each(|(k, &(node_id, node_dof_index))| {
                // Perturb position in positive direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] += x_perturbation;
                state_perturb.calc_step_end(1.);

                // Calculate constraints and extract B matrix
                self.elements
                    .assemble_system(&state_perturb, 1., self.r.as_mut());
                self.constraints
                    .assemble(&state_perturb, self.lambda.as_ref());
                let ma_plus = (&self.elements.beams.m_sp + &self.elements.masses.m_sp) * &acc;
                let btlambda_plus = &self.constraints.b_sp.transpose() * &lambda;
                let bxddot_plus = &self.constraints.b_sp * &acc;

                // Perturb position in negative direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] -= x_perturbation;
                state_perturb.calc_step_end(1.);

                // Calculate constraints and extract B matrix
                self.elements
                    .assemble_system(&state_perturb, 1., self.r.as_mut());
                self.constraints
                    .assemble(&state_perturb, self.lambda.as_ref());
                let ma_minus = (&self.elements.beams.m_sp + &self.elements.masses.m_sp) * &acc;
                let btlambda_minus = &self.constraints.b_sp.transpose() * &lambda;
                let bxddot_minus = &self.constraints.b_sp * &acc;

                //----------------------------------------------------------

                // Compute change in mass matrix times acceleration for this dof
                let d = (&ma_plus - &ma_minus).subrows(0, self.n_system) / (2. * x_perturbation);
                dmx_ddot_dx.col_mut(k).copy_from(d);

                // Compute change in constraint gradient transpose times lambda for this dof
                let d = (&btlambda_plus - &btlambda_minus).subrows(0, self.n_system)
                    / (2. * x_perturbation);
                dbt_lambda_dx.col_mut(k).copy_from(d);

                // Compute change in constraint gradient times acceleration for this dof
                let d = (&bxddot_plus - &bxddot_minus).subrows(self.n_system, self.n_lambda)
                    / (2. * x_perturbation);
                dbx_ddot_dx.col_mut(k).copy_from(d);
            });

        //----------------------------------------------------------------------
        // Calculate dd/dx and dd/dx_dot matrices
        //----------------------------------------------------------------------

        // Get vector of velocities
        let mut vel = Col::zeros(self.n_dofs);
        self.nfm
            .node_dofs
            .iter()
            .enumerate()
            .for_each(|(node_id, dofs)| {
                dofs.node_dof_indices().enumerate().for_each(|(i, i_node)| {
                    vel[dofs.first_dof_index + i] = state.v[(i_node, node_id)];
                });
            });

        let mut dd_dx = Mat::<f64>::zeros(self.n_lambda, self.n_system);
        let mut dd_dx_dot = Mat::<f64>::zeros(self.n_lambda, self.n_system);

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Iterate through node perturbation indices and perturb position
        perturb_node_indices
            .iter()
            .enumerate()
            .for_each(|(k_col, &(node_id, node_dof_index))| {
                // Perturb position in positive direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] += x_perturbation;
                state_perturb.calc_step_end(1.);
                let d_plus = self.calc_d(&state_perturb, x_perturbation);

                // Perturb position in negative direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] -= x_perturbation;
                state_perturb.calc_step_end(1.);
                let d_minus = self.calc_d(&state_perturb, x_perturbation);

                dd_dx
                    .col_mut(k_col)
                    .copy_from((d_plus - d_minus) / (2. * x_perturbation));
            });

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Iterate through node perturbation indices and perturb velocity
        perturb_node_indices
            .iter()
            .enumerate()
            .for_each(|(k_col, &(node_id, node_dof_index))| {
                // Perturb velocity in positive direction ------------------
                state_perturb.v.copy_from(&state.v);
                state_perturb.v[(node_dof_index, node_id)] += v_perturbation;
                let d_plus = self.calc_d(&state_perturb, x_perturbation);

                // Perturb velocity in negative direction ------------------
                state_perturb.v.copy_from(&state.v);
                state_perturb.v[(node_dof_index, node_id)] -= v_perturbation;
                let d_minus = self.calc_d(&state_perturb, x_perturbation);

                dd_dx_dot
                    .col_mut(k_col)
                    .copy_from((d_plus - d_minus) / (2. * v_perturbation));
            });

        //----------------------------------------------------------------------
        // Construct B0 matrix
        //----------------------------------------------------------------------

        let mut b0 = Mat::<f64>::zeros(self.n_dofs, 2 * self.n_system);

        // Quadrant (1,1)
        b0.submatrix_mut(0, 0, self.n_system, self.n_system)
            .copy_from(&dq_dx - &dmx_ddot_dx - &dbt_lambda_dx);

        // Quadrant (1,2)
        b0.submatrix_mut(0, self.n_system, self.n_system, self.n_system)
            .copy_from(&dq_dx_dot);

        // Quadrant (2,1)
        b0.submatrix_mut(self.n_system, 0, self.n_lambda, self.n_system)
            .copy_from(-&dd_dx - &dbx_ddot_dx);

        // Quadrant (2,2)
        b0.submatrix_mut(self.n_system, self.n_system, self.n_lambda, self.n_system)
            .copy_from(-&dd_dx_dot);

        //----------------------------------------------------------------------
        // Solve for F0 = A0^-1 B0
        //----------------------------------------------------------------------

        let f0 = a0.partial_piv_lu().solve(b0);

        // Create state space A matrix
        let mut a_ss = Mat::<f64>::zeros(2 * self.n_system, 2 * self.n_system);

        a_ss.submatrix_mut(0, self.n_system, self.n_system, self.n_system)
            .copy_from(&Mat::<f64>::identity(self.n_system, self.n_system));

        a_ss.submatrix_mut(self.n_system, 0, self.n_system, 2 * self.n_system)
            .copy_from(&f0.submatrix(0, 0, self.n_system, 2 * self.n_system));

        a_ss
    }

    pub fn linearize2(&mut self, state: &State) -> Mat<f64> {
        // Create base state for perturbation (no acceleration)
        let mut base_state = state.clone();
        base_state.calc_step_end(self.p.h);
        base_state.u_prev.copy_from(&base_state.u);
        base_state.u_delta.fill(0.);
        base_state.vd.fill(0.);

        // Position perturbation size (divided by time step as it is multiplied by h during application)
        let x_perturbation = 1e-5;

        // Velocity perturbation size
        let v_perturbation = 1e-5;

        // Create a vector of node identifiers and active dof indices
        let perturb_node_indices = self
            .nfm
            .node_dofs
            .iter()
            .enumerate()
            .flat_map(|(node_id, dofs)| {
                dofs.node_dof_indices()
                    .map(|node_dof_index| (node_id, node_dof_index))
                    .collect_vec()
            })
            .collect_vec();

        // Matrices to hold partial derivatives wrt position
        let mut a_ss = Mat::<f64>::zeros(2 * self.n_system, 2 * self.n_system);
        a_ss.submatrix_mut(0, self.n_system, self.n_system, self.n_system)
            .copy_from(&Mat::<f64>::identity(self.n_system, self.n_system));

        //----------------------------------------------------------------------
        // Position finite difference
        //----------------------------------------------------------------------

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Loop through each node and dof
        perturb_node_indices
            .iter()
            .enumerate()
            .for_each(|(k_col, &(node_id, node_dof_index))| {
                // Perturb position in positive direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] += x_perturbation;
                state_perturb.calc_step_end(1.);
                let (acc_plus, _) = self.calc_acceleration(&state_perturb);

                // Perturb position in negative direction ------------------
                state_perturb.u_delta.fill(0.);
                state_perturb.u_delta[(node_dof_index, node_id)] -= x_perturbation;
                state_perturb.calc_step_end(1.);
                let (acc_minus, _) = self.calc_acceleration(&state_perturb);

                // Compute change in acceleration for this dof
                a_ss.col_mut(k_col)
                    .subrows_mut(self.n_system, self.n_system)
                    .copy_from((&acc_plus - &acc_minus) / (2. * x_perturbation));
            });

        //----------------------------------------------------------------------
        // Velocity finite difference
        //----------------------------------------------------------------------

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Loop through each node and dof
        perturb_node_indices
            .iter()
            .enumerate()
            .for_each(|(k_col, &(node_id, node_dof_index))| {
                // Perturb position in positive direction ------------------
                state_perturb.v.copy_from(&base_state.v);
                state_perturb.v[(node_dof_index, node_id)] += v_perturbation;
                let (acc_plus, _) = self.calc_acceleration(&state_perturb);

                // Perturb position in negative direction ------------------
                state_perturb.v.copy_from(&base_state.v);
                state_perturb.v[(node_dof_index, node_id)] -= v_perturbation;
                let (acc_minus, _) = self.calc_acceleration(&state_perturb);

                // Compute change in acceleration for this dof
                a_ss.col_mut(k_col + self.n_system)
                    .subrows_mut(self.n_system, self.n_system)
                    .copy_from((&acc_plus - &acc_minus) / (2. * x_perturbation));
            });

        a_ss
    }

    pub fn linearize3<
        Fperturb: Fn(usize, f64, bool) -> (Col<f64>, Col<f64>),
        Fvd: Fn(Col<f64>, Col<f64>, Col<f64>) -> (Col<f64>, Col<f64>),
    >(
        &mut self,
        state: &State,
        perturb_fn: Fperturb,
        vd_solve: Fvd,
    ) -> Mat<f64> {
        // Create base state for perturbation (no acceleration)
        let mut base_state = state.clone();
        base_state.calc_step_end(self.p.h);
        base_state.u_prev.copy_from(&base_state.u);
        base_state.u_delta.fill(0.);
        base_state.vd.fill(0.);

        // Create a vector of node identifiers and active dof indices
        let perturb_node_indices = self
            .nfm
            .node_dofs
            .iter()
            .enumerate()
            .flat_map(|(node_id, dofs)| {
                dofs.node_dof_indices()
                    .map(|node_dof_index| (node_id, node_dof_index))
                    .collect_vec()
            })
            .collect_vec();

        // State Space A matrix
        let mut a_ss = Mat::<f64>::zeros(2 * self.n_system, 2 * self.n_system);
        a_ss.submatrix_mut(0, self.n_system, self.n_system, self.n_system)
            .copy_from(&Mat::<f64>::identity(self.n_system, self.n_system));

        let perturbation = 1e-6;

        //----------------------------------------------------------------------
        // Position finite difference
        //----------------------------------------------------------------------

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Loop through each node and dof
        (0..perturb_node_indices.len()).for_each(|k_col| {
            let (perturb_x, perturb_v) = perturb_fn(k_col, perturbation, false);

            // Perturb position in positive direction ------------------
            state_perturb.u_delta.fill(0.);
            perturb_node_indices.iter().zip(perturb_x.iter()).for_each(
                |(&(node_id, node_dof_index), &p)| {
                    state_perturb.u_delta[(node_dof_index, node_id)] += p;
                },
            );
            state_perturb.calc_step_end(1.);
            let (acc_plus, _) = self.calc_acceleration(&state_perturb);

            // Perturb position in negative direction ------------------
            state_perturb.u_delta.fill(0.);
            perturb_node_indices.iter().zip(perturb_x.iter()).for_each(
                |(&(node_id, node_dof_index), &p)| {
                    state_perturb.u_delta[(node_dof_index, node_id)] -= p;
                },
            );
            state_perturb.calc_step_end(1.);
            let (acc_minus, _) = self.calc_acceleration(&state_perturb);

            let (v_nr, vd_nr) = vd_solve(acc_plus - acc_minus, perturb_x, perturb_v);

            // Compute change in acceleration for this dof
            a_ss.col_mut(k_col)
                .subrows_mut(0, self.n_system)
                .copy_from(v_nr / (2. * perturbation));
            a_ss.col_mut(k_col)
                .subrows_mut(self.n_system, self.n_system)
                .copy_from(vd_nr / (2. * perturbation));
        });

        //----------------------------------------------------------------------
        // Velocity finite difference
        //----------------------------------------------------------------------

        // Clone the state for perturbation
        let mut state_perturb = base_state.clone();

        // Loop through each node and dof
        (0..perturb_node_indices.len()).for_each(|k_col| {
            // Get perturbation vector for velocity
            let (perturb_x, perturb_v) = perturb_fn(k_col, perturbation, true);

            // Perturb velocity in positive direction ------------------
            state_perturb.v.copy_from(&base_state.v);
            perturb_node_indices.iter().zip(perturb_v.iter()).for_each(
                |(&(node_id, node_dof_index), &p)| {
                    state_perturb.v[(node_dof_index, node_id)] += p;
                },
            );
            let (acc_plus, _) = self.calc_acceleration(&state_perturb);

            // Perturb velocity in negative direction ------------------
            state_perturb.v.copy_from(&base_state.v);
            perturb_node_indices.iter().zip(perturb_v.iter()).for_each(
                |(&(node_id, node_dof_index), &p)| {
                    state_perturb.v[(node_dof_index, node_id)] -= p;
                },
            );
            let (acc_minus, _) = self.calc_acceleration(&state_perturb);

            let (v_nr, vd_nr) = vd_solve(acc_plus - acc_minus, perturb_x, perturb_v);

            // Compute change in acceleration for this dof
            a_ss.col_mut(k_col)
                .subrows_mut(0, self.n_system)
                .copy_from(v_nr / (2. * perturbation));
            a_ss.col_mut(k_col)
                .subrows_mut(self.n_system, self.n_system)
                .copy_from(vd_nr / (2. * perturbation));
        });

        a_ss
    }

    pub fn calc_acceleration(&mut self, state: &State) -> (Col<f64>, Col<f64>) {
        let x_perturbation = 1e-6;

        // Reset the matrices
        self.r.fill(0.);
        self.st_sp.val_mut().iter_mut().for_each(|v| *v = 0.);

        // Add external loads to residual
        self.add_external_loads_to_residual(&state);

        // Add elements to system
        self.elements.assemble_system(&state, 1., self.r.as_mut());

        //------------------------------------------------------------------
        // Build system matrix and residual vector
        //------------------------------------------------------------------

        // Add mass matrices to St matrix
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.beams.m_sp.as_ref());
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.elements.masses.m_sp.as_ref());

        // Add B to St matrix
        sparse::ops::add_assign(self.st_sp.rb_mut(), self.constraints.b_sp.as_ref());

        // Add B^T to St matrix
        let b_sp_t = self.constraints.b_sp.transpose().to_col_major().unwrap();
        sparse::ops::add_assign(self.st_sp.rb_mut(), b_sp_t.as_ref());

        // Populate right-hand side
        let d = self.calc_d(&state, x_perturbation);
        self.rhs.subrows_mut(0, self.n_system).copy_from(-&self.r);
        self.rhs
            .subrows_mut(self.n_system, self.n_lambda)
            .copy_from(&-d);

        let lu = sparse::linalg::solvers::Lu::try_new_with_symbolic(
            self.lu_sym.clone(),
            self.st_sp.as_ref(),
        )
        .unwrap();

        let x = lu.solve(&self.rhs);

        // Return the acceleration vector and lambda vector
        (
            x.subrows(0, self.n_system).to_owned(),
            x.subrows(self.n_system, self.n_lambda).to_owned(),
        )
    }

    fn calc_d(&mut self, state: &State, x_perturbation: f64) -> Col<f64> {
        // Matrix to hold the partial derivative of dD(x)x_dot/dx
        let mut ddx_dot_dx = Mat::<f64>::zeros(self.n_lambda, self.n_system);

        // Extract vector of velocities
        let mut vel = Col::zeros(self.n_dofs);
        self.nfm
            .node_dofs
            .iter()
            .enumerate()
            .for_each(|(node_id, dofs)| {
                let mut vr = vel.subrows_mut(dofs.first_dof_index, dofs.n_dofs);
                dofs.node_dof_indices().enumerate().for_each(|(i, idx)| {
                    vr[i] = state.v[(idx, node_id)];
                });
            });

        // State perturbation copy
        let mut sp = state.clone();

        // Iterate through node perturbation indices
        let k_col = 0;
        self.nfm
            .node_dofs
            .iter()
            .enumerate()
            .for_each(|(node_id, dofs)| {
                dofs.node_dof_indices().for_each(|i_dof| {
                    // Perturb position in positive direction ------------------
                    sp.u_delta.fill(0.);
                    sp.u_delta[(i_dof, node_id)] += x_perturbation;
                    sp.calc_step_end(1.);
                    self.constraints.assemble(&sp, self.lambda.as_ref());
                    let bv_plus = &self.constraints.b_sp * &vel;

                    // Perturb position in negative direction ------------------
                    sp.u_delta.fill(0.);
                    sp.u_delta[(i_dof, node_id)] -= x_perturbation;
                    sp.calc_step_end(1.);
                    self.constraints.assemble(&sp, self.lambda.as_ref());
                    let bv_minus = &self.constraints.b_sp * &vel;

                    ddx_dot_dx.col_mut(k_col).copy_from(
                        (bv_plus - bv_minus).subrows(self.n_system, self.n_lambda)
                            / (2. * x_perturbation),
                    );
                });
            });

        ddx_dot_dx * &vel.subrows(0, self.n_system)
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
            &self.matmul_info,
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
        self.rhs
            .subrows_mut(0, self.n_system)
            .copy_from(&self.r * self.p.conditioner);

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
        self.x.copy_from(-&self.rhs);

        // Remove conditioning from solution vector
        self.x
            .subrows_mut(self.n_system, self.n_lambda)
            .div_assign(self.p.conditioner);
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

    // Calculate convergence error (https://doi.org/10.1115/1.4033441)
    fn calculate_convergence_error_separate(&self) -> (f64, f64) {
        (self.r.norm_l2(), self.constraints.phi.norm_l2())
    }

    fn update_lambda(&mut self) {
        self.lambda += self.x.subrows(self.n_system, self.n_lambda);
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
                    rv.copy_from(r_delta);
                    rv *= self.p.h;

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
                    tan += &mt * b;

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
        mut _dres_mat: MatMut<f64>,
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
        self.lambda += &self.x.subrows(self.n_system, self.n_lambda);

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
        self.constraints.assemble(state, self.lambda.as_ref());

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

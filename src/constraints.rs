use crate::{
    node::NodeFreedomMap,
    state::State,
    util::{
        axial_vector_of_matrix, cross_product, dot_product, matrix_ax, matrix_ax2,
        quat_as_matrix_alloc, quat_compose_alloc, quat_from_rotation_vector,
        quat_from_rotation_vector_alloc, quat_inverse_alloc, quat_rotate_vector,
        quat_rotate_vector_alloc, sparse_matrix_from_triplets, vec_tilde_alloc,
    },
};
use faer::{linalg::matmul::matmul, prelude::*, sparse::*, Accum, Par};
use itertools::Itertools;
use std::cmp;

#[derive(Clone, Copy)]
pub enum ConstraintKind {
    Prescribed,
    Rigid,
    Revolute,
    Rotation,
}

pub struct ConstraintInput {
    pub id: usize,
    pub kind: ConstraintKind,
    pub node_id_base: usize,
    pub node_id_target: usize,
    pub x0: Col<f64>,
    pub vec: Col<f64>,
}

pub struct Constraints {
    pub n_rows: usize,
    pub phi: Col<f64>,
    pub b_sp: SparseColMat<usize, f64>,
    b_order: Vec<usize>,
    pub constraints: Vec<Constraint>,
}

impl Constraints {
    pub fn new(inputs: &[ConstraintInput], nfm: &NodeFreedomMap) -> Self {
        let mut n_rows = 0;

        //----------------------------------------------------------------------
        // Create vector of constraints
        //----------------------------------------------------------------------

        let mut constraints = inputs
            .iter()
            .map(|inp| {
                let c = Constraint::new(n_rows, inp, nfm);
                n_rows += c.n_rows;
                c
            })
            .collect_vec();

        //----------------------------------------------------------------------
        // Build constraint stiffness matrices
        //----------------------------------------------------------------------

        constraints.iter_mut().for_each(|c| {
            (c.k_sp, c.k_order) = {
                sparse_matrix_from_triplets(
                    nfm.n_system_dofs + n_rows,
                    nfm.n_system_dofs + n_rows,
                    &c.get_k_triplets(),
                )
            }
        });

        //----------------------------------------------------------------------
        // Create constraint gradient sparse matrix
        //----------------------------------------------------------------------

        // Create a vector of triplets for the b sparse matrix
        let mut b_triplets = Vec::new();
        constraints.iter().for_each(|c| {
            (0..c.b_base.ncols())
                .cartesian_product(0..c.b_base.nrows())
                .for_each(|(j, i)| {
                    b_triplets.push(Triplet::new(
                        c.first_row_index + nfm.n_system_dofs + i,
                        c.base_col_index + j,
                        0.,
                    ))
                });
            (0..c.b_target.ncols())
                .cartesian_product(0..c.b_target.nrows())
                .for_each(|(j, i)| {
                    b_triplets.push(Triplet::new(
                        c.first_row_index + nfm.n_system_dofs + i,
                        c.target_col_index + j,
                        0.,
                    ))
                });
        });

        let (b_sp, b_order) = sparse_matrix_from_triplets(
            nfm.n_system_dofs + n_rows,
            nfm.n_system_dofs + n_rows,
            &b_triplets,
        );

        //----------------------------------------------------------------------
        // Return Constraints struct
        //----------------------------------------------------------------------

        Self {
            n_rows,
            phi: Col::<f64>::zeros(n_rows),
            b_sp,
            b_order,
            constraints,
        }
    }

    pub fn assemble_constraints(&mut self, state: &State, lambda: ColRef<f64>) {
        // Loop through constraints and calculate residual and gradient
        self.constraints.iter_mut().for_each(|c| {
            // Get base and target node
            let u_base = state.u.col(c.node_id_base).subrows(0, 3);
            let r_base = state.u.col(c.node_id_base).subrows(3, 4);
            let u_target = state.u.col(c.node_id_target).subrows(0, 3);
            let r_target = state.u.col(c.node_id_target).subrows(3, 4);
            let lambda = lambda.subrows(c.first_row_index, c.n_rows);

            // Switch calculation based on constraint type
            match c.kind {
                ConstraintKind::Prescribed => c.calculate_prescribed(u_target, r_target, lambda),
                ConstraintKind::Rigid | ConstraintKind::Rotation => {
                    c.calculate_rigid(u_base, r_base, u_target, r_target, lambda)
                }
                ConstraintKind::Revolute => {
                    c.calculate_revolute(u_base, r_base, u_target, r_target, lambda);
                }
            }

            // Populate k sparse matrix values
            c.update_k_values();

            // Add constraint residual to phi vector
            self.phi
                .subrows_mut(c.first_row_index, c.n_rows)
                .copy_from(&c.phi);
        });

        // Populate b sparse matrix values
        self.update_b_values();
    }

    // Update b values based on constraints
    // order of values must indices produced by b_pairs
    fn update_b_values(&mut self) {
        let values = self.b_sp.val_mut();
        let mut k = 0;
        self.constraints.iter().for_each(|c| {
            (0..c.b_base.ncols())
                .cartesian_product(0..c.b_base.nrows())
                .for_each(|(j, i)| {
                    values[self.b_order[k]] = c.b_base[(i, j)];
                    k += 1;
                });
            (0..c.b_target.ncols())
                .cartesian_product(0..c.b_target.nrows())
                .for_each(|(j, i)| {
                    values[self.b_order[k]] = c.b_target[(i, j)];
                    k += 1;
                });
        });
    }
}

pub struct Constraint {
    kind: ConstraintKind,
    first_row_index: usize,
    base_col_index: usize,
    target_col_index: usize,
    n_rows: usize,
    node_id_base: usize,
    node_id_target: usize,
    x0: Col<f64>,
    phi: Col<f64>,
    b_base: Mat<f64>,
    b_target: Mat<f64>,
    input: Col<f64>,
    /// Stiffness matrix for base node `[12,12]`
    k_b: Mat<f64>,
    k_t: Mat<f64>,
    k_bt: Mat<f64>,
    k_tb: Mat<f64>,
    /// Rotation matrix `[3,3]`
    axes: Mat<f64>,
    pub k_sp: SparseColMat<usize, f64>,
    k_order: Vec<usize>,
}

impl Constraint {
    fn new(first_dof_index: usize, input: &ConstraintInput, nfm: &NodeFreedomMap) -> Self {
        let n_dofs_base = match input.kind {
            ConstraintKind::Prescribed => 0,
            _ => nfm.node_dofs[input.node_id_base].n_dofs,
        };

        let n_dofs_target = nfm.node_dofs[input.node_id_target].n_dofs;
        let base_col_index = nfm.node_dofs[input.node_id_base].first_dof_index;
        let target_col_index = nfm.node_dofs[input.node_id_target].first_dof_index;

        // Get number of constraint DOFs (rows of phi vector or B matrix)
        let n_rows = match input.kind {
            ConstraintKind::Prescribed => n_dofs_target,
            ConstraintKind::Rigid | ConstraintKind::Rotation => {
                cmp::min(n_dofs_base, n_dofs_target)
            }
            ConstraintKind::Revolute => 5,
        };

        // Calculate constraint axes
        let axes = match input.kind {
            ConstraintKind::Revolute | ConstraintKind::Rotation => {
                let x = col![1., 0., 0.];
                let x_hat = &input.vec / input.vec.norm_l2();

                // Create rotation matrix to rotate x to match vector
                let mut cp = Col::<f64>::zeros(3);
                cross_product(x.as_ref(), x_hat.as_ref(), cp.as_mut());
                let dp = dot_product(x_hat.as_ref(), x.as_ref());
                let k = 1. / (1. + dp);

                // Set orthogonal unit vectors from the rotation matrix
                // columns are x,y,z axes unit vectors
                mat![
                    [
                        cp[0] * cp[0] * k + dp,
                        cp[0] * cp[1] * k - cp[2],
                        cp[0] * cp[2] * k + cp[1],
                    ],
                    [
                        cp[1] * cp[0] * k + cp[2],
                        cp[1] * cp[1] * k + dp,
                        cp[1] * cp[2] * k - cp[0],
                    ],
                    [
                        cp[2] * cp[0] * k - cp[1],
                        cp[2] * cp[1] * k + cp[0],
                        cp[2] * cp[2] * k + dp,
                    ]
                ]
            }
            _ => Mat::<f64>::identity(3, 3),
        };

        let k_t = match input.kind {
            ConstraintKind::Rigid => {
                if n_dofs_target == 6 {
                    Mat::<f64>::zeros(3, 3)
                } else {
                    Mat::<f64>::new()
                }
            }
            _ => Mat::<f64>::zeros(3, 3),
        };

        let k_b = match input.kind {
            ConstraintKind::Prescribed => Mat::<f64>::new(),
            _ => Mat::<f64>::zeros(3, 3),
        };

        let k_bt = if k_t.ncols() == 0 || k_b.ncols() == 0 {
            Mat::<f64>::new()
        } else {
            Mat::<f64>::zeros(3, 3)
        };

        //----------------------------------------------------------------------
        // Constraint structure
        //----------------------------------------------------------------------

        Self {
            kind: input.kind,
            first_row_index: first_dof_index,
            base_col_index,
            target_col_index,
            n_rows,
            node_id_base: input.node_id_base,
            node_id_target: input.node_id_target,
            x0: input.x0.clone(),
            input: col![0., 0., 0., 1., 0., 0., 0.],
            phi: Col::<f64>::zeros(n_rows),
            b_base: -1. * Mat::<f64>::identity(n_rows, n_dofs_base),
            b_target: Mat::<f64>::identity(n_rows, n_dofs_target),
            k_b,
            k_t,
            k_tb: k_bt.clone(),
            k_bt,
            axes,
            k_sp: SparseColMat::try_new_from_triplets(0, 0, &[]).unwrap(),
            k_order: Vec::new(),
        }
    }

    pub fn get_k_triplets(&self) -> Vec<Triplet<usize, usize, f64>> {
        let indices = (0..3usize).cartesian_product(0..3usize).collect_vec();

        let mut k_triplets = Vec::new();

        if self.k_b.ncols() > 0 {
            indices.iter().for_each(|&(j, i)| {
                k_triplets.push(Triplet::new(
                    self.base_col_index + 3 + i,
                    self.base_col_index + 3 + j,
                    0.,
                ));
            });
        }
        if self.k_bt.ncols() > 0 {
            // k_bt
            indices.iter().for_each(|&(j, i)| {
                k_triplets.push(Triplet::new(
                    self.base_col_index + 3 + i,
                    self.target_col_index + 3 + j,
                    0.,
                ));
            });
            // k_tb
            indices.iter().for_each(|&(j, i)| {
                k_triplets.push(Triplet::new(
                    self.target_col_index + 3 + i,
                    self.base_col_index + 3 + j,
                    0.,
                ));
            });
        }
        if self.k_t.ncols() > 0 {
            indices.iter().for_each(|&(j, i)| {
                k_triplets.push(Triplet::new(
                    self.target_col_index + 3 + i,
                    self.target_col_index + 3 + j,
                    0.,
                ));
            });
        }
        k_triplets
    }

    // Set displacement for prescribed displacement constraint
    pub fn set_displacement(&mut self, x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) {
        let mut q = col![0., 0., 0., 0.];
        quat_from_rotation_vector(col![rx, ry, rz].as_ref(), q.as_mut());
        self.input[0] = x;
        self.input[1] = y;
        self.input[2] = z;
        self.input[3] = q[0];
        self.input[4] = q[1];
        self.input[5] = q[2];
        self.input[6] = q[3];
    }

    // Set rotation angle for prescribed rotation constraint
    pub fn set_rotation(&mut self, angle: f64) {
        self.input[0] = angle;
    }

    fn calculate_prescribed(
        &mut self,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
        lambda: ColRef<f64>,
    ) {
        let u_base = self.input.subrows(0, 3);
        let r_base = self.input.subrows(3, 4);

        // Phi(0:3) = ut + X0 - ub - Rb*X0
        let rb_x0 = quat_rotate_vector_alloc(r_base.as_ref(), self.x0.as_ref());
        zip!(
            &mut self.phi.subrows_mut(0, 3),
            &u_base,
            &u_target,
            &self.x0,
            &rb_x0
        )
        .for_each(|unzip!(phi, ub, ut, x0, rb_x0)| *phi = *ut + *x0 - *ub - *rb_x0);

        // If only position is prescribed, return
        if self.n_rows == 3 {
            return;
        }

        // Combination of all rotations
        let r_base_inv = quat_inverse_alloc(r_base.rb());
        let rc = quat_compose_alloc(r_target.rb(), r_base_inv.rb());

        // Angular residual:  Phi(3:6) = axial(Rt*inv(Rb))
        let c = quat_as_matrix_alloc(rc.rb());
        axial_vector_of_matrix(c.as_ref(), self.phi.subrows_mut(3, 3));

        // Constraint Gradient
        // B(0:3,0:3) = I (set at init)
        // B(3:6,3:6) = AX(Rb*inv(Rt)) = transpose(AX(Rt*inv(Rb)))
        matrix_ax(
            c.as_ref(),
            self.b_target.submatrix_mut(3, 3, 3, 3).transpose_mut(),
        );

        // Rotational stiffness matrix for target node
        let lambda_tilde = vec_tilde_alloc(lambda.subrows(3, 3));
        let c = quat_as_matrix_alloc(r_target);
        matrix_ax((-&c * &lambda_tilde).rb(), self.k_t.as_mut());
    }

    fn update_k_values(&mut self) {
        let values = self.k_sp.val_mut();
        let mut k = 0;

        self.k_b.col_iter().for_each(|col| {
            col.iter().for_each(|&v| {
                values[self.k_order[k]] = v;
                k += 1;
            });
        });
        self.k_bt.col_iter().for_each(|col| {
            col.iter().for_each(|&v| {
                values[self.k_order[k]] = v;
                k += 1;
            });
        });
        self.k_tb.col_iter().for_each(|col| {
            col.iter().for_each(|&v| {
                values[self.k_order[k]] = v;
                k += 1;
            });
        });
        self.k_t.col_iter().for_each(|col| {
            col.iter().for_each(|&v| {
                values[self.k_order[k]] = v;
                k += 1;
            });
        });
    }

    fn calculate_rigid(
        &mut self,
        u_base: ColRef<f64>,
        r_base: ColRef<f64>,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
        lambda: ColRef<f64>,
    ) {
        //----------------------------------------------------------------------
        // Position residual
        //----------------------------------------------------------------------

        // Phi(0:3) = ut + X0 - ub - Rb*X0
        let rb_x0 = quat_rotate_vector_alloc(r_base.as_ref(), self.x0.as_ref());
        zip!(
            &mut self.phi.subrows_mut(0, 3),
            &u_base,
            &u_target,
            &self.x0,
            &rb_x0
        )
        .for_each(|unzip!(phi, ub, ut, x0, rb_x0)| *phi = *ut + *x0 - *ub - *rb_x0);

        //----------------------------------------------------------------------
        // Stiffness matrix for base node
        //----------------------------------------------------------------------

        // Lambda from translational terms
        let lambda_1 = lambda.subrows(0, 3);
        let lambda_1_tilde = vec_tilde_alloc(lambda_1);

        let rb_x0_tilde = vec_tilde_alloc(rb_x0.as_ref());
        matmul(
            self.k_b.rb_mut(),
            Accum::Replace,
            &lambda_1_tilde,
            &rb_x0_tilde,
            -1.,
            Par::Seq,
        );

        //----------------------------------------------------------------------
        // Target constraint gradient
        //----------------------------------------------------------------------

        // B(0:3,0:3) = I (set at init)

        //----------------------------------------------------------------------
        // Base constraint gradient
        //----------------------------------------------------------------------

        // B(0:3,0:3) = -I (set at init)
        // B(0:3,3:6) = tilde(Rb*X0)
        self.b_base
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&rb_x0_tilde);

        // Return if only position is constrained
        if self.n_rows == 3 {
            return;
        }

        // Calculate control rotation quaternion
        let r_control = quat_from_rotation_vector_alloc((&self.axes.col(0) * &self.input[0]).rb());

        // Combination of all rotations
        let rc = quat_compose_alloc(
            quat_compose_alloc(r_target.rb(), quat_inverse_alloc(r_control.rb()).rb()).rb(),
            quat_inverse_alloc(r_base.rb()).rb(),
        );

        //----------------------------------------------------------------------
        // Angular residual
        //----------------------------------------------------------------------

        // Phi(3:6) = axial(Rt*inv(rb))
        let c = quat_as_matrix_alloc(rc.rb());
        axial_vector_of_matrix(c.rb(), self.phi.subrows_mut(3, 3));

        //----------------------------------------------------------------------
        // Target constraint gradient
        //----------------------------------------------------------------------

        // B(3:6,3:6) = transpose(AX(Rt*inv(Rb)))
        let mut ax = Mat::<f64>::zeros(3, 3);
        matrix_ax(c.rb(), ax.as_mut());
        self.b_target
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(ax.transpose());

        //----------------------------------------------------------------------
        // Base constraint gradient
        //----------------------------------------------------------------------

        // B(3:6,3:6) = -AX(Rt*inv(Rb))
        self.b_base.submatrix_mut(3, 3, 3, 3).copy_from(-&ax);

        //----------------------------------------------------------------------
        // Stiffness matrix
        //----------------------------------------------------------------------

        // Rotation matrix of rt_rbinv
        let m_rc = quat_as_matrix_alloc(rc.rb());

        // lambda_tilde from rotational lambda terms
        let lambda_2 = lambda.subrows(3, 3);
        let lambda_2_tilde = vec_tilde_alloc(lambda_2);

        // Stiffness matrix components
        matrix_ax((&m_rc * &lambda_2_tilde).rb(), ax.as_mut());
        zip!(&mut self.k_t, &ax).for_each(|unzip!(k, ax)| *k += *ax);
        matrix_ax2(m_rc.transpose(), lambda_2, ax.as_mut());
        zip!(&mut self.k_tb, &ax).for_each(|unzip!(k, ax)| *k += *ax);
        matrix_ax2(m_rc.rb(), lambda_2, ax.as_mut());
        zip!(&mut self.k_bt, &ax).for_each(|unzip!(k, ax)| *k -= *ax);
        matrix_ax((&m_rc.transpose() * &lambda_2_tilde).rb(), ax.as_mut());
        zip!(&mut self.k_b, &ax).for_each(|unzip!(k, ax)| *k -= *ax);
    }

    fn calculate_revolute(
        &mut self,
        u_base: ColRef<f64>,
        r_base: ColRef<f64>,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
        lambda: ColRef<f64>,
    ) {
        let rb_x0 = quat_rotate_vector_alloc(r_base, self.x0.rb());
        let mut x = Col::<f64>::zeros(3);
        let mut y = Col::<f64>::zeros(3);
        let mut z = Col::<f64>::zeros(3);
        let mut xcz = Col::<f64>::zeros(3);
        let mut xcy = Col::<f64>::zeros(3);
        quat_rotate_vector(r_base, self.axes.col(0), x.as_mut());
        quat_rotate_vector(r_target, self.axes.col(1), y.as_mut());
        quat_rotate_vector(r_target, self.axes.col(2), z.as_mut());
        cross_product(x.rb(), z.rb(), xcz.as_mut());
        cross_product(x.rb(), y.rb(), xcy.as_mut());

        // Position residual: Phi[0:2] = u2 + X0 - u1 - Rb*X0
        self.phi[0] = u_target[0] + self.x0[0] - u_base[0] - rb_x0[0];
        self.phi[1] = u_target[1] + self.x0[1] - u_base[1] - rb_x0[1];
        self.phi[2] = u_target[2] + self.x0[2] - u_base[2] - rb_x0[2];

        // Phi[3] = dot(Rt * z0_hat, Rb * x0_hat)
        self.phi[3] = dot_product(z.rb(), x.rb());

        // Phi[4] = dot(Rt * y0_hat, Rb * x0_hat)
        self.phi[4] = dot_product(y.rb(), x.rb());

        //----------------------------------------------------------------------
        // Target node constraint gradient
        //----------------------------------------------------------------------

        // B(3, 3:6) = -cross(R1 * x0_hat, transpose(R2 * z0_hat))
        self.b_target
            .row_mut(3)
            .subcols_mut(3, 3)
            .copy_from(-xcz.transpose());

        // B(4, 3:6) = -cross(R1 * x0_hat, transpose(R2 * y0_hat))
        self.b_target
            .row_mut(4)
            .subcols_mut(3, 3)
            .copy_from(-xcy.transpose());

        //----------------------------------------------------------------------
        // Base node constraint gradient
        //----------------------------------------------------------------------

        // B(3,3:6) = cross(R1 * x0_hat, transpose(R2 * z0_hat))
        self.b_base
            .row_mut(3)
            .subcols_mut(3, 3)
            .copy_from(xcz.transpose());

        // B(4,3:6) = cross(R1 * x0_hat, transpose(R2 * y0_hat))
        self.b_base
            .row_mut(4)
            .subcols_mut(3, 3)
            .copy_from(xcy.transpose());

        //----------------------------------------------------------------------
        // Stiffness matrix
        //----------------------------------------------------------------------

        let lambda_2 = lambda[3];
        let lambda_3 = lambda[4];

        let x_tilde = vec_tilde_alloc(x.rb());
        let y_tilde = vec_tilde_alloc(y.rb());
        let z_tilde = vec_tilde_alloc(z.rb());

        self.k_t
            .copy_from(lambda_2 * &x_tilde * &z_tilde + lambda_3 * &x_tilde * &y_tilde);
        self.k_tb
            .copy_from(-lambda_2 * &z_tilde * &x_tilde - lambda_3 * &y_tilde * &x_tilde);
        self.k_bt
            .copy_from(-lambda_2 * &x_tilde * &z_tilde - lambda_3 * &x_tilde * &y_tilde);
        self.k_b
            .copy_from(lambda_2 * &z_tilde * &x_tilde + lambda_3 * &y_tilde * &x_tilde);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::node::{ActiveDOFs, Node};
    use equator::assert;
    use faer::{
        dyn_stack::{MemBuffer, MemStack},
        sparse::linalg::matmul::{
            sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
            sparse_sparse_matmul_symbolic,
        },
        utils::approx::{ApproxEq, CwiseMat},
    };

    fn create_nfm() -> NodeFreedomMap {
        NodeFreedomMap::new(&[
            Node {
                id: 0,
                s: 0.,
                x: [0.; 7],
                u: [0.; 7],
                v: [0.; 6],
                vd: [0.; 6],
                active_dofs: ActiveDOFs::All,
            },
            Node {
                id: 1,
                s: 0.,
                x: [0.; 7],
                u: [0.; 7],
                v: [0.; 6],
                vd: [0.; 6],
                active_dofs: ActiveDOFs::All,
            },
        ])
    }

    #[test]
    fn test_rigid_constraint() {
        let nfm = create_nfm();

        let mut c = Constraint::new(
            0,
            &ConstraintInput {
                id: 0,
                kind: ConstraintKind::Rigid,
                node_id_base: 0,
                node_id_target: 1,
                x0: col![1., 2., 3.],
                vec: col![0., 0., 0.],
            },
            &nfm,
        );

        c.calculate_rigid(
            col![18., 19., 20.,].as_ref(),
            col![21., 22., 23., 24.].as_ref(),
            col![11., 12., 13.].as_ref(),
            col![14., 15., 16., 17.].as_ref(),
            col![0., 0., 0., 0., 0., 0.].as_ref(),
        );

        let approx_eq = CwiseMat(ApproxEq::eps() * 1000.);

        assert!(c.phi ~ col![-5900., -2385., -4162., 19.310344827586249, -6.5558669604115494e-14, 38.620689655172455]);
        assert!(c.b_base ~ mat![
            [-1., 0., 0., 0., -4158., 2380.],
            [0., -1., 0., 4158., 0., -5894.],
            [0., 0., -1., -2380., 5894., 0.],
            [0., 0., 0., -965.42068965517251, -19.310344827586228, 0.1931034482758299],
            [0., 0., 0., 19.310344827586228, -965.51724137931046, -9.6551724137931245],
            [0., 0., 0., 0.19310344827589546, 9.6551724137931245, -965.13103448275876],
        ]);
        assert!(c.b_target ~ mat![
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 965.42068965517251, -19.310344827586228, -0.19310344827589564],
            [0., 0., 0., 19.310344827586228, 965.51724137931046, -9.6551724137931245],
            [0., 0., 0., -0.1931034482758299, 9.6551724137931245, 965.13103448275876],
        ]);
    }

    #[test]
    fn test_revolute_joint() {
        let nfm = create_nfm();

        let mut c = Constraint::new(
            0,
            &ConstraintInput {
                id: 0,
                kind: ConstraintKind::Revolute,
                node_id_base: 0,
                node_id_target: 1,
                x0: col![1., 2., 3.],
                vec: col![0., 0., 0.],
            },
            &nfm,
        );
        c.axes = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
            .transpose()
            .cloned();

        c.calculate_revolute(
            col![18., 19., 20.,].as_ref(),
            col![21., 22., 23., 24.].as_ref(),
            col![11., 12., 13.].as_ref(),
            col![14., 15., 16., 17.].as_ref(),
            col![0., 0., 0., 0., 0., 0.].as_ref(),
        );

        let approx_eq = CwiseMat(ApproxEq::eps() * 1000.);
        println!("c.phi: {:?}", c.phi);
        assert!(c.phi ~ col![-5900., -2385., -4162., 97314000., 62379744.]);
        let b_base_exp = mat![
            [-1., 0., 0., 0., 0., 0.],
            [0., -1., 0., 0., 0., 0.],
            [0., 0., -1., 0., 0., 0.],
            [0., 0., 0., -10930136., -15850520., 24566248.],
            [0., 0., 0., -5585804., -8091272., 12549292.],
        ];
        assert!(c.b_base ~ b_base_exp);
        assert!(c.b_target ~ -b_base_exp);
    }

    #[test]
    fn test_prescribed() {
        let nfm = create_nfm();

        let mut c = Constraint::new(
            0,
            &ConstraintInput {
                id: 0,
                kind: ConstraintKind::Prescribed,
                node_id_base: 0,
                node_id_target: 1,
                x0: col![1., 2., 3.],
                vec: col![0., 0., 0.],
            },
            &nfm,
        );

        // Set prescribed displacement
        c.input = col![4., 5., 6., 7., 8., 9., 10.];

        // Calculate prescribed constraint
        c.calculate_prescribed(
            col![11., 12., 13.].as_ref(),
            col![14., 15., 16., 17.].as_ref(),
            col![0., 0., 0., 0., 0., 0.].as_ref(),
        );

        let approx_eq = CwiseMat(ApproxEq::eps() * 1000.);
        println!("c.phi: {:?}", c.phi);
        assert!(c.phi ~ col![-790., -411., -620., -50.666666666666657, -7.1054273576010019e-15, -101.33333333333343]);
        assert!(c.b_target ~ mat![
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 961.99999999999977, 50.666666666666714, -1.3333333333333379],
            [0., 0., 0., -50.666666666666714, 962.6666666666664, 25.333333333333329],
            [0., 0., 0., -1.3333333333333308, -25.333333333333329, 959.99999999999977],
        ]);
    }

    #[test]
    fn test_sparse_order() {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![
            Triplet::new(0, 0, 1.),
            Triplet::new(0, 2, 2.),
            Triplet::new(2, 0, 3.),
        ];

        let m_sp = SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap();

        let approx_eq = CwiseMat(ApproxEq::eps());

        assert!(m_sp.to_dense() ~ mat![
            [1., 0., 2.],
            [0., 0., 0.],
            [3., 0., 0.]
        ]);

        assert!(Col::from_iter(m_sp.val().iter().cloned()) ~ col![1., 3., 2.]);
    }

    #[test]
    fn test_sparse_merge() {
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            5,
            &vec![
                Triplet::new(0, 0, 2.),
                Triplet::new(1, 1, 1.),
                Triplet::new(2, 2, 1.),
                Triplet::new(3, 3, 1.),
                Triplet::new(4, 4, 1.),
            ],
        )
        .unwrap();

        let b = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            5,
            &vec![
                Triplet::new(3, 0, 1.),
                Triplet::new(4, 2, 2.),
                Triplet::new(4, 1, 3.),
            ],
        )
        .unwrap();

        // Get symbolic multiplication of a and b sparse matrices
        let (c_sym, mminfo) = sparse_sparse_matmul_symbolic(b.symbolic(), a.symbolic()).unwrap();

        let c_sym = ops::union_symbolic(
            c_sym.as_ref(),
            b.symbolic().transpose().to_col_major().unwrap().as_ref(),
        )
        .unwrap();

        let c_data = vec![0.; c_sym.compute_nnz()];
        let mut c = SparseColMat::<usize, f64>::new(c_sym, c_data);

        let mut mem_buffer = MemBuffer::try_new(
            sparse_sparse_matmul_numeric_scratch::<usize, f64>(c.symbolic(), Par::Seq),
        )
        .unwrap();
        sparse_sparse_matmul_numeric(
            c.as_shape_mut(5, 5),
            Accum::Add,
            b.as_ref(),
            a.as_ref(),
            1.,
            &mminfo,
            Par::Seq,
            MemStack::new(&mut mem_buffer),
        );

        ops::add_assign(
            c.as_shape_mut(5, 5),
            b.transpose().to_col_major().unwrap().as_ref(),
        );

        let approx_eq = CwiseMat(ApproxEq::eps());

        assert!(c.to_dense() ~ mat![
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 3.],
            [0., 0., 0., 0., 2.],
            [2., 0., 0., 0., 0.],
            [0., 3., 2., 0., 0.],
        ]);

        assert!(Col::from_iter(c.val().iter().cloned()) ~ col![2., 3., 2., 1., 3., 2.]);
    }
}

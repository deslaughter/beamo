use crate::{
    node::NodeFreedomMap,
    state::State,
    util::{
        axial_vector_of_matrix, cross_product, dot_product, matrix_ax, matrix_ax2,
        quat_as_matrix_alloc, quat_compose, quat_from_rotation_vector, quat_inverse,
        quat_rotate_vector, quat_rotate_vector_alloc, vec_tilde_alloc,
    },
};
use faer::{linalg::matmul::matmul, prelude::*, Accum, Par};
use itertools::Itertools;
use std::cmp;

#[derive(Clone, Copy)]
pub enum ConstraintKind {
    Prescribed,
    Rigid,
    Revolute,
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
    pub constraints: Vec<Constraint>,
}

impl Constraints {
    pub fn new(inputs: &[ConstraintInput], nfm: &NodeFreedomMap) -> Self {
        let mut n_dofs = 0;
        let constraints = inputs
            .iter()
            .map(|inp| {
                let c = Constraint::new(n_dofs, inp, nfm);
                n_dofs += c.n_rows;
                c
            })
            .collect_vec();
        Self {
            n_rows: n_dofs,
            constraints,
        }
    }

    pub fn assemble_constraints(
        &mut self,
        nfm: &NodeFreedomMap,
        state: &State,
        lambda: ColRef<f64>,
        mut phi: ColMut<f64>,
        mut b: MatMut<f64>,
        mut kt: MatMut<f64>,
    ) {
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
                ConstraintKind::Rigid => {
                    c.calculate_rigid(u_base, r_base, u_target, r_target, lambda)
                }
                ConstraintKind::Revolute => {
                    c.calculate_revolute(u_base, r_base, u_target, r_target, lambda);
                }
            }
        });

        // Assemble residual and gradient into global matrix and array
        self.constraints.iter_mut().for_each(|c| {
            // Subsection of residual for this constraint
            let mut phi_c = phi.as_mut().subrows_mut(c.first_row_index, c.phi.nrows());

            // Assemble residual and gradient based on type
            let dofs_target = &nfm.node_dofs[c.node_id_target];
            let mut b_target = b.as_mut().submatrix_mut(
                c.first_row_index,
                dofs_target.first_dof_index,
                c.b_target.nrows(),
                c.b_target.ncols(),
            );

            // c.k.fill(0.);

            match c.kind {
                ConstraintKind::Prescribed => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                    if dofs_target.n_dofs == 6 {
                        let mut kt_sub = kt.as_mut().submatrix_mut(
                            dofs_target.first_dof_index,
                            dofs_target.first_dof_index,
                            6,
                            6,
                        );
                        zip!(&mut kt_sub, &c.k.submatrix(0, 0, 6, 6))
                            .for_each(|unzip!(kt_sub, k_const)| *kt_sub += *k_const);
                    }
                }
                _ => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                    let dofs_base = &nfm.node_dofs[c.node_id_base];
                    let mut b_base = b.rb_mut().submatrix_mut(
                        c.first_row_index,
                        dofs_base.first_dof_index,
                        c.n_rows,
                        c.b_base.ncols(),
                    );
                    b_base.copy_from(&c.b_base);

                    // Base node stiffness matrix
                    zip!(
                        &mut kt.rb_mut().submatrix_mut(
                            dofs_base.first_dof_index,
                            dofs_base.first_dof_index,
                            6,
                            6,
                        ),
                        &c.k.submatrix(6, 6, 6, 6)
                    )
                    .for_each(|unzip!(kt, kc)| *kt += *kc);

                    // If target node has 6 DOFs, add remainder of the stiffness matrix
                    if dofs_target.n_dofs == 6 {
                        // Upper right corner of stiffness matrix
                        zip!(
                            &mut kt.rb_mut().submatrix_mut(
                                dofs_target.first_dof_index,
                                dofs_base.first_dof_index,
                                6,
                                6,
                            ),
                            &c.k.submatrix(0, 6, 6, 6)
                        )
                        .for_each(|unzip!(kt, kc)| *kt += *kc);

                        // Lower left corner of stiffness matrix
                        zip!(
                            &mut kt.rb_mut().submatrix_mut(
                                dofs_base.first_dof_index,
                                dofs_target.first_dof_index,
                                6,
                                6,
                            ),
                            &c.k.submatrix(6, 0, 6, 6)
                        )
                        .for_each(|unzip!(kt, kc)| *kt += *kc);

                        // Target node stiffness matrix
                        zip!(
                            &mut kt.as_mut().submatrix_mut(
                                dofs_target.first_dof_index,
                                dofs_target.first_dof_index,
                                6,
                                6,
                            ),
                            &c.k.submatrix(0, 0, 6, 6)
                        )
                        .for_each(|unzip!(kt, kc)| *kt += *kc);
                    }
                }
            }
        });
    }
}

pub struct Constraint {
    kind: ConstraintKind,
    first_row_index: usize,
    n_rows: usize,
    node_id_base: usize,
    node_id_target: usize,
    x0: Col<f64>,
    phi: Col<f64>,
    b_base: Mat<f64>,
    b_target: Mat<f64>,
    input: Col<f64>,
    rbinv: Col<f64>,
    rt_rbinv: Col<f64>,
    /// Stiffness matrix for base node `[12,12]`
    k: Mat<f64>,
    /// Rotation matrix `[3,3]`
    axes: Mat<f64>,
}

impl Constraint {
    fn new(first_dof_index: usize, input: &ConstraintInput, nfm: &NodeFreedomMap) -> Self {
        let n_dofs_base = nfm.node_dofs[input.node_id_base].n_dofs;
        let n_dofs_target = nfm.node_dofs[input.node_id_target].n_dofs;

        // Get number of constraint DOFs (rows of phi vector or B matrix)
        let n_rows = match input.kind {
            ConstraintKind::Prescribed => n_dofs_target,
            ConstraintKind::Rigid => cmp::min(n_dofs_base, n_dofs_target),
            ConstraintKind::Revolute => 5,
        };

        // Calculate constraint axes
        let axes = match input.kind {
            ConstraintKind::Revolute => {
                let x = col![1., 0., 0.];
                let x_hat = if input.vec.rb().norm_l2() != 0. {
                    &input.vec / input.vec.norm_l2()
                } else {
                    &input.x0 / input.x0.norm_l2()
                };

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
            _ => {
                let mut axes = Mat::<f64>::zeros(3, 3);
                axes.col_mut(0).copy_from(&input.vec);
                axes
            }
        };

        Self {
            kind: input.kind,
            first_row_index: first_dof_index,
            n_rows,
            node_id_base: input.node_id_base,
            node_id_target: input.node_id_target,
            x0: input.x0.clone(),
            input: col![0., 0., 0., 1., 0., 0., 0.],
            phi: Col::<f64>::zeros(n_rows),
            b_base: -1. * Mat::<f64>::identity(n_rows, n_dofs_base),
            b_target: Mat::<f64>::identity(n_rows, n_dofs_target),
            rbinv: Col::<f64>::zeros(4),
            rt_rbinv: Col::<f64>::zeros(4),
            k: Mat::<f64>::zeros(12, 12),
            axes,
        }
    }

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

    fn calculate_prescribed(
        &mut self,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
        lambda: ColRef<f64>,
    ) {
        let u_base = self.input.subrows(0, 3);
        let r_base = self.input.subrows(3, 4);

        // Position residual: Phi(0:3) = u2 - u1
        self.phi[0] = u_target[0] - u_base[0];
        self.phi[1] = u_target[1] - u_base[1];
        self.phi[2] = u_target[2] - u_base[2];

        // If only position is prescribed, return
        if self.n_rows == 3 {
            return;
        }

        // Angular residual:  Phi(3:6) = axial(Rt*inv(Rb))
        quat_inverse(r_base, self.rbinv.as_mut());
        quat_compose(
            r_target.as_ref(),
            self.rbinv.as_ref(),
            self.rt_rbinv.as_mut(),
        );
        let c = quat_as_matrix_alloc(self.rt_rbinv.as_ref());
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
        matrix_ax((-&c * &lambda_tilde).rb(), self.k.submatrix_mut(3, 3, 3, 3));
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
        let rb_x0_tilde = vec_tilde_alloc(rb_x0.as_ref());
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

        self.k.fill(0.);
        matmul(
            self.k.submatrix_mut(9, 9, 3, 3),
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

        //----------------------------------------------------------------------
        // Angular residual
        //----------------------------------------------------------------------

        // Phi(3:6) = axial(Rt*inv(rb))
        quat_inverse(r_base, self.rbinv.as_mut());
        quat_compose(r_target, self.rbinv.rb(), self.rt_rbinv.as_mut());
        let c = quat_as_matrix_alloc(self.rt_rbinv.rb());
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
        let m_rt_rbinv = quat_as_matrix_alloc(self.rt_rbinv.as_ref());

        // lambda_tilde from rotational lambda terms
        let lambda_2 = lambda.subrows(3, 3);
        let lambda_2_tilde = vec_tilde_alloc(lambda_2);

        // Stiffness matrix components
        matrix_ax((&m_rt_rbinv * &lambda_2_tilde).rb(), ax.as_mut());
        zip!(&mut self.k.submatrix_mut(3, 3, 3, 3), &ax).for_each(|unzip!(k, ax)| *k += *ax);
        matrix_ax2(m_rt_rbinv.transpose(), lambda_2, ax.as_mut());
        zip!(&mut self.k.submatrix_mut(3, 9, 3, 3), &ax).for_each(|unzip!(k, ax)| *k += *ax);
        matrix_ax2(m_rt_rbinv.rb(), lambda_2, ax.as_mut());
        zip!(&mut self.k.submatrix_mut(9, 3, 3, 3), &ax).for_each(|unzip!(k, ax)| *k -= *ax);
        matrix_ax(
            (&m_rt_rbinv.transpose() * &lambda_2_tilde).rb(),
            ax.as_mut(),
        );
        zip!(&mut self.k.submatrix_mut(9, 9, 3, 3), &ax).for_each(|unzip!(k, ax)| *k -= *ax);
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

        self.k
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(lambda_2 * &x_tilde * &z_tilde + lambda_3 * &x_tilde * &y_tilde);
        self.k
            .submatrix_mut(3, 9, 3, 3)
            .copy_from(-lambda_2 * &z_tilde * &x_tilde - lambda_3 * &y_tilde * &x_tilde);
        self.k
            .submatrix_mut(9, 3, 3, 3)
            .copy_from(-lambda_2 * &x_tilde * &z_tilde - lambda_3 * &x_tilde * &y_tilde);
        self.k
            .submatrix_mut(9, 9, 3, 3)
            .copy_from(lambda_2 * &z_tilde * &x_tilde + lambda_3 * &y_tilde * &x_tilde);
    }
}

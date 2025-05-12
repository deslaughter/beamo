use crate::{
    node::NodeFreedomMap,
    state::State,
    util::{
        axial_vector_of_matrix, cross_product, dot_product, matrix_ax, quat_as_matrix,
        quat_compose, quat_from_rotation_vector, quat_inverse, quat_rotate_vector, vec_tilde,
    },
};
use faer::{linalg::matmul::matmul, prelude::*, Accum, Par};
use itertools::Itertools;
use std::cmp;

#[derive(Clone, Copy)]
pub enum ConstraintKind {
    Rigid,
    Prescribed,
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
            let lambda_target = lambda.subrows(c.first_row_index, c.phi.nrows());

            // Switch calculation based on constraint type
            match c.kind {
                ConstraintKind::Prescribed => c.calculate_prescribed(u_target, r_target),
                ConstraintKind::Rigid => {
                    c.calculate_rigid(u_base, r_base, u_target, r_target, lambda_target)
                }
                ConstraintKind::Revolute => {
                    c.calculate_revolute(u_base, r_base, u_target, r_target);
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
            match c.kind {
                ConstraintKind::Prescribed => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                }
                _ => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                    let dofs_base = &nfm.node_dofs[c.node_id_base];
                    let mut b_base = b.as_mut().submatrix_mut(
                        c.first_row_index,
                        dofs_base.first_dof_index,
                        c.b_base.nrows(),
                        c.b_base.ncols(),
                    );
                    b_base.copy_from(&c.b_base);
                    if dofs_base.n_dofs == 6 {
                        let mut k_base = kt.as_mut().submatrix_mut(
                            dofs_base.first_dof_index,
                            dofs_base.first_dof_index,
                            6,
                            6,
                        );
                        zip!(&mut k_base, &c.k_base).for_each(|unzip!(k, k_const)| *k += *k_const);
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
    u_prescribed: Col<f64>,
    rbinv: Col<f64>,
    rt_rbinv: Col<f64>,
    r_x0: Col<f64>,
    /// Rotation matrix `[3,3]`
    c: Mat<f64>,
    /// Axial vector of rotation matrix
    ax: Mat<f64>,
    /// Stiffness matrix for target node `[6,6]`
    k_base: Mat<f64>,
    r_x0_tilde: Mat<f64>,
    lambda_tilde: Mat<f64>,
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
            u_prescribed: col![0., 0., 0., 1., 0., 0., 0.],
            phi: Col::<f64>::zeros(n_rows),
            b_base: -1. * Mat::<f64>::identity(n_rows, n_dofs_base),
            b_target: Mat::<f64>::identity(n_rows, n_dofs_target),
            r_x0: Col::<f64>::zeros(3),
            rbinv: Col::<f64>::zeros(4),
            rt_rbinv: Col::<f64>::zeros(4),
            c: Mat::<f64>::zeros(3, 3),
            ax: Mat::zeros(3, 3),
            k_base: Mat::<f64>::zeros(n_dofs_base, n_dofs_base),
            r_x0_tilde: Mat::<f64>::zeros(3, 3),
            lambda_tilde: Mat::<f64>::zeros(3, 3),
            axes,
        }
    }

    pub fn set_displacement(&mut self, x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) {
        let mut q = col![0., 0., 0., 0.];
        quat_from_rotation_vector(col![rx, ry, rz].as_ref(), q.as_mut());
        self.u_prescribed[0] = x;
        self.u_prescribed[1] = y;
        self.u_prescribed[2] = z;
        self.u_prescribed[3] = q[0];
        self.u_prescribed[4] = q[1];
        self.u_prescribed[5] = q[2];
        self.u_prescribed[6] = q[3];
    }

    fn calculate_prescribed(&mut self, u_target: ColRef<f64>, r_target: ColRef<f64>) {
        let u_base = self.u_prescribed.subrows(0, 3);
        let r_base = self.u_prescribed.subrows(3, 4);

        // Position residual: Phi(0:3) = u2 - u1
        self.phi[0] = u_target[0] - u_base[0];
        self.phi[1] = u_target[1] - u_base[1];
        self.phi[2] = u_target[2] - u_base[2];

        if self.n_rows == 3 {
            return;
        }

        // Angular residual:  Phi(3:6) = axial(R2*inv(R1))
        quat_inverse(r_base, self.rbinv.as_mut());
        quat_compose(
            r_target.as_ref(),
            self.rbinv.as_ref(),
            self.rt_rbinv.as_mut(),
        );
        quat_as_matrix(self.rt_rbinv.as_ref(), self.c.as_mut());
        axial_vector_of_matrix(self.c.as_ref(), self.phi.subrows_mut(3, 3));

        // Constraint Gradient
        // Set at initialization B(0:3,0:3) = I
        // B(3:6,3:6) = AX(R1*inv(R2)) = transpose(AX(R2*inv(R1)))
        matrix_ax(
            self.c.as_ref(),
            self.b_target.submatrix_mut(3, 3, 3, 3).transpose_mut(),
        );
    }

    fn calculate_rigid(
        &mut self,
        u_base: ColRef<f64>,
        r_base: ColRef<f64>,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
        lambda: ColRef<f64>,
    ) {
        // Position residual: Phi(0:3) = u2 + X0 - u1 - R1*X0
        quat_rotate_vector(r_base.as_ref(), self.x0.as_ref(), self.r_x0.as_mut());
        vec_tilde(self.r_x0.as_ref(), self.r_x0_tilde.as_mut());
        zip!(
            &mut self.phi.subrows_mut(0, 3),
            &u_base,
            &u_target,
            &self.x0,
            &self.r_x0
        )
        .for_each(|unzip!(phi, u1, u2, x0, rb_x0)| *phi = *u2 + *x0 - *u1 - *rb_x0);

        // Stiffness matrix for target node
        vec_tilde(lambda.subrows(0, 3), self.lambda_tilde.as_mut());
        if self.k_base.shape() == (6, 6) {
            matmul(
                self.k_base.submatrix_mut(3, 3, 3, 3),
                Accum::Replace,
                &self.lambda_tilde,
                &self.r_x0_tilde,
                -1.,
                Par::Seq,
            );
        }

        // Angular residual:  Phi(3:6) = axial(R2*inv(rb))
        if self.n_rows == 6 {
            quat_inverse(r_base, self.rbinv.as_mut());
            quat_compose(r_target, self.rbinv.as_ref(), self.rt_rbinv.as_mut());
            quat_as_matrix(self.rt_rbinv.as_ref(), self.c.as_mut());
            axial_vector_of_matrix(self.c.as_ref(), self.phi.subrows_mut(3, 3));
        }

        // Base constraint gradient
        // Set at initialization B(0:3,0:3) = -I
        // B(0:3,3:6) = tilde(R1*X0)
        self.b_base
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&self.r_x0_tilde);
        if self.n_rows == 6 {
            // AX(c)
            matrix_ax(self.c.as_ref(), self.ax.as_mut());
            // B(3:6,3:6) = -AX(R2*inv(R1))
            zip!(&mut self.b_base.submatrix_mut(3, 3, 3, 3), &self.ax)
                .for_each(|unzip!(b, ax)| *b = -*ax);
        }

        // Target constraint gradient
        // Set at initialization B(0:3,0:3) = I
        if self.n_rows == 6 {
            // B(3:6,3:6) = transpose(AX(R2*inv(R1)))
            self.b_target
                .submatrix_mut(3, 3, 3, 3)
                .copy_from(self.ax.transpose());
        }
    }

    fn calculate_revolute(
        &mut self,
        u_base: ColRef<f64>,
        r_base: ColRef<f64>,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
    ) {
        quat_rotate_vector(r_base.as_ref(), self.x0.as_ref(), self.r_x0.as_mut());
        let mut x = Col::<f64>::zeros(3);
        let mut y = Col::<f64>::zeros(3);
        let mut z = Col::<f64>::zeros(3);
        let mut xcz = Col::<f64>::zeros(3);
        let mut xcy = Col::<f64>::zeros(3);
        quat_rotate_vector(r_base.as_ref(), self.axes.col(0), x.as_mut());
        quat_rotate_vector(r_target.as_ref(), self.axes.col(1), y.as_mut());
        quat_rotate_vector(r_target.as_ref(), self.axes.col(2), z.as_mut());
        cross_product(x.as_ref(), z.as_ref(), xcz.as_mut());
        cross_product(x.as_ref(), y.as_ref(), xcy.as_mut());

        // Position residual: Phi(0:3) = u2 + X0 - u1 - R1*X0
        self.phi[0] = u_base[0] + self.x0[0] - u_target[0] - self.r_x0[0];
        self.phi[1] = u_base[1] + self.x0[1] - u_target[1] - self.r_x0[1];
        self.phi[2] = u_base[2] + self.x0[2] - u_target[2] - self.r_x0[2];

        // Phi(3) = dot(R2 * z0_hat, R1 * x0_hat)
        self.phi[3] = dot_product(x.as_ref(), y.as_ref());

        // Phi(4) = dot(R2 * y0_hat, R1 * x0_hat)
        self.phi[4] = dot_product(x.as_ref(), z.as_ref());

        // Target node constraint gradient
        self.b_target
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(Mat::<f64>::identity(3, 3));
        self.b_target
            .submatrix_mut(3, 3, 1, 3)
            .copy_from(-xcz.as_mat());
        self.b_target
            .submatrix_mut(4, 3, 1, 3)
            .copy_from(-xcy.as_mat());

        // Base node constraint gradient
        self.b_base
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(-Mat::<f64>::identity(3, 3));
        self.b_base
            .submatrix_mut(3, 3, 1, 3)
            .copy_from(xcz.as_mat());
        self.b_base
            .submatrix_mut(4, 3, 1, 3)
            .copy_from(xcy.as_mat());
    }
}

use crate::{
    beams::{ColAsMatMut, ColAsMatRef, Damping},
    quaternion::{quat_as_matrix, quat_derivative, quat_rotate_vector, Quat},
};
use faer::{
    linalg::matmul::matmul,
    mat::{AsMatMut, AsMatRef},
    unzipped, zipped, Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism,
};
use itertools::izip;

pub struct BeamQPs {
    /// Integration weights `[n_qps]`
    pub weight: Col<f64>,
    /// Jacobian vector `[n_qps]`
    pub jacobian: Col<f64>,
    /// Mass matrix in material frame `[6][6][n_qps]`
    pub m_star: Mat<f64>,
    /// Stiffness matrix in material frame `[6][6][n_qps]`
    pub c_star: Mat<f64>,
    /// Current position/orientation `[7][n_qps]`
    pub x: Mat<f64>,
    /// Initial position `[7][n_qps]`
    pub x0: Mat<f64>,
    /// Initial position derivative `[7][n_qps]`
    pub x0_prime: Mat<f64>,
    /// State: displacement `[7][n_qps]`
    pub u: Mat<f64>,
    /// State: displacement derivative wrt x`[7][n_qps]`
    pub u_prime: Mat<f64>,
    /// State: velocity `[6][n_qps]`
    pub v: Mat<f64>,
    /// State: velocity derivative wrt x `[6][n_qps]`
    pub v_prime: Mat<f64>,
    /// State: acceleration `[6][n_qps]`
    pub vd: Mat<f64>,
    /// tilde(x0_prime + u_prime) `[3][3][n_qps]`
    pub e1_tilde: Mat<f64>,
    /// mass `[n_qps]`
    pub m: Col<f64>,
    /// mass `[3][n_qps]`
    pub eta: Mat<f64>,
    /// mass `[3][3][n_qps]`
    pub rho: Mat<f64>,
    /// Strain `[6][n_qps]`
    pub strain: Mat<f64>,
    /// Strain Rate `[6][n_qps]`
    pub strain_dot: Mat<f64>,
    /// Elastic force C `[6][n_qps]`
    pub fe_c: Mat<f64>,
    /// Elastic force D `[6][n_qps]`
    pub fe_d: Mat<f64>,
    /// Dissipative force C `[6][n_qps]`
    pub fd_c: Mat<f64>,
    /// Dissipative force D `[6][n_qps]`
    pub fd_d: Mat<f64>,
    /// Inertial force `[6][n_qps]`
    pub fi: Mat<f64>,
    /// External force `[6][n_qps]`
    pub fx: Mat<f64>,
    /// Gravity force `[6][n_qps]`
    pub fg: Mat<f64>,
    /// Global rotation `[6][6][n_qps]`
    pub rr0: Mat<f64>,
    /// Inertial mass matrices `[6][6][n_qps]`
    pub muu: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub cuu: Mat<f64>,
    /// Damped stiffness in global frame `[6][6][n_qps]`
    pub mu_cuu: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub oe: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub pe: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub qe: Mat<f64>,
    /// Inertial gyroscopic matrices `[6][6][n_qps]`
    pub gi: Mat<f64>,
    /// Inertial stiffness matrices `[6][6][n_qps]`
    pub ki: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub pd: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub qd: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub xd: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub yd: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub sd: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub od: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub gd: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub ed: Mat<f64>,
}

impl BeamQPs {
    pub fn new(weights: &[f64]) -> Self {
        let n_qps = weights.len();
        BeamQPs {
            weight: Col::from_fn(n_qps, |i| weights[i]),
            jacobian: Col::ones(n_qps),
            m_star: Mat::zeros(6 * 6, n_qps),
            c_star: Mat::zeros(6 * 6, n_qps),
            x: Mat::zeros(7, n_qps),
            x0: Mat::zeros(7, n_qps),
            x0_prime: Mat::zeros(7, n_qps),
            u: Mat::zeros(7, n_qps),
            u_prime: Mat::zeros(7, n_qps),
            v: Mat::zeros(6, n_qps),
            v_prime: Mat::zeros(6, n_qps),
            vd: Mat::zeros(6, n_qps),
            e1_tilde: Mat::zeros(3 * 3, n_qps),
            m: Col::zeros(n_qps),
            eta: Mat::zeros(3, n_qps),
            rho: Mat::zeros(3 * 3, n_qps),
            strain: Mat::zeros(6, n_qps),
            strain_dot: Mat::zeros(6, n_qps),
            fe_c: Mat::zeros(6, n_qps),
            fe_d: Mat::zeros(6, n_qps),
            fd_c: Mat::zeros(6, n_qps),
            fd_d: Mat::zeros(6, n_qps),
            fi: Mat::zeros(6, n_qps),
            fx: Mat::zeros(6, n_qps),
            fg: Mat::zeros(6, n_qps),
            rr0: Mat::zeros(6 * 6, n_qps),
            muu: Mat::zeros(6 * 6, n_qps),
            cuu: Mat::zeros(6 * 6, n_qps),
            mu_cuu: Mat::zeros(6 * 6, n_qps),
            oe: Mat::zeros(6 * 6, n_qps),
            pe: Mat::zeros(6 * 6, n_qps),
            qe: Mat::zeros(6 * 6, n_qps),
            gi: Mat::zeros(6 * 6, n_qps),
            ki: Mat::zeros(6 * 6, n_qps),
            pd: Mat::zeros(6 * 6, n_qps),
            qd: Mat::zeros(6 * 6, n_qps),
            xd: Mat::zeros(6 * 6, n_qps),
            yd: Mat::zeros(6 * 6, n_qps),
            sd: Mat::zeros(6 * 6, n_qps),
            od: Mat::zeros(6 * 6, n_qps),
            gd: Mat::zeros(6 * 6, n_qps),
            ed: Mat::zeros(6 * 6, n_qps),
        }
    }

    pub fn calc(&mut self, gravity: ColRef<f64>, damping: &Damping) {
        calc_x(self.x.as_mut(), self.x0.as_ref(), self.u.as_ref());
        calc_rr0(self.rr0.as_mut(), self.x.as_ref());
        calc_rot_mat(self.muu.as_mut(), self.m_star.as_ref(), self.rr0.as_ref());
        calc_rot_mat(self.cuu.as_mut(), self.c_star.as_ref(), self.rr0.as_ref());
        calc_mu_cuu(
            damping,
            self.mu_cuu.as_mut(),
            self.cuu.as_ref(),
            self.rr0.as_ref(),
        );
        calc_m_eta_rho(
            self.m.as_mut(),
            self.eta.as_mut(),
            self.rho.as_mut(),
            self.muu.as_ref(),
        );
        calc_strain(
            self.strain.as_mut(),
            self.x0_prime.subrows(0, 3),
            self.u.subrows(3, 4),
            self.u_prime.subrows(0, 3),
            self.u_prime.subrows(3, 4),
        );
        calc_strain_dot(
            self.strain_dot.as_mut(),
            self.v.as_ref(),
            self.v_prime.as_ref(),
            self.e1_tilde.as_ref(),
        );
        calc_e1_tilde(
            self.e1_tilde.as_mut(),
            self.x0_prime.subrows(0, 3),
            self.u_prime.subrows(0, 3),
        );
        calc_fe_c(self.fe_c.as_mut(), self.cuu.as_ref(), self.strain.as_ref());
        calc_fe_d(
            self.fe_d.as_mut(),
            self.fe_c.as_ref(),
            self.e1_tilde.as_ref(),
        );
        calc_fd_c(
            self.fd_c.as_mut(),
            self.mu_cuu.as_ref(),
            self.strain_dot.as_ref(),
        );
        calc_fd_d(
            self.fd_d.as_mut(),
            self.fd_c.as_ref(),
            self.e1_tilde.as_ref(),
        );
        calc_fi(
            self.fi.as_mut(),
            self.m.as_ref(),
            self.v.subrows(3, 3).as_ref(),
            self.vd.subrows(0, 3).as_ref(),
            self.vd.subrows(3, 3).as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
        );
        calc_fg(
            self.fg.as_mut(),
            gravity.as_ref(),
            self.m.as_ref(),
            self.eta.as_ref(),
        );
        calc_ouu(
            self.oe.as_mut(),
            self.cuu.as_ref(),
            self.e1_tilde.as_ref(),
            self.fe_c.as_ref(),
        );
        calc_puu(
            self.pe.as_mut(),
            self.cuu.as_ref(),
            self.e1_tilde.as_ref(),
            self.fe_c.as_ref(),
        );
        calc_quu(
            self.qe.as_mut(),
            self.cuu.as_ref(),
            self.e1_tilde.as_ref(),
            self.fe_c.as_ref(),
        );
        calc_guu(
            self.gi.as_mut(),
            self.m.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
        );
        calc_kuu(
            self.ki.as_mut(),
            self.m.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
            self.vd.subrows(0, 3),
            self.vd.subrows(3, 3),
        );
        calc_sd_pd_od_qd_gd_xd_yd(
            self.sd.as_mut(),
            self.pd.as_mut(),
            self.od.as_mut(),
            self.qd.as_mut(),
            self.gd.as_mut(),
            self.xd.as_mut(),
            self.yd.as_mut(),
            self.mu_cuu.as_ref(),
            self.v_prime.subrows(0, 3).as_ref(),
            self.v.subrows(3, 3).as_ref(),
            self.fd_c.as_ref(),
            self.e1_tilde.as_ref(),
        )
    }
}

#[inline]
/// Calculate current position and rotation (x0 + u)
fn calc_x(x: MatMut<f64>, x0: MatRef<f64>, u: MatRef<f64>) {
    izip!(x.col_iter_mut(), x0.col_iter(), u.col_iter()).for_each(|(mut x, x0, u)| {
        x[0] = x0[0] + u[0];
        x[1] = x0[1] + u[1];
        x[2] = x0[2] + u[2];
        x.subrows_mut(3, 4)
            .quat_compose(u.subrows(3, 4), x0.subrows(3, 4));
    });
}

#[inline]
/// Calculate rotation matrix from material to inertial coordinates
fn calc_rr0(rr0: MatMut<f64>, x: MatRef<f64>) {
    let mut m = Mat::<f64>::zeros(3, 3);
    izip!(rr0.col_iter_mut(), x.subrows(3, 4).col_iter()).for_each(|(col, r)| {
        let mut rr0 = col.as_mat_mut(6, 6);
        quat_as_matrix(r, m.as_mut());
        rr0.as_mut().submatrix_mut(0, 0, 3, 3).copy_from(&m);
        rr0.as_mut().submatrix_mut(3, 3, 3, 3).copy_from(&m);
    });
}

#[inline]
/// Rotate material into inertial coordinates
fn calc_rot_mat(mat: MatMut<f64>, mat_star: MatRef<f64>, rr0: MatRef<f64>) {
    let mut mat_tmp = Mat::<f64>::zeros(6, 6);
    izip!(mat.col_iter_mut(), mat_star.col_iter(), rr0.col_iter()).for_each(
        |(mat_col, mat_star_col, rr0_col)| {
            let mat = mat_col.as_mat_mut(6, 6);
            let mat_star = mat_star_col.as_mat_ref(6, 6);
            let rr0 = rr0_col.as_mat_ref(6, 6);
            matmul(mat_tmp.as_mut(), rr0, mat_star, None, 1., Parallelism::None);
            matmul(
                mat,
                mat_tmp.as_ref(),
                rr0.transpose(),
                None,
                1.,
                Parallelism::None,
            );
        },
    );
}

#[inline]
fn calc_mu_cuu(damping: &Damping, mu_cuu: MatMut<f64>, cuu: MatRef<f64>, rr0: MatRef<f64>) {
    match damping {
        Damping::None => {}
        Damping::Mu(mu) => {
            let mut mu_mat = Mat::<f64>::zeros(6, 6);
            let mut rr0_mat_rr0t = Mat::<f64>::zeros(6, 6);
            let mut tmp6 = Mat::<f64>::zeros(6, 6);
            mu_mat.diagonal_mut().column_vector_mut().copy_from(&mu);
            izip!(mu_cuu.col_iter_mut(), cuu.col_iter(), rr0.col_iter()).for_each(
                |(mu_cuu_col, cuu_col, rr0_col)| {
                    let mut mu_cuu = mu_cuu_col.as_mat_mut(6, 6);
                    let cuu = cuu_col.as_mat_ref(6, 6);
                    let rr0 = rr0_col.as_mat_ref(6, 6);

                    // Rotate mu damping coefficients into inertial frame
                    matmul(
                        tmp6.as_mut(),
                        rr0,
                        mu_mat.as_ref(),
                        None,
                        1.,
                        Parallelism::None,
                    );
                    matmul(
                        rr0_mat_rr0t.as_mut(),
                        tmp6.as_ref(),
                        rr0.transpose(),
                        None,
                        1.,
                        Parallelism::None,
                    );

                    // Multiply damping coefficients by stiffness matrix
                    matmul(
                        mu_cuu.as_mut(),
                        rr0_mat_rr0t.as_ref(),
                        cuu,
                        None,
                        1.0,
                        Parallelism::None,
                    );
                },
            );
        }
    }
}

#[inline]
fn calc_m_eta_rho(m: ColMut<f64>, eta: MatMut<f64>, rho: MatMut<f64>, muu: MatRef<f64>) {
    izip!(
        m.iter_mut(),
        eta.col_iter_mut(),
        rho.col_iter_mut(),
        muu.col_iter()
    )
    .for_each(|(m, mut eta, rho_col, muu_col)| {
        let muu = muu_col.as_mat_ref(6, 6);
        *m = muu[(0, 0)];
        if *m == 0. {
            eta.fill(0.);
        } else {
            eta[0] = muu[(5, 1)] / *m;
            eta[1] = -muu[(5, 0)] / *m;
            eta[2] = muu[(4, 0)] / *m;
        }
        let mut rho = rho_col.as_mat_mut(3, 3);
        rho.copy_from(muu.submatrix(3, 3, 3, 3));
    });
}

#[inline]
fn calc_strain(
    strain: MatMut<f64>,
    x0_prime: MatRef<f64>,
    r: MatRef<f64>,
    u_prime: MatRef<f64>,
    r_prime: MatRef<f64>,
) {
    let mut r_x0_prime = Col::<f64>::zeros(3);
    let mut r_deriv = Mat::<f64>::zeros(3, 4);
    izip!(
        strain.col_iter_mut(),
        x0_prime.col_iter(),
        u_prime.col_iter(),
        r_prime.col_iter(),
        r.col_iter()
    )
    .for_each(|(mut qp_strain, x0_prime, u_prime, r_prime, r)| {
        quat_rotate_vector(r, x0_prime, r_x0_prime.as_mut());
        zipped!(
            &mut qp_strain.as_mut().subrows_mut(0, 3),
            &x0_prime,
            &u_prime,
            &r_x0_prime
        )
        .for_each(|unzipped!(mut strain, x0_prime, u_prime, r_x0_prime)| {
            *strain = *x0_prime + *u_prime - *r_x0_prime
        });

        quat_derivative(r, r_deriv.as_mut());
        matmul(
            qp_strain.subrows_mut(3, 3),
            r_deriv.as_ref(),
            r_prime,
            None,
            2.,
            Parallelism::None,
        );
    });
}

/// Calculate strain rate
fn calc_strain_dot(
    strain_dot: MatMut<f64>,
    v: MatRef<f64>,
    v_prime: MatRef<f64>,
    e1_tilde: MatRef<f64>,
) {
    let mut e1_tilde_omega = Col::<f64>::zeros(3);
    izip!(
        strain_dot.col_iter_mut(),
        v.col_iter(),
        v_prime.col_iter(),
        e1_tilde.col_iter()
    )
    .for_each(|(mut strain_dot, v, v_prime, e1_tilde_col)| {
        let e1_tilde = e1_tilde_col.as_mat_ref(3, 3);
        let omega = v.subrows(3, 3);
        let u_dot_prime = v_prime.subrows(0, 3);
        let omega_prime = v_prime.subrows(3, 3);
        matmul(
            e1_tilde_omega.as_mut(),
            e1_tilde,
            omega,
            None,
            1.,
            Parallelism::None,
        );
        zipped!(
            &mut strain_dot.as_mut().subrows_mut(0, 3),
            &u_dot_prime,
            &e1_tilde_omega
        )
        .for_each(|unzipped!(mut strain_dot, u_dot_prime, e1_tilde_omega)| {
            *strain_dot = *u_dot_prime + *e1_tilde_omega
        });
        strain_dot.subrows_mut(3, 3).copy_from(omega_prime);
    });
}

#[inline]
fn calc_e1_tilde(e1_tilde: MatMut<f64>, x0_prime: MatRef<f64>, u_prime: MatRef<f64>) {
    let mut x0pup = Col::<f64>::zeros(3);
    izip!(
        e1_tilde.col_iter_mut(),
        x0_prime.col_iter(),
        u_prime.col_iter()
    )
    .for_each(|(x0pupss_col, x0_prime, u_prime)| {
        zipped!(&mut x0pup, x0_prime, u_prime)
            .for_each(|unzipped!(mut x0pup, x0p, up)| *x0pup = *x0p + *up);
        vec_tilde(x0pup.as_ref(), x0pupss_col.as_mat_mut(3, 3));
    });
}

#[inline]
fn calc_fe_c(fc: MatMut<f64>, cuu: MatRef<f64>, strain: MatRef<f64>) {
    izip!(fc.col_iter_mut(), cuu.col_iter(), strain.col_iter()).for_each(
        |(fc, cuu_col, strain)| {
            matmul(
                fc,
                cuu_col.as_mat_ref(6, 6),
                strain,
                None,
                1.,
                Parallelism::None,
            );
        },
    );
}

#[inline]
fn calc_fe_d(fd: MatMut<f64>, fc: MatRef<f64>, e1_tilde: MatRef<f64>) {
    izip!(fd.col_iter_mut(), fc.col_iter(), e1_tilde.col_iter(),).for_each(
        |(fd, fc, e1_tilde_col)| {
            matmul(
                fd.subrows_mut(3, 3),
                e1_tilde_col.as_mat_ref(3, 3).transpose(),
                fc.subrows(0, 3), // N
                None,
                1.0,
                Parallelism::None,
            );
        },
    );
}

/// Calculate dissipative force Fd^C
fn calc_fd_c(
    mut fd_c: impl AsMatMut<f64>,
    mu_cuu: impl AsMatRef<f64>,
    strain_dot: impl AsMatRef<f64>,
) {
    izip!(
        fd_c.as_mat_mut().col_iter_mut(),
        mu_cuu.as_mat_ref().col_iter(),
        strain_dot.as_mat_ref().col_iter()
    )
    .for_each(|(fd_c, mu_cuu_col, strain_dot)| {
        matmul(
            fd_c,
            mu_cuu_col.as_mat_ref(6, 6),
            strain_dot,
            None,
            1.,
            Parallelism::None,
        );
    });
}

/// Calculate dissipative force Fd^D
fn calc_fd_d(fd_d: MatMut<f64>, fd_c: MatRef<f64>, e1_tilde: MatRef<f64>) {
    izip!(fd_d.col_iter_mut(), fd_c.col_iter(), e1_tilde.col_iter(),).for_each(
        |(fd_d, fd_c, e1_tilde_col)| {
            matmul(
                fd_d.subrows_mut(3, 3),
                e1_tilde_col.as_mat_ref(3, 3).transpose(),
                fd_c.subrows(0, 3), // Nd
                None,
                1.0,
                Parallelism::None,
            );
        },
    );
}

#[inline]
fn calc_fi(
    fi: MatMut<f64>,
    m: ColRef<f64>,
    omega: MatRef<f64>,
    u_ddot: MatRef<f64>,
    omega_dot: MatRef<f64>,
    eta: MatRef<f64>,
    rho: MatRef<f64>,
) {
    let mut mat = Mat::<f64>::zeros(3, 3);
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_dot_tilde = Mat::<f64>::zeros(3, 3);
    izip!(
        fi.col_iter_mut(),
        m.iter(),
        omega.col_iter(),
        u_ddot.col_iter(),
        omega_dot.col_iter(),
        eta.col_iter(),
        rho.col_iter(),
    )
    .for_each(|(mut fi, &m, omega, u_ddot, omega_dot, eta, rho_col)| {
        vec_tilde(eta, eta_tilde.as_mut());
        vec_tilde(omega, omega_tilde.as_mut());
        vec_tilde(omega_dot, omega_dot_tilde.as_mut());
        matmul(
            mat.as_mut(),
            omega_tilde.as_ref(),
            omega_tilde.as_ref(),
            None,
            m,
            Parallelism::None,
        );
        zipped!(mat.as_mut(), omega_dot_tilde.as_ref()).for_each(
            |unzipped!(mut mat, omega_dot_tilde)| {
                *mat += m * *omega_dot_tilde;
            },
        );
        let mut fi1 = fi.as_mut().subrows_mut(0, 3);
        matmul(fi1.as_mut(), mat.as_ref(), eta, None, 1., Parallelism::None);
        zipped!(&mut fi1, &u_ddot).for_each(|unzipped!(mut fi1, u_ddot)| *fi1 += *u_ddot * m);

        let mut fi2 = fi.as_mut().subrows_mut(3, 3);
        let rho = rho_col.as_mat_ref(3, 3);
        matmul(
            fi2.as_mut(),
            eta_tilde.as_ref(),
            u_ddot,
            None,
            m,
            Parallelism::None,
        );
        matmul(
            fi2.as_mut(),
            rho,
            omega_dot,
            Some(1.),
            1.,
            Parallelism::None,
        );
        matmul(
            mat.as_mut(),
            omega_tilde.as_ref(),
            rho,
            None,
            1.,
            Parallelism::None,
        );
        matmul(
            fi2.as_mut(),
            mat.as_ref(),
            omega,
            Some(1.),
            1.,
            Parallelism::None,
        );
    });
}

#[inline]
fn calc_fg(fg: MatMut<f64>, gravity: ColRef<f64>, m: ColRef<f64>, eta: MatRef<f64>) {
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    izip!(fg.col_iter_mut(), m.iter(), eta.col_iter(),).for_each(|(mut fg, &m, eta)| {
        vec_tilde(eta, eta_tilde.as_mut());
        zipped!(&mut fg.as_mut().subrows_mut(0, 3), &gravity)
            .for_each(|unzipped!(mut fg, g)| *fg = *g * m);
        matmul(
            fg.as_mut().subrows_mut(3, 3),
            eta_tilde.as_ref(),
            gravity.as_ref(),
            None,
            m,
            Parallelism::None,
        );
    });
}

#[inline]
fn calc_ouu(ouu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
    izip!(
        ouu.col_iter_mut(),
        cuu.col_iter(),
        x0pupss.col_iter(),
        fc.col_iter(),
    )
    .for_each(|(ouu_col, cuu_col, x0pupss_col, fc)| {
        let mut ouu = ouu_col.as_mat_mut(6, 6);
        let cuu = cuu_col.as_mat_ref(6, 6);
        let x0pupss = x0pupss_col.as_mat_ref(3, 3);

        let mut ouu12 = ouu.as_mut().submatrix_mut(0, 3, 3, 3);
        let c11 = cuu.submatrix(0, 0, 3, 3);
        vec_tilde(fc.subrows(0, 3), ouu12.as_mut()); // n_tilde
        matmul(
            ouu12.as_mut(),
            c11,
            x0pupss,
            Some(-1.),
            1.,
            Parallelism::None,
        );

        let mut ouu22 = ouu.as_mut().submatrix_mut(3, 3, 3, 3);
        let c21 = cuu.submatrix(3, 0, 3, 3);
        vec_tilde(fc.subrows(3, 3), ouu22.as_mut()); // m_tilde
        matmul(
            ouu22.as_mut(),
            c21,
            x0pupss,
            Some(-1.),
            1.,
            Parallelism::None,
        );
    });
}

#[inline]
fn calc_puu(puu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
    izip!(
        puu.col_iter_mut(),
        cuu.col_iter(),
        x0pupss.col_iter(),
        fc.col_iter(),
    )
    .for_each(|(mut puu_col, cuu_col, x0pupss_col, fc)| {
        puu_col.fill(0.);
        let mut puu = puu_col.as_mat_mut(6, 6);
        let cuu = cuu_col.as_mat_ref(6, 6);
        let x0pupss = x0pupss_col.as_mat_ref(3, 3);

        let c11 = cuu.submatrix(0, 0, 3, 3);
        let c12 = cuu.submatrix(0, 3, 3, 3);

        let mut puu21 = puu.as_mut().submatrix_mut(3, 0, 3, 3);
        vec_tilde(fc.subrows(0, 3), puu21.as_mut());
        matmul(
            puu21.as_mut(),
            x0pupss.transpose(),
            c11,
            Some(1.),
            1.,
            Parallelism::None,
        );

        let mut puu22 = puu.as_mut().submatrix_mut(3, 3, 3, 3);
        matmul(
            puu22.as_mut(),
            x0pupss.transpose(),
            c12,
            None,
            1.,
            Parallelism::None,
        );
    });
}

#[inline]
fn calc_quu(quu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
    let mut mat = Mat::<f64>::zeros(3, 3);
    izip!(
        quu.col_iter_mut(),
        cuu.col_iter(),
        x0pupss.col_iter(),
        fc.col_iter(),
    )
    .for_each(|(mut quu_col, cuu_col, x0pupss_col, fc)| {
        quu_col.fill(0.);
        let mut quu = quu_col.as_mat_mut(6, 6);
        let cuu = cuu_col.as_mat_ref(6, 6);
        let x0pupss = x0pupss_col.as_mat_ref(3, 3);
        vec_tilde(fc.subrows(0, 3), mat.as_mut()); // n_tilde

        let mut quu22 = quu.as_mut().submatrix_mut(3, 3, 3, 3);
        let c11 = cuu.submatrix(0, 0, 3, 3);
        matmul(mat.as_mut(), c11, x0pupss, Some(-1.), 1., Parallelism::None);
        matmul(
            quu22.as_mut(),
            x0pupss.transpose(),
            mat.as_ref(),
            None,
            1.,
            Parallelism::None,
        );
    });
}

#[inline]
fn calc_guu(
    guu: MatMut<f64>,
    m: ColRef<f64>,
    eta: MatRef<f64>,
    rho: MatRef<f64>,
    omega: MatRef<f64>,
) {
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut m_omega_tilde_eta = Col::<f64>::zeros(3);
    let mut m_omega_tilde_eta_tilde = Mat::<f64>::zeros(3, 3);
    let mut m_omega_tilde_eta_g_tilde = Mat::<f64>::zeros(3, 3);
    let mut rho_omega = Col::<f64>::zeros(3);
    let mut rho_omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde_rho = Mat::<f64>::zeros(3, 3);

    izip!(
        guu.col_iter_mut(),
        m.iter(),
        eta.col_iter(),
        rho.col_iter(),
        omega.col_iter(),
    )
    .for_each(|(guu_col, &m, eta, rho_col, omega)| {
        let mut guu = guu_col.as_mat_mut(6, 6);
        let rho = rho_col.as_mat_ref(3, 3);
        vec_tilde(eta, eta_tilde.as_mut());
        vec_tilde(omega, omega_tilde.as_mut());

        let mut guu12 = guu.as_mut().submatrix_mut(0, 3, 3, 3);
        matmul(
            m_omega_tilde_eta.as_mut(),
            omega_tilde.as_ref(),
            eta,
            None,
            m,
            Parallelism::None,
        );
        matmul(
            m_omega_tilde_eta_tilde.as_mut(),
            omega_tilde.as_ref(),
            eta_tilde.as_ref(),
            None,
            m,
            Parallelism::None,
        );
        vec_tilde(
            m_omega_tilde_eta.as_ref(),
            m_omega_tilde_eta_g_tilde.as_mut(),
        );
        zipped!(
            &mut guu12,
            &m_omega_tilde_eta_tilde,
            &m_omega_tilde_eta_g_tilde.transpose()
        )
        .for_each(|unzipped!(mut guu12, a, b)| *guu12 = *a + *b);

        let mut guu22 = guu.as_mut().submatrix_mut(3, 3, 3, 3);
        matmul(rho_omega.as_mut(), rho, omega, None, 1.0, Parallelism::None);
        vec_tilde(rho_omega.as_ref(), rho_omega_tilde.as_mut());
        matmul(
            omega_tilde_rho.as_mut(),
            omega_tilde.as_ref(),
            rho,
            None,
            1.0,
            Parallelism::None,
        );
        zipped!(&mut guu22, &omega_tilde_rho, &rho_omega_tilde)
            .for_each(|unzipped!(mut guu22, a, b)| *guu22 = *a - *b);
    });
}

fn calc_sd_pd_od_qd_gd_xd_yd(
    sd: MatMut<f64>,
    pd: MatMut<f64>,
    od: MatMut<f64>,
    qd: MatMut<f64>,
    gd: MatMut<f64>,
    xd: MatMut<f64>,
    yd: MatMut<f64>,
    mu_cuu: MatRef<f64>,
    u_dot_prime: MatRef<f64>,
    omega: MatRef<f64>,
    fd_c: MatRef<f64>,
    e1_tilde: MatRef<f64>,
) {
    let mut alpha = Mat::<f64>::zeros(3, 3);
    let mut b11 = Mat::<f64>::zeros(3, 3);
    let mut b12 = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut nd_tilde = Mat::<f64>::zeros(3, 3);
    let mut md_tilde = Mat::<f64>::zeros(3, 3);
    izip!(
        sd.col_iter_mut(),
        pd.col_iter_mut(),
        od.col_iter_mut(),
        qd.col_iter_mut(),
        gd.col_iter_mut(),
        xd.col_iter_mut(),
        yd.col_iter_mut(),
        mu_cuu.col_iter(),
        u_dot_prime.col_iter(),
        omega.col_iter(),
        fd_c.col_iter(),
        e1_tilde.col_iter(),
    )
    .for_each(
        |(
            sd_col,
            pd_col,
            od_col,
            qd_col,
            gd_col,
            xd_col,
            yd_col,
            mu_cuu_col,
            u_dot_prime,
            omega,
            fd_c,
            e1_tilde_col,
        )| {
            let e1_tilde = e1_tilde_col.as_mat_ref(3, 3);
            let mu_cuu = mu_cuu_col.as_mat_ref(6, 6);
            vec_tilde(fd_c.subrows(0, 3), nd_tilde.as_mut());
            vec_tilde(fd_c.subrows(3, 3), md_tilde.as_mut());
            matmul(
                b11.as_mut(),
                e1_tilde,
                mu_cuu.submatrix(0, 0, 3, 3), // d11
                None,
                -1.,
                Parallelism::None,
            );
            matmul(
                b12.as_mut(),
                e1_tilde,
                mu_cuu.submatrix(0, 3, 3, 3), // d12
                None,
                -1.,
                Parallelism::None,
            );

            // Components of mu*Cuu
            let d11 = mu_cuu.submatrix(0, 0, 3, 3);
            let d12 = mu_cuu.submatrix(0, 3, 3, 3);
            let d21 = mu_cuu.submatrix(3, 0, 3, 3);
            let d22 = mu_cuu.submatrix(3, 3, 3, 3);

            // Sd - stiffness matrix
            let mut sd = sd_col.as_mat_mut(6, 6);
            let sd11 = sd.as_mut().submatrix_mut(0, 0, 3, 3);
            matmul(sd11, d11, &omega_tilde, None, -1., Parallelism::None);
            let sd12 = sd.as_mut().submatrix_mut(0, 3, 3, 3);
            matmul(sd12, d12, &omega_tilde, None, -1., Parallelism::None);
            let sd21 = sd.as_mut().submatrix_mut(3, 0, 3, 3);
            matmul(sd21, d21, &omega_tilde, None, -1., Parallelism::None);
            let sd22 = sd.as_mut().submatrix_mut(3, 3, 3, 3);
            matmul(sd22, d22, &omega_tilde, None, -1., Parallelism::None);

            // Pd - stiffness matrix
            let mut pd = pd_col.as_mat_mut(6, 6);
            let mut pd21 = pd.as_mut().submatrix_mut(3, 0, 3, 3);
            vec_tilde(omega, omega_tilde.as_mut());
            pd21.copy_from(&nd_tilde);
            matmul(
                pd21.as_mut(),
                b11.as_ref(),
                omega_tilde.as_ref(),
                Some(1.),
                -1.,
                Parallelism::None,
            );
            let mut pd22 = pd.as_mut().submatrix_mut(3, 3, 3, 3);
            matmul(
                pd22.as_mut(),
                b12.as_ref(),
                omega_tilde.as_ref(),
                None,
                -1.,
                Parallelism::None,
            );

            // Od - stiffness matrix
            vec_tilde(u_dot_prime, alpha.as_mut());
            matmul(
                alpha.as_mut(),
                omega_tilde.as_ref(),
                e1_tilde,
                Some(1.),
                1.,
                Parallelism::None,
            );
            let mut od: MatMut<'_, f64> = od_col.as_mat_mut(6, 6);
            let mut od12 = od.as_mut().submatrix_mut(0, 3, 3, 3);
            od12.copy_from(&nd_tilde);
            matmul(
                od12.as_mut(),
                d11,
                alpha.as_ref(),
                Some(-1.),
                1.,
                Parallelism::None,
            );
            let mut od22 = od.as_mut().submatrix_mut(3, 3, 3, 3);
            od22.copy_from(&md_tilde);
            matmul(
                od22.as_mut(),
                d21,
                alpha.as_ref(),
                Some(-1.),
                1.,
                Parallelism::None,
            );

            // Qd - stiffness matrix
            let qd = qd_col.as_mat_mut(6, 6);
            let qd22 = qd.submatrix_mut(3, 3, 3, 3);
            let od12 = od.submatrix(0, 3, 3, 3);
            matmul(qd22, e1_tilde, od12, None, -1., Parallelism::None);

            // Gd - gyroscopic matrix
            let mut gd = gd_col.as_mat_mut(6, 6);
            let mut gd12 = gd.as_mut().submatrix_mut(0, 3, 3, 3);
            gd12.copy_from(b11.transpose());
            let mut gd22 = gd.as_mut().submatrix_mut(3, 3, 3, 3);
            gd22.copy_from(b12.transpose());

            // Xd - gyroscopic matrix
            let xd = xd_col.as_mat_mut(6, 6);
            let xd22 = xd.submatrix_mut(3, 3, 3, 3);
            let gd12 = gd.submatrix(0, 3, 3, 3);
            matmul(xd22, e1_tilde, gd12, None, -1., Parallelism::None);

            // Yd - gyroscopic matrix
            let mut yd = yd_col.as_mat_mut(6, 6);
            yd.as_mut().submatrix_mut(3, 0, 3, 3).copy_from(&b11);
            yd.as_mut().submatrix_mut(3, 3, 3, 3).copy_from(&b12);
        },
    );
}

#[inline]
fn calc_kuu(
    kuu: MatMut<f64>,
    m: ColRef<f64>,
    eta: MatRef<f64>,
    rho: MatRef<f64>,
    omega: MatRef<f64>,
    u_ddot: MatRef<f64>,
    omega_dot: MatRef<f64>,
) {
    let mut rho_omega = Col::<f64>::zeros(3);
    let mut rho_omega_dot = Col::<f64>::zeros(3);
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_dot_tilde = Mat::<f64>::zeros(3, 3);
    let mut u_ddot_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde_sq = Mat::<f64>::zeros(3, 3);
    let mut rho_omega_tilde = Mat::<f64>::zeros(3, 3);
    let mut rho_omega_g_tilde = Mat::<f64>::zeros(3, 3);
    let mut rho_omega_dot_tilde = Mat::<f64>::zeros(3, 3);
    let mut rho_omega_dot_g_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_dot_tilde_plus_omega_tilde_sq = Mat::<f64>::zeros(3, 3);
    let mut m_u_ddot_tilde_eta_tilde = Mat::<f64>::zeros(3, 3);
    let mut rho_omega_tilde_minus_rho_omega_g_tilde = Mat::<f64>::zeros(3, 3);
    let mut omega_tilde_rho_omega_tilde_minus_rho_omega_g_tilde = Mat::<f64>::zeros(3, 3);
    izip!(
        kuu.col_iter_mut(),
        m.iter(),
        eta.col_iter(),
        rho.col_iter(),
        omega.col_iter(),
        u_ddot.col_iter(),
        omega_dot.col_iter(),
    )
    .for_each(
        |(mut kuu_col, &m, eta, rho_col, omega, u_ddot, omega_dot)| {
            kuu_col.fill(0.);
            let mut kuu = kuu_col.as_mat_mut(6, 6);
            let rho = rho_col.as_mat_ref(3, 3);
            matmul(rho_omega.as_mut(), rho, omega, None, 1., Parallelism::None);
            matmul(
                rho_omega_dot.as_mut(),
                rho,
                omega_dot,
                None,
                1.,
                Parallelism::None,
            );
            vec_tilde(eta, eta_tilde.as_mut());
            vec_tilde(omega, omega_tilde.as_mut());
            vec_tilde(omega_dot, omega_dot_tilde.as_mut());
            vec_tilde(u_ddot, u_ddot_tilde.as_mut());
            vec_tilde(rho_omega.as_ref(), rho_omega_g_tilde.as_mut());
            vec_tilde(rho_omega_dot.as_ref(), rho_omega_dot_g_tilde.as_mut());

            let mut kuu12 = kuu.as_mut().submatrix_mut(0, 3, 3, 3);
            matmul(
                omega_tilde_sq.as_mut(),
                omega_tilde.as_ref(),
                omega_tilde.as_ref(),
                None,
                1.,
                Parallelism::None,
            );
            zipped!(
                &mut omega_dot_tilde_plus_omega_tilde_sq,
                &omega_dot_tilde,
                &omega_tilde_sq
            )
            .for_each(|unzipped!(mut c, a, b)| *c = *a + *b);
            matmul(
                kuu12.as_mut(),
                omega_dot_tilde_plus_omega_tilde_sq.as_ref(),
                eta_tilde.transpose(),
                None,
                m,
                Parallelism::None,
            );

            let mut kuu22 = kuu.as_mut().submatrix_mut(3, 3, 3, 3);
            matmul(
                m_u_ddot_tilde_eta_tilde.as_mut(),
                u_ddot_tilde.as_ref(),
                eta_tilde.as_ref(),
                None,
                m,
                Parallelism::None,
            );
            matmul(
                rho_omega_dot_tilde.as_mut(),
                rho,
                omega_dot_tilde.as_ref(),
                None,
                1.,
                Parallelism::None,
            );
            matmul(
                rho_omega_tilde.as_mut(),
                rho,
                omega_tilde.as_ref(),
                None,
                1.,
                Parallelism::None,
            );
            zipped!(
                &mut rho_omega_tilde_minus_rho_omega_g_tilde,
                &rho_omega_tilde,
                &rho_omega_g_tilde
            )
            .for_each(|unzipped!(mut c, a, b)| *c = *a - *b);
            matmul(
                omega_tilde_rho_omega_tilde_minus_rho_omega_g_tilde.as_mut(),
                omega_tilde.as_ref(),
                rho_omega_tilde_minus_rho_omega_g_tilde.as_ref(),
                None,
                1.,
                Parallelism::None,
            );
            zipped!(
                &mut kuu22,
                &m_u_ddot_tilde_eta_tilde,
                &rho_omega_dot_tilde,
                &rho_omega_dot_g_tilde,
                &omega_tilde_rho_omega_tilde_minus_rho_omega_g_tilde
            )
            .for_each(|unzipped!(mut k, a, b, c, d)| *k = *a + *b - *c + *d);
        },
    );
}

pub fn vec_tilde(v: ColRef<f64>, mut m: MatMut<f64>) {
    // [0., -v[2], v[1]]
    // [v[2], 0., -v[0]]
    // [-v[1], v[0], 0.]
    m[(0, 0)] = 0.;
    m[(1, 0)] = v[2];
    m[(2, 0)] = -v[1];
    m[(0, 1)] = -v[2];
    m[(1, 1)] = 0.;
    m[(2, 1)] = v[0];
    m[(0, 2)] = v[1];
    m[(1, 2)] = -v[0];
    m[(2, 2)] = 0.;
}

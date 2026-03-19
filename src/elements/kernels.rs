use crate::util::{
    quat_as_matrix, quat_compose, quat_derivative, quat_rotate_vector, quat_rotate_vector_alloc,
    vec_tilde, vec_tilde_alloc, ColMutReshape, ColRefReshape,
};
use faer::{linalg::matmul::matmul, prelude::*, Accum};
use itertools::izip;

#[inline]
/// Calculate current position and rotation (x0 + u)
pub fn calc_x(x: MatMut<f64>, x0: MatRef<f64>, u: MatRef<f64>) {
    izip!(x.col_iter_mut(), x0.col_iter(), u.col_iter()).for_each(|(mut x, x0, u)| {
        x[0] = x0[0] + u[0];
        x[1] = x0[1] + u[1];
        x[2] = x0[2] + u[2];
        quat_compose(u.subrows(3, 4), x0.subrows(3, 4), x.subrows_mut(3, 4));
    });
}

#[inline]
/// Calculate rotation matrix from material to inertial coordinates
pub fn calc_rr0(rr0: MatMut<f64>, x: MatRef<f64>) {
    let mut m = Mat::<f64>::zeros(3, 3);
    izip!(rr0.col_iter_mut(), x.subrows(3, 4).col_iter()).for_each(|(col, r)| {
        let mut rr0 = col.reshape_mut(6, 6);
        quat_as_matrix(r, m.as_mut());
        rr0.as_mut().submatrix_mut(0, 0, 3, 3).copy_from(&m);
        rr0.as_mut().submatrix_mut(3, 3, 3, 3).copy_from(&m);
    });
}

#[inline]
/// Rotate material into inertial coordinates
pub fn calc_global_matrix(mat: MatMut<f64>, mat_star: MatRef<f64>, rr0: MatRef<f64>) {
    let mut mat_tmp = Mat::<f64>::zeros(6, 6);
    izip!(mat.col_iter_mut(), mat_star.col_iter(), rr0.col_iter()).for_each(
        |(mat_col, mat_star_col, rr0_col)| {
            let mat = mat_col.reshape_mut(6, 6);
            let mat_star = mat_star_col.reshape(6, 6);
            let rr0 = rr0_col.reshape(6, 6);
            matmul(
                mat_tmp.as_mut(),
                Accum::Replace,
                rr0,
                mat_star,
                1.,
                Par::Seq,
            );
            matmul(
                mat,
                Accum::Replace,
                mat_tmp.as_ref(),
                rr0.transpose(),
                1.,
                Par::Seq,
            );
        },
    );
}

// rotate a column into the inertial frame
pub fn rotate_col_to_sectional(col_star: MatMut<f64>, col: MatRef<f64>, rr0: MatRef<f64>) {
    izip!(col_star.col_iter_mut(), col.col_iter(), rr0.col_iter()).for_each(
        |(col_star_col, col_col, rr0_col)| {
            let rr0 = rr0_col.reshape(6, 6);
            matmul(
                col_star_col,
                Accum::Replace,
                rr0.transpose(),
                col_col,
                1.,
                Par::Seq,
            );
        },
    );
}

#[inline]
pub fn calc_m_eta_rho(m: ColMut<f64>, eta: MatMut<f64>, rho: MatMut<f64>, muu: MatRef<f64>) {
    izip!(
        m.iter_mut(),
        eta.col_iter_mut(),
        rho.col_iter_mut(),
        muu.col_iter()
    )
    .for_each(|(m, mut eta, rho_col, muu_col)| {
        let muu = muu_col.reshape(6, 6);
        *m = muu[(0, 0)];
        if *m == 0. {
            eta.fill(0.);
        } else {
            eta[0] = muu[(5, 1)] / *m;
            eta[1] = -muu[(5, 0)] / *m;
            eta[2] = muu[(4, 0)] / *m;
        }
        let mut rho = rho_col.reshape_mut(3, 3);
        rho.copy_from(muu.submatrix(3, 3, 3, 3));
    });
}

#[inline]
/// Calculate inertial force on mass matrix
pub fn calc_fi(
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
            Accum::Replace,
            omega_tilde.as_ref(),
            omega_tilde.as_ref(),
            m,
            Par::Seq,
        );
        zip!(mat.as_mut(), omega_dot_tilde.as_ref()).for_each(|unzip!(mat, omega_dot_tilde)| {
            *mat += m * *omega_dot_tilde;
        });
        let mut fi1 = fi.as_mut().subrows_mut(0, 3);
        matmul(
            fi1.as_mut(),
            Accum::Replace,
            mat.as_ref(),
            eta,
            1.,
            Par::Seq,
        );
        zip!(&mut fi1, &u_ddot).for_each(|unzip!(fi1, u_ddot)| *fi1 += *u_ddot * m);

        let mut fi2 = fi.as_mut().subrows_mut(3, 3);
        let rho = rho_col.reshape(3, 3);
        matmul(
            fi2.as_mut(),
            Accum::Replace,
            eta_tilde.as_ref(),
            u_ddot,
            m,
            Par::Seq,
        );
        matmul(fi2.as_mut(), Accum::Add, rho, omega_dot, 1., Par::Seq);
        matmul(
            mat.as_mut(),
            Accum::Replace,
            omega_tilde.as_ref(),
            rho,
            1.,
            Par::Seq,
        );
        matmul(fi2.as_mut(), Accum::Add, mat.as_ref(), omega, 1., Par::Seq);
    });
}

#[inline]
/// Calculate gravitational force on mass matrix
pub fn calc_fg(fg: MatMut<f64>, gravity: ColRef<f64>, m: ColRef<f64>, eta: MatRef<f64>) {
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    izip!(fg.col_iter_mut(), m.iter(), eta.col_iter(),).for_each(|(mut fg, &m, eta)| {
        vec_tilde(eta, eta_tilde.as_mut());
        zip!(&mut fg.as_mut().subrows_mut(0, 3), &gravity).for_each(|unzip!(fg, g)| *fg = *g * m);
        matmul(
            fg.as_mut().subrows_mut(3, 3),
            Accum::Replace,
            eta_tilde.as_ref(),
            gravity.as_ref(),
            m,
            Par::Seq,
        );
    });
}

#[inline]
/// Calculate inertial damping matrix
pub fn calc_gi(
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
        let mut guu = guu_col.reshape_mut(6, 6);
        let rho = rho_col.reshape(3, 3);
        vec_tilde(eta, eta_tilde.as_mut());
        vec_tilde(omega, omega_tilde.as_mut());

        let mut guu12 = guu.as_mut().submatrix_mut(0, 3, 3, 3);
        matmul(
            m_omega_tilde_eta.as_mut(),
            Accum::Replace,
            omega_tilde.as_ref(),
            eta,
            m,
            Par::Seq,
        );
        matmul(
            m_omega_tilde_eta_tilde.as_mut(),
            Accum::Replace,
            omega_tilde.as_ref(),
            eta_tilde.as_ref(),
            m,
            Par::Seq,
        );
        vec_tilde(
            m_omega_tilde_eta.as_ref(),
            m_omega_tilde_eta_g_tilde.as_mut(),
        );
        zip!(
            &mut guu12,
            &m_omega_tilde_eta_tilde,
            &m_omega_tilde_eta_g_tilde.transpose()
        )
        .for_each(|unzip!(guu12, a, b)| *guu12 = *a + *b);

        let mut guu22 = guu.as_mut().submatrix_mut(3, 3, 3, 3);
        matmul(
            rho_omega.as_mut(),
            Accum::Replace,
            rho,
            omega,
            1.0,
            Par::Seq,
        );
        vec_tilde(rho_omega.as_ref(), rho_omega_tilde.as_mut());
        matmul(
            omega_tilde_rho.as_mut(),
            Accum::Replace,
            omega_tilde.as_ref(),
            rho,
            1.0,
            Par::Seq,
        );
        zip!(&mut guu22, &omega_tilde_rho, &rho_omega_tilde)
            .for_each(|unzip!(guu22, a, b)| *guu22 = *a - *b);
    });
}

#[inline]
pub fn calc_ki(
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
            let mut kuu = kuu_col.reshape_mut(6, 6);
            let rho = rho_col.reshape(3, 3);
            matmul(rho_omega.as_mut(), Accum::Replace, rho, omega, 1., Par::Seq);
            matmul(
                rho_omega_dot.as_mut(),
                Accum::Replace,
                rho,
                omega_dot,
                1.,
                Par::Seq,
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
                Accum::Replace,
                omega_tilde.as_ref(),
                omega_tilde.as_ref(),
                1.,
                Par::Seq,
            );
            zip!(
                &mut omega_dot_tilde_plus_omega_tilde_sq,
                &omega_dot_tilde,
                &omega_tilde_sq
            )
            .for_each(|unzip!(c, a, b)| *c = *a + *b);
            matmul(
                kuu12.as_mut(),
                Accum::Replace,
                omega_dot_tilde_plus_omega_tilde_sq.as_ref(),
                eta_tilde.transpose(),
                m,
                Par::Seq,
            );

            let mut kuu22 = kuu.as_mut().submatrix_mut(3, 3, 3, 3);
            matmul(
                m_u_ddot_tilde_eta_tilde.as_mut(),
                Accum::Replace,
                u_ddot_tilde.as_ref(),
                eta_tilde.as_ref(),
                m,
                Par::Seq,
            );
            matmul(
                rho_omega_dot_tilde.as_mut(),
                Accum::Replace,
                rho,
                omega_dot_tilde.as_ref(),
                1.,
                Par::Seq,
            );
            matmul(
                rho_omega_tilde.as_mut(),
                Accum::Replace,
                rho,
                omega_tilde.as_ref(),
                1.,
                Par::Seq,
            );
            zip!(
                &mut rho_omega_tilde_minus_rho_omega_g_tilde,
                &rho_omega_tilde,
                &rho_omega_g_tilde
            )
            .for_each(|unzip!(c, a, b)| *c = *a - *b);
            matmul(
                omega_tilde_rho_omega_tilde_minus_rho_omega_g_tilde.as_mut(),
                Accum::Replace,
                omega_tilde.as_ref(),
                rho_omega_tilde_minus_rho_omega_g_tilde.as_ref(),
                1.,
                Par::Seq,
            );
            zip!(
                &mut kuu22,
                &m_u_ddot_tilde_eta_tilde,
                &rho_omega_dot_tilde,
                &rho_omega_dot_g_tilde,
                &omega_tilde_rho_omega_tilde_minus_rho_omega_g_tilde
            )
            .for_each(|unzip!(k, a, b, c, d)| *k = *a + *b - *c + *d);
        },
    );
}

#[inline]
pub fn calc_mu_cuu(mu: ColRef<f64>, mu_cuu: MatMut<f64>, cuu: MatRef<f64>, rr0: MatRef<f64>) {
    let mut rr0_mat_rr0t = Mat::<f64>::zeros(6, 6);
    let mut tmp6 = Mat::<f64>::zeros(6, 6);

    // Create matrix from mu, same for all qps
    let mut mu_mat = Mat::<f64>::zeros(6, 6);
    mu_mat.diagonal_mut().column_vector_mut().copy_from(&mu);

    izip!(mu_cuu.col_iter_mut(), cuu.col_iter(), rr0.col_iter()).for_each(
        |(mu_cuu_col, cuu_col, rr0_col)| {
            let mut mu_cuu = mu_cuu_col.reshape_mut(6, 6);
            let cuu = cuu_col.reshape(6, 6);
            let rr0 = rr0_col.reshape(6, 6);

            // Rotate mu damping coefficients into inertial frame
            matmul(
                tmp6.as_mut(),
                Accum::Replace,
                rr0,
                mu_mat.as_ref(),
                1.,
                Par::Seq,
            );
            matmul(
                rr0_mat_rr0t.as_mut(),
                Accum::Replace,
                tmp6.as_ref(),
                rr0.transpose(),
                1.,
                Par::Seq,
            );

            // Multiply damping coefficients by stiffness matrix
            matmul(
                mu_cuu.as_mut(),
                Accum::Replace,
                rr0_mat_rr0t.as_ref(),
                cuu,
                1.0,
                Par::Seq,
            );
        },
    );
}

#[inline]
pub fn calc_strain(
    strain: MatMut<f64>,
    xr_prime: MatRef<f64>,
    r: MatRef<f64>,
    u_prime: MatRef<f64>,
    r_prime: MatRef<f64>,
) {
    let mut r_xr_prime = Col::<f64>::zeros(3);
    let mut r_deriv = Mat::<f64>::zeros(3, 4);
    izip!(
        strain.col_iter_mut(),
        xr_prime.col_iter(),
        u_prime.col_iter(),
        r_prime.col_iter(),
        r.col_iter()
    )
    .for_each(|(mut qp_strain, xr_prime, u_prime, r_prime, r)| {
        quat_rotate_vector(r, xr_prime, r_xr_prime.as_mut());
        zip!(
            &mut qp_strain.as_mut().subrows_mut(0, 3),
            &xr_prime,
            &u_prime,
            &r_xr_prime
        )
        .for_each(|unzip!(strain, xr_prime, u_prime, r_xr_prime)| {
            *strain = *xr_prime + *u_prime - *r_xr_prime
        });

        quat_derivative(r, r_deriv.as_mut());
        matmul(
            qp_strain.subrows_mut(3, 3),
            Accum::Replace,
            r_deriv.as_ref(),
            r_prime,
            2.,
            Par::Seq,
        );
    });
}

/// Calculate strain rate
pub fn calc_strain_dot(
    strain_dot: MatMut<f64>,
    strain: MatRef<f64>,
    u: MatRef<f64>,
    v: MatRef<f64>,
    v_prime: MatRef<f64>,
    xr_prime: MatRef<f64>,
) {
    let mut r_xr_prime = Col::<f64>::zeros(3);
    let mut omega_tilde = Mat::<f64>::zeros(3, 3);
    izip!(
        strain_dot.col_iter_mut(),
        strain.subrows(3, 3).col_iter(),  // kappa
        u.subrows(3, 4).col_iter(),       // R
        v.subrows(3, 3).col_iter(),       // omega
        v_prime.subrows(0, 3).col_iter(), // u'_dot
        v_prime.subrows(3, 3).col_iter(), // omega'
        xr_prime.col_iter()               // x'_ref
    )
    .for_each(
        |(mut strain_dot, kappa, r, omega, u_dot_prime, omega_prime, xr_prime)| {
            vec_tilde(omega, omega_tilde.as_mut());
            quat_rotate_vector(r, xr_prime, r_xr_prime.as_mut());

            // epsilon dot = u'_dot - tilde(omega) * R * xr'
            strain_dot
                .rb_mut()
                .subrows_mut(0, 3)
                .copy_from(u_dot_prime - &omega_tilde * &r_xr_prime);

            // kappa dot = tilde(omega) * kappa + omega'_dot
            strain_dot
                .subrows_mut(3, 3)
                .copy_from(&omega_tilde * kappa + omega_prime);
        },
    );
}

#[inline]
// tilde(x'^r) + tilde(u')
pub fn calc_e1_tilde(e1_tilde: MatMut<f64>, xr_prime: MatRef<f64>, u_prime: MatRef<f64>) {
    let mut xrp_up = Col::<f64>::zeros(3);
    izip!(
        e1_tilde.col_iter_mut(),
        xr_prime.col_iter(),
        u_prime.col_iter()
    )
    .for_each(|(e1_tilde_col, xr_prime, u_prime)| {
        zip!(&mut xrp_up, xr_prime, u_prime)
            .for_each(|unzip!(xrp_up, xrp, up)| *xrp_up = *xrp + *up);
        vec_tilde(xrp_up.as_ref(), e1_tilde_col.reshape_mut(3, 3));
    });
}

#[inline]
pub fn calc_f_e1(fc: MatMut<f64>, cuu: MatRef<f64>, strain: MatRef<f64>) {
    izip!(fc.col_iter_mut(), cuu.col_iter(), strain.col_iter()).for_each(
        |(fc, cuu_col, strain)| {
            matmul(
                fc,
                Accum::Replace,
                cuu_col.reshape(6, 6),
                strain,
                1.,
                Par::Seq,
            );
        },
    );
}

#[inline]
pub fn calc_f_e2(fd: MatMut<f64>, f_e1: MatRef<f64>, e1_tilde: MatRef<f64>) {
    izip!(fd.col_iter_mut(), f_e1.col_iter(), e1_tilde.col_iter(),).for_each(
        |(fd, f_e1, e1_tilde_col)| {
            matmul(
                fd.subrows_mut(3, 3),
                Accum::Replace,
                e1_tilde_col.reshape(3, 3).transpose(),
                f_e1.subrows(0, 3), // N
                1.0,
                Par::Seq,
            );
        },
    );
}

/// Calculate dissipative force Fd^C
pub fn calc_f_d1(f_d1: MatMut<f64>, d: MatRef<f64>, strain_dot: MatRef<f64>) {
    izip!(
        f_d1.col_iter_mut(),
        d.col_iter(), // d
        strain_dot.col_iter()
    )
    .for_each(|(f_d1, d_col, strain_dot)| {
        matmul(
            f_d1,
            Accum::Replace,
            d_col.reshape(6, 6),
            strain_dot,
            1.,
            Par::Seq,
        );
    });
}

/// Calculate dissipative force Fd^D
pub fn calc_f_d2(
    f_d2: MatMut<f64>,
    d: MatRef<f64>,
    xr_prime: MatRef<f64>,
    u_prime: MatRef<f64>,
    strain_dot: MatRef<f64>,
) {
    izip!(
        f_d2.col_iter_mut(),
        d.col_iter(),
        xr_prime.col_iter(),
        u_prime.col_iter(),
        strain_dot.subrows(0, 3).col_iter(), // epsilon dot
        strain_dot.subrows(3, 3).col_iter(), // kappa dot
    )
    .for_each(|(f_d2, d_col, xr_prime, u_prime, eps_dot, kappa_dot)| {
        let d = d_col.reshape(6, 6);
        let d_11 = d.submatrix(0, 0, 3, 3);
        let d_12 = d.submatrix(0, 3, 3, 3);

        let mut f_d2_2 = f_d2.subrows_mut(3, 3);
        f_d2_2.copy_from(
            vec_tilde_alloc((xr_prime + u_prime).as_ref()).transpose()
                * (d_11 * eps_dot + d_12 * kappa_dot),
        );
    });
}

/// Calculate viscoelastic forces into f_d1
pub fn calc_f_d1_viscoelastic(
    f_d1: MatMut<f64>,
    h: f64,
    kv_i: MatRef<f64>,
    tau_i: f64,
    rr0: MatRef<f64>,
    strain_dot_n: MatRef<f64>,
    strain_dot_n1: MatRef<f64>,
    visco_hist: MatRef<f64>,
) {
    // Viscoelastic history decay
    let tmp = -1. * h / tau_i;

    izip!(
        f_d1.col_iter_mut(),
        rr0.col_iter(),
        strain_dot_n.col_iter(),
        strain_dot_n1.col_iter(),
        visco_hist.col_iter(),
    )
    .for_each(|(f_d1, rr0_col, sd_n, sd_n1, visc_col)| {
        let visco_curr =
            Scale(tmp.exp()) * visc_col + Scale(h / 2. * tmp.exp()) * sd_n + Scale(h / 2.) * sd_n1;

        let mut fd_tmp = Col::<f64>::zeros(6);

        // force in sectional coordinates at quadrature
        matmul(
            fd_tmp.as_mut(),
            Accum::Replace,
            kv_i,
            visco_curr,
            1.,
            Par::Seq,
        );

        let rr0 = rr0_col.reshape(6, 6);

        // global force at quadrature point
        matmul(f_d1, Accum::Replace, rr0, fd_tmp.as_ref(), 1., Par::Seq);
    });
}

/// Calculate viscoelastic forces into f_d1
pub fn update_viscoelastic(
    visco_hist: MatMut<f64>,
    strain_dot_n: MatRef<f64>,
    strain_dot_n1: MatRef<f64>,
    h: f64,
    tau_i: f64,
) {
    // Viscoelastic history decay
    let tmp = -1. * h / tau_i;

    izip!(
        visco_hist.col_iter_mut(),
        strain_dot_n.col_iter(),
        strain_dot_n1.col_iter(),
    )
    .for_each(|(mut hist, sd_n, sd_n1)| {
        // Decay previous history
        hist *= Scale(tmp.exp());

        // Add history from this time step
        hist += Scale(h / 2. * tmp.exp()) * sd_n + Scale(h / 2.) * sd_n1;
    });
}

#[inline]
pub fn calc_k_e1(k_e1: MatMut<f64>, c: MatRef<f64>, e1_tilde: MatRef<f64>, f_e1: MatRef<f64>) {
    izip!(
        k_e1.col_iter_mut(),
        c.col_iter(),
        e1_tilde.col_iter(),
        f_e1.col_iter(),
    )
    .for_each(|(k_e1_col, c_col, e1_tilde_col, f_e1)| {
        let mut k_e1 = k_e1_col.reshape_mut(6, 6);
        let c = c_col.reshape(6, 6);
        let e1_tilde = e1_tilde_col.reshape(3, 3);

        let mut ouu12 = k_e1.as_mut().submatrix_mut(0, 3, 3, 3);
        let c11 = c.submatrix(0, 0, 3, 3);
        vec_tilde(f_e1.subrows(0, 3), ouu12.as_mut()); // n_tilde
        ouu12 *= -1.;
        matmul(ouu12.as_mut(), Accum::Add, c11, e1_tilde, 1., Par::Seq);

        let mut ouu22 = k_e1.as_mut().submatrix_mut(3, 3, 3, 3);
        let c21 = c.submatrix(3, 0, 3, 3);
        vec_tilde(f_e1.subrows(3, 3), ouu22.as_mut()); // m_tilde
        ouu22 *= -1.;
        matmul(ouu22.as_mut(), Accum::Add, c21, e1_tilde, 1., Par::Seq);
    });
}

#[inline]
pub fn calc_p_e2(puu: MatMut<f64>, c: MatRef<f64>, e1_tilde: MatRef<f64>, fc: MatRef<f64>) {
    izip!(
        puu.col_iter_mut(),
        c.col_iter(),
        e1_tilde.col_iter(),
        fc.col_iter(),
    )
    .for_each(|(mut p_e2_col, c_col, e1_tilde_col, fc)| {
        p_e2_col.fill(0.);
        let mut p_e2 = p_e2_col.reshape_mut(6, 6);
        let c = c_col.reshape(6, 6);
        let e1_tilde = e1_tilde_col.reshape(3, 3);

        let c11 = c.submatrix(0, 0, 3, 3);
        let c12 = c.submatrix(0, 3, 3, 3);

        let mut puu21 = p_e2.as_mut().submatrix_mut(3, 0, 3, 3);
        vec_tilde(fc.subrows(0, 3), puu21.as_mut());
        matmul(
            puu21.as_mut(),
            Accum::Add,
            e1_tilde.transpose(),
            c11,
            1.,
            Par::Seq,
        );

        let mut puu22 = p_e2.as_mut().submatrix_mut(3, 3, 3, 3);
        matmul(
            puu22.as_mut(),
            Accum::Replace,
            e1_tilde.transpose(),
            c12,
            1.,
            Par::Seq,
        );
    });
}

#[inline]
pub fn calc_k_e2(quu: MatMut<f64>, cuu: MatRef<f64>, e1_tilde: MatRef<f64>, f_e1: MatRef<f64>) {
    let mut mat = Mat::<f64>::zeros(3, 3);
    izip!(
        quu.col_iter_mut(),
        cuu.col_iter(),
        e1_tilde.col_iter(),
        f_e1.col_iter(),
    )
    .for_each(|(mut k_e2_col, c_col, e1_tilde_col, f_e1)| {
        k_e2_col.fill(0.);
        let mut k_e2 = k_e2_col.reshape_mut(6, 6);
        let c = c_col.reshape(6, 6);
        let e1_tilde = e1_tilde_col.reshape(3, 3);
        vec_tilde(f_e1.subrows(0, 3), mat.as_mut()); // n_tilde

        let mut k_e2_22 = k_e2.as_mut().submatrix_mut(3, 3, 3, 3);
        let c11 = c.submatrix(0, 0, 3, 3);
        mat *= -1.;
        matmul(mat.as_mut(), Accum::Add, c11, e1_tilde, 1., Par::Seq);
        matmul(
            k_e2_22.as_mut(),
            Accum::Replace,
            e1_tilde.transpose(),
            mat.as_ref(),
            1.,
            Par::Seq,
        );
    });
}

pub fn calc_damping_matrices(
    d_d1: MatMut<f64>,
    d_d2: MatMut<f64>,
    g_d1: MatMut<f64>,
    g_d2: MatMut<f64>,
    p_d2: MatMut<f64>,
    k_d1: MatMut<f64>,
    k_d2: MatMut<f64>,
    u: MatRef<f64>,
    v: MatRef<f64>,
    d: MatRef<f64>,
    strain: MatRef<f64>,
    strain_dot: MatRef<f64>,
    xr_prime: MatRef<f64>,
    u_prime: MatRef<f64>,
) {
    izip!(
        d_d1.col_iter_mut(),
        d_d2.col_iter_mut(),
        g_d1.col_iter_mut(),
        g_d2.col_iter_mut(),
        p_d2.col_iter_mut(),
        k_d1.col_iter_mut(),
        k_d2.col_iter_mut(),
        u.subrows(3, 4).col_iter(), // rotation displacement (r)
        v.subrows(3, 3).col_iter(), // rotational velocity (omega)
        d.col_iter(),
        xr_prime.subrows(0, 3).col_iter(),
        u_prime.subrows(0, 3).col_iter(),
        strain.subrows(3, 3).col_iter(),     // kappa
        strain_dot.subrows(0, 3).col_iter(), // epsilon dot
        strain_dot.subrows(3, 3).col_iter(), // kappa dot
    )
    .for_each(
        |(
            d_d1_col,
            d_d2_col,
            g_d1_col,
            g_d2_col,
            p_d2_col,
            k_d1_col,
            k_d2_col,
            r,
            omega,
            d_col,
            xr_prime,
            u_prime,
            kappa,
            eps_dot,
            kappa_dot,
        )| {
            let tilde_omega = vec_tilde_alloc(omega);
            let tilde_kappa = vec_tilde_alloc(kappa);
            let tilde_eps_dot = vec_tilde_alloc(eps_dot);
            let tilde_kappa_dot = vec_tilde_alloc(kappa_dot);
            let tilde_xp_up = vec_tilde_alloc((&(xr_prime + u_prime)).as_ref());
            let transpose_tilde_xp_up = tilde_xp_up.transpose();

            let r_xr_prime = quat_rotate_vector_alloc(r, xr_prime);
            let tilde_r_xr_prime = vec_tilde_alloc(r_xr_prime.as_ref());

            // Components of D
            let d = d_col.reshape(6, 6);
            let d11 = d.submatrix(0, 0, 3, 3);
            let d12 = d.submatrix(0, 3, 3, 3);
            let d21 = d.submatrix(3, 0, 3, 3);
            let d22 = d.submatrix(3, 3, 3, 3);

            // D^D1
            let mut d_d1 = d_d1_col.reshape_mut(6, 6);
            let mut d_d1_12 = d_d1.as_mut().submatrix_mut(0, 3, 3, 3);
            d_d1_12.copy_from(d12 * &tilde_omega);
            let mut d_d1_22 = d_d1.as_mut().submatrix_mut(3, 3, 3, 3);
            d_d1_22.copy_from(d22 * &tilde_omega);

            // D^D2
            let mut d_d2 = d_d2_col.reshape_mut(6, 6);
            let mut d_d2_21 = d_d2.as_mut().submatrix_mut(3, 0, 3, 3);
            d_d2_21.copy_from(transpose_tilde_xp_up * d11);
            let mut d_d2_22 = d_d2.as_mut().submatrix_mut(3, 3, 3, 3);
            d_d2_22.copy_from(transpose_tilde_xp_up * d12);

            // G^D1
            let mut g_d1 = g_d1_col.reshape_mut(6, 6);
            let mut g_d1_12 = g_d1.as_mut().submatrix_mut(0, 3, 3, 3);
            g_d1_12.copy_from(d11 * &tilde_r_xr_prime - d12 * &tilde_kappa);
            let mut g_d1_22 = g_d1.as_mut().submatrix_mut(3, 3, 3, 3);
            g_d1_22.copy_from(d21 * &tilde_r_xr_prime - d22 * &tilde_kappa);

            // G^D2
            let mut g_d2 = g_d2_col.reshape_mut(6, 6);
            let mut g_d2_22 = g_d2.as_mut().submatrix_mut(3, 3, 3, 3);
            g_d2_22
                .copy_from(transpose_tilde_xp_up * (d11 * &tilde_r_xr_prime - d12 * &tilde_kappa));

            // Auxiliary calculations for P^D2, K^D1, K^D2
            let tilde_d11_eps_dot = vec_tilde_alloc((&d11 * &eps_dot).as_ref());
            let d11_tilde_eps_dot = d11 * &tilde_eps_dot;
            let tilde_d12_kappa_dot = vec_tilde_alloc((&d12 * &kappa_dot).as_ref());
            let d12_tilde_kappa_dot = d12 * &tilde_kappa_dot;
            let tilde_d21_eps_dot = vec_tilde_alloc((&d21 * &eps_dot).as_ref());
            let d21_tilde_eps_dot = d21 * &tilde_eps_dot;
            let tilde_d22_kappa_dot = vec_tilde_alloc((&d22 * &kappa_dot).as_ref());
            let d22_tilde_kappa_dot = d22 * &tilde_kappa_dot;

            // P^D2
            let mut p_d2 = p_d2_col.reshape_mut(6, 6);
            let mut p_d2_21 = p_d2.as_mut().submatrix_mut(3, 0, 3, 3);
            p_d2_21.copy_from(-&tilde_d11_eps_dot - &tilde_d12_kappa_dot);
            let mut p_d2_22 = p_d2.as_mut().submatrix_mut(3, 3, 3, 3);
            p_d2_22.copy_from(transpose_tilde_xp_up * d12 * &tilde_omega);

            // K^D1
            let mut k_d1 = k_d1_col.reshape_mut(6, 6);
            let mut k_d1_12 = k_d1.as_mut().submatrix_mut(0, 3, 3, 3);
            k_d1_12.copy_from(
                -&tilde_d11_eps_dot + d11_tilde_eps_dot - &tilde_d12_kappa_dot
                    + d12_tilde_kappa_dot
                    + d11 * &tilde_omega * &tilde_r_xr_prime
                    - d12 * &tilde_omega * &tilde_kappa,
            );
            let mut k_d1_22 = k_d1.as_mut().submatrix_mut(3, 3, 3, 3);
            k_d1_22.copy_from(
                -&tilde_d21_eps_dot + d21_tilde_eps_dot - &tilde_d22_kappa_dot
                    + d22_tilde_kappa_dot
                    + d21 * &tilde_omega * &tilde_r_xr_prime
                    - d22 * &tilde_omega * &tilde_kappa,
            );

            // K^D2
            let mut k_d2 = k_d2_col.reshape_mut(6, 6);
            let mut k_d2_22 = k_d2.as_mut().submatrix_mut(3, 3, 3, 3);
            k_d2_22.copy_from(&k_d1.as_ref().submatrix(0, 3, 3, 3));
        },
    );
}

#[cfg(test)]
mod tests {

    use crate::util::ColRefReshape;

    use super::*;

    #[test]
    fn test_calc_f_d2() {
        let mut f_d2 = Mat::<f64>::zeros(6, 1);

        // 1., 2., 3., 4., 5., 6.,
        // 7., 8., 9., 10., 11., 12.,
        // 13., 14., 15., 16., 17., 18.,
        // 19., 20., 21., 22., 23., 24.,
        // 25., 26., 27., 28., 29., 30.,
        // 31., 32., 33., 34., 35., 36.,

        let d = faer::MatRef::from_column_major_slice(
            &[
                1., 7., 13., 19., 25., 31., 2., 8., 14., 20., 26., 32., 3., 9., 15., 21., 27., 33.,
                4., 10., 16., 22., 28., 34., 5., 11., 17., 23., 29., 35., 6., 12., 18., 24., 30.,
                36.,
            ],
            36,
            1,
        );

        let xr_prime = faer::MatRef::from_column_major_slice(&[37., 38., 39.], 3, 1);
        let u_prime = faer::MatRef::from_column_major_slice(&[40., 41., 42.], 3, 1);
        let strain_dot =
            faer::MatRef::from_column_major_slice(&[43., 44., 45., 46., 47., 48.], 6, 1);

        calc_f_d2(f_d2.as_mut(), d, xr_prime, u_prime, strain_dot);
    }

    #[test]
    fn test_calc_damping_matrices() {
        let mut d_d1 = Mat::<f64>::zeros(36, 1);
        let mut d_d2 = Mat::<f64>::zeros(36, 1);
        let mut g_d1 = Mat::<f64>::zeros(36, 1);
        let mut g_d2 = Mat::<f64>::zeros(36, 1);
        let mut p_d2 = Mat::<f64>::zeros(36, 1);
        let mut k_d1 = Mat::<f64>::zeros(36, 1);
        let mut k_d2 = Mat::<f64>::zeros(36, 1);

        // 1., 2., 3., 4., 5., 6.,
        // 7., 8., 9., 10., 11., 12.,
        // 13., 14., 15., 16., 17., 18.,
        // 19., 20., 21., 22., 23., 24.,
        // 25., 26., 27., 28., 29., 30.,
        // 31., 32., 33., 34., 35., 36.,
        let d = faer::MatRef::from_column_major_slice(
            &[
                1., 7., 13., 19., 25., 31., 2., 8., 14., 20., 26., 32., 3., 9., 15., 21., 27., 33.,
                4., 10., 16., 22., 28., 34., 5., 11., 17., 23., 29., 35., 6., 12., 18., 24., 30.,
                36.,
            ],
            36,
            1,
        );

        let u = faer::MatRef::from_column_major_slice(&[37., 38., 39., 40., 41., 42., 43.], 7, 1);
        let v = faer::MatRef::from_column_major_slice(&[44., 45., 46., 47., 48., 49.], 6, 1);
        let strain = faer::MatRef::from_column_major_slice(&[50., 51., 52., 53., 54., 55.], 6, 1);
        let strain_dot =
            faer::MatRef::from_column_major_slice(&[56., 57., 58., 59., 60., 61.], 6, 1);
        let xr_prime = faer::MatRef::from_column_major_slice(&[62., 63., 64.], 3, 1);
        let u_prime = faer::MatRef::from_column_major_slice(&[65., 66., 67.], 3, 1);

        calc_damping_matrices(
            d_d1.as_mut(),
            d_d2.as_mut(),
            g_d1.as_mut(),
            g_d2.as_mut(),
            p_d2.as_mut(),
            k_d1.as_mut(),
            k_d2.as_mut(),
            u,
            v,
            d,
            strain,
            strain_dot,
            xr_prime,
            u_prime,
        );

        print!("\nd_d1\n{:?}", d_d1.col(0).reshape(6, 6));
        print!("\nd_d2\n{:?}", d_d2.col(0).reshape(6, 6));
        print!("\ng_d1\n{:?}", g_d1.col(0).reshape(6, 6));
        print!("\ng_d2\n{:?}", g_d2.col(0).reshape(6, 6));
        print!("\np_d2\n{:?}", p_d2.col(0).reshape(6, 6));
        print!("\nk_d1\n{:?}", k_d1.col(0).reshape(6, 6));
        print!("\nk_d2\n{:?}", k_d2.col(0).reshape(6, 6));
    }
}

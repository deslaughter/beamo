use faer::{Col, ColRef, Mat};

use super::kernels::{
    calc_e1_tilde, calc_fe_c, calc_fe_d, calc_fg, calc_fi, calc_gi, calc_inertial_matrix, calc_ki,
    calc_m_eta_rho, calc_ouu, calc_puu, calc_quu, calc_rr0, calc_strain, calc_strain_dot, calc_x,
};

/// Beam quadrature point data
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

    pub fn calc(&mut self, gravity: ColRef<f64>) {
        calc_x(self.x.as_mut(), self.x0.as_ref(), self.u.as_ref());
        calc_rr0(self.rr0.as_mut(), self.x.as_ref());
        calc_inertial_matrix(self.muu.as_mut(), self.m_star.as_ref(), self.rr0.as_ref());
        calc_inertial_matrix(self.cuu.as_mut(), self.c_star.as_ref(), self.rr0.as_ref());
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
            self.strain.as_ref(),
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
        calc_gi(
            self.gi.as_mut(),
            self.m.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
        );
        calc_ki(
            self.ki.as_mut(),
            self.m.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
            self.vd.subrows(0, 3),
            self.vd.subrows(3, 3),
        );
    }
}

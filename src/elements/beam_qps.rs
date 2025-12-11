use faer::{Col, ColRef, Mat};

use super::kernels::{
    calc_damping_matrices, calc_e1_tilde, calc_f_d1, calc_f_d2, calc_f_e1, calc_f_e2, calc_fg,
    calc_fi, calc_gi, calc_global_matrix, calc_k_e1, calc_k_e2, calc_ki, calc_m_eta_rho, calc_p_e2,
    calc_rr0, calc_strain, calc_strain_dot, calc_x,
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
    /// Damping matrix in material frame `[6][6][n_qps]`
    pub d_star: Mat<f64>,
    /// Current position/orientation `[7][n_qps]`
    pub x: Mat<f64>,
    /// Initial position `[7][n_qps]`
    pub x0: Mat<f64>,
    /// Initial position derivative `[7][n_qps]`
    pub xr_prime: Mat<f64>,
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
    /// tilde(xr_prime + u_prime) `[3][3][n_qps]`
    pub e1_tilde: Mat<f64>,
    /// mass `[n_qps]`
    pub mass: Col<f64>,
    /// mass `[3][n_qps]`
    pub eta: Mat<f64>,
    /// mass `[3][3][n_qps]`
    pub rho: Mat<f64>,
    /// Strain `[6][n_qps]`
    pub strain: Mat<f64>,
    /// Strain Rate `[6][n_qps]`
    pub strain_dot: Mat<f64>,
    /// Elastic force C `[6][n_qps]`
    pub f_e1: Mat<f64>,
    /// Elastic force D `[6][n_qps]`
    pub f_e2: Mat<f64>,
    /// Dissipative force C `[6][n_qps]`
    pub f_d1: Mat<f64>,
    /// Dissipative force D `[6][n_qps]`
    pub f_d2: Mat<f64>,
    /// Inertial force `[6][n_qps]`
    pub fi: Mat<f64>,
    /// External force `[6][n_qps]`
    pub fx: Mat<f64>,
    /// Gravity force `[6][n_qps]`
    pub fg: Mat<f64>,
    /// Global rotation `[6][6][n_qps]`
    pub rr0: Mat<f64>,
    /// Inertial mass matrices `[6][6][n_qps]`
    pub m: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub c: Mat<f64>,
    /// Damped stiffness in global frame `[6][6][n_qps]`
    pub d: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub k_e1: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub p_e2: Mat<f64>,
    /// Elastic stiffness matrices `[6][6][n_qps]`
    pub k_e2: Mat<f64>,
    /// Inertial gyroscopic matrices `[6][6][n_qps]`
    pub gi: Mat<f64>,
    /// Inertial stiffness matrices `[6][6][n_qps]`
    pub ki: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub d_d2: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub g_d2: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub k_d1: Mat<f64>,
    /// Dissipative elastic matrices `[6][6][n_qps]`
    pub k_d2: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub d_d1: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub g_d1: Mat<f64>,
    /// Dissipative inertial matrices `[6][6][n_qps]`
    pub p_d2: Mat<f64>,
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
            d_star: Mat::zeros(6 * 6, n_qps),
            x: Mat::zeros(7, n_qps),
            x0: Mat::zeros(7, n_qps),
            xr_prime: Mat::zeros(7, n_qps),
            u: Mat::zeros(7, n_qps),
            u_prime: Mat::zeros(7, n_qps),
            v: Mat::zeros(6, n_qps),
            v_prime: Mat::zeros(6, n_qps),
            vd: Mat::zeros(6, n_qps),
            e1_tilde: Mat::zeros(3 * 3, n_qps),
            mass: Col::zeros(n_qps),
            eta: Mat::zeros(3, n_qps),
            rho: Mat::zeros(3 * 3, n_qps),
            strain: Mat::zeros(6, n_qps),
            strain_dot: Mat::zeros(6, n_qps),
            f_e1: Mat::zeros(6, n_qps),
            f_e2: Mat::zeros(6, n_qps),
            f_d1: Mat::zeros(6, n_qps),
            f_d2: Mat::zeros(6, n_qps),
            fi: Mat::zeros(6, n_qps),
            fx: Mat::zeros(6, n_qps),
            fg: Mat::zeros(6, n_qps),
            rr0: Mat::zeros(6 * 6, n_qps),
            m: Mat::zeros(6 * 6, n_qps),
            c: Mat::zeros(6 * 6, n_qps),
            d: Mat::zeros(6 * 6, n_qps),
            k_e1: Mat::zeros(6 * 6, n_qps),
            p_e2: Mat::zeros(6 * 6, n_qps),
            k_e2: Mat::zeros(6 * 6, n_qps),
            gi: Mat::zeros(6 * 6, n_qps),
            ki: Mat::zeros(6 * 6, n_qps),
            d_d2: Mat::zeros(6 * 6, n_qps),
            g_d2: Mat::zeros(6 * 6, n_qps),
            k_d1: Mat::zeros(6 * 6, n_qps),
            k_d2: Mat::zeros(6 * 6, n_qps),
            d_d1: Mat::zeros(6 * 6, n_qps),
            g_d1: Mat::zeros(6 * 6, n_qps),
            p_d2: Mat::zeros(6 * 6, n_qps),
            ed: Mat::zeros(6 * 6, n_qps),
        }
    }

    pub fn calc(&mut self, gravity: ColRef<f64>) {
        // Calculate global position/orientation (reference + displacement)
        calc_x(self.x.as_mut(), self.x0.as_ref(), self.u.as_ref());

        // Calculate matrix for rotating from material to global frame
        calc_rr0(self.rr0.as_mut(), self.x.as_ref());

        // Calculate mass and stiffness matrices in global frame
        calc_global_matrix(self.m.as_mut(), self.m_star.as_ref(), self.rr0.as_ref());
        calc_global_matrix(self.c.as_mut(), self.c_star.as_ref(), self.rr0.as_ref());

        // Extract mass, eta, rho from mass matrix
        calc_m_eta_rho(
            self.mass.as_mut(),
            self.eta.as_mut(),
            self.rho.as_mut(),
            self.m.as_ref(),
        );

        // Calculate the strain
        calc_strain(
            self.strain.as_mut(),
            self.xr_prime.subrows(0, 3),
            self.u.subrows(3, 4),
            self.u_prime.subrows(0, 3),
            self.u_prime.subrows(3, 4),
        );

        // Calculate the strain rate
        calc_strain_dot(
            self.strain_dot.as_mut(),
            self.strain.as_ref(),
            self.u.as_ref(),
            self.v.as_ref(),
            self.v_prime.as_ref(),
            self.xr_prime.as_ref(),
        );

        calc_e1_tilde(
            self.e1_tilde.as_mut(),
            self.xr_prime.subrows(0, 3),
            self.u_prime.subrows(0, 3),
        );

        // Calculate the elastic forces
        calc_f_e1(self.f_e1.as_mut(), self.c.as_ref(), self.strain.as_ref());
        calc_f_e2(
            self.f_e2.as_mut(),
            self.f_e1.as_ref(),
            self.e1_tilde.as_ref(),
        );

        // Calculate the inertial force
        calc_fi(
            self.fi.as_mut(),
            self.mass.as_ref(),
            self.v.subrows(3, 3).as_ref(),
            self.vd.subrows(0, 3).as_ref(),
            self.vd.subrows(3, 3).as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
        );

        // Calculate the gravity force
        calc_fg(
            self.fg.as_mut(),
            gravity.as_ref(),
            self.mass.as_ref(),
            self.eta.as_ref(),
        );

        calc_k_e1(
            self.k_e1.as_mut(),
            self.c.as_ref(),
            self.e1_tilde.as_ref(),
            self.f_e1.as_ref(),
        );
        calc_p_e2(
            self.p_e2.as_mut(),
            self.c.as_ref(),
            self.e1_tilde.as_ref(),
            self.f_e1.as_ref(),
        );
        calc_k_e2(
            self.k_e2.as_mut(),
            self.c.as_ref(),
            self.e1_tilde.as_ref(),
            self.f_e1.as_ref(),
        );

        // Calculate the gyroscopic matrix
        calc_gi(
            self.gi.as_mut(),
            self.mass.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
        );

        // Calculate the inertial stiffness matrix
        calc_ki(
            self.ki.as_mut(),
            self.mass.as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
            self.v.subrows(3, 3),
            self.vd.subrows(0, 3),
            self.vd.subrows(3, 3),
        );
    }

    pub fn calculate_mu_damping(&mut self, i_qp_start: usize, n_qps: usize) {
        // Calculate the damping matrix in global frame
        calc_global_matrix(
            self.d.subcols_mut(i_qp_start, n_qps),
            self.d_star.subcols(i_qp_start, n_qps),
            self.rr0.subcols(i_qp_start, n_qps),
        );

        // Calculate dissipative forces
        calc_f_d1(
            self.f_d1.subcols_mut(i_qp_start, n_qps),
            self.d.subcols(i_qp_start, n_qps),
            self.strain_dot.subcols(i_qp_start, n_qps),
        );
        calc_f_d2(
            self.f_d2.subcols_mut(i_qp_start, n_qps),
            self.d.subcols(i_qp_start, n_qps),
            self.xr_prime.subcols(i_qp_start, n_qps),
            self.u_prime.subcols(i_qp_start, n_qps),
            self.strain_dot.subcols(i_qp_start, n_qps),
        );

        // Calculate damping matrices
        calc_damping_matrices(
            self.d_d1.subcols_mut(i_qp_start, n_qps),
            self.d_d2.subcols_mut(i_qp_start, n_qps),
            self.g_d1.subcols_mut(i_qp_start, n_qps),
            self.g_d2.subcols_mut(i_qp_start, n_qps),
            self.p_d2.subcols_mut(i_qp_start, n_qps),
            self.k_d1.subcols_mut(i_qp_start, n_qps),
            self.k_d2.subcols_mut(i_qp_start, n_qps),
            self.u.subcols(i_qp_start, n_qps),
            self.v.subcols(i_qp_start, n_qps),
            self.d.subcols(i_qp_start, n_qps),
            self.strain.subcols(i_qp_start, n_qps),
            self.strain_dot.subcols(i_qp_start, n_qps),
            self.xr_prime.subcols(i_qp_start, n_qps),
            self.u_prime.subcols(i_qp_start, n_qps),
        );
    }
}

use crate::node::Node;
use crate::node::NodeFreedomMap;
use crate::state::State;
use crate::util::ColRefReshape;
use faer::prelude::*;
use itertools::{izip, Itertools};
use std::ops::Rem;

use super::kernels::{
    calc_fg, calc_fi, calc_gi, calc_inertial_matrix, calc_ki, calc_m_eta_rho, calc_rr0, calc_x,
};

pub struct MassElement {
    pub id: usize,
    pub node_id: usize,
    pub m: Mat<f64>,
}

pub struct Masses {
    n_elem: usize,
    /// Node ID for each element
    node_ids: Vec<usize>,
    /// Gravity vector
    gravity: Col<f64>,
    /// Initial position/rotation `[7][n_nodes]`
    pub x0: Mat<f64>,
    /// State: translation/rotation displacement `[7][n_nodes]`
    pub u: Mat<f64>,
    /// State: translation/rotation velocity `[6][n_nodes]`
    pub v: Mat<f64>,
    /// State: translation/rotation acceleration `[6][n_nodes]`
    pub vd: Mat<f64>,
    /// Current position/orientation `[7][n_nodes]`
    pub x: Mat<f64>,
    /// Mass matrix in material frame `[6][6][n_nodes]`
    pub m_star: Mat<f64>,
    /// Global rotation `[6][6][n_nodes]`
    pub rr0: Mat<f64>,
    /// mass `[n_nodes]`
    pub m: Col<f64>,
    /// mass `[3][n_nodes]`
    pub eta: Mat<f64>,
    /// mass `[3][3][n_nodes]`
    pub rho: Mat<f64>,
    /// Inertial force `[6][n_nodes]`
    pub fi: Mat<f64>,
    /// Gravity force `[6][n_nodes]`
    pub fg: Mat<f64>,
    /// Inertial mass matrices `[6][6][n_nodes]`
    pub muu: Mat<f64>,
    /// gyro matrices `[6][6][n_nodes]`
    pub gi: Mat<f64>,
    /// stiff matrices `[6][6][n_nodes]`
    pub ki: Mat<f64>,
}

impl Masses {
    pub fn new(elements: &[MassElement], gravity: &[f64; 3], nodes: &[Node]) -> Self {
        // Total number of elements
        let n_elem = elements.len();

        Self {
            n_elem,
            node_ids: elements.iter().map(|e| e.node_id).collect_vec(),
            x: Mat::zeros(7, n_elem),
            x0: Mat::from_fn(7, n_elem, |i, j| nodes[j].x[i]),
            u: Mat::from_fn(7, n_elem, |i, j| nodes[j].u[i]),
            v: Mat::from_fn(6, n_elem, |i, j| nodes[j].v[i]),
            vd: Mat::from_fn(6, n_elem, |i, j| nodes[j].vd[i]),
            gravity: Col::from_fn(3, |i| gravity[i]),
            m_star: Mat::<f64>::from_fn(6 * 6, n_elem, |i, j| elements[j].m[(i.rem(6), i / 6)]),
            rr0: Mat::zeros(6 * 6, n_elem),
            m: Col::zeros(n_elem),
            eta: Mat::zeros(3, n_elem),
            rho: Mat::zeros(3 * 3, n_elem),
            fi: Mat::zeros(6, n_elem),
            fg: Mat::zeros(6, n_elem),
            muu: Mat::zeros(6 * 6, n_elem),
            gi: Mat::zeros(6 * 6, n_elem),
            ki: Mat::zeros(6 * 6, n_elem),
        }
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &mut self,
        nfm: &NodeFreedomMap,
        state: &State,
        mut m: MatMut<f64>, // Mass
        mut g: MatMut<f64>, // Damping
        mut k: MatMut<f64>, // Stiffness
        mut r: ColMut<f64>, // Residual
    ) {
        if self.n_elem == 0 {
            return;
        }

        // Copy displacement, velocity, and acceleration data from state nodes to element nodes
        izip!(
            self.node_ids.iter(),
            self.u.col_iter_mut(),
            self.v.col_iter_mut(),
            self.vd.col_iter_mut()
        )
        .for_each(|(&id, mut u, mut v, mut vd)| {
            u.copy_from(state.u.col(id));
            v.copy_from(state.v.col(id));
            vd.copy_from(state.vd.col(id));
        });

        //----------------------------------------------------------------------
        // Calculate matrices and vectors
        //----------------------------------------------------------------------

        // Calculate current position/rotation
        calc_x(self.x.as_mut(), self.x0.as_ref(), self.u.as_ref());

        // Convert current rotation to matrix
        calc_rr0(self.rr0.as_mut(), self.x.as_ref());

        // Rotate material mass matrix to inertial frame
        calc_inertial_matrix(self.muu.as_mut(), self.m_star.as_ref(), self.rr0.as_ref());

        // Extract components of mass matrix
        calc_m_eta_rho(
            self.m.as_mut(),
            self.eta.as_mut(),
            self.rho.as_mut(),
            self.muu.as_ref(),
        );

        // Calculate gravity force
        calc_fg(
            self.fg.as_mut(),
            self.gravity.as_ref(),
            self.m.as_ref(),
            self.eta.as_ref(),
        );

        // Calculate inertial force vector
        calc_fi(
            self.fi.as_mut(),
            self.m.as_ref(),
            self.v.subrows(3, 3).as_ref(),
            self.vd.subrows(0, 3).as_ref(),
            self.vd.subrows(3, 3).as_ref(),
            self.eta.as_ref(),
            self.rho.as_ref(),
        );

        // Calculate inertial damping matrix
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

        //----------------------------------------------------------------------
        // Assemble elements into global matrices and vectors
        //----------------------------------------------------------------------

        // Get first node dof index of each element
        let elem_first_dof_indices = self
            .node_ids
            .iter()
            .map(|&node_id| nfm.node_dofs[node_id].first_dof_index)
            .collect_vec();

        // Mass matrix
        izip!(elem_first_dof_indices.iter(), self.muu.col_iter()).for_each(|(&i, me_col)| {
            zip!(
                &mut m.as_mut().submatrix_mut(i, i, 6, 6),
                me_col.reshape(6, 6)
            )
            .for_each(|unzip!(m, me)| *m += *me);
        });

        // Gyroscopic matrix
        izip!(elem_first_dof_indices.iter(), self.gi.col_iter()).for_each(|(&i, ge_col)| {
            zip!(
                &mut g.as_mut().submatrix_mut(i, i, 6, 6),
                ge_col.reshape(6, 6)
            )
            .for_each(|unzip!(g, ge)| *g += *ge);
        });

        // Stiffness matrix
        izip!(elem_first_dof_indices.iter(), self.ki.col_iter()).for_each(|(&i, ke_col)| {
            zip!(
                &mut k.as_mut().submatrix_mut(i, i, 6, 6),
                ke_col.reshape(6, 6)
            )
            .for_each(|unzip!(k, ke)| *k += *ke);
        });

        // Residual vector
        izip!(
            elem_first_dof_indices.iter(),
            self.fi.col_iter(),
            self.fg.col_iter()
        )
        .for_each(|(&i, fi, fg)| {
            zip!(&mut r.as_mut().subrows_mut(i, 6), fi, fg)
                .for_each(|unzip!(r, fi, fg)| *r += *fi - *fg);
        });
    }
}

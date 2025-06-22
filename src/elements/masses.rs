use crate::node::Node;
use crate::node::NodeFreedomMap;
use crate::state::State;
use crate::util::sparse_matrix_from_triplets;
use faer::prelude::*;
use faer::sparse::*;
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
    /// Element first dof indices
    elem_first_dof_indices: Vec<usize>,
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

    /// Sparse mass matrix
    pub m_sp: SparseColMat<usize, f64>,
    /// Sparse gyroscopic matrix
    pub g_sp: SparseColMat<usize, f64>,
    /// Sparse stiffness matrix
    pub k_sp: SparseColMat<usize, f64>,
    /// Order of data in sparse matrices
    order_sp: Vec<usize>,
}

impl Masses {
    pub fn new(
        elements: &[MassElement],
        gravity: &[f64; 3],
        nodes: &[Node],
        nfm: &NodeFreedomMap,
    ) -> Self {
        // Total number of elements
        let n_elem = elements.len();

        let node_ids = elements.iter().map(|e| e.node_id).collect_vec();

        //----------------------------------------------------------------------
        // Initialize sparse matrices
        //----------------------------------------------------------------------

        // Get first node dof index of each element
        let elem_first_dof_indices = node_ids
            .iter()
            .map(|&node_id| nfm.node_dofs[node_id].first_dof_index)
            .collect_vec();

        // Sparsity pattern for mass, gyroscopic, and stiffness matrices
        let sp_triplets = elem_first_dof_indices
            .iter()
            .flat_map(|&i| {
                (0..6)
                    .cartesian_product(0..6)
                    .map(|(k, j)| Triplet::new(i + j, i + k, 0.))
                    .collect_vec()
            })
            .collect_vec();

        // Create sparse matrices from triplets and get data order
        let (sp, sp_order) = sparse_matrix_from_triplets(nfm.n_dofs(), nfm.n_dofs(), &sp_triplets);

        //----------------------------------------------------------------------
        // Populate structure
        //----------------------------------------------------------------------

        Self {
            n_elem,
            node_ids,
            elem_first_dof_indices,
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
            m_sp: sp.clone(),
            g_sp: sp.clone(),
            k_sp: sp.clone(),
            order_sp: sp_order,
        }
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &mut self,
        state: &State,
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
        // Populate the residual vector
        //----------------------------------------------------------------------

        izip!(
            self.elem_first_dof_indices.iter(),
            self.fi.col_iter(),
            self.fg.col_iter()
        )
        .for_each(|(&i, fi, fg)| {
            zip!(&mut r.as_mut().subrows_mut(i, 6), fi, fg)
                .for_each(|unzip!(r, fi, fg)| *r += *fi - *fg);
        });

        //----------------------------------------------------------------------
        // Update values in sparse matrices
        // Order of data in sparse matrices is given by `sp_order`
        //----------------------------------------------------------------------

        // Mass matrix
        let mut i = 0;
        let m_values = self.m_sp.val_mut();
        self.muu.col_iter().for_each(|m_col| {
            m_col.iter().for_each(|&v| {
                m_values[self.order_sp[i]] = v;
                i += 1;
            })
        });

        // Gyroscopic matrix
        let mut i = 0;
        let g_values = self.g_sp.val_mut();
        self.gi.col_iter().for_each(|g_col| {
            g_col.iter().for_each(|&v| {
                g_values[self.order_sp[i]] = v;
                i += 1;
            })
        });

        // Stiffness matrix
        let mut i = 0;
        let k_values = self.k_sp.val_mut();
        self.ki.col_iter().for_each(|ke_col| {
            ke_col.iter().for_each(|&v| {
                k_values[self.order_sp[i]] = v;
                i += 1;
            })
        });
    }
}

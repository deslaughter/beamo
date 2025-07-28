use std::ops::Rem;

use crate::elements::beam_qps::BeamQPs;
use crate::interp::{shape_deriv_matrix, shape_interp_matrix};
use crate::node::{Node, NodeFreedomMap};
use crate::quadrature::Quadrature;
use crate::state::State;
use crate::util::{sparse_matrix_from_triplets, ColMutReshape, ColRefReshape};
use faer::linalg::matmul::matmul;
use faer::sparse::Triplet;
use faer::{prelude::*, Accum};
use itertools::{izip, multiunzip, Itertools};

use super::kernels::{
    calc_fd_c, calc_fd_c_viscoelastic, calc_fd_d, calc_inertial_matrix, calc_sd_pd_od_qd_gd_xd_yd,
    rotate_col_to_sectional, update_viscoelastic,
};

pub struct BeamElement {
    pub id: usize,
    pub node_ids: Vec<usize>,
    pub sections: Vec<BeamSection>,
    pub quadrature: Quadrature,
    pub damping: Damping,
}

#[derive(Clone, Debug)]
pub struct BeamSection {
    /// Distance along centerline from first point
    pub s: f64,
    /// Mass matrix `[6][6]`
    pub m_star: Mat<f64>,
    /// Stiffness matrix `[6][6]`
    pub c_star: Mat<f64>,
}

#[derive(Clone, Debug)]
pub enum Damping {
    None,
    Mu(Col<f64>),
    ModalElement(Col<f64>),
    Viscoelastic(Mat<f64>, Col<f64>),
}

#[derive(Clone, Debug)]
pub struct ElemIndex {
    elem_id: usize,
    n_nodes: usize,
    n_qps: usize,
    i_node_start: usize,
    i_qp_start: usize,
    i_mat_start: usize,
    damping: Damping,
    mi_c1: Mat<f64>, // Matrix integration constants 1
    mi_c2: Mat<f64>, // Matrix integration constants 2
    mi_c3: Mat<f64>, // Matrix integration constants 3
    mi_c4: Mat<f64>, // Matrix integration constants 4
    fi_c1: Mat<f64>, // Force integration constants 1
    fi_c2: Mat<f64>, // Force integration constants 2
}

pub struct Beams {
    elem_index: Vec<ElemIndex>,
    pub node_ids: Vec<usize>,
    pub node_first_dof: Vec<usize>,
    pub gravity: Col<f64>, // Gravity components `[3]`
    /// Initial position/rotation `[7][n_nodes]`
    pub node_x0: Mat<f64>,
    /// State: translation/rotation displacement `[7][n_nodes]`
    pub node_u: Mat<f64>,
    /// State: translation/rotation velocity `[6][n_nodes]`
    pub node_v: Mat<f64>,
    /// State: translation/rotation acceleration `[6][n_nodes]`
    pub node_vd: Mat<f64>,
    /// Elastic forces `[6][n_nodes]`
    pub node_fe: Mat<f64>,
    /// Dissipative forces `[6][n_nodes]`
    pub node_fd: Mat<f64>,
    /// Internal forces `[6][n_nodes]`
    pub node_fi: Mat<f64>,
    /// External forces `[6][n_nodes]`
    pub node_fx: Mat<f64>,
    /// Gravity forces `[6][n_nodes]`
    pub node_fg: Mat<f64>,
    /// total forces `[6][n_nodes]`
    pub node_f: Mat<f64>,
    /// mass matrices `[6][n_nodes*max_nodes]`
    pub node_muu: Mat<f64>,
    /// gyro matrices `[6][n_nodes*max_nodes]`
    pub node_guu: Mat<f64>,
    /// stiff matrices `[6][n_nodes*max_nodes]`
    pub node_kuu: Mat<f64>,

    /// Shape function values `[n_qps][max_nodes]`
    pub shape_interp: Mat<f64>,
    /// Shape function derivatives `[n_qps][max_nodes]`
    pub shape_deriv: Mat<f64>,

    pub qp: BeamQPs,

    pub m_sp: SparseColMat<usize, f64>, // Mass triplets
    pub g_sp: SparseColMat<usize, f64>, // Gyro triplets
    pub k_sp: SparseColMat<usize, f64>, // Stiffness triplets
    order_sp: Vec<usize>,               // Sparse matrix order
}

impl Beams {
    pub fn new(
        elements: &[BeamElement],
        gravity: &[f64; 3],
        nodes: &[Node],
        nfm: &NodeFreedomMap,
    ) -> Self {
        // Total number of nodes to allocate (multiple of 8)
        let total_nodes = elements.iter().map(|e| (e.node_ids.len())).sum::<usize>();
        let alloc_nodes = total_nodes;

        // Total number of quadrature points (multiple of 8)
        let total_qps = elements
            .iter()
            .map(|e| (e.quadrature.points.len()))
            .sum::<usize>();
        let alloc_qps = total_qps;

        // Max number of nodes in any element
        let max_elem_nodes = elements
            .iter()
            .map(|e| (e.node_ids.len()))
            .max()
            .unwrap_or(0);

        // Build element index
        let mut elem_index: Vec<ElemIndex> = vec![];
        let mut start_node = 0;
        let mut start_qp = 0;
        let mut start_mat = 0;
        for (i, e) in elements.iter().enumerate() {
            let n_nodes = e.node_ids.len();
            let n_qps = e.quadrature.points.len();
            elem_index.push(ElemIndex {
                elem_id: i,
                n_nodes,
                i_node_start: start_node,
                n_qps,
                i_qp_start: start_qp,
                i_mat_start: start_mat,
                damping: e.damping.clone(),
                mi_c1: Mat::<f64>::zeros(n_qps, n_nodes * n_nodes),
                mi_c2: Mat::<f64>::zeros(n_qps, n_nodes * n_nodes),
                mi_c3: Mat::<f64>::zeros(n_qps, n_nodes * n_nodes),
                mi_c4: Mat::<f64>::zeros(n_qps, n_nodes * n_nodes),
                fi_c1: Mat::<f64>::zeros(n_qps, n_nodes),
                fi_c2: Mat::<f64>::zeros(n_qps, n_nodes),
            });
            start_node += n_nodes;
            start_qp += n_qps;
            start_mat += n_nodes * n_nodes;
        }

        // let (x0, u, v, vd): (Vec<[f64; 7]>, Vec<[f64; 7]>, Vec<[f64; 6]>, Vec<[f64; 6]>) = inp
        let (x0, u, v, vd): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(
            elements
                .iter()
                .flat_map(|e| {
                    e.node_ids
                        .iter()
                        .map(|&node_id| {
                            let node = &nodes[node_id];
                            (node.x, node.u, node.v, node.vd)
                        })
                        .collect_vec()
                })
                .collect_vec(),
        );

        let qp_weights = elements
            .iter()
            .flat_map(|e| e.quadrature.weights.clone())
            .collect_vec();

        let node_ids = elements
            .iter()
            .flat_map(|e| e.node_ids.to_owned())
            .collect_vec();

        // First degree of freedom for each node
        let node_first_dof = node_ids
            .iter()
            .map(|&node_id| nfm.node_dofs[node_id].first_dof_index)
            .collect_vec();

        //----------------------------------------------------------------------
        // Sparse matrices
        //----------------------------------------------------------------------

        let node_indices: Vec<(usize, usize)> = (0..6).cartesian_product(0..6).collect_vec();

        // Get element node data triplets for sparse matrices
        let sp_triplets = elem_index
            .iter()
            .flat_map(|ei| {
                // Get starting degree of freedom for each node
                let ids = &node_ids[ei.i_node_start..ei.i_node_start + ei.n_nodes];
                ids.iter().cartesian_product(ids).flat_map(|(&i, &j)| {
                    let i_dof = nfm.node_dofs[i].first_dof_index;
                    let j_dof = nfm.node_dofs[j].first_dof_index;
                    node_indices
                        .iter()
                        .cloned()
                        .map(|(n, m)| Triplet::new(i_dof + m, j_dof + n, 0.))
                        .collect_vec()
                })
            })
            .collect_vec();

        let (sp, order_sp) = sparse_matrix_from_triplets(nfm.n_dofs(), nfm.n_dofs(), &sp_triplets);

        //----------------------------------------------------------------------
        // Create beams structure
        //----------------------------------------------------------------------

        let mut beams = Self {
            elem_index,
            node_ids,
            node_first_dof,
            gravity: Col::from_fn(3, |i| gravity[i]),

            // Nodes
            node_x0: Mat::from_fn(7, alloc_nodes, |i, j| x0[j][i]),
            node_u: Mat::from_fn(7, alloc_nodes, |i, j| u[j][i]),
            node_v: Mat::from_fn(6, alloc_nodes, |i, j| v[j][i]),
            node_vd: Mat::from_fn(6, alloc_nodes, |i, j| vd[j][i]),
            node_fe: Mat::zeros(6, alloc_nodes),
            node_fd: Mat::zeros(6, alloc_nodes),
            node_fi: Mat::zeros(6, alloc_nodes),
            node_fx: Mat::zeros(6, alloc_nodes),
            node_fg: Mat::zeros(6, alloc_nodes),
            node_f: Mat::zeros(6, alloc_nodes),
            node_muu: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),
            node_guu: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),
            node_kuu: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),

            // Quadrature points
            qp: BeamQPs::new(&qp_weights),

            // Shape function matrices
            shape_interp: Mat::zeros(alloc_qps, max_elem_nodes),
            shape_deriv: Mat::zeros(alloc_qps, max_elem_nodes),

            // Sparse matrices
            m_sp: sp.clone(), // Mass triplets
            g_sp: sp.clone(), // Gyro triplets
            k_sp: sp.clone(), // Stiffness triplets
            order_sp,         // Sparse matrix order
        };

        //----------------------------------------------------------------------
        // Shape functions
        //----------------------------------------------------------------------

        // Initialize element shape functions for interpolation and derivative
        for (ei, e) in izip!(beams.elem_index.iter(), elements.iter()) {
            // Get node positions along beam [-1, 1]
            let node_xi = e
                .node_ids
                .iter()
                .map(|&node_id| nodes[node_id].s * 2. - 1.)
                .collect_vec();

            // Get shape interpolation matrix for this element
            let mut shape_interp =
                beams
                    .shape_interp
                    .submatrix_mut(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Create interpolation matrix from quadrature point and node locations
            shape_interp_matrix(&e.quadrature.points, &node_xi, shape_interp.as_mut());

            // Get shape derivative matrix for this element
            let mut shape_deriv =
                beams
                    .shape_deriv
                    .submatrix_mut(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Create derivative matrix from quadrature point and node locations
            shape_deriv_matrix(&e.quadrature.points, &node_xi, shape_deriv.as_mut());
        }

        //----------------------------------------------------------------------
        // Quadrature points
        //----------------------------------------------------------------------

        for ei in beams.elem_index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp = beams
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Interpolate initial position
            let node_x0 = beams.node_x0.subcols(ei.i_node_start, ei.n_nodes);
            let mut qp_x0 = beams.qp.x0.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_x0.as_mut().transpose_mut(),
                faer::Accum::Replace,
                shape_interp,
                node_x0.transpose(),
                1.,
                Par::Seq,
            );
            qp_x0.subrows_mut(3, 4).col_iter_mut().for_each(|mut c| {
                let m = c.norm_l2();
                if m != 0. {
                    c /= m;
                }
            });

            // Get shape derivative matrix for this element
            let shape_deriv = beams
                .shape_deriv
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Calculate Jacobian
            let qp_jacobian = beams.qp.jacobian.subrows_mut(ei.i_qp_start, ei.n_qps);
            let mut qp_x0_prime = beams.qp.x0_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_x0_prime.as_mut().transpose_mut(),
                faer::Accum::Replace,
                shape_deriv,
                node_x0.transpose(),
                1.0,
                Par::Seq,
            );
            izip!(qp_jacobian.iter_mut(), qp_x0_prime.col_iter_mut()).for_each(
                |(j, mut x0_prime)| {
                    *j = x0_prime.as_mut().subrows(0, 3).norm_l2();
                    x0_prime /= *j;
                },
            );

            // Create vector of element sections
            let sections = &elements[ei.elem_id].sections;
            let section_s = sections.iter().map(|section| section.s).collect_vec();
            let section_m_star = sections
                .iter()
                .map(|s| Col::from_fn(6 * 6, |i| s.m_star[(i.rem(6), i / 6)]))
                .collect_vec();
            let section_c_star = sections
                .iter()
                .map(|s| Col::from_fn(6 * 6, |i| s.c_star[(i.rem(6), i / 6)]))
                .collect_vec();

            // Get quadrature point locations on range of [0,1]
            let qp_s = Col::<f64>::from_fn(ei.n_qps, |i| {
                (elements[ei.elem_id].quadrature.points[i] + 1.) / 2.
            });

            // Linearly interpolate mass and stiffness matrices from sections to quadrature points
            qp_s.iter().enumerate().for_each(|(i, &qp_s)| {
                let mut qp_m_star = beams
                    .qp
                    .m_star
                    .subcols_mut(ei.i_qp_start, ei.n_qps)
                    .col_mut(i);
                let mut qp_c_star = beams
                    .qp
                    .c_star
                    .subcols_mut(ei.i_qp_start, ei.n_qps)
                    .col_mut(i);
                match section_s.iter().position(|&ss| ss > qp_s) {
                    None => {
                        qp_m_star.copy_from(&section_m_star[sections.len() - 1]);
                        qp_c_star.copy_from(&section_c_star[sections.len() - 1]);
                    }
                    Some(0) => {
                        qp_m_star.copy_from(&section_m_star[0]);
                        qp_c_star.copy_from(&section_c_star[0]);
                    }
                    Some(j) => {
                        let alpha = (qp_s - section_s[j - 1]) / (section_s[j] - section_s[j - 1]);
                        qp_m_star.copy_from(
                            &section_m_star[j - 1] * Scale(1. - alpha)
                                + &section_m_star[j] * Scale(alpha),
                        );
                        qp_c_star.copy_from(
                            &section_c_star[j - 1] * Scale(1. - alpha)
                                + &section_c_star[j] * Scale(alpha),
                        );
                    }
                }
            });

            // Match based on damping type
            match &ei.damping {
                Damping::Mu(mu) => {
                    // Calculate damping matrix (g_star = mu*c_star) for each qp in element
                    let mut mu_mat = Mat::<f64>::zeros(6, 6);
                    mu_mat.diagonal_mut().column_vector_mut().copy_from(&mu);
                    let qp_g_star = beams.qp.g_star.subcols_mut(ei.i_qp_start, ei.n_qps);
                    let qp_c_star = beams.qp.c_star.subcols(ei.i_qp_start, ei.n_qps);
                    izip!(qp_g_star.col_iter_mut(), qp_c_star.col_iter()).for_each(
                        |(g_star_col, c_star_col)| {
                            let mut g_star = g_star_col.reshape_mut(6, 6);
                            let c_star = c_star_col.reshape(6, 6);
                            matmul(
                                g_star.as_mut(),
                                Accum::Replace,
                                &mu_mat,
                                c_star,
                                1.,
                                Par::Seq,
                            );
                        },
                    );
                }
                _ => {}
            }
        }

        //----------------------------------------------------------------------
        // Populate integration constant matrices
        //----------------------------------------------------------------------

        beams.elem_index.iter_mut().for_each(|ei| {
            let weight = beams.qp.weight.subrows(ei.i_qp_start, ei.n_qps);
            let jacobian = beams.qp.jacobian.subrows(ei.i_qp_start, ei.n_qps);
            let phi = beams.shape_interp.subrows(ei.i_qp_start, ei.n_qps);
            let phi_prime = beams.shape_deriv.subrows(ei.i_qp_start, ei.n_qps);

            // Force integration constant
            // c = w * j * phi
            ei.fi_c1
                .col_iter_mut()
                .enumerate()
                .for_each(|(i, mut c_col)| {
                    zip!(&mut c_col, &weight, &jacobian, &phi.col(i),)
                        .for_each(|unzip!(cc, w, j, p)| *cc = *w * *j * *p);
                });

            // Force integration constant
            // c = w * phi_prime
            ei.fi_c2
                .col_iter_mut()
                .enumerate()
                .for_each(|(i, mut c_col)| {
                    zip!(&mut c_col, &weight, &phi_prime.col(i),)
                        .for_each(|unzip!(cc, w, p)| *cc = *w * *p);
                });

            // Node combinations for matrix integration
            let node_ij = (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .collect_vec();

            // Matrix integration constant
            // c = w * j * phi_i * phi_j
            node_ij.iter().enumerate().for_each(|(col, &(i, j))| {
                zip!(
                    &mut ei.mi_c1.col_mut(col),
                    &weight,
                    &jacobian,
                    &phi.col(i),
                    &phi.col(j)
                )
                .for_each(|unzip!(cc, w, j, p_i, p_j)| *cc = *w * *j * *p_i * *p_j);
            });

            // Matrix integration constant
            // c = w * phi_i * phi_prime_j
            node_ij.iter().enumerate().for_each(|(col, &(i, j))| {
                zip!(
                    &mut ei.mi_c2.col_mut(col),
                    &weight,
                    &phi.col(i),
                    &phi_prime.col(j)
                )
                .for_each(|unzip!(cc, w, p_i, pp_j)| *cc = *w * *p_i * *pp_j);
            });

            // Matrix integration constant
            // c = w * phi_prime_i * phi_j
            node_ij.iter().enumerate().for_each(|(col, &(i, j))| {
                zip!(
                    &mut ei.mi_c3.col_mut(col),
                    &weight,
                    &phi_prime.col(i),
                    &phi.col(j)
                )
                .for_each(|unzip!(cc, w, pp_i, p_j)| *cc = *w * *pp_i * *p_j);
            });

            // Matrix integration constant
            // c = w * phi_prime_i * phi_prime_j / j
            node_ij.iter().enumerate().for_each(|(col, &(i, j))| {
                zip!(
                    &mut ei.mi_c4.col_mut(col),
                    &weight,
                    &jacobian,
                    &phi_prime.col(i),
                    &phi_prime.col(j)
                )
                .for_each(|unzip!(cc, w, j, pp_i, pp_j)| *cc = *w * *pp_i * *pp_j / *j);
            });
        });

        beams
    }

    /// Calculate strain rate
    pub fn calculate_strain_dot(&mut self, state: &mut State) {
        // Calculate the strain rate at the quadrature points
        // and save the data into state.
        // Calculated strains in the local/sectional frame.

        // Copy displacement, velocity, and acceleration data from state nodes to beam nodes
        izip!(
            self.node_ids.iter(),
            self.node_u.col_iter_mut(),
            self.node_v.col_iter_mut(),
            self.node_vd.col_iter_mut()
        )
        .for_each(|(&id, mut u, mut v, mut vd)| {
            u.copy_from(state.u.col(id));
            v.copy_from(state.v.col(id));
            vd.copy_from(state.vd.col(id));
        });

        // Interpolate node data to quadrature points
        self.interp_to_qps();

        // Calculate quadrature point values
        self.qp.calc(self.gravity.as_ref());

        // Rotate strain rate into the sectional coordinate system
        self.elem_index.iter().for_each(|ei| {
            rotate_col_to_sectional(
                state.strain_dot_n.subcols_mut(ei.i_qp_start, ei.n_qps),
                self.qp.strain_dot.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.rr0.subcols(ei.i_qp_start, ei.n_qps),
            );
        });
    }

    /// Update the viscoelastic history variables
    pub fn update_viscoelastic_history(&mut self, state: &mut State, h: f64) {
        // strain rate at start of time step
        let strain_dot_n = state.strain_dot_n.clone();

        // calculate strain rates at n+1 -> state.strain_dot_n
        self.calculate_strain_dot(state);

        // Update the viscoelastic history variables
        self.elem_index.iter().for_each(|ei| match &ei.damping {
            Damping::Viscoelastic(_, tau_i) => {
                tau_i.iter().enumerate().for_each(|(index, tau_i_curr)| {
                    update_viscoelastic(
                        state
                            .visco_hist
                            .subcols_mut(ei.i_qp_start, ei.n_qps)
                            .subrows_mut(6 * index, 6),
                        strain_dot_n.subcols(ei.i_qp_start, ei.n_qps),
                        state.strain_dot_n.subcols(ei.i_qp_start, ei.n_qps), //time n+1
                        h,
                        *tau_i_curr,
                    );
                });
            }
            _ => (),
        });
    }

    /// Calculate element properties
    pub fn calculate_system(&mut self, state: &State, h: f64) {
        // Copy displacement, velocity, and acceleration data from state nodes to beam nodes
        izip!(
            self.node_ids.iter(),
            self.node_u.col_iter_mut(),
            self.node_v.col_iter_mut(),
            self.node_vd.col_iter_mut()
        )
        .for_each(|(&id, mut u, mut v, mut vd)| {
            u.copy_from(state.u.col(id));
            v.copy_from(state.v.col(id));
            vd.copy_from(state.vd.col(id));
        });

        // Interpolate node data to quadrature points
        self.interp_to_qps();

        // Calculate quadrature point values
        self.qp.calc(self.gravity.as_ref());

        let mut strain_dot_local = Mat::zeros(6, state.strain_dot_n.shape().1);

        // Loop through elements and handle quadrature-point damping
        self.elem_index.iter().for_each(|ei| match &ei.damping {
            Damping::Mu(_) => {
                calculate_mu_damping(
                    self.qp.g_star.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.rr0.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.strain_dot.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.e1_tilde.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.v.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.v_prime.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.mu_cuu.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.fd_c.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.fd_d.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.sd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.pd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.od.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.qd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.gd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.xd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.yd.subcols_mut(ei.i_qp_start, ei.n_qps),
                );
            }
            Damping::Viscoelastic(kv_i, tau_i) => {
                rotate_col_to_sectional(
                    strain_dot_local.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.strain_dot.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.rr0.subcols(ei.i_qp_start, ei.n_qps),
                );

                calculate_viscoelastic_force(
                    kv_i.as_ref(),
                    tau_i.as_ref(),
                    self.qp.rr0.subcols(ei.i_qp_start, ei.n_qps),
                    state.strain_dot_n.subcols(ei.i_qp_start, ei.n_qps),
                    strain_dot_local.subcols(ei.i_qp_start, ei.n_qps),
                    state.visco_hist.subcols(ei.i_qp_start, ei.n_qps),
                    h,
                    self.qp.e1_tilde.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.v.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.v_prime.subcols(ei.i_qp_start, ei.n_qps),
                    self.qp.mu_cuu.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.fd_c.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.fd_d.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.sd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.pd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.od.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.qd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.gd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.xd.subcols_mut(ei.i_qp_start, ei.n_qps),
                    self.qp.yd.subcols_mut(ei.i_qp_start, ei.n_qps),
                );
            }
            _ => (),
        });

        // Integrate forces
        self.integrate_forces();

        // Integrate matrices
        self.integrate_matrices();

        // Loop through elements and handle element-based damping
        self.elem_index.iter().for_each(|ei| {
            match &ei.damping {
                Damping::ModalElement(zeta) => {
                    // Create matrices to store mass and stiffness
                    let n_dofs = ei.n_nodes * 6;

                    // Get starting degree of freedom for each node
                    let dof_start_pairs = (0..ei.n_nodes)
                        .cartesian_product(0..ei.n_nodes)
                        .map(|(i, j)| (i * 6, j * 6))
                        .collect_vec();

                    // Mass matrix
                    let mut m = Mat::<f64>::zeros(n_dofs, n_dofs);
                    izip!(
                        dof_start_pairs.iter(),
                        self.node_muu
                            .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                            .col_iter()
                    )
                    .for_each(|(&(i, j), muu)| {
                        let mut me = m.as_mut().submatrix_mut(i, j, 6, 6);
                        zip!(&mut me, &muu.reshape(6, 6)).for_each(|unzip!(me, muu)| *me += *muu);
                    });

                    // Stiffness matrix
                    let mut k = Mat::<f64>::zeros(n_dofs, n_dofs);
                    izip!(
                        dof_start_pairs.iter(),
                        self.node_kuu
                            .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                            .col_iter()
                    )
                    .for_each(|(&(i, j), kuu)| {
                        let mut ke = k.as_mut().submatrix_mut(i, j, 6, 6);
                        zip!(&mut ke, &kuu.reshape(6, 6)).for_each(|unzip!(ke, kuu)| *ke += *kuu);
                    });

                    // Solve for A matrix given M and K
                    let n_dof_bc = n_dofs - 6;
                    let lu = m.submatrix(6, 6, n_dof_bc, n_dof_bc).partial_piv_lu();
                    let a = lu.solve(k.submatrix(6, 6, n_dof_bc, n_dof_bc));

                    // Perform eigendecomposition on A matrix to get values and vectors
                    let eig = a.eigen().unwrap();
                    let eig_val_raw = eig.S().column_vector();
                    let eig_vec_raw = eig.U();

                    // Get the order of the eigenvalues
                    let mut eig_order: Vec<_> = (0..eig_val_raw.nrows()).collect();
                    eig_order
                        .sort_by(|&i, &j| eig_val_raw.get(i).re.total_cmp(&eig_val_raw.get(j).re));

                    // Get sorted eigenvalue vector
                    let omega = Col::<f64>::from_fn(eig_val_raw.nrows(), |i| {
                        eig_val_raw[eig_order[i]].re.sqrt()
                    });

                    // Get sorted eigenvector matrix
                    let psi = Mat::<f64>::from_fn(n_dofs, eig_vec_raw.ncols(), |i, j| {
                        if i < 6 {
                            0.
                        } else {
                            eig_vec_raw[(i - 6, eig_order[j])].re
                        }
                    });

                    // Calculate modal mass
                    let m_modal = &psi.transpose() * &m * &psi;

                    // Build mass normalized eigenvalue matrix Phi^T * M * Phi = I
                    let mut phi = psi;
                    phi.col_iter_mut()
                        .zip(m_modal.diagonal().column_vector().iter())
                        .for_each(|(mut phi_i, &m_i)| phi_i /= m_i.sqrt());

                    // Calculate reduced z and phi matrices based on specified modes
                    let phi_d = Mat::<f64>::from_fn(phi.nrows(), zeta.nrows(), |i, j| phi[(i, j)]);
                    let z_d = Mat::<f64>::from_fn(zeta.nrows(), zeta.nrows(), |i, j| {
                        if i == j {
                            2. * omega[i] * zeta[i]
                        } else {
                            0.
                        }
                    });

                    // Build the damping matrix
                    let c_d = &m.transpose() * (&phi_d * &z_d * &phi_d.transpose()) * &m;

                    // Add components of the damping matrix to node gyroscopic matrices
                    izip!(
                        dof_start_pairs.iter(),
                        self.node_guu
                            .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                            .col_iter_mut()
                    )
                    .for_each(|(&(i, j), guu)| {
                        zip!(&mut guu.reshape_mut(6, 6), &c_d.submatrix(i, j, 6, 6))
                            .for_each(|unzip!(guu, c)| *guu += *c);
                    });

                    // Calculate the damping force on each node
                    let mut v = Col::<f64>::zeros(n_dofs);
                    v.as_mut()
                        .reshape_mut(6, ei.n_nodes)
                        .copy_from(self.node_v.subcols(ei.i_node_start, ei.n_nodes));
                    let f_d = &c_d * &v;

                    // Add to node dissipative force
                    zip!(
                        &mut self.node_fd.subcols_mut(ei.i_node_start, ei.n_nodes),
                        &f_d.as_ref().reshape(6, ei.n_nodes)
                    )
                    .for_each(|unzip!(node_fd, f_d)| *node_fd += *f_d);
                }
                _ => {}
            }
        });

        // Combine force components
        zip!(
            &mut self.node_f, // total
            &self.node_fe,    // elastic
            &self.node_fd,    // dissipative
            &self.node_fg,    // gravity
            &self.node_fi,    // internal
            &self.node_fx     // external (distributed)
        )
        .for_each(|unzip!(f, fe, fd, fg, fi, fx)| *f = *fi + *fe + *fd - *fx - *fg);
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &mut self,
        mut r: ColMut<f64>, // Residual
    ) {
        let mut m_k = 0;
        let mut g_k = 0;
        let mut k_k = 0;
        let m_vals = self.m_sp.val_mut();
        let g_vals = self.g_sp.val_mut();
        let k_vals = self.k_sp.val_mut();

        // Loop through elements
        self.elem_index.iter().for_each(|ei| {
            //------------------------------------------------------------------
            // Sparse matrices (order must match)
            //------------------------------------------------------------------

            // Mass
            self.node_muu
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                .col_iter()
                .for_each(|muu| {
                    muu.iter().for_each(|&v| {
                        m_vals[self.order_sp[m_k]] = v;
                        m_k += 1;
                    });
                });

            // Gyro
            self.node_guu
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                .col_iter()
                .for_each(|guu| {
                    guu.iter().for_each(|&v| {
                        g_vals[self.order_sp[g_k]] = v;
                        g_k += 1;
                    });
                });

            // Stiffness
            self.node_kuu
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                .col_iter()
                .for_each(|kuu| {
                    kuu.iter().for_each(|&v| {
                        k_vals[self.order_sp[k_k]] = v;
                        k_k += 1;
                    });
                });

            //------------------------------------------------------------------
            // Residual vector
            //------------------------------------------------------------------

            // Residual vector
            izip!(
                self.node_first_dof[ei.i_node_start..ei.i_node_start + ei.n_nodes].iter(),
                self.node_f.subcols(ei.i_node_start, ei.n_nodes).col_iter()
            )
            .for_each(|(&i, f)| {
                let mut residual = r.as_mut().subrows_mut(i, 6);
                zip!(&mut residual, &f).for_each(|unzip!(r, f)| *r += *f);
            });
        });
    }

    fn interp_to_qps(&mut self) {
        for ei in self.elem_index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp = self
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Get shape derivative matrix for this element
            let shape_deriv = self
                .shape_deriv
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Node values for this element
            let node_u = self.node_u.subcols(ei.i_node_start, ei.n_nodes);
            let node_v = self.node_v.subcols(ei.i_node_start, ei.n_nodes);
            let node_vd = self.node_vd.subcols(ei.i_node_start, ei.n_nodes);

            // Interpolate displacement
            let mut u = self.qp.u.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                u.as_mut().transpose_mut(),
                Accum::Replace,
                shape_interp,
                node_u.transpose(),
                1.,
                Par::Seq,
            );

            // Displacement derivative
            let mut u_prime = self.qp.u_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                u_prime.as_mut().transpose_mut(),
                Accum::Replace,
                shape_deriv,
                node_u.transpose(),
                1.0,
                Par::Seq,
            );

            // Interpolate velocity
            let mut v = self.qp.v.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                v.as_mut().transpose_mut(),
                Accum::Replace,
                shape_interp,
                node_v.transpose(),
                1.0,
                Par::Seq,
            );

            // Velocity derivative
            let mut qp_v_prime = self.qp.v_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_v_prime.as_mut().transpose_mut(),
                Accum::Replace,
                shape_deriv,
                node_v.transpose(),
                1.0,
                Par::Seq,
            );

            // Interpolate acceleration
            let mut vd = self.qp.vd.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                vd.as_mut().transpose_mut(),
                Accum::Replace,
                shape_interp,
                node_vd.transpose(),
                1.0,
                Par::Seq,
            );
        }

        // Divide each column of u_prime by Jacobian
        izip!(self.qp.u_prime.col_iter_mut(), self.qp.jacobian.iter()).for_each(
            |(mut col, &jacobian)| zip!(&mut col).for_each(|unzip!(col)| *col /= jacobian),
        );

        // Divide each column of v_prime by Jacobian
        izip!(self.qp.v_prime.col_iter_mut(), self.qp.jacobian.iter()).for_each(
            |(mut col, &jacobian)| zip!(&mut col).for_each(|unzip!(col)| *col /= jacobian),
        );

        // Normalize quaternion and quaternion derivative
        izip!(
            self.qp.u.subrows_mut(3, 4).col_iter_mut(),
            self.qp.u_prime.subrows_mut(3, 4).col_iter_mut()
        )
        .for_each(|(mut q, mut q_prime)| {
            let m = q.norm_l2();
            if m != 0. {
                q /= m;
                let a =
                    (q[0] * q_prime[0] + q[1] * q_prime[1] + q[2] * q_prime[2] + q[3] * q_prime[3])
                        / m.powi(3);
                zip!(&mut q_prime, q).for_each(|unzip!(qp, q)| *qp = *qp / m - *q * a)
            }
        });
    }

    #[inline]
    fn integrate_forces(&mut self) {
        // Zero node matrices
        self.node_fe.fill(0.);
        // self.node_fd.fill(0.);
        self.node_fi.fill(0.);
        // self.node_fx.fill(0.);
        self.node_fg.fill(0.);

        // Loop through elements
        for ei in self.elem_index.iter() {
            integrate_element_forces(
                self.node_fe.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.node_fd.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.node_fi.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.node_fx.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.node_fg.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fe_c.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fe_d.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fd_c.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fd_d.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fi.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fg.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fx.subcols(ei.i_qp_start, ei.n_qps),
                ei.fi_c1.rb(),
                ei.fi_c2.rb(),
            );
        }
    }

    #[inline]
    fn integrate_matrices(&mut self) {
        // Loop through elements
        for ei in self.elem_index.iter() {
            integrate_element_matrices(
                self.node_muu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.node_guu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.node_kuu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.qp.muu.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.gi.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.ki.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.pe.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.qe.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.cuu.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.oe.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.sd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.pd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.od.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.qd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.gd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.xd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.yd.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.mu_cuu.subcols(ei.i_qp_start, ei.n_qps),
                ei.mi_c1.rb(),
                ei.mi_c2.rb(),
                ei.mi_c3.rb(),
                ei.mi_c4.rb(),
            );
        }
    }
}

#[inline]
fn integrate_element_forces(
    mut node_fe: MatMut<f64>,
    mut node_fd: MatMut<f64>,
    mut node_fi: MatMut<f64>,
    mut node_fx: MatMut<f64>,
    mut node_fg: MatMut<f64>,
    qp_fe_c: MatRef<f64>,
    qp_fe_d: MatRef<f64>,
    qp_fd_c: MatRef<f64>,
    qp_fd_d: MatRef<f64>,
    qp_fi: MatRef<f64>,
    qp_fg: MatRef<f64>,
    qp_fx: MatRef<f64>,
    c1: MatRef<f64>,
    c2: MatRef<f64>,
) {
    // Internal forces
    matmul(node_fi.rb_mut(), Accum::Replace, &qp_fi, &c1, 1., Par::Seq);

    // Gravity forces
    matmul(node_fg.rb_mut(), Accum::Replace, &qp_fg, &c1, 1., Par::Seq);

    // External distributed forces
    matmul(node_fx.rb_mut(), Accum::Replace, &qp_fx, &c1, 1., Par::Seq);

    // Elastic forces part 1
    matmul(
        node_fe.rb_mut(),
        Accum::Replace,
        &qp_fe_d,
        &c1,
        1.,
        Par::Seq,
    );

    // Dissipative forces part 1
    matmul(
        node_fd.rb_mut(),
        Accum::Replace,
        &qp_fd_d,
        &c1,
        1.,
        Par::Seq,
    );

    // Elastic forces part 2
    matmul(node_fe.rb_mut(), Accum::Add, &qp_fe_c, &c2, 1., Par::Seq);

    // Dissipative forces part 2
    matmul(node_fd.rb_mut(), Accum::Add, &qp_fd_c, &c2, 1., Par::Seq);
}

fn integrate_element_matrices(
    mut node_m: MatMut<f64>,
    mut node_g: MatMut<f64>,
    mut node_k: MatMut<f64>,
    qp_muu: MatRef<f64>,
    qp_gi: MatRef<f64>,
    qp_ki: MatRef<f64>,
    qp_pe: MatRef<f64>,
    qp_qe: MatRef<f64>,
    qp_cuu: MatRef<f64>,
    qp_oe: MatRef<f64>,
    qp_sd: MatRef<f64>,
    qp_pd: MatRef<f64>,
    qp_od: MatRef<f64>,
    qp_qd: MatRef<f64>,
    qp_gd: MatRef<f64>,
    qp_xd: MatRef<f64>,
    qp_yd: MatRef<f64>,
    qp_mu_cuu: MatRef<f64>,
    c1: MatRef<f64>,
    c2: MatRef<f64>,
    c3: MatRef<f64>,
    c4: MatRef<f64>,
) {
    matmul(node_m.rb_mut(), Accum::Replace, qp_muu, c1, 1., Par::Seq);

    matmul(node_g.rb_mut(), Accum::Replace, qp_gi, c1, 1., Par::Seq);
    matmul(node_g.rb_mut(), Accum::Add, qp_xd, c1, 1., Par::Seq);
    matmul(node_g.rb_mut(), Accum::Add, qp_yd, c2, 1., Par::Seq);
    matmul(node_g.rb_mut(), Accum::Add, qp_gd, c3, 1., Par::Seq);
    matmul(node_g.rb_mut(), Accum::Add, qp_mu_cuu, c4, 1., Par::Seq);

    matmul(node_k.rb_mut(), Accum::Replace, qp_ki, c1, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_qe, c1, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_qd, c1, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_pe, c2, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_pd, c2, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_oe, c3, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_od, c3, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_cuu, c4, 1., Par::Seq);
    matmul(node_k.rb_mut(), Accum::Add, qp_sd, c4, 1., Par::Seq);
}

pub fn calculate_mu_damping(
    g_star: MatRef<f64>,
    rr0: MatRef<f64>,
    strain_dot: MatRef<f64>,
    e1_tilde: MatRef<f64>,
    v: MatRef<f64>,
    v_prime: MatRef<f64>,
    mut mu_cuu: MatMut<f64>,
    mut fd_c: MatMut<f64>,
    fd_d: MatMut<f64>,
    sd: MatMut<f64>,
    pd: MatMut<f64>,
    od: MatMut<f64>,
    qd: MatMut<f64>,
    gd: MatMut<f64>,
    xd: MatMut<f64>,
    yd: MatMut<f64>,
) {
    calc_inertial_matrix(mu_cuu.as_mut(), g_star, rr0);
    calc_fd_c(fd_c.as_mut(), mu_cuu.as_ref(), strain_dot);
    calc_fd_d(fd_d, fd_c.as_ref(), e1_tilde);
    calc_sd_pd_od_qd_gd_xd_yd(
        sd,
        pd,
        od,
        qd,
        gd,
        xd,
        yd,
        mu_cuu.as_ref(),
        v_prime.subrows(0, 3),
        v.subrows(3, 3),
        fd_c.as_ref(),
        e1_tilde,
    )
}

pub fn calculate_viscoelastic_force(
    kv_i: MatRef<f64>, //[36][n_prony]
    tau_i: ColRef<f64>,
    rr0: MatRef<f64>,
    strain_dot_n: MatRef<f64>,
    strain_dot_n1: MatRef<f64>,
    visco_hist: MatRef<f64>, //[6*n_prony][n_qps in elem]
    h: f64,
    e1_tilde: MatRef<f64>,
    v: MatRef<f64>,
    v_prime: MatRef<f64>,
    mut mu_cuu: MatMut<f64>,
    mut fd_c: MatMut<f64>,
    mut fd_d: MatMut<f64>,
    mut sd: MatMut<f64>,
    mut pd: MatMut<f64>,
    mut od: MatMut<f64>,
    mut qd: MatMut<f64>,
    mut gd: MatMut<f64>,
    mut xd: MatMut<f64>,
    mut yd: MatMut<f64>,
) {
    // fill everything with zeros
    mu_cuu.fill(0.);
    fd_c.fill(0.);
    fd_d.fill(0.);
    sd.fill(0.);
    pd.fill(0.);
    od.fill(0.);
    qd.fill(0.);
    gd.fill(0.);
    xd.fill(0.);
    yd.fill(0.);

    // copy kv_i needs to be reshaped to match formatting of mu_cuu
    let mut flat_kvi: Mat<f64> = faer::Mat::zeros(36, mu_cuu.shape().1);

    izip!(kv_i.col_iter().enumerate(), tau_i.iter()).for_each(|((index, kvi_col), tau_i_curr)| {
        let kvi_mat = kvi_col.reshape(6, 6);

        let mut mu_cuu_tmp: Mat<f64> = faer::Mat::zeros(mu_cuu.shape().0, mu_cuu.shape().1);
        let mut fd_c_tmp: Mat<f64> = faer::Mat::zeros(fd_c.shape().0, fd_c.shape().1);
        let mut fd_d_tmp: Mat<f64> = faer::Mat::zeros(fd_d.shape().0, fd_d.shape().1);
        let mut sd_tmp: Mat<f64> = faer::Mat::zeros(sd.shape().0, sd.shape().1);
        let mut pd_tmp: Mat<f64> = faer::Mat::zeros(pd.shape().0, pd.shape().1);
        let mut od_tmp: Mat<f64> = faer::Mat::zeros(od.shape().0, od.shape().1);
        let mut qd_tmp: Mat<f64> = faer::Mat::zeros(qd.shape().0, qd.shape().1);
        let mut gd_tmp: Mat<f64> = faer::Mat::zeros(gd.shape().0, gd.shape().1);
        let mut xd_tmp: Mat<f64> = faer::Mat::zeros(xd.shape().0, xd.shape().1);
        let mut yd_tmp: Mat<f64> = faer::Mat::zeros(yd.shape().0, yd.shape().1);
        let mut flat_kvi_tmp: Mat<f64> = faer::Mat::zeros(flat_kvi.shape().0, flat_kvi.shape().1);

        // Quadrature viscoelastic forces saved into fd_c
        calc_fd_c_viscoelastic(
            fd_c_tmp.as_mut(),
            h,
            kvi_mat,
            *tau_i_curr,
            rr0,
            strain_dot_n,
            strain_dot_n1,
            visco_hist.subrows(6 * index, 6),
        );

        // Additional components similar to fd_d
        calc_fd_d(fd_d_tmp.as_mut(), fd_c_tmp.as_ref(), e1_tilde);

        flat_kvi_tmp.col_iter_mut().for_each(|col_kvi| {
            let mut mat_kvi_grad = col_kvi.reshape_mut(6, 6);
            mat_kvi_grad.copy_from(h / 2. * kvi_mat);
        });

        // Gradient of global forces w.r.t. global strain rate at n+1
        // saved into mu_cuu
        calc_inertial_matrix(mu_cuu_tmp.as_mut(), flat_kvi_tmp.as_ref(), rr0);

        calc_sd_pd_od_qd_gd_xd_yd(
            sd_tmp.as_mut(),
            pd_tmp.as_mut(),
            od_tmp.as_mut(),
            qd_tmp.as_mut(),
            gd_tmp.as_mut(),
            xd_tmp.as_mut(),
            yd_tmp.as_mut(),
            mu_cuu_tmp.as_ref(),
            v_prime.subrows(0, 3),
            v.subrows(3, 3),
            fd_c_tmp.as_ref(),
            e1_tilde,
        );

        fd_c += fd_c_tmp;
        fd_d += fd_d_tmp;

        flat_kvi += flat_kvi_tmp;
        mu_cuu += mu_cuu_tmp;

        sd += sd_tmp;
        pd += pd_tmp;
        od += od_tmp;
        qd += qd_tmp;
        gd += gd_tmp;
        xd += xd_tmp;
        yd += yd_tmp;
    });
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        interp::gauss_legendre_lobotto_points,
        model::Model,
        quadrature::Quadrature,
        util::{quat_compose, quat_from_rotation_matrix, vec_tilde},
    };
    use equator::assert;
    use faer::utils::approx::*;

    fn create_beams(h: f64) -> Beams {
        // Mass matrix 6x6
        let m_star = mat![
            [2., 0., 0., 0., 0.6, -0.4],
            [0., 2., 0., -0.6, 0., 0.2],
            [0., 0., 2., 0.4, -0.2, 0.],
            [0., -0.6, 0.4, 1., 2., 3.],
            [0.6, 0., -0.2, 2., 4., 6.],
            [-0.4, 0.2, 0., 3., 6., 9.],
        ];

        // Stiffness matrix 6x6
        let c_star = mat![
            [1., 2., 3., 4., 5., 6.],
            [2., 4., 6., 8., 10., 12.],
            [3., 6., 9., 12., 15., 18.],
            [4., 8., 12., 16., 20., 24.],
            [5., 10., 15., 20., 25., 30.],
            [6., 12., 18., 24., 30., 36.],
        ];

        let node_s = vec![0., 0.1726731646460114, 0.5, 0.82732683535398865, 1.0];

        let mut model = Model::new();

        let node_ids = vec![
            model
                .add_node()
                .element_location(node_s[0])
                .position(
                    0.,
                    0.,
                    0.,
                    0.9778215200524469,
                    -0.01733607539094763,
                    -0.09001900002195001,
                    -0.18831121859148398,
                )
                .build(),
            model
                .add_node()
                .element_location(node_s[1])
                .position(
                    0.863365823230057,
                    -0.2558982639254171,
                    0.11304112106827427,
                    0.9950113028068008,
                    -0.002883848832932071,
                    -0.030192109815745303,
                    -0.09504013471947484,
                )
                .displacement(
                    0.002981602178886856,
                    -0.00246675949494302,
                    0.003084570715675624,
                    0.9999627302042724,
                    0.008633550973807708,
                    0.,
                    0.,
                )
                .velocity(
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624,
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624,
                )
                .acceleration(
                    0.01726731646460114,
                    -0.011304112106827427,
                    0.00606617289456248,
                    0.01726731646460114,
                    -0.014285714285714285,
                    -0.014285714285714285,
                )
                .build(),
            model
                .add_node()
                .element_location(node_s[2])
                .position(
                    2.5,
                    -0.25,
                    0.,
                    0.9904718430204884,
                    -0.009526411091536478,
                    0.09620741150793366,
                    0.09807604012323785,
                )
                .displacement(
                    0.025,
                    -0.0125,
                    0.0275,
                    0.9996875162757026,
                    0.02499739591471221,
                    0.,
                    0.,
                )
                .velocity(0.05, -0.025, 0.0275, 0.05, -0.025, 0.0275)
                .acceleration(0.05, 0., 0.0525, 0.05, -0.025, -0.025)
                .build(),
            model
                .add_node()
                .element_location(node_s[3])
                .position(
                    4.1366341767699435,
                    0.39875540678256005,
                    -0.5416125496397031,
                    0.9472312341234699,
                    -0.04969214162931507,
                    0.18127630174800594,
                    0.25965858850765167,
                )
                .displacement(
                    0.06844696924968459,
                    -0.011818954790771264,
                    0.07977257214146725,
                    0.9991445348823055,
                    0.04135454527402512,
                    0.,
                    0.,
                )
                .velocity(
                    0.08273268353539887,
                    -0.01428571428571428,
                    0.07977257214146725,
                    0.08273268353539887,
                    -0.01428571428571428,
                    0.07977257214146725,
                )
                .acceleration(
                    0.08273268353539887,
                    0.05416125496397031,
                    0.14821954139115184,
                    0.08273268353539887,
                    -0.01428571428571428,
                    -0.01428571428571428,
                )
                .build(),
            model
                .add_node()
                .element_location(node_s[4])
                .position(
                    5.,
                    1.,
                    -1.,
                    0.9210746582719719,
                    -0.07193653093139739,
                    0.20507529985516368,
                    0.32309554437664584,
                )
                .displacement(
                    0.1,
                    0.,
                    0.12,
                    0.9987502603949663,
                    0.04997916927067825,
                    0.,
                    0.,
                )
                .velocity(0.1, 0., 0.12, 0.1, 0., 0.12)
                .acceleration(0.1, 0.1, 0.22, 0.1, 0., 0.)
                .build(),
        ];

        model.set_gravity(0., 0., 9.81);

        model.add_beam_element(
            &node_ids,
            &Quadrature {
                points: vec![
                    -0.9491079123427585,
                    -0.7415311855993943,
                    -0.40584515137739696,
                    6.123233995736766e-17,
                    0.4058451513773971,
                    0.7415311855993945,
                    0.9491079123427585,
                ],
                weights: vec![
                    0.1294849661688697,
                    0.27970539148927664,
                    0.3818300505051189,
                    0.4179591836734694,
                    0.3818300505051189,
                    0.27970539148927664,
                    0.1294849661688697,
                ],
            },
            &[
                BeamSection {
                    s: 0.,
                    m_star: m_star.clone(),
                    c_star: c_star.clone(),
                },
                BeamSection {
                    s: 1.,
                    m_star: m_star.clone(),
                    c_star: c_star.clone(),
                },
            ],
            &Damping::None,
        );

        let nfm = model.create_node_freedom_map();
        let mut elements = model.create_elements(&nfm);
        let state = model.create_state();
        elements.beams.calculate_system(&state, h);

        elements.beams
    }

    #[test]
    fn test_node_x0() {
        let approx_eq = CwiseMat(ApproxEq::eps() * 100.);

        //Only matters for viscoelastic material, but needs to be passed to create_beams
        let h = 0.001;

        let beams = create_beams(h);
        assert!(
            beams.node_x0.subcols(0, 2).transpose() ~
            mat![
                [
                    0.,
                    0.,
                    0.,
                    0.9778215200524469,
                    -0.01733607539094763,
                    -0.09001900002195001,
                    -0.18831121859148398
                ],
                [
                    0.863365823230057,
                    -0.2558982639254171,
                    0.11304112106827427,
                    0.9950113028068008,
                    -0.002883848832932071,
                    -0.030192109815745303,
                    -0.09504013471947484
                ],
            ]
        );

        assert!(
            beams.node_u.subcols(0, 2).transpose() ~
            mat![
                [0., 0., 0., 1., 0., 0., 0.],
                [
                    0.002981602178886856,
                    -0.00246675949494302,
                    0.003084570715675624,
                    0.9999627302042724,
                    0.008633550973807708,
                    0.,
                    0.
                ],
            ]
        );

        assert!(
            beams.node_v.subcols(0, 2).transpose() ~
            mat![
                [0., 0., 0., 0., 0., 0.],
                [
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624,
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624
                ],
            ]
        );

        assert!(
            beams.node_vd.subcols(0, 2).transpose() ~
            mat![
                [0., 0., 0., 0., 0., 0.],
                [
                    0.01726731646460114,
                    -0.011304112106827427,
                    0.00606617289456248,
                    0.01726731646460114,
                    -0.014285714285714285,
                    -0.014285714285714285
                ],
            ]
        );

        assert!(
            beams.qp.m_star.col(0).reshape(6, 6) ~
            mat![
                [2., 0., 0., 0., 0.6, -0.4],
                [0., 2., 0., -0.6, 0., 0.2],
                [0., 0., 2., 0.4, -0.2, 0.],
                [0., -0.6, 0.4, 1., 2., 3.],
                [0.6, 0., -0.2, 2., 4., 6.],
                [-0.4, 0.2, 0., 3., 6., 9.],
            ]
        );

        assert!(
            beams.qp.c_star.col(0).reshape(6, 6) ~
            mat![
                [1., 2., 3., 4., 5., 6.],
                [2., 4., 6., 8., 10., 12.],
                [3., 6., 9., 12., 15., 18.],
                [4., 8., 12., 16., 20., 24.],
                [5., 10., 15., 20., 25., 30.],
                [6., 12., 18., 24., 30., 36.],
            ]
        );

        assert!(
            beams.qp.x0.col(0) ~
            col![
                0.12723021914310376,
                -0.048949584217657216,
                0.024151041535564563,
                0.98088441332097497,
                -0.014472327094052504,
                -0.082444330164641907,
                -0.17566801608760002
            ]
        );

        assert!(
            beams.qp.x0_prime.col(0) ~
            col![
                0.92498434449987588,
                -0.34174910719483215,
                0.16616711516322963,
                0.023197240723437866,
                0.01993094516117577,
                0.056965007432292485,
                0.09338617439225547
            ]
        );

        assert!(
            beams.qp.u.col(0) ~
            col![
                0.000064750114652809492,
                -0.000063102480397445355,
                0.000065079641503882152,
                0.9999991906236807,
                0.0012723018445566286,
                0.,
                0.
            ]
        );

        assert!(
            beams.qp.u_prime.col(0) ~
            col![
                0.00094148768683727929,
                -0.00090555198142222483,
                0.00094867482792029139,
                -0.000011768592508100627,
                0.0092498359395732574,
                0.,
                0.
            ]
        );

        assert!(
            beams.qp.x.col(0) ~
            col![
                0.12729496925775657,
                -0.049012686698054662,
                0.024216121177068447,
                0.9809020325848155,
                -0.013224334332128548,
                -0.08222076069525557,
                -0.1757727679794095,
            ]
        );

        assert!(
            beams.qp.jacobian.subrows(0, beams.elem_index[0].n_qps) ~
            col![
                2.7027484463552831,
                2.5851972184835246,
                2.5041356900076868,
                2.5980762113533156,
                2.8809584014451262,
                3.223491986410379,
                3.4713669823269537,
            ]
        );

        assert!(
            beams.qp.strain.subcols(0, 2).transpose() ~
            mat![
                [
                    0.0009414876868371058,
                    -0.0004838292834870028,
                    0.0018188281296873665,
                    0.0184996868523541,
                    0.,
                    0.
                ],
                [
                    0.004999015404948938,
                    -0.0028423419905453384,
                    0.008261426556751703,
                    0.0193408842119465,
                    0.,
                    0.
                ]
            ]
        );

        assert!(
            beams.qp.rr0.col(0).reshape(6, 6) ~
            mat![
                [
                    0.9246873610951006,
                    0.34700636042507577,
                    -0.156652066872805,
                    0.,
                    0.,
                    0.
                ],
                [
                    -0.3426571011111718,
                    0.937858102036658,
                    0.05484789423748749,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.16594997827377847,
                    0.002960788533623304,
                    0.9861297269843315,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.9246873610951006,
                    0.34700636042507577,
                    -0.156652066872805
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.3426571011111718,
                    0.937858102036658,
                    0.05484789423748749
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.16594997827377847,
                    0.002960788533623304,
                    0.9861297269843315
                ],
            ]
        );

        assert!(
            beams.qp.muu.col(0).reshape(6, 6) ~
            mat![
                [
                    2.000000000000001,
                    5.204170427930421e-17,
                    -0.00000000000000005551115123125783,
                    -0.00000000000000004163336342344337,
                    0.626052147258804,
                    -0.3395205571349214
                ],
                [
                    5.204170427930421e-17,
                    2.0000000000000018,
                    0.000000000000000013877787807814457,
                    -0.6260521472588039,
                    -0.000000000000000003469446951953614,
                    0.22974877626536766
                ],
                [
                    -0.00000000000000005551115123125783,
                    0.000000000000000013877787807814457,
                    2.0000000000000013,
                    0.33952055713492146,
                    -0.22974877626536772,
                    -1.3877787807814457e-17
                ],
                [
                    0.00000000000000004163336342344337,
                    -0.626052147258804,
                    0.3395205571349214,
                    1.3196125048858467,
                    1.9501108129670985,
                    3.5958678677753957
                ],
                [
                    0.6260521472588039,
                    0.000000000000000003469446951953614,
                    -0.22974877626536766,
                    1.9501108129670985,
                    2.881855217930184,
                    5.313939345820573
                ],
                [
                    -0.33952055713492146,
                    0.22974877626536772,
                    1.3877787807814457e-17,
                    3.5958678677753957,
                    5.3139393458205735,
                    9.79853227718398
                ],
            ]
        );

        assert!(
            beams.qp.cuu.col(0).reshape(6, 6) ~
            mat![
                [
                    1.3196125048858467,
                    1.9501108129670968,
                    3.595867867775392,
                    5.1623043394880055,
                    4.190329885612304,
                    7.576404967559336
                ],
                [
                    1.9501108129670985,
                    2.881855217930184,
                    5.313939345820573,
                    7.628804270184899,
                    6.192429663690275,
                    11.196339225304031
                ],
                [
                    3.5958678677753957,
                    5.3139393458205735,
                    9.79853227718398,
                    14.066981200400345,
                    11.418406945420463,
                    20.64526599682174
                ],
                [
                    5.162304339488006,
                    7.628804270184899,
                    14.066981200400342,
                    20.194857198478893,
                    16.392507703808057,
                    29.638782670624547
                ],
                [
                    4.190329885612305,
                    6.192429663690274,
                    11.418406945420463,
                    16.392507703808064,
                    13.306076204373788,
                    24.058301996624227
                ],
                [
                    7.576404967559343,
                    11.196339225304024,
                    20.64526599682174,
                    29.63878267062455,
                    24.058301996624223,
                    43.499066597147355
                ],
            ]
        );

        assert!(
            beams.qp.fe_c.subcols(0, 2).transpose() ~
            mat![
                [
                    0.10234015755301376,
                    0.15123731179112526,
                    0.27887101915557216,
                    0.4003531623743676,
                    0.32497344417766216,
                    0.5875743638338231
                ],
                [
                    0.1330671495178759,
                    0.19697673529625795,
                    0.3533112877630318,
                    0.513335151687895,
                    0.45018629955393274,
                    0.7288106297098473,
                ]
            ]
        );

        assert!(
            beams.qp.fi.subcols(0, 2).transpose() ~
            mat![
                [
                    0.0043751995416213951,
                    -0.0069967574749430095,
                    0.001685428032356653,
                    -0.008830739650908432,
                    -0.013790343428970871,
                    -0.029753242214998223
                ],
                [
                    0.02268836256808221,
                    -0.029926522347008034,
                    0.013688107068036134,
                    -0.038866587003176072,
                    -0.060824965744675892,
                    -0.12868857402180306
                ]
            ]
        );

        assert!(
            beams.qp.fe_d.subcols(0, 2).transpose() ~
            mat![
                [
                    0.,
                    0.,
                    0.,
                    0.12083059685899902,
                    0.24111122420708941,
                    -0.17510186558425117,
                ],
                [
                    0.,
                    0.,
                    0.,
                    0.10453073708428925,
                    0.33031057987442675,
                    -0.22352273933363573,
                ]
            ]
        );

        assert!(
            beams.qp.fg.subcols(0, 2).transpose() ~
            mat![
                [0., 0., 19.62, 3.330696665493577, -2.2538354951632567, 0.],
                [0., 0., 19.62, 3.3957558632056069, -2.2939945293324624, 0.]
            ]
        );

        assert!(
            beams.qp.oe.col(0).reshape(6, 6) ~
            mat![
                [
                    0.,
                    0.,
                    0.,
                    1.558035187754702,
                    3.3878498808227686,
                    -2.4090666622503765
                ],
                [
                    0.,
                    0.,
                    0.,
                    2.023578567654383,
                    4.594419401889352,
                    -3.234258589323782
                ],
                [
                    0.,
                    0.,
                    0.,
                    4.396793221398986,
                    8.36944769597985,
                    -6.152454589644055
                ],
                [
                    0.,
                    0.,
                    0.,
                    6.095010301161761,
                    12.74985307030108,
                    -9.157568020649526
                ],
                [
                    0.,
                    0.,
                    0.,
                    4.359848751597229,
                    9.872327664027363,
                    -6.7692134860262945
                ],
                [
                    0.,
                    0.,
                    0.,
                    9.270255102567303,
                    17.44949503500229,
                    -12.963070176574703
                ],
            ]
        );

        assert!(
            beams.qp.pe.col(0).reshape(6, 6) ~
            mat![
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [
                    1.558035187754702,
                    2.0235785676543827,
                    4.396793221398986,
                    6.095010301161762,
                    4.947423115431053,
                    8.945281658389643
                ],
                [
                    3.3878498808227677,
                    4.594419401889353,
                    8.36944769597985,
                    12.162278706467262,
                    9.872327664027365,
                    17.849848197376673
                ],
                [
                    -2.4090666622503765,
                    -3.234258589323782,
                    -6.152454589644055,
                    -8.832594576471866,
                    -7.169566648400663,
                    -12.963070176574703
                ],
            ]
        );

        assert!(
            beams.qp.qe.col(0).reshape(6, 6) ~
            mat![
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [
                    0.,
                    0.,
                    0.,
                    1.8447536136896567,
                    3.635630275656868,
                    -2.64866290970237,
                ],
                [
                    0.,
                    0.,
                    0.,
                    3.810732141241119,
                    7.183324613638193,
                    -5.294123557503817,
                ],
                [
                    0.,
                    0.,
                    0.,
                    -2.4075516854952808,
                    -5.414954154362818,
                    3.820161491912928,
                ],
            ]
        );

        assert!(
            beams.qp.gi.col(0).reshape(6, 6) ~
            mat![
                [
                    0.,
                    0.,
                    0.,
                    0.0008012182534494834,
                    0.0008639454977572939,
                    0.0015930550378149653
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.0005697434834375273,
                    -0.0006253629923483913,
                    -0.0015525180895013208
                ],
                [
                    0.,
                    0.,
                    0.,
                    0.0000149519679953059,
                    0.000022095876141539064,
                    0.00025734175971377
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.009044140792420115,
                    -0.016755394335528064,
                    -0.022806270184157214
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.005674132164451402,
                    -0.013394960837037522,
                    -0.025943451454082944
                ],
                [
                    0.,
                    0.,
                    0.,
                    0.006396216051163168,
                    0.013413253109011812,
                    0.022439101629457635
                ],
            ]
        );

        assert!(
            beams.qp.ki.col(0).reshape(6, 6) ~
            mat![
                [
                    0.,
                    0.,
                    0.,
                    -0.0023904728226588536,
                    0.0005658527664274542,
                    0.0005703830914904407
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.0008599439459226316,
                    -0.000971811812092634,
                    0.0008426153626567674
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.0015972403418206974,
                    0.0015555222717217175,
                    -0.000257435063678694
                ],
                [
                    0.,
                    0.,
                    0.,
                    0.004762288305421506,
                    -0.016524233223710137,
                    0.007213755243428677
                ],
                [
                    0.,
                    0.,
                    0.,
                    0.035164381478288514,
                    0.017626317482204206,
                    -0.022463736936512112
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.0025828596476940593,
                    0.04278211835291491,
                    -0.022253736971069835
                ],
            ]
        );
    }

    fn setup_test(h: f64) -> Beams {
        let fz = |t: f64| -> f64 { t - 2. * t * t };
        let fy = |t: f64| -> f64 { -2. * t + 3. * t * t };
        let fx = |t: f64| -> f64 { 5. * t };
        // let ft = |t: f64| -> f64 { 0. * t * t };

        // Displacement
        let scale = 0.1;
        let ux = |t: f64| -> f64 { scale * t * t };
        let uy = |t: f64| -> f64 { scale * (t * t * t - t * t) };
        let uz = |t: f64| -> f64 { scale * (t * t + 0.2 * t * t * t) };
        let rot = |t: f64| -> Mat<f64> {
            mat![
                [1., 0., 0.,],
                [0., (scale * t).cos(), -(scale * t).sin(),],
                [0., (scale * t).sin(), (scale * t).cos(),]
            ]
        };

        // Velocities
        let vx = |s: f64| -> f64 { scale * (s) };
        let vy = |s: f64| -> f64 { scale * (s * s - s) };
        let vz = |s: f64| -> f64 { scale * (s * s + 0.2 * s * s * s) };
        let omega_x = |s: f64| -> f64 { scale * (s) };
        let omega_y = |s: f64| -> f64 { scale * (s * s - s) };
        let omega_z = |s: f64| -> f64 { scale * (s * s + 0.2 * s * s * s) };

        // Accelerations
        let ax = |s: f64| -> f64 { scale * (s) };
        let ay = |s: f64| -> f64 { scale * (2. * s * s - s) };
        let az = |s: f64| -> f64 { scale * (2. * s * s + 0.2 * s * s * s) };
        let omega_dot_x = |s: f64| -> f64 { scale * (s) };
        let omega_dot_y = |s: f64| -> f64 { scale * (s * s - s) };
        let omega_dot_z = |s: f64| -> f64 { scale * (s * s - s) };

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let xi = gauss_legendre_lobotto_points(4);
        let node_s = xi.iter().map(|&xi| (xi + 1.) / 2.).collect_vec();

        let r0 = mat![
            [
                0.9778215200524469, // node 0
                -0.01733607539094763,
                -0.09001900002195001,
                -0.18831121859148398
            ],
            [
                0.9950113028068008, // node 1
                -0.002883848832932071,
                -0.030192109815745303,
                -0.09504013471947484
            ],
            [
                0.9904718430204884, // node 2
                -0.009526411091536478,
                0.09620741150793366,
                0.09807604012323785
            ],
            [
                0.9472312341234699, // node 3
                -0.04969214162931507,
                0.18127630174800594,
                0.25965858850765167
            ],
            [
                0.9210746582719719, // node 4
                -0.07193653093139739,
                0.20507529985516368,
                0.32309554437664584
            ],
        ];

        let mut model = Model::new();

        let node_ids = node_s
            .iter()
            .enumerate()
            .map(|(i, &si)| {
                let mut r = Col::<f64>::zeros(4);
                quat_from_rotation_matrix(rot(si).as_ref(), r.as_mut());
                model
                    .add_node()
                    .element_location(si)
                    .position(
                        fx(si),
                        fy(si),
                        fz(si),
                        r0[(i, 0)],
                        r0[(i, 1)],
                        r0[(i, 2)],
                        r0[(i, 3)],
                    )
                    .displacement(ux(si), uy(si), uz(si), r[0], r[1], r[2], r[3])
                    .velocity(
                        vx(si),
                        vy(si),
                        vz(si),
                        omega_x(si),
                        omega_y(si),
                        omega_z(si),
                    )
                    .acceleration(
                        ax(si),
                        ay(si),
                        az(si),
                        omega_dot_x(si),
                        omega_dot_y(si),
                        omega_dot_z(si),
                    )
                    .build()
            })
            .collect_vec();

        // Construct mass matrix
        let m = 2.0;
        let eta_star = col![0.1, 0.2, 0.3];
        let mut eta_star_tilde = Mat::<f64>::zeros(3, 3);
        let mut m_star = Mat::<f64>::zeros(6, 6);
        vec_tilde(eta_star.as_ref(), eta_star_tilde.as_mut());
        m_star
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(Mat::<f64>::identity(3, 3) * m);
        m_star
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(m * eta_star_tilde.transpose());
        m_star
            .submatrix_mut(3, 0, 3, 3)
            .copy_from(m * eta_star_tilde);
        m_star
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(Mat::from_fn(3, 3, |i, j| ((i + 1) * (j + 1)) as f64));

        let c_star = Mat::from_fn(6, 6, |i, j| ((i + 1) * (j + 1)) as f64);

        // Create quadrature points and weights
        let gq = Quadrature::gauss(7);

        model.set_gravity(0., 0., 9.81);

        model.add_beam_element(
            &node_ids,
            &gq,
            &[
                BeamSection {
                    s: 0.,
                    m_star: m_star.clone(),
                    c_star: c_star.clone(),
                },
                BeamSection {
                    s: 1.,
                    m_star: m_star.clone(),
                    c_star: c_star.clone(),
                },
            ],
            &Damping::None,
        );
        let nfm = model.create_node_freedom_map();
        let mut elements = model.create_elements(&nfm);
        let state = model.create_state();
        elements.beams.calculate_system(&state, h);

        elements.beams
    }

    #[test]
    fn test_setup_test() {
        let approx_eq = CwiseMat(ApproxEq::eps() * 100.);
        //Only matters for viscoelastic material, but needs to be passed to create_beams
        let h = 0.001;

        let beams = setup_test(h);

        assert!(
            beams.qp.m_star.col(0).reshape(6, 6) ~
            mat![
                [2., 0., 0., 0., 0.6, -0.4], // column 1
                [0., 2., 0., -0.6, 0., 0.2], // column 2
                [0., 0., 2., 0.4, -0.2, 0.], // column 3
                [0., -0.6, 0.4, 1., 2., 3.], // column 4
                [0.6, 0., -0.2, 2., 4., 6.], // column 5
                [-0.4, 0.2, 0., 3., 6., 9.], // column 6
            ]
        );

        assert!(
            beams.qp.c_star.col(0).reshape(6, 6) ~
            mat![
                [1., 2., 3., 4., 5., 6.],      // column 1
                [2., 4., 6., 8., 10., 12.],    // column 2
                [3., 6., 9., 12., 15., 18.],   // column 3
                [4., 8., 12., 16., 20., 24.],  // column 4
                [5., 10., 15., 20., 25., 30.], // column 5
                [6., 12., 18., 24., 30., 36.], // column 6
            ]
        );

        assert!(
            beams.qp.u.col(0) ~
            col![
                6.475011465280995e-5,
                -6.310248039744534e-5,
                6.5079641503883e-5,
                0.9999991906236807,
                0.0012723018445566566,
                0.0,
                0.0
            ]
        );

        assert!(
            beams.qp.u_prime.col(0) ~
            col![
                0.0009414876868372848,
                -0.0009055519814222241,
                0.0009486748279202956,
                -0.000011768592508141705,
                0.009249835939573446,
                0.0,
                0.0
            ]
        );

        let mut rr0 = Col::<f64>::zeros(4);
        quat_compose(
            beams.qp.u.col(0).subrows(3, 4),
            beams.qp.x0.col(0).subrows(3, 4),
            rr0.as_mut(),
        );
        assert!(
            rr0 ~
            col![
                0.9809020325848156,
                -0.013224334332128542,
                -0.08222076069525554,
                -0.17577276797940944
            ]
        );

        assert!(
            beams.qp.strain.col(0) ~
            col![
                0.0009414876868371058,
                -0.0004838292834870028,
                0.0018188281296873665,
                0.018499686852354473,
                0.0,
                0.0
            ]
        );

        assert!(
            beams.qp.cuu.col(0).reshape(6, 6) ~
            mat![
                [
                    1.3196125048858467,
                    1.9501108129670985,
                    3.5958678677753957,
                    5.162304339488006,
                    4.190329885612305,
                    7.576404967559343
                ],
                [
                    1.9501108129670985,
                    2.881855217930184,
                    5.3139393458205735,
                    7.628804270184899,
                    6.192429663690274,
                    11.196339225304024
                ],
                [
                    3.5958678677753957,
                    5.313939345820573,
                    9.79853227718398,
                    14.066981200400342,
                    11.418406945420463,
                    20.64526599682174
                ],
                [
                    5.1623043394880055,
                    7.628804270184899,
                    14.066981200400345,
                    20.194857198478893,
                    16.392507703808064,
                    29.63878267062455
                ],
                [
                    4.190329885612304,
                    6.1924296636902705,
                    11.418406945420463,
                    16.392507703808057,
                    13.306076204373786,
                    24.058301996624223
                ],
                [
                    7.576404967559343,
                    11.196339225304024,
                    20.64526599682174,
                    29.638782670624547,
                    24.058301996624227,
                    43.499066597147355
                ]
            ]
            .transpose()
        );

        assert!(
            beams.qp.fe_c.col(0) ~
            col![
                0.1023401575530157,
                0.15123731179112812,
                0.2788710191555775,
                0.40035316237437524,
                0.3249734441776684,
                0.5875743638338343
            ]
        );

        assert!(
            beams.qp.fe_d.col(0) ~
            col![
                0.0,
                0.0,
                0.0,
                0.12083059685900131,
                0.24111122420709402,
                -0.1751018655842545
            ]
        );

        assert!(
            beams.qp.fi.col(0) ~
            col![
                0.004375199541621397,
                -0.006996757474943007,
                0.0016854280323566574,
                -0.008830739650908434,
                -0.01379034342897087,
                -0.02975324221499824
            ]
        );

        assert!(
            beams.node_fe.col(0) ~
            col![
                -0.11121183449279251,
                -0.1614948289968797,
                -0.30437442031624906,
                -0.4038524317172822,
                -0.29275354335734394,
                -0.6838427114868927
            ]
        );

        assert!(
            beams.node_fi.col(0) ~
            col![
                0.0001160455640893056,
                -0.0006507362696177845,
                -0.0006134866787566601,
                0.0006142322011934131,
                -0.002199479688149198,
                -0.002486843354672648,
            ]
        );

        assert!(
            beams.node_fg.col(0) ~
            col![
                0.0,
                0.0,
                5.387595382846479,
                0.9155947038768218,
                -0.6120658127519634,
                0.0,
            ]
        );

        assert!(
            beams.node_f.col(0) ~
            col![
                -0.1110957889287032,
                -0.16214556526649748,
                -5.692583289841486,
                -1.3188329033929111,
                0.3171127897064705,
                -0.6863295548415653
            ]

        );
    }
}

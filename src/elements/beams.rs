use std::ops::Rem;

use crate::elements::beam_qps::BeamQPs;
use crate::interp::{shape_deriv_matrix, shape_interp_matrix};
use crate::node::{Node, NodeFreedomMap};
use crate::quadrature::Quadrature;
use crate::state::State;
use crate::util::ColAsMatRef;
use faer::linalg::matmul::matmul;
use faer::{unzipped, zipped, Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism, Scale};
use itertools::{izip, multiunzip, Itertools};

pub struct BeamElement {
    pub id: usize,
    pub node_ids: Vec<usize>,
    pub sections: Vec<BeamSection>,
    pub quadrature: Quadrature,
    pub damping: Damping,
}

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
}

#[derive(Clone, Copy, Debug)]
pub struct ElemIndex {
    elem_id: usize,
    n_nodes: usize,
    n_qps: usize,
    i_node_start: usize,
    i_qp_start: usize,
    i_mat_start: usize,
}

pub struct Beams {
    enable_damping: bool,
    elem_index: Vec<ElemIndex>,
    pub node_ids: Vec<usize>,
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
}

impl Beams {
    pub fn new(
        elements: &[BeamElement],
        gravity: &[f64; 3],
        nodes: &[Node],
        enable_damping: bool,
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
        let mut index: Vec<ElemIndex> = vec![];
        let mut start_node = 0;
        let mut start_qp = 0;
        let mut start_mat = 0;
        for (i, e) in elements.iter().enumerate() {
            let n_nodes = e.node_ids.len();
            let n_qps = e.quadrature.points.len();
            index.push(ElemIndex {
                elem_id: i,
                n_nodes,
                i_node_start: start_node,
                n_qps,
                i_qp_start: start_qp,
                i_mat_start: start_mat,
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

        let mut beams = Self {
            enable_damping,
            elem_index: index,
            node_ids: elements
                .iter()
                .flat_map(|e| e.node_ids.to_owned())
                .collect_vec(),
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

            qp: BeamQPs::new(&qp_weights),

            shape_interp: Mat::zeros(alloc_qps, max_elem_nodes),
            shape_deriv: Mat::zeros(alloc_qps, max_elem_nodes),
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

        for (i, ei) in beams.elem_index.iter().enumerate() {
            // Get shape derivative matrix for this element
            let shape_interp = beams
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);

            // Interpolate initial position
            let node_x0 = beams.node_x0.subcols(ei.i_node_start, ei.n_nodes);
            let mut qp_x0 = beams.qp.x0.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_x0.as_mut().transpose_mut(),
                shape_interp,
                node_x0.transpose(),
                None,
                1.,
                Parallelism::None,
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
                shape_deriv,
                node_x0.transpose(),
                None,
                1.0,
                Parallelism::None,
            );
            izip!(qp_jacobian.iter_mut(), qp_x0_prime.col_iter_mut()).for_each(
                |(j, mut x0_prime)| {
                    *j = x0_prime.as_mut().subrows(0, 3).norm_l2();
                    x0_prime /= *j;
                },
            );

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

            // Interpolate mass and stiffness matrices
            let qp_s = Col::<f64>::from_fn(ei.n_qps, |i| {
                (elements[ei.elem_id].quadrature.points[i] + 1.) / 2.
            });

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

            // Populate damping
            let qp_mu = beams.qp.mu.subcols_mut(ei.i_qp_start, ei.n_qps);
            match &elements[i].damping {
                Damping::None => (),
                Damping::Mu(mu) => {
                    qp_mu.col_iter_mut().for_each(|mut c| c.copy_from(mu));
                }
            }
        }

        beams
    }

    /// Calculate element properties
    pub fn calculate_system(&mut self, state: &State) {
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

        // Calculate quadrature point damping values
        if self.enable_damping {
            self.qp.calc_bauchau_damping();
        }

        // Integrate forces
        self.integrate_forces();

        // Integrate matrices
        self.integrate_matrices();
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &self,
        nfm: &NodeFreedomMap,
        mut m: MatMut<f64>, // Mass
        mut g: MatMut<f64>, // Damping
        mut k: MatMut<f64>, // Stiffness
        mut r: ColMut<f64>, // Residual
    ) {
        // Loop through elements
        self.elem_index.iter().for_each(|ei| {
            // Get slice of node ids for this element
            let node_ids = &self.node_ids[ei.i_node_start..ei.i_node_start + ei.n_nodes];

            // Get starting degree of freedom for each node
            let elem_dof_start_pairs = node_ids
                .iter()
                .cartesian_product(node_ids)
                .map(|(&i, &j)| {
                    (
                        nfm.node_dofs[i].first_dof_index,
                        nfm.node_dofs[j].first_dof_index,
                    )
                })
                .collect_vec();

            // Mass matrix
            izip!(
                elem_dof_start_pairs.iter(),
                self.node_muu
                    .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                    .col_iter()
            )
            .for_each(|((i, j), muu)| {
                let mut me = m.as_mut().submatrix_mut(*i, *j, 6, 6);
                zipped!(&mut me, &muu.as_mat_ref(6, 6))
                    .for_each(|unzipped!(mut me, muu)| *me += *muu);
            });

            // Damping matrix
            izip!(
                elem_dof_start_pairs.iter(),
                self.node_guu
                    .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                    .col_iter(),
            )
            .for_each(|((i, j), guu)| {
                let mut ge = g.as_mut().submatrix_mut(*i, *j, 6, 6);
                zipped!(&mut ge, &guu.as_mat_ref(6, 6))
                    .for_each(|unzipped!(mut ge, guu)| *ge += *guu);
            });

            // Stiffness matrix
            izip!(
                elem_dof_start_pairs.iter(),
                self.node_kuu
                    .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes)
                    .col_iter()
            )
            .for_each(|((i, j), kuu)| {
                let mut ke = k.as_mut().submatrix_mut(*i, *j, 6, 6);
                zipped!(&mut ke, &kuu.as_mat_ref(6, 6))
                    .for_each(|unzipped!(mut ke, kuu)| *ke += *kuu);
            });

            // Get first node dof index of each element
            let elem_first_dof_indices = self
                .node_ids
                .iter()
                .map(|&node_id| nfm.node_dofs[node_id].first_dof_index)
                .collect_vec();

            // Residual vector
            izip!(
                elem_first_dof_indices.iter(),
                self.node_f.subcols(ei.i_node_start, ei.n_nodes).col_iter()
            )
            .for_each(|(&i, f)| {
                let mut residual = r.as_mut().subrows_mut(i, 6);
                zipped!(&mut residual, &f).for_each(|unzipped!(mut r, f)| *r += *f);
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
                shape_interp,
                node_u.transpose(),
                None,
                1.,
                Parallelism::None,
            );
            u.as_mut()
                .subrows_mut(3, 4)
                .col_iter_mut()
                .for_each(|mut c| {
                    let m = c.norm_l2();
                    if m != 0. {
                        c /= m;
                    }
                });

            // Displacement derivative
            let mut u_prime = self.qp.u_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                u_prime.as_mut().transpose_mut(),
                shape_deriv,
                node_u.transpose(),
                None,
                1.0,
                Parallelism::None,
            );

            // Interpolate velocity
            let mut v = self.qp.v.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                v.as_mut().transpose_mut(),
                shape_interp,
                node_v.transpose(),
                None,
                1.0,
                Parallelism::None,
            );

            // Velocity derivative
            let mut qp_v_prime = self.qp.v_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_v_prime.as_mut().transpose_mut(),
                shape_deriv,
                node_v.transpose(),
                None,
                1.0,
                Parallelism::None,
            );

            // Interpolate acceleration
            let mut vd = self.qp.vd.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                vd.as_mut().transpose_mut(),
                shape_interp,
                node_vd.transpose(),
                None,
                1.0,
                Parallelism::None,
            );
        }

        // Divide each column of u_prime by Jacobian
        izip!(self.qp.u_prime.col_iter_mut(), self.qp.jacobian.iter()).for_each(
            |(mut col, &jacobian)| {
                zipped!(&mut col).for_each(|unzipped!(mut col)| *col /= jacobian)
            },
        );

        // Divide each column of v_prime by Jacobian
        izip!(self.qp.v_prime.col_iter_mut(), self.qp.jacobian.iter()).for_each(
            |(mut col, &jacobian)| {
                zipped!(&mut col).for_each(|unzipped!(mut col)| *col /= jacobian)
            },
        );
    }

    #[inline]
    fn integrate_forces(&mut self) {
        // Zero matrices
        self.node_fe.fill_zero();
        self.node_fi.fill_zero();
        self.node_fx.fill_zero();
        self.node_fg.fill_zero();

        // Loop through elements
        for ei in self.elem_index.iter() {
            let shape_interp = self
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let shape_deriv = self
                .shape_deriv
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let qp_w = self.qp.weight.subrows(ei.i_qp_start, ei.n_qps);
            let qp_j = self.qp.jacobian.subrows(ei.i_qp_start, ei.n_qps);

            // Elastic forces
            integrate_fe(
                self.node_fe.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fe_c.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fe_d.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                shape_deriv,
                qp_w,
                qp_j,
            );

            // Dissipative forces
            integrate_fe(
                self.node_fe.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fd_c.subcols(ei.i_qp_start, ei.n_qps),
                self.qp.fd_d.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                shape_deriv,
                qp_w,
                qp_j,
            );

            // Inertial forces
            integrate_f(
                self.node_fi.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fi.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                qp_w,
                qp_j,
            );

            // External (distributed) forces
            integrate_f(
                self.node_fx.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fx.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                qp_w,
                qp_j,
            );

            // Gravity forces
            integrate_f(
                self.node_fg.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp.fg.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                qp_w,
                qp_j,
            );
        }

        // Combine force components
        zipped!(
            &mut self.node_f,
            &self.node_fe,
            &self.node_fg,
            &self.node_fi,
            &self.node_fx
        )
        .for_each(|unzipped!(mut f, fe, fg, fi, fx)| *f = *fi + *fe - *fx - *fg);
    }

    #[inline]
    fn integrate_matrices(&mut self) {
        self.node_muu.fill_zero();
        self.node_guu.fill_zero();
        self.node_kuu.fill_zero();

        // Loop through elements
        for ei in self.elem_index.iter() {
            let node_ij = (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .collect_vec();

            integrate_element_matrices(
                &node_ij,
                self.node_muu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.node_guu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.node_kuu
                    .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes),
                self.shape_interp
                    .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes), // columns are phi
                self.shape_deriv
                    .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes), // columns are phi_prime
                self.qp.weight.subrows(ei.i_qp_start, ei.n_qps),
                self.qp.jacobian.subrows(ei.i_qp_start, ei.n_qps),
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
            );
        }
    }
}

#[inline]
fn integrate_fe(
    node_fe: MatMut<f64>,
    qp_fc: MatRef<f64>,
    qp_fd: MatRef<f64>,
    shape_interp: MatRef<f64>,
    shape_deriv: MatRef<f64>,
    qp_w: ColRef<f64>,
    qp_j: ColRef<f64>,
) {
    let mut acc = Col::<f64>::zeros(6);
    izip!(
        node_fe.col_iter_mut(),
        shape_interp.col_iter(),
        shape_deriv.col_iter()
    )
    .for_each(|(mut fe, phi, phi_prime)| {
        acc.fill_zero();
        izip!(
            qp_w.iter(),
            qp_j.iter(),
            phi.iter(),
            phi_prime.iter(),
            qp_fc.col_iter(),
            qp_fd.col_iter()
        )
        .for_each(|(&w, &j, &phi, &phi_prime, fc, fd)| {
            zipped!(&mut acc, &fc, &fd).for_each(|unzipped!(mut acc, fc, fd)| {
                *acc += w * (*fc * phi_prime + *fd * phi * j)
            });
        });

        // Add values to node matrix
        zipped!(&mut fe, &acc).for_each(|unzipped!(mut fe, acc)| *fe += *acc);
    });
}

#[inline]
fn integrate_f(
    node_f: MatMut<f64>,
    qp_f: MatRef<f64>,
    shape_interp: MatRef<f64>,
    qp_w: ColRef<f64>,
    qp_j: ColRef<f64>,
) {
    let mut acc = Col::<f64>::zeros(6);
    izip!(node_f.col_iter_mut(), shape_interp.col_iter()).for_each(|(mut node_f, phi)| {
        acc.fill_zero();
        izip!(qp_w.iter(), qp_j.iter(), phi.iter(), qp_f.col_iter(),).for_each(
            |(&w, &j, &phi, qp_f)| {
                zipped!(&mut acc, &qp_f).for_each(|unzipped!(mut acc, f)| *acc += *f * phi * j * w);
            },
        );

        // Add values to node matrix
        zipped!(&mut node_f, &acc).for_each(|unzipped!(mut f, acc)| *f += *acc);
    });
}

fn integrate_element_matrices(
    node_ij: &[(usize, usize)],
    node_m: MatMut<f64>,
    node_g: MatMut<f64>,
    node_k: MatMut<f64>,
    phi: MatRef<f64>,
    phi_prime: MatRef<f64>,
    weight: ColRef<f64>,
    jacobian: ColRef<f64>,
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
) {
    let mut acc = Col::<f64>::zeros(6 * 6);
    let mut c = Col::<f64>::zeros(weight.nrows());

    // Mass
    node_ij
        .iter()
        .zip(node_m.col_iter_mut())
        .for_each(|(&(i, j), mut m_col)| {
            // Reset accumulator
            acc.fill_zero();

            // c = w * j * phi_i * phi_j
            zipped!(&mut c, &weight, &jacobian, &phi.col(i), &phi.col(j))
                .for_each(|unzipped!(mut c, w, j, phi_i, phi_j)| *c = *w * *j * *phi_i * *phi_j);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_muu);

            // Copy accumulator to node
            m_col.copy_from(&acc);
        });

    // Gyroscopic
    node_ij
        .iter()
        .zip(node_g.col_iter_mut())
        .for_each(|(&(i, j), mut g_col)| {
            // Reset accumulator
            acc.fill_zero();

            // c = w * j * phi_i * phi_j
            zipped!(&mut c, &weight, &jacobian, &phi.col(i), &phi.col(j))
                .for_each(|unzipped!(mut c, w, j, phi_i, phi_j)| *c = *w * *j * *phi_i * *phi_j);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_gi);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_xd);

            // c = w * phi_i * phi_prime_j
            zipped!(&mut c, &weight, &phi.col(i), &phi_prime.col(j)).for_each(
                |unzipped!(mut c, w, phi_i, phi_prime_j)| *c = *w * *phi_i * *phi_prime_j,
            );
            integrate_mat(acc.as_mut(), c.as_ref(), qp_yd);

            // c = w * phi_prime_i * phi_j
            zipped!(&mut c, &weight, &phi_prime.col(i), &phi.col(j)).for_each(
                |unzipped!(mut c, w, phi_prime_i, phi_j)| *c = *w * *phi_prime_i * *phi_j,
            );
            integrate_mat(acc.as_mut(), c.as_ref(), qp_gd);

            // c = w * phi_prime_i * phi_prime_j / j
            zipped!(
                &mut c,
                &weight,
                &jacobian,
                &phi_prime.col(i),
                &phi_prime.col(j)
            )
            .for_each(|unzipped!(mut c, w, j, phi_prime_i, phi_prime_j)| {
                *c = *w * *phi_prime_i * *phi_prime_j / *j
            });
            integrate_mat(acc.as_mut(), c.as_ref(), qp_mu_cuu);

            // Copy accumulator to node
            g_col.copy_from(&acc);
        });

    // Stiffness
    node_ij
        .iter()
        .zip(node_k.col_iter_mut())
        .for_each(|(&(i, j), mut k_col)| {
            // Reset accumulator
            acc.fill_zero();

            // c = w * j * phi_i * phi_j
            zipped!(&mut c, &weight, &jacobian, &phi.col(i), &phi.col(j))
                .for_each(|unzipped!(mut c, w, j, phi_i, phi_j)| *c = *w * *j * *phi_i * *phi_j);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_ki);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_qe);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_qd);

            // c = w * phi_i * phi_prime_j
            zipped!(&mut c, &weight, &phi.col(i), &phi_prime.col(j)).for_each(
                |unzipped!(mut c, w, phi_i, phi_prime_j)| *c = *w * *phi_i * *phi_prime_j,
            );
            integrate_mat(acc.as_mut(), c.as_ref(), qp_pe);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_pd);

            // c = w * phi_prime_i * phi_j
            zipped!(&mut c, &weight, &phi_prime.col(i), &phi.col(j)).for_each(
                |unzipped!(mut c, w, phi_prime_i, phi_j)| *c = *w * *phi_prime_i * *phi_j,
            );
            integrate_mat(acc.as_mut(), c.as_ref(), qp_oe);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_od);

            // c = w * phi_prime_i * phi_prime_j / j
            zipped!(
                &mut c,
                &weight,
                &jacobian,
                &phi_prime.col(i),
                &phi_prime.col(j)
            )
            .for_each(|unzipped!(mut c, w, j, phi_prime_i, phi_prime_j)| {
                *c = *w * *phi_prime_i * *phi_prime_j / *j
            });
            integrate_mat(acc.as_mut(), c.as_ref(), qp_cuu);
            integrate_mat(acc.as_mut(), c.as_ref(), qp_sd);

            // Copy accumulator to node
            k_col.copy_from(&acc);
        });
}

#[inline]
fn integrate_mat(mut node_mat: ColMut<f64>, c: ColRef<f64>, qp_mat: MatRef<f64>) {
    izip!(qp_mat.col_iter(), c.iter()).for_each(|(qp_mat, &c)| {
        zipped!(&mut node_mat, &qp_mat)
            .for_each(|unzipped!(mut node_mat, qp_mat)| *node_mat += *qp_mat * c)
    });
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        interp::gauss_legendre_lobotto_points, model::Model, quadrature::Quadrature,
        util::vec_tilde, util::Quat,
    };
    use faer::{assert_matrix_eq, col, mat};

    fn create_beams() -> Beams {
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
            Damping::None,
        );

        let mut beams = model.create_beams();
        let state = model.create_state();
        beams.calculate_system(&state);

        beams
    }

    #[test]
    fn test_node_x0() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.node_x0.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_node_u() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.node_u.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_node_v() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.node_v.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_node_vd() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.node_vd.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_m_star() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.m_star.col(0).as_mat_ref(6, 6),
            mat![
                [2., 0., 0., 0., 0.6, -0.4],
                [0., 2., 0., -0.6, 0., 0.2],
                [0., 0., 2., 0.4, -0.2, 0.],
                [0., -0.6, 0.4, 1., 2., 3.],
                [0.6, 0., -0.2, 2., 4., 6.],
                [-0.4, 0.2, 0., 3., 6., 9.],
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_c_star() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.c_star.col(0).as_mat_ref(6, 6),
            mat![
                [1., 2., 3., 4., 5., 6.],
                [2., 4., 6., 8., 10., 12.],
                [3., 6., 9., 12., 15., 18.],
                [4., 8., 12., 16., 20., 24.],
                [5., 10., 15., 20., 25., 30.],
                [6., 12., 18., 24., 30., 36.],
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_x0() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.x0.col(0).as_2d().transpose(),
            mat![[
                0.12723021914310376,
                -0.048949584217657216,
                0.024151041535564563,
                0.98088441332097497,
                -0.014472327094052504,
                -0.082444330164641907,
                -0.17566801608760002
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_x0_prime() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.x0_prime.col(0).as_2d().transpose(),
            mat![[
                0.92498434449987588,
                -0.34174910719483215,
                0.16616711516322963,
                0.023197240723437866,
                0.01993094516117577,
                0.056965007432292485,
                0.09338617439225547
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_u() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.u.col(0).as_2d().transpose(),
            mat![[
                0.000064750114652809492,
                -0.000063102480397445355,
                0.000065079641503882152,
                0.9999991906236807,
                0.0012723018445567188,
                0.,
                0.
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_u_prime() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.u_prime.col(0).as_2d().transpose(),
            mat![[
                0.00094148768683727929,
                -0.00090555198142222483,
                0.00094867482792029139,
                -0.000011768592508690223,
                0.0092498359395732574,
                0.,
                0.
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_x() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.x.col(0).as_2d().transpose(),
            mat![[
                0.12729496925775657,
                -0.049012686698054662,
                0.024216121177068447,
                0.9809020325848155,
                -0.013224334332128548,
                -0.08222076069525557,
                -0.1757727679794095,
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_jacobian() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams
                .qp
                .jacobian
                .subrows(0, beams.elem_index[0].n_qps)
                .as_2d()
                .transpose(),
            mat![[
                2.7027484463552831,
                2.5851972184835246,
                2.5041356900076868,
                2.5980762113533156,
                2.8809584014451262,
                3.223491986410379,
                3.4713669823269537,
            ]],
            comp = float
        );
    }

    #[test]
    fn test_qp_strain() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.strain.subcols(0, 2).transpose(),
            mat![
                [
                    0.0009414876868372797,
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_rr0() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.rr0.col(0).as_mat_ref(6, 6),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_muu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.muu.col(0).as_mat_ref(6, 6),
            mat![
                [
                    2.000000000000001,
                    5.204170427930421e-17,
                    -5.551115123125783e-17,
                    -4.163336342344337e-17,
                    0.626052147258804,
                    -0.3395205571349214
                ],
                [
                    5.204170427930421e-17,
                    2.0000000000000018,
                    1.3877787807814457e-17,
                    -0.6260521472588039,
                    -3.469446951953614e-18,
                    0.22974877626536766
                ],
                [
                    -5.551115123125783e-17,
                    1.3877787807814457e-17,
                    2.0000000000000013,
                    0.33952055713492146,
                    -0.22974877626536772,
                    -1.3877787807814457e-17
                ],
                [
                    4.163336342344337e-17,
                    -0.626052147258804,
                    0.3395205571349214,
                    1.3196125048858467,
                    1.9501108129670985,
                    3.5958678677753957
                ],
                [
                    0.6260521472588039,
                    3.469446951953614e-18,
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_cuu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.cuu.col(0).as_mat_ref(6, 6),
            mat![
                [
                    1.3196125048858467,
                    1.9501108129670985,
                    3.5958678677753957,
                    5.1623043394880055,
                    4.190329885612304,
                    7.576404967559343
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_fc() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.fe_c.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_fi() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.fi.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_fd() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.fe_d.subcols(0, 2).transpose(),
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_fg() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.fg.subcols(0, 2).transpose(),
            mat![
                [0., 0., 19.62, 3.330696665493577, -2.2538354951632567, 0.],
                [0., 0., 19.62, 3.3957558632056069, -2.2939945293324624, 0.]
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_ouu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.oe.col(0).as_mat_ref(6, 6),
            mat![
                [
                    0.,
                    0.,
                    0.,
                    1.558035187754702,
                    3.387860395787122,
                    -2.409072364725116
                ],
                [
                    0.,
                    0.,
                    0.,
                    2.0235680526900297,
                    4.594419401889352,
                    -3.2342547305394884
                ],
                [
                    0.,
                    0.,
                    0.,
                    4.3967989238737255,
                    8.369443837195558,
                    -6.152454589644055
                ],
                [
                    0.,
                    0.,
                    0.,
                    6.095010301161761,
                    12.749875225071804,
                    -9.157580273927733
                ],
                [
                    0.,
                    0.,
                    0.,
                    4.359826596826506,
                    9.872327664027363,
                    -6.7691983905199775
                ],
                [
                    0.,
                    0.,
                    0.,
                    9.27026735584551,
                    17.449479939495973,
                    -12.963070176574703
                ],
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_puu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.pe.col(0).as_mat_ref(6, 6),
            mat![
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [
                    1.558035187754702,
                    2.0235680526900293,
                    4.3967989238737255,
                    6.095010301161762,
                    4.947423115431053,
                    8.945281658389643
                ],
                [
                    3.387860395787121,
                    4.594419401889353,
                    8.369443837195558,
                    12.162278706467262,
                    9.872327664027365,
                    17.849848197376673
                ],
                [
                    -2.409072364725116,
                    -3.2342547305394884,
                    -6.152454589644055,
                    -8.832594576471866,
                    -7.169566648400663,
                    -12.963070176574703
                ],
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_quu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.qe.col(0).as_mat_ref(6, 6),
            mat![
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [
                    0.,
                    0.,
                    0.,
                    1.8447538104526202,
                    3.635628953426451,
                    -2.6486622648385847,
                ],
                [
                    0.,
                    0.,
                    0.,
                    3.8107374213097884,
                    7.183319283473562,
                    -5.294122604530246,
                ],
                [
                    0.,
                    0.,
                    0.,
                    -2.4075419494181616,
                    -5.414957757364345,
                    3.8201598729444077,
                ],
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_guu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.gi.col(0).as_mat_ref(6, 6),
            mat![
                [
                    0.,
                    0.,
                    0.,
                    -0.0008012182534494841,
                    0.002003432464632351,
                    0.0015631511018243545
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.002297634478952118,
                    0.0006253629923483924,
                    -0.0015967098417843995
                ],
                [
                    0.,
                    0.,
                    0.,
                    -0.0031711581076346268,
                    0.0031271320551441812,
                    -0.0002573417597137699
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
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_kuu() {
        let beams = create_beams();
        assert_matrix_eq!(
            beams.qp.ki.col(0).as_mat_ref(6, 6),
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
            ],
            comp = float
        );
    }

    fn setup_test() -> Beams {
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
                r.as_mut().quat_from_rotation_matrix(rot(si).as_ref());
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
            Damping::None,
        );

        let mut beams = model.create_beams();

        let state = model.create_state();

        beams.calculate_system(&state);

        beams
    }

    #[test]
    fn test_qp_m_star2() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.m_star.col(0).as_mat_ref(6, 6),
            mat![
                [2., 0., 0., 0., 0.6, -0.4], // column 1
                [0., 2., 0., -0.6, 0., 0.2], // column 2
                [0., 0., 2., 0.4, -0.2, 0.], // column 3
                [0., -0.6, 0.4, 1., 2., 3.], // column 4
                [0.6, 0., -0.2, 2., 4., 6.], // column 5
                [-0.4, 0.2, 0., 3., 6., 9.], // column 6
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_c_star2() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.c_star.col(0).as_mat_ref(6, 6),
            mat![
                [1., 2., 3., 4., 5., 6.],      // column 1
                [2., 4., 6., 8., 10., 12.],    // column 2
                [3., 6., 9., 12., 15., 18.],   // column 3
                [4., 8., 12., 16., 20., 24.],  // column 4
                [5., 10., 15., 20., 25., 30.], // column 5
                [6., 12., 18., 24., 30., 36.], // column 6
            ],
            comp = float
        );
    }

    #[test]
    fn test_qp_u2() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.u.col(0).as_2d(),
            col![
                6.475011465280995e-5,
                -6.310248039744534e-5,
                6.5079641503883e-5,
                0.9999991906236807,
                0.001272301844556629,
                0.0,
                0.0
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_u_prime2() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.u_prime.col(0).as_2d(),
            col![
                0.0009414876868372848,
                -0.0009055519814222241,
                0.0009486748279202956,
                -1.1768592508490864e-5,
                0.009249835939573259,
                0.0,
                0.0
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_rr03() {
        let beams = setup_test();
        let mut rr0 = Col::<f64>::zeros(4);
        rr0.as_mut().quat_compose(
            beams.qp.u.col(0).subrows(3, 4),
            beams.qp.x0.col(0).subrows(3, 4),
        );
        assert_eq!(rr0.norm_l2(), 1.);
        assert_matrix_eq!(
            rr0.as_2d(),
            col![
                0.9809020325848156,
                -0.013224334332128542,
                -0.08222076069525554,
                -0.17577276797940944
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_strain3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.strain.col(0).as_2d(),
            col![
                0.0009414876868373279,
                -0.00048382928348705834,
                0.0018188281296873943,
                0.0184996868523541,
                0.0,
                0.0
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_cuu3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.cuu.col(0).as_mat_ref(6, 6),
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
                    11.196339225304033
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
                    13.3060762043738,
                    24.058301996624223
                ],
                [
                    7.576404967559343,
                    11.196339225304031,
                    20.64526599682174,
                    29.638782670624547,
                    24.058301996624227,
                    43.499066597147355
                ]
            ]
            .transpose(),
            comp = float
        );
    }

    #[test]
    fn test_qp_fc3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.fe_c.col(0).as_2d(),
            col![
                0.1023401575530157,
                0.15123731179112812,
                0.2788710191555775,
                0.40035316237437524,
                0.3249734441776684,
                0.5875743638338343
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_fd3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.fe_d.col(0).as_2d(),
            col![
                0.0,
                0.0,
                0.0,
                0.12083059685900131,
                0.24111122420709402,
                -0.1751018655842545
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_qp_fi3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp.fi.col(0).as_2d(),
            col![
                0.004375199541621397,
                -0.006996757474943007,
                0.0016854280323566574,
                -0.008830739650908434,
                -0.01379034342897087,
                -0.02975324221499824
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_node_fe3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.node_fe.col(0).as_2d(),
            col![
                -0.11121183449279251,
                -0.1614948289968797,
                -0.30437442031624906,
                -0.4038524317172822,
                -0.29275354335734394,
                -0.6838427114868927
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_node_fi3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.node_fi.col(0).as_2d(),
            col![
                0.0001160455640892761,
                -0.0006507362696178125,
                -0.0006134866787567512,
                0.0006142322011934131,
                -0.002199479688149198,
                -0.002486843354672648,
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_node_fg3() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.node_fg.col(0).as_2d(),
            col![
                0.0,
                0.0,
                5.387595382846484,
                0.9155947038768231,
                -0.6120658127519644,
                0.0,
            ]
            .as_2d(),
            comp = float
        );
    }

    #[test]
    fn test_node_f() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.node_f.col(0).as_2d(),
            col![
                -0.1110957889287032,
                -0.16214556526649748,
                -5.692583289841486,
                -1.3188329033929111,
                0.3171127897064705,
                -0.6863295548415653
            ]
            .as_2d(),
            comp = float
        );
    }
}

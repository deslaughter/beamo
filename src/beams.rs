use std::ops::Rem;
use std::rc::Rc;

use crate::interp::shape_interp_matrix;
use crate::quadrature::Quadrature;
use crate::quaternion::Quat;
use crate::{interp::shape_deriv_matrix, node::Node};
use faer::col::ColBatch;
use faer::linalg::matmul;
use faer::{col, mat, Col, ColRef, Mat, MatMut, MatRef, Parallelism, Row, Scale};
use itertools::{izip, multiunzip, Itertools};

pub struct BeamInput {
    gravity: [f64; 3],
    elements: Vec<BeamElement>,
}

pub struct BeamElement {
    nodes: Vec<BeamNode>,
    sections: Vec<BeamSection>,
    quadrature: Quadrature,
}

pub struct BeamNode {
    si: f64,
    node: Rc<Node>,
}

pub struct BeamSection {
    pub s: f64,           // Distance along centerline from first point
    pub m_star: Mat<f64>, // [6][6] Mass matrix
    pub c_star: Mat<f64>, // [6][6] Stiffness matrix
}

pub struct ElemIndex {
    elem_id: usize,
    n_nodes: usize,
    n_qps: usize,
    node_range: [usize; 2],
    qp_range: [usize; 2],
}

pub struct Beams {
    index: Vec<ElemIndex>,
    node_ids: Vec<usize>,
    gravity: Row<f64>, // [3] gravity

    // Node-based data
    node_x0: Mat<f64>, // [n_nodes][7] Inital position/rotation
    node_u: Mat<f64>,  // [n_nodes][7] State: translation/rotation displacement
    node_v: Mat<f64>,  // [n_nodes][6] State: translation/rotation velocity
    node_vd: Mat<f64>, // [n_nodes][6] State: translation/rotation acceleration
    node_fx: Mat<f64>, // [n_nodes][6] External forces

    // Quadrature point data
    qp_weight: Col<f64>,   // [n_qps]        Integration weights
    qp_jacobian: Col<f64>, // [n_qps]        Jacobian vector
    qp_m_star: Mat<f64>,   // [6][6][n_qps]  Mass matrix in material frame
    qp_c_star: Mat<f64>,   // [6][6][n_qps]  Stiffness matrix in material frame
    qp_x: Mat<f64>,        // [7][n_qps]     Current position/orientation
    qp_x0: Mat<f64>,       // [7][n_qps]     Initial position
    qp_x0_prime: Mat<f64>, // [7][n_qps]     Initial position derivative
    qp_u: Mat<f64>,        // [7][n_qps]     State: translation displacement
    qp_u_prime: Mat<f64>,  // [7][n_qps]     State: translation displacement derivative
    qp_v: Mat<f64>,        // [6][n_qps]     State: translation velocity
    qp_vd: Mat<f64>,       // [6][n_qps]     State: translation acceleration
    qp_e: Mat<f64>,        // [3][4][n_qps]  Quaternion derivative
    qp_x0pupss: Mat<f64>,  // [3][3][n_qps] skew_symmetric(x0_prime + u_prime)
    qp_eta: Mat<f64>,      // [3][n_qps]     mass
    qp_rho: Mat<f64>,      // [3][3][n_qps]  mass
    qp_strain: Mat<f64>,   // [6][n_qps]     Strain
    qp_fc: Mat<f64>,       // [6][n_qps]     Elastic force
    qp_fd: Mat<f64>,       // [6][n_qps]     Elastic force
    qp_fi: Mat<f64>,       // [6][n_qps]     Inertial force
    qp_fx: Mat<f64>,       // [6][n_qps]     External force
    qp_fg: Mat<f64>,       // [6][n_qps]     Gravity force
    qp_rr0: Mat<f64>,      // [6][6][n_qps]  Global rotation
    qp_muu: Mat<f64>,      // [6][6][n_qps]  Mass in global frame
    qp_cuu: Mat<f64>,      // [6][6][n_qps]  Stiffness in global frame
    qp_ouu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    qp_puu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    qp_quu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    qp_guu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    qp_kuu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices

    residual_vector_terms: Mat<f64>,  // [6][n_qps]
    stiffness_matrix_terms: Mat<f64>, // [6][6][n_qps]
    inertia_matrix_terms: Mat<f64>,   // [6][6][n_qps]

    // Shape Function data
    shape_interp: Mat<f64>, // [n_qps][max_nodes] Shape function values
    shape_deriv: Mat<f64>,  // [n_qps][max_nodes] Shape function derivatives
}

impl Beams {
    fn new(inp: &BeamInput) -> Self {
        // Total number of nodes to allocate (multiple of 8)
        let total_nodes = inp.elements.iter().map(|e| (e.nodes.len())).sum::<usize>();
        let alloc_nodes = total_nodes + 7 & !7;

        // Total number of quadrature points (multiple of 8)
        let total_qps = inp
            .elements
            .iter()
            .map(|e| (e.quadrature.points.len()))
            .sum::<usize>();
        let alloc_qps = total_qps + 7 & !7;

        // Max number of nodes in any element
        let max_elem_nodes = inp.elements.iter().map(|e| (e.nodes.len())).max().unwrap();

        // Build element index
        let mut index: Vec<ElemIndex> = vec![];
        let mut start_node = 0;
        let mut start_qp = 0;
        for (i, e) in inp.elements.iter().enumerate() {
            let n_nodes = e.nodes.len();
            let n_qps = e.quadrature.points.len();
            index.push(ElemIndex {
                elem_id: i,
                n_nodes,
                node_range: [start_node, start_node + n_nodes],
                n_qps,
                qp_range: [start_qp, start_qp + n_qps],
            });
            start_node += n_nodes;
            start_qp += n_qps;
        }

        // Get node IDs
        let node_ids = inp
            .elements
            .iter()
            .flat_map(|e| e.nodes.iter().map(|n| n.node.id).collect_vec())
            .collect_vec();

        // let (x0, u, v, vd): (Vec<[f64; 7]>, Vec<[f64; 7]>, Vec<[f64; 6]>, Vec<[f64; 6]>) = inp
        let (x0, u, v, vd): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(
            inp.elements
                .iter()
                .flat_map(|e| {
                    e.nodes
                        .iter()
                        .map(|n| (n.node.x, n.node.u, n.node.v, n.node.vd))
                        .collect_vec()
                })
                .collect_vec(),
        );

        let qp_weights = inp
            .elements
            .iter()
            .flat_map(|e| e.quadrature.weights.clone())
            .collect_vec();

        let mut beams = Self {
            index,
            node_ids,
            gravity: Row::from_fn(3, |i| inp.gravity[i]),

            // Nodes
            node_x0: Mat::from_fn(
                7,
                alloc_nodes,
                |i, j| if j < x0.len() { x0[j][i] } else { 0. },
            ),
            node_u: Mat::from_fn(
                7,
                alloc_nodes,
                |i, j| if j < u.len() { u[j][i] } else { 0. },
            ),
            node_v: Mat::from_fn(
                6,
                alloc_nodes,
                |i, j| if j < v.len() { v[j][i] } else { 0. },
            ),
            node_vd: Mat::from_fn(
                6,
                alloc_nodes,
                |i, j| if j < vd.len() { vd[j][i] } else { 0. },
            ),
            node_fx: Mat::zeros(6, alloc_nodes),

            // Quadrature points
            qp_weight: Col::from_fn(alloc_qps, |i| {
                if i < qp_weights.len() {
                    qp_weights[0]
                } else {
                    0.
                }
            }),
            qp_jacobian: Col::ones(alloc_qps),
            qp_m_star: Mat::zeros(6 * 6, alloc_qps),
            qp_c_star: Mat::zeros(6 * 6, alloc_qps),
            qp_x: Mat::zeros(7, alloc_qps),
            qp_x0: Mat::zeros(7, alloc_qps),
            qp_x0_prime: Mat::zeros(7, alloc_qps),
            qp_u: Mat::zeros(7, alloc_qps),
            qp_u_prime: Mat::zeros(7, alloc_qps),
            qp_v: Mat::zeros(6, alloc_qps),
            qp_vd: Mat::zeros(6, alloc_qps),
            qp_e: Mat::zeros(3 * 4, alloc_qps),
            qp_x0pupss: Mat::zeros(3 * 3, alloc_qps),
            qp_eta: Mat::zeros(3, alloc_qps),
            qp_rho: Mat::zeros(3 * 3, alloc_qps),
            qp_strain: Mat::zeros(6, alloc_qps),
            qp_fc: Mat::zeros(6, alloc_qps),
            qp_fd: Mat::zeros(6, alloc_qps),
            qp_fi: Mat::zeros(6, alloc_qps),
            qp_fx: Mat::zeros(6, alloc_qps),
            qp_fg: Mat::zeros(6, alloc_qps),
            qp_rr0: Mat::zeros(6 * 6, alloc_qps),
            qp_muu: Mat::zeros(6 * 6, alloc_qps),
            qp_cuu: Mat::zeros(6 * 6, alloc_qps),
            qp_ouu: Mat::zeros(6 * 6, alloc_qps),
            qp_puu: Mat::zeros(6 * 6, alloc_qps),
            qp_quu: Mat::zeros(6 * 6, alloc_qps),
            qp_guu: Mat::zeros(6 * 6, alloc_qps),
            qp_kuu: Mat::zeros(6 * 6, alloc_qps),

            residual_vector_terms: Mat::zeros(alloc_nodes, 6),
            stiffness_matrix_terms: Mat::zeros(alloc_nodes, 6 * 6), //6x6
            inertia_matrix_terms: Mat::zeros(alloc_nodes, 6 * 6),   //6x6

            shape_interp: Mat::zeros(alloc_qps, max_elem_nodes),
            shape_deriv: Mat::zeros(alloc_qps, max_elem_nodes),
        };

        //----------------------------------------------------------------------
        // Shape functions
        //----------------------------------------------------------------------

        // Initialize element shape functions for interpolation and derivative
        for (ei, e) in izip!(beams.index.iter(), inp.elements.iter()) {
            // Get node positions along beam [-1, 1]
            let node_xi = e.nodes.iter().map(|n| 2. * n.si - 1.).collect_vec();

            // Get shape interpolation matrix for this element
            let mut shape_interp =
                beams
                    .shape_interp
                    .submatrix_mut(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Create interpolation matrix from quadrature point and node locations
            shape_interp_matrix(&e.quadrature.points, &node_xi, shape_interp.as_mut());

            // Get shape derivative matrix for this element
            let mut shape_deriv =
                beams
                    .shape_deriv
                    .submatrix_mut(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Create derivative matrix from quadrature point and node locations
            shape_deriv_matrix(&e.quadrature.points, &node_xi, shape_deriv.as_mut());
        }

        //----------------------------------------------------------------------
        // Quadrature points
        //----------------------------------------------------------------------

        for ei in beams.index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp =
                beams
                    .shape_interp
                    .submatrix(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Interpolate initial position
            let node_x0 = beams.node_x0.subcols(ei.node_range[0], ei.n_nodes);
            let mut qp_x0 = beams.qp_x0.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
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
                .submatrix(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Calculate Jacobian
            let qp_jacobian = beams.qp_jacobian.subrows_mut(ei.qp_range[0], ei.n_qps);
            let mut qp_x0_prime = beams.qp_x0_prime.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
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

            let sections = &inp.elements[ei.elem_id].sections;
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
                (inp.elements[ei.elem_id].quadrature.points[i] + 1.) / 2.
            });

            qp_s.iter().enumerate().for_each(|(i, &s)| {
                let mut qp_m_star = beams
                    .qp_m_star
                    .subcols_mut(ei.qp_range[0], ei.n_qps)
                    .col_mut(i);
                let mut qp_c_star = beams
                    .qp_c_star
                    .subcols_mut(ei.qp_range[0], ei.n_qps)
                    .col_mut(i);
                match section_s.iter().position(|&ss| s > ss) {
                    None => {
                        qp_m_star.copy_from(&section_m_star[0]);
                        qp_c_star.copy_from(&section_c_star[0]);
                    }
                    Some(j) => {
                        if j == sections.len() {
                            qp_m_star.copy_from(&section_m_star[j]);
                            qp_c_star.copy_from(&section_c_star[j]);
                        } else {
                            let alpha = (s - section_s[j]) / (section_s[j + 1] - section_s[j]);
                            qp_m_star.copy_from(
                                &section_m_star[j] * Scale(1. - alpha)
                                    + &section_m_star[j + 1] * Scale(alpha),
                            );
                            qp_c_star.copy_from(
                                &section_c_star[j] * Scale(1. - alpha)
                                    + &section_c_star[j + 1] * Scale(alpha),
                            );
                        }
                    }
                }
            });
        }

        //----------------------------------------------------------------------
        // Jacobian
        //----------------------------------------------------------------------

        // Interpolate nodes to quadrature points
        beams.interp_to_qps();

        beams
    }

    fn interp_to_qps(&mut self) {
        for ei in self.index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp = self
                .shape_interp
                .submatrix(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Get shape derivative matrix for this element
            let shape_deriv = self
                .shape_deriv
                .submatrix(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Interpolate displacement
            let node_u = self.node_u.subcols(ei.node_range[0], ei.n_nodes);
            let mut qp_u = self.qp_u.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_u.as_mut().transpose_mut(),
                shape_interp,
                node_u.transpose(),
                None,
                1.,
                Parallelism::None,
            );
            qp_u.as_mut()
                .subrows_mut(3, 4)
                .col_iter_mut()
                .for_each(|mut c| {
                    let m = c.norm_l2();
                    if m != 0. {
                        c /= m;
                    }
                });

            // Get Jacobians for this element
            let qp_jacobian = self.qp_jacobian.subrows(ei.qp_range[0], ei.n_qps);

            // Displacement derivative
            let mut qp_u_prime = self.qp_u_prime.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_u_prime.as_mut().transpose_mut(),
                shape_deriv,
                node_u.transpose(),
                None,
                1.0,
                Parallelism::None,
            );
            izip!(qp_u_prime.col_iter_mut(), qp_jacobian.iter())
                .for_each(|(mut col, &jacobian)| col /= jacobian);

            // Interpolate velocity
            let node_v = self.node_v.subcols(ei.node_range[0], ei.n_nodes);
            let mut qp_v = self.qp_v.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_v.as_mut().transpose_mut(),
                shape_interp,
                node_v.transpose(),
                None,
                1.0,
                Parallelism::None,
            );

            // Interpolate acceleration
            let node_vd = self.node_vd.subcols(ei.node_range[0], ei.n_nodes);
            let mut qp_vd = self.qp_vd.subcols_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_vd.as_mut().transpose_mut(),
                shape_interp,
                node_vd.transpose(),
                None,
                1.0,
                Parallelism::None,
            );
        }
    }

    fn calc_qp(&mut self) {
        self.calc_qp_x();
        self.calc_qp_rr0();
        self.calc_qp_muu();
        self.calc_qp_cuu();
        self.calc_qp_eta_rho();
        self.calc_qp_strain();
        self.calc_qp_x0pupss();
        self.calc_qp_fc();
    }

    // Calculate current position and rotation (x0 + u)
    fn calc_qp_x(&mut self) {
        izip!(
            self.qp_x.col_iter_mut(),
            self.qp_x0.col_iter(),
            self.qp_u.col_iter()
        )
        .for_each(|(mut x, x0, u)| {
            x[0] = x0[0] + u[0];
            x[1] = x0[1] + u[1];
            x[2] = x0[2] + u[2];
            x.subrows_mut(3, 4)
                .quat_compose(u.subrows(3, 4), x0.subrows(3, 4));
        });
    }

    fn calc_qp_rr0(&mut self) {
        izip!(
            self.qp_rr0.col_iter_mut(),
            self.qp_x.subrows_mut(3, 4).col_iter_mut()
        )
        .for_each(|(col, mut x)| {
            let mut rr0: MatMut<f64> =
                mat::from_column_major_slice_mut(col.try_as_slice_mut().unwrap(), 6, 6);
            let mut m = Mat::<f64>::zeros(3, 3);
            x.as_mut().quat_as_matrix(m.as_mut());
            for j in 0..3 {
                for i in 0..3 {
                    rr0[(i, j)] = m[(i, j)];
                    rr0[(i + 3, j + 3)] = m[(i, j)];
                }
            }
        });
    }

    fn calc_qp_muu(&mut self) {
        izip!(
            self.qp_muu.col_iter_mut(),
            self.qp_m_star.col_iter(),
            self.qp_rr0.col_iter()
        )
        .for_each(|(muu_col, m_star_col, rr0_col)| {
            let muu: MatMut<f64> =
                mat::from_column_major_slice_mut(muu_col.try_as_slice_mut().unwrap(), 6, 6);
            let m_star: MatRef<f64> =
                mat::from_column_major_slice(m_star_col.try_as_slice().unwrap(), 6, 6);
            let rr0: MatRef<f64> =
                mat::from_column_major_slice(rr0_col.try_as_slice().unwrap(), 6, 6);
            let mut muu_tmp = Mat::<f64>::zeros(6, 6);
            matmul::matmul(muu_tmp.as_mut(), rr0, m_star, None, 1., Parallelism::None);
            matmul::matmul(
                muu,
                muu_tmp.as_ref(),
                rr0.transpose(),
                None,
                1.,
                Parallelism::None,
            );
        });
    }

    fn calc_qp_cuu(&mut self) {
        izip!(
            self.qp_cuu.col_iter_mut(),
            self.qp_c_star.col_iter(),
            self.qp_rr0.col_iter()
        )
        .for_each(|(cuu_col, c_star_col, rr0_col)| {
            let cuu: MatMut<f64> =
                mat::from_column_major_slice_mut(cuu_col.try_as_slice_mut().unwrap(), 6, 6);
            let c_star: MatRef<f64> =
                mat::from_column_major_slice(c_star_col.try_as_slice().unwrap(), 6, 6);
            let rr0: MatRef<f64> =
                mat::from_column_major_slice(rr0_col.try_as_slice().unwrap(), 6, 6);
            let mut cuu_tmp = Mat::<f64>::zeros(6, 6);
            matmul::matmul(cuu_tmp.as_mut(), rr0, c_star, None, 1., Parallelism::None);
            matmul::matmul(
                cuu,
                cuu_tmp.as_ref(),
                rr0.transpose(),
                None,
                1.,
                Parallelism::None,
            );
        });
    }

    fn calc_qp_x0pupss(&mut self) {
        izip!(
            self.qp_x0pupss.col_iter_mut(),
            self.qp_x0_prime.subrows(0, 3).col_iter(),
            self.qp_u_prime.subrows(0, 3).col_iter()
        )
        .for_each(|(x0pupss_col, x0_prime, u_prime)| {
            let x0pupss: MatMut<f64> =
                mat::from_column_major_slice_mut(x0pupss_col.try_as_slice_mut().unwrap(), 3, 3);
            let x0pup = col![
                x0_prime[0] + u_prime[0],
                x0_prime[1] + u_prime[1],
                x0_prime[2] + u_prime[2]
            ];
            vec_tilde(x0pup.as_ref(), x0pupss);
        });
    }

    fn calc_qp_strain(&mut self) {
        izip!(
            self.qp_strain.col_iter_mut(),
            self.qp_x0_prime.subrows(0, 3).col_iter(),
            self.qp_u_prime.subrows(0, 3).col_iter(),
            self.qp_u_prime.subrows(3, 4).col_iter(),
            self.qp_u.subrows_mut(3, 4).col_iter_mut()
        )
        .for_each(|(mut qp_strain, x0_prime, u_prime, r_prime, r)| {
            let mut r_x0_prime = x0_prime.to_owned();
            r.quat_rotate_vector(r_x0_prime.as_mut());
            r_x0_prime[0] -= x0_prime[0] + u_prime[0];
            r_x0_prime[1] -= x0_prime[1] + u_prime[1];
            r_x0_prime[2] -= x0_prime[2] + u_prime[2];
            let mut r_deriv = Mat::<f64>::zeros(3, 4);
            r.quat_derivative(r_deriv.as_mut());
            let mut e2 = Col::<f64>::zeros(3);
            matmul::matmul(e2.as_mut(), r_deriv, r_prime, None, 2., Parallelism::None);
            qp_strain[0] = -r_x0_prime[0];
            qp_strain[1] = -r_x0_prime[1];
            qp_strain[2] = -r_x0_prime[2];
            qp_strain[3] = e2[0];
            qp_strain[4] = e2[1];
            qp_strain[5] = e2[2];
        });
    }

    fn calc_qp_eta_rho(&mut self) {
        izip!(
            self.qp_eta.col_iter_mut(),
            self.qp_rho.col_iter_mut(),
            self.qp_muu.col_iter()
        )
        .for_each(|(mut eta, rho_col, muu_col)| {
            let muu: MatRef<f64> =
                mat::from_column_major_slice(muu_col.try_as_slice().unwrap(), 6, 6);
            let m = muu[(0, 0)];
            if m == 0. {
                eta.fill(0.);
            } else {
                eta[0] = muu[(5, 1)] / m;
                eta[1] = -muu[(5, 0)] / m;
                eta[2] = muu[(4, 0)] / m;
            }
            let mut rho: MatMut<f64> =
                mat::from_column_major_slice_mut(rho_col.try_as_slice_mut().unwrap(), 3, 3);
            rho.copy_from(muu.submatrix(3, 3, 3, 3));
        });
    }

    fn calc_qp_fc(&mut self) {
        izip!(
            self.qp_fc.col_iter_mut(),
            self.qp_cuu.col_iter(),
            self.qp_strain.col_iter(),
        )
        .for_each(|(fc, cuu, strain)| {
            matmul::matmul(fc, cuu, strain, None, 1., Parallelism::None);
        });
    }

    fn calc_qp_fi(&mut self) {
        izip!(
            self.qp_fi.col_iter_mut(),
            self.qp_muu.col_iter(),
            self.qp_v.subrows(3, 3).col_iter(),  // omega
            self.qp_vd.subrows(0, 3).col_iter(), // u_ddot
            self.qp_vd.subrows(3, 3).col_iter(), // omega_dot
            self.qp_eta.col_iter(),
            self.qp_rho.col_iter(),
        )
        .for_each(|(fi, muu, omega, u_ddot, omega_dot, eta, rho)| {});
    }

    // fn calc_qp_ouu(&mut self) {
    //     izip!(
    //         self.qp_ouu.col_iter_mut(),
    //         self.qp_c_star.col_iter(),
    //         self.qp_rr0.col_iter()
    //     )
    //     .for_each(|(ouu_col, c_star_col, rr0_col)| {
    //         let ouu: MatMut<f64> =
    //             mat::from_column_major_slice_mut(ouu_col.try_as_slice_mut().unwrap(), 6, 6);
    //         let c_star: MatRef<f64> =
    //             mat::from_column_major_slice(c_star_col.try_as_slice().unwrap(), 6, 6);
    //         let rr0: MatRef<f64> =
    //             mat::from_column_major_slice(rr0_col.try_as_slice().unwrap(), 6, 6);
    //         matmul::matmul(ouu_tmp.as_mut(), rr0, c_star, None, 1., Parallelism::None);
    //         matmul::matmul(
    //             ouu,
    //             ouu_tmp.as_ref(),
    //             rr0.transpose(),
    //             None,
    //             1.,
    //             Parallelism::None,
    //         );
    //     });
    // }
}

fn vec_tilde(v: ColRef<f64>, mut m: MatMut<f64>) {
    m[(0, 0)] = 0.;
    m[(0, 1)] = -v[2];
    m[(0, 2)] = v[1];
    m[(1, 0)] = v[2];
    m[(1, 1)] = 0.;
    m[(1, 2)] = -v[0];
    m[(2, 0)] = -v[1];
    m[(2, 1)] = v[0];
    m[(2, 2)] = 0.;
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use faer::{assert_matrix_eq, mat, row, Scale};
    use itertools::Itertools;

    use crate::interp::gauss_legendre_lobotto_points;
    use crate::quadrature::Quadrature;

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

        let node_s = vec![0., 0.1726731646460114, 0.5, 0.82732683535398865, 1.];

        let nodes = vec![
            Rc::new(Node {
                id: 0,
                x: [
                    0.,
                    0.,
                    0.,
                    0.9778215200524469,
                    -0.01733607539094763,
                    -0.09001900002195001,
                    -0.18831121859148398,
                ],
                u: [0., 0., 0., 1., 0., 0., 0.],
                v: [0., 0., 0., 0., 0., 0.],
                vd: [0., 0., 0., 0., 0., 0.],
            }),
            Rc::new(Node {
                id: 1,
                x: [
                    0.863365823230057,
                    -0.2558982639254171,
                    0.11304112106827427,
                    0.9950113028068008,
                    -0.002883848832932071,
                    -0.030192109815745303,
                    -0.09504013471947484,
                ],
                u: [
                    0.002981602178886856,
                    -0.00246675949494302,
                    0.003084570715675624,
                    0.9999627302042724,
                    0.008633550973807708,
                    0.,
                    0.,
                ],
                v: [
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624,
                    0.01726731646460114,
                    -0.014285714285714285,
                    0.003084570715675624,
                ],
                vd: [
                    0.01726731646460114,
                    -0.011304112106827427,
                    0.00606617289456248,
                    0.01726731646460114,
                    -0.014285714285714285,
                    -0.014285714285714285,
                ],
            }),
            Rc::new(Node {
                id: 2,
                x: [
                    2.5,
                    -0.25,
                    0.,
                    0.9904718430204884,
                    -0.009526411091536478,
                    0.09620741150793366,
                    0.09807604012323785,
                ],
                u: [
                    0.025,
                    -0.0125,
                    0.0275,
                    0.9996875162757026,
                    0.02499739591471221,
                    0.,
                    0.,
                ],
                v: [0.05, -0.025, 0.0275, 0.05, -0.025, 0.0275],
                vd: [0.05, 0., 0.0525, 0.05, -0.025, -0.025],
            }),
            Rc::new(Node {
                id: 3,
                x: [
                    4.1366341767699435,
                    0.39875540678256005,
                    -0.5416125496397031,
                    0.9472312341234699,
                    -0.04969214162931507,
                    0.18127630174800594,
                    0.25965858850765167,
                ],
                u: [
                    0.06844696924968459,
                    -0.011818954790771264,
                    0.07977257214146725,
                    0.9991445348823055,
                    0.04135454527402512,
                    0.,
                    0.,
                ],
                v: [
                    0.08273268353539887,
                    -0.01428571428571428,
                    0.07977257214146725,
                    0.08273268353539887,
                    -0.01428571428571428,
                    0.07977257214146725,
                ],
                vd: [
                    0.08273268353539887,
                    0.05416125496397031,
                    0.14821954139115184,
                    0.08273268353539887,
                    -0.01428571428571428,
                    -0.01428571428571428,
                ],
            }),
            Rc::new(Node {
                id: 4,
                x: [
                    5.,
                    1.,
                    -1.,
                    0.9210746582719719,
                    -0.07193653093139739,
                    0.20507529985516368,
                    0.32309554437664584,
                ],
                u: [
                    0.1,
                    0.,
                    0.12,
                    0.9987502603949663,
                    0.04997916927067825,
                    0.,
                    0.,
                ],
                v: [0.1, 0., 0.12, 0.1, 0., 0.12],
                vd: [0.1, 0.1, 0.22, 0.1, 0., 0.],
            }),
        ];

        let input = BeamInput {
            gravity: [0., 0., 0.],
            elements: vec![BeamElement {
                nodes: izip!(node_s.iter(), nodes.iter())
                    .map(|(&s, n)| BeamNode {
                        si: s,
                        node: n.clone(),
                    })
                    .collect(),
                quadrature: Quadrature {
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
                sections: vec![
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
            }],
        };

        let mut beams = Beams::new(&input);
        beams.calc_qp();
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
            mat::from_column_major_slice(beams.qp_m_star.col(0).try_as_slice().unwrap(), 6, 6),
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
            mat::from_column_major_slice(beams.qp_c_star.col(0).try_as_slice().unwrap(), 6, 6),
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
            beams.qp_x0.col(0).as_2d().transpose(),
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
            beams.qp_x0_prime.col(0).as_2d().transpose(),
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
            beams.qp_u.col(0).as_2d().transpose(),
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
            beams.qp_u_prime.col(0).as_2d().transpose(),
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
            beams.qp_x.col(0).as_2d().transpose(),
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
                .qp_jacobian
                .subrows(0, beams.index[0].n_qps)
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
            beams.qp_strain.subcols(0, 2).transpose(),
            mat![
                [
                    0.0009414876868372797,
                    -0.0004838292834870028,
                    0.0018199012459517439,
                    0.0184996868523541,
                    0.,
                    0.
                ],
                [
                    0.004999015404948938,
                    -0.0028423419905453384,
                    0.008276774088180464,
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
            mat::from_column_major_slice(beams.qp_rr0.col(0).try_as_slice().unwrap(), 6, 6),
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
            mat::from_column_major_slice(beams.qp_muu.col(0).try_as_slice().unwrap(), 6, 6),
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
            mat::from_column_major_slice(beams.qp_cuu.col(0).try_as_slice().unwrap(), 6, 6),
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
                    11.196339225304033,
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
    fn test_me() {
        let xi = gauss_legendre_lobotto_points(5);
        let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

        // Quadrature rule
        let gq = Quadrature::gauss(7);

        // Node initial position and rotation
        let r0 = row![1., 0., 0., 0.];
        let fx = |s: f64| -> f64 { 10. * s + 2. };

        // Mass matrix 6x6
        let m_star = mat![
            [8.538, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 8.538, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 8.538, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 1.4433, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.40972, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 1.0336],
        ] * Scale(1e-2);

        // Stiffness matrix 6x6
        let c_star = mat![
            [1368.17, 0., 0., 0., 0., 0.],
            [0., 88.56, 0., 0., 0., 0.],
            [0., 0., 38.78, 0., 0., 0.],
            [0., 0., 0., 16.960, 17.610, -0.351],
            [0., 0., 0., 17.610, 59.120, -0.370],
            [0., 0., 0., -0.351, -0.370, 141.47],
        ] * Scale(1e3);

        //----------------------------------------------------------------------
        // Create element
        //----------------------------------------------------------------------

        let input = BeamInput {
            gravity: [0., 0., 0.],
            elements: vec![BeamElement {
                nodes: s
                    .iter()
                    .enumerate()
                    .map(|(i, &si)| BeamNode {
                        si: si,
                        node: Rc::new(Node {
                            id: i,
                            x: [fx(si), 0., 0., r0[0], r0[1], r0[2], r0[3]],
                            u: [0., 0., 0., 1., 0., 0., 0.],
                            v: [0., 0., 0., 0., 0., 0.],
                            vd: [0., 0., 0., 0., 0., 0.],
                        }),
                    })
                    .collect(),
                quadrature: gq,
                sections: vec![
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
            }],
        };

        let _beams = Beams::new(&input);
    }
}

use std::ops::Rem;

use crate::interp::shape_interp_matrix;
use crate::quadrature::Quadrature;
use crate::quaternion::{quat_as_matrix, quat_derivative, quat_rotate_vector, Quat};
use crate::state::State;
use crate::{interp::shape_deriv_matrix, node::Node};
use faer::linalg::matmul::matmul;
use faer::{
    mat, unzipped, zipped, Col, ColMut, ColRef, Entity, Mat, MatMut, MatRef, Parallelism, Scale,
};
use itertools::{izip, multiunzip, Itertools};

pub struct BeamInput {
    pub gravity: [f64; 3],
    pub elements: Vec<BeamElement>,
}

pub struct BeamElement {
    pub nodes: Vec<BeamNode>,
    pub sections: Vec<BeamSection>,
    pub quadrature: Quadrature,
}

pub struct BeamNode {
    pub si: f64,
    pub node_id: usize,
}

impl BeamNode {
    pub fn new(s: f64, node: &Node) -> Self {
        Self {
            si: s,
            node_id: node.id,
        }
    }
}

pub struct BeamSection {
    pub s: f64,           // Distance along centerline from first point
    pub m_star: Mat<f64>, // [6][6] Mass matrix
    pub c_star: Mat<f64>, // [6][6] Stiffness matrix
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
    elem_index: Vec<ElemIndex>,
    node_ids: Vec<usize>,
    gravity: Col<f64>, // [3] gravity

    // Node-based data
    node_x0: Mat<f64>,     // [7][n_nodes] Initial position/rotation
    node_u: Mat<f64>,      // [7][n_nodes] State: translation/rotation displacement
    node_v: Mat<f64>,      // [6][n_nodes] State: translation/rotation velocity
    node_vd: Mat<f64>,     // [6][n_nodes] State: translation/rotation acceleration
    pub node_fe: Mat<f64>, // [6][n_nodes] Elastic forces
    pub node_fi: Mat<f64>, // [6][n_nodes] Internal forces
    pub node_fx: Mat<f64>, // [6][n_nodes] External forces
    pub node_fg: Mat<f64>, // [6][n_nodes] Gravity forces
    pub node_f: Mat<f64>,  // [6][n_nodes] total forces
    node_muu: Mat<f64>,    // [6][n_nodes*max_nodes] mass matrices
    node_guu: Mat<f64>,    // [6][n_nodes*max_nodes] gyro matrices
    node_ke: Mat<f64>,     // [6][n_nodes*max_nodes] stiff matrices
    node_ki: Mat<f64>,     // [6][n_nodes*max_nodes] stiff matrices

    // Quadrature point data
    pub qp_weight: Col<f64>,   // [n_qps]        Integration weights
    pub qp_jacobian: Col<f64>, // [n_qps]        Jacobian vector
    pub qp_m_star: Mat<f64>,   // [6][6][n_qps]  Mass matrix in material frame
    pub qp_c_star: Mat<f64>,   // [6][6][n_qps]  Stiffness matrix in material frame
    pub qp_x: Mat<f64>,        // [7][n_qps]     Current position/orientation
    pub qp_x0: Mat<f64>,       // [7][n_qps]     Initial position
    pub qp_x0_prime: Mat<f64>, // [7][n_qps]     Initial position derivative
    pub qp_u: Mat<f64>,        // [7][n_qps]     State: displacement
    pub qp_u_prime: Mat<f64>,  // [7][n_qps]     State: displacement derivative
    pub qp_v: Mat<f64>,        // [6][n_qps]     State: velocity
    pub qp_vd: Mat<f64>,       // [6][n_qps]     State: acceleration
    pub qp_x0pupss: Mat<f64>,  // [3][3][n_qps]  skew_symmetric(x0_prime + u_prime)
    pub qp_m: Col<f64>,        // [n_qps]        mass
    pub qp_eta: Mat<f64>,      // [3][n_qps]     mass
    pub qp_rho: Mat<f64>,      // [3][3][n_qps]  mass
    pub qp_strain: Mat<f64>,   // [6][n_qps]     Strain
    pub qp_fc: Mat<f64>,       // [6][n_qps]     Elastic force
    pub qp_fd: Mat<f64>,       // [6][n_qps]     Elastic force
    pub qp_fi: Mat<f64>,       // [6][n_qps]     Inertial force
    pub qp_fx: Mat<f64>,       // [6][n_qps]     External force
    pub qp_fg: Mat<f64>,       // [6][n_qps]     Gravity force
    pub qp_rr0: Mat<f64>,      // [6][6][n_qps]  Global rotation
    pub qp_muu: Mat<f64>,      // [6][6][n_qps]  Mass in global frame
    pub qp_cuu: Mat<f64>,      // [6][6][n_qps]  Stiffness in global frame
    pub qp_ouu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    pub qp_puu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    pub qp_quu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    pub qp_guu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices
    pub qp_kuu: Mat<f64>,      // [6][6][n_qps]  Linearization matrices

    // Shape Function data
    pub shape_interp: Mat<f64>, // [n_qps][max_nodes] Shape function values
    pub shape_deriv: Mat<f64>,  // [n_qps][max_nodes] Shape function derivatives
}

impl Beams {
    pub fn new(inp: &BeamInput, nodes: &[Node]) -> Self {
        // Total number of nodes to allocate (multiple of 8)
        let total_nodes = inp.elements.iter().map(|e| (e.nodes.len())).sum::<usize>();
        let alloc_nodes = total_nodes;

        // Total number of quadrature points (multiple of 8)
        let total_qps = inp
            .elements
            .iter()
            .map(|e| (e.quadrature.points.len()))
            .sum::<usize>();
        let alloc_qps = total_qps;

        // Max number of nodes in any element
        let max_elem_nodes = inp.elements.iter().map(|e| (e.nodes.len())).max().unwrap();

        // Build element index
        let mut index: Vec<ElemIndex> = vec![];
        let mut start_node = 0;
        let mut start_qp = 0;
        let mut start_mat = 0;
        for (i, e) in inp.elements.iter().enumerate() {
            let n_nodes = e.nodes.len();
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

        // Get node IDs
        let node_ids = inp
            .elements
            .iter()
            .flat_map(|e| e.nodes.iter().map(|n| n.node_id).collect_vec())
            .collect_vec();

        // let (x0, u, v, vd): (Vec<[f64; 7]>, Vec<[f64; 7]>, Vec<[f64; 6]>, Vec<[f64; 6]>) = inp
        let (x0, u, v, vd): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(
            inp.elements
                .iter()
                .flat_map(|e| {
                    e.nodes
                        .iter()
                        .map(|n| {
                            let node = &nodes[n.node_id];
                            (node.x, node.u, node.v, node.vd)
                        })
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
            elem_index: index,
            node_ids,
            gravity: Col::from_fn(3, |i| inp.gravity[i]),

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
            node_fe: Mat::zeros(6, alloc_nodes),
            node_fi: Mat::zeros(6, alloc_nodes),
            node_fx: Mat::zeros(6, alloc_nodes),
            node_fg: Mat::zeros(6, alloc_nodes),
            node_f: Mat::zeros(6, alloc_nodes),
            node_muu: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),
            node_guu: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),
            node_ke: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),
            node_ki: Mat::zeros(6 * 6, alloc_nodes * max_elem_nodes),

            // Quadrature points
            qp_weight: Col::from_fn(alloc_qps, |i| {
                if i < qp_weights.len() {
                    qp_weights[i]
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
            qp_x0pupss: Mat::zeros(3 * 3, alloc_qps),
            qp_m: Col::zeros(alloc_qps),
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

            // residual_vector_terms: Mat::zeros(6, alloc_nodes),
            // stiffness_matrix_terms: Mat::zeros(6 * 6, alloc_nodes), //6x6
            // inertia_matrix_terms: Mat::zeros(6 * 6, alloc_nodes),   //6x6
            shape_interp: Mat::zeros(alloc_qps, max_elem_nodes),
            shape_deriv: Mat::zeros(alloc_qps, max_elem_nodes),
        };

        //----------------------------------------------------------------------
        // Shape functions
        //----------------------------------------------------------------------

        // Initialize element shape functions for interpolation and derivative
        for (ei, e) in izip!(beams.elem_index.iter(), inp.elements.iter()) {
            // Get node positions along beam [-1, 1]
            let node_xi = e.nodes.iter().map(|n| 2. * n.si - 1.).collect_vec();

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
            let mut qp_x0 = beams.qp_x0.subcols_mut(ei.i_qp_start, ei.n_qps);
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
            let qp_jacobian = beams.qp_jacobian.subrows_mut(ei.i_qp_start, ei.n_qps);
            let mut qp_x0_prime = beams.qp_x0_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
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
                    .subcols_mut(ei.i_qp_start, ei.n_qps)
                    .col_mut(i);
                let mut qp_c_star = beams
                    .qp_c_star
                    .subcols_mut(ei.i_qp_start, ei.n_qps)
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

        // Calculate quadrature point data
        beams.calc_qp_values();

        // Integrate forces
        beams.integrate_forces();

        // Integrate matrices
        beams.integrate_matrices();

        beams
    }

    pub fn calculate_system(&mut self, state: &State) {
        // Copy displacement, velocity, and acceleration from nodes to beams
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

        // Calculate quadrature point data
        self.calc_qp_values();

        // Integrate forces
        self.integrate_forces();

        // Integrate matrices
        self.integrate_matrices();
    }

    pub fn assemble_system(
        &self,
        mut m: MatMut<f64>,
        mut g: MatMut<f64>,
        mut k: MatMut<f64>,
        mut r: ColMut<f64>,
    ) {
        // Loop through elements
        self.elem_index.iter().for_each(|ei| {
            // Get slice of node ids for this element
            let node_ids = &self.node_ids[ei.i_node_start..ei.i_node_start + ei.n_nodes];

            // Get starting degree of freedom for each node
            let elem_dof_start_pairs = node_ids
                .iter()
                .cartesian_product(node_ids)
                .map(|(&i, &j)| (i * 6, j * 6))
                .collect_vec();

            // Mass matrix
            let muu = self
                .node_muu
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            izip!(elem_dof_start_pairs.iter(), muu.col_iter()).for_each(|((i, j), muu)| {
                let mut me = m.as_mut().submatrix_mut(*i, *j, 6, 6);
                zipped!(&mut me, &muu.as_mat_ref(6, 6))
                    .for_each(|unzipped!(mut me, muu)| *me += *muu);
            });

            // Gyroscopic matrix
            let guu = self
                .node_guu
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            izip!(elem_dof_start_pairs.iter(), guu.col_iter()).for_each(|((i, j), guu)| {
                let mut ge = g.as_mut().submatrix_mut(*i, *j, 6, 6);
                zipped!(&mut ge, &guu.as_mat_ref(6, 6))
                    .for_each(|unzipped!(mut ge, guu)| *ge += *guu);
            });

            // Stiffness matrix
            let ke = self
                .node_ke
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            let ki = self
                .node_ki
                .subcols(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            izip!(elem_dof_start_pairs.iter(), ke.col_iter(), ki.col_iter()).for_each(
                |((i, j), ke, ki)| {
                    let mut kt = k.as_mut().submatrix_mut(*i, *j, 6, 6);
                    zipped!(&mut kt, &ke.as_mat_ref(6, 6), &ki.as_mat_ref(6, 6))
                        .for_each(|unzipped!(mut kt, ke, ki)| *kt += *ke + *ki);
                },
            );
        });

        // Residual vector
        for (i, f) in self
            .node_ids
            .iter()
            .map(|id| id * 6)
            .zip(self.node_f.col_iter())
        {
            let mut resid = r.as_mut().subrows_mut(i, 6);
            zipped!(&mut resid, &f).for_each(|unzipped!(mut r, f)| *r += *f);
        }
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

            // Interpolate displacement
            let node_u = self.node_u.subcols(ei.i_node_start, ei.n_nodes);
            let mut qp_u = self.qp_u.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
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
            let qp_jacobian = self.qp_jacobian.subrows(ei.i_qp_start, ei.n_qps);

            // Displacement derivative
            let mut qp_u_prime = self.qp_u_prime.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
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
            let node_v = self.node_v.subcols(ei.i_node_start, ei.n_nodes);
            let mut qp_v = self.qp_v.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_v.as_mut().transpose_mut(),
                shape_interp,
                node_v.transpose(),
                None,
                1.0,
                Parallelism::None,
            );

            // Interpolate acceleration
            let node_vd = self.node_vd.subcols(ei.i_node_start, ei.n_nodes);
            let mut qp_vd = self.qp_vd.subcols_mut(ei.i_qp_start, ei.n_qps);
            matmul(
                qp_vd.as_mut().transpose_mut(),
                shape_interp,
                node_vd.transpose(),
                None,
                1.0,
                Parallelism::None,
            );
        }
    }

    fn calc_qp_values(&mut self) {
        calc_qp_x(self.qp_x.as_mut(), self.qp_x0.as_ref(), self.qp_u.as_ref());
        calc_qp_rr0(self.qp_rr0.as_mut(), self.qp_x.as_ref());
        calc_qp_mat(
            self.qp_muu.as_mut(),
            self.qp_m_star.as_ref(),
            self.qp_rr0.as_ref(),
        );
        calc_qp_mat(
            self.qp_cuu.as_mut(),
            self.qp_c_star.as_ref(),
            self.qp_rr0.as_ref(),
        );
        calc_qp_m_eta_rho(
            self.qp_m.as_mut(),
            self.qp_eta.as_mut(),
            self.qp_rho.as_mut(),
            self.qp_muu.as_ref(),
        );
        calc_qp_strain(
            self.qp_strain.as_mut(),
            self.qp_x0_prime.subrows(0, 3),
            self.qp_u.subrows(3, 4),
            self.qp_u_prime.subrows(0, 3),
            self.qp_u_prime.subrows(3, 4),
        );
        calc_qp_x0pupss(
            self.qp_x0pupss.as_mut(),
            self.qp_x0_prime.subrows(0, 3),
            self.qp_u_prime.subrows(0, 3),
        );
        calc_qp_fc(
            self.qp_fc.as_mut(),
            self.qp_cuu.as_ref(),
            self.qp_strain.as_ref(),
        );
        calc_qp_fd(
            self.qp_fd.as_mut(),
            self.qp_fc.as_ref(),
            self.qp_x0pupss.as_ref(),
        );
        calc_qp_fi(
            self.qp_fi.as_mut(),
            self.qp_m.as_ref(),
            self.qp_v.subrows(3, 3).as_ref(),  // omega
            self.qp_vd.subrows(0, 3).as_ref(), // u_ddot
            self.qp_vd.subrows(3, 3).as_ref(), // omega_dot
            self.qp_eta.as_ref(),
            self.qp_rho.as_ref(),
        );
        calc_qp_fg(
            self.qp_fg.as_mut(),
            self.gravity.as_ref(),
            self.qp_m.as_ref(),
            self.qp_eta.as_ref(),
        );
        calc_qp_ouu(
            self.qp_ouu.as_mut(),
            self.qp_cuu.as_ref(),
            self.qp_x0pupss.as_ref(),
            self.qp_fc.as_ref(),
        );
        calc_qp_puu(
            self.qp_puu.as_mut(),
            self.qp_cuu.as_ref(),
            self.qp_x0pupss.as_ref(),
            self.qp_fc.as_ref(),
        );
        calc_qp_quu(
            self.qp_quu.as_mut(),
            self.qp_cuu.as_ref(),
            self.qp_x0pupss.as_ref(),
            self.qp_fc.as_ref(),
        );
        calc_qp_guu(
            self.qp_guu.as_mut(),
            self.qp_m.as_ref(),
            self.qp_eta.as_ref(),
            self.qp_rho.as_ref(),
            self.qp_v.subrows(3, 3),
        );
        calc_qp_kuu(
            self.qp_kuu.as_mut(),
            self.qp_m.as_ref(),
            self.qp_eta.as_ref(),
            self.qp_rho.as_ref(),
            self.qp_v.subrows(3, 3),
            self.qp_vd.subrows(0, 3),
            self.qp_vd.subrows(3, 3),
        );
    }

    #[inline]
    fn integrate_forces(&mut self) {
        for ei in self.elem_index.iter() {
            let shape_interp = self
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let shape_deriv = self
                .shape_deriv
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let qp_w = self.qp_weight.subrows(ei.i_qp_start, ei.n_qps);
            let qp_j = self.qp_jacobian.subrows(ei.i_qp_start, ei.n_qps);
            integrate_fe(
                self.node_fe.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp_fc.subcols(ei.i_qp_start, ei.n_qps),
                self.qp_fd.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                shape_deriv,
                qp_w,
                qp_j,
            );
            integrate_f(
                self.node_fi.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp_fi.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                qp_w,
                qp_j,
            );
            integrate_f(
                self.node_fx.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp_fx.subcols(ei.i_qp_start, ei.n_qps),
                shape_interp,
                qp_w,
                qp_j,
            );
            integrate_f(
                self.node_fg.subcols_mut(ei.i_node_start, ei.n_nodes),
                self.qp_fg.subcols(ei.i_qp_start, ei.n_qps),
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
        for ei in self.elem_index.iter() {
            let shape_interp = self
                .shape_interp
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let shape_deriv = self
                .shape_deriv
                .submatrix(ei.i_qp_start, 0, ei.n_qps, ei.n_nodes);
            let qp_w = self.qp_weight.subrows(ei.i_qp_start, ei.n_qps);
            let qp_j = self.qp_jacobian.subrows(ei.i_qp_start, ei.n_qps);

            let node_muu = self
                .node_muu
                .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            let qp_muu = self.qp_muu.subcols(ei.i_qp_start, ei.n_qps);
            (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .zip(node_muu.col_iter_mut())
                .for_each(|((i, j), mut muu_col)| {
                    integrate_matrix(i, j, muu_col.as_mut(), qp_muu, shape_interp, qp_w, qp_j);
                });

            let node_guu = self
                .node_guu
                .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            let qp_guu = self.qp_guu.subcols(ei.i_qp_start, ei.n_qps);
            (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .zip(node_guu.col_iter_mut())
                .for_each(|((i, j), guu_col)| {
                    integrate_matrix(i, j, guu_col, qp_guu, shape_interp, qp_w, qp_j);
                });

            let mut node_ki = self
                .node_ki
                .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            let qp_kuu = self.qp_kuu.subcols(ei.i_qp_start, ei.n_qps);
            (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .zip(node_ki.as_mut().col_iter_mut())
                .for_each(|((i, j), mut ki_col)| {
                    integrate_matrix(
                        i,
                        j,
                        ki_col.as_mut(),
                        qp_kuu.as_ref(),
                        shape_interp.as_ref(),
                        qp_w.as_ref(),
                        qp_j.as_ref(),
                    );
                });

            let mut node_ke = self
                .node_ke
                .subcols_mut(ei.i_mat_start, ei.n_nodes * ei.n_nodes);
            let qp_puu = self.qp_puu.subcols(ei.i_qp_start, ei.n_qps);
            let qp_quu = self.qp_quu.subcols(ei.i_qp_start, ei.n_qps);
            let qp_cuu = self.qp_cuu.subcols(ei.i_qp_start, ei.n_qps);
            let qp_ouu = self.qp_ouu.subcols(ei.i_qp_start, ei.n_qps);
            (0..ei.n_nodes)
                .cartesian_product(0..ei.n_nodes)
                .zip(node_ke.as_mut().col_iter_mut())
                .for_each(|((i, j), mut ke_col)| {
                    integrate_elastic_stiffness_matrix(
                        i,
                        j,
                        ke_col.as_mut(),
                        qp_puu.as_ref(),
                        qp_quu.as_ref(),
                        qp_cuu.as_ref(),
                        qp_ouu.as_ref(),
                        shape_interp.as_ref(),
                        shape_deriv.as_ref(),
                        qp_w.as_ref(),
                        qp_j.as_ref(),
                    );
                });
        }
    }
}

#[inline]
// Calculate current position and rotation (x0 + u)
fn calc_qp_x(x: MatMut<f64>, x0: MatRef<f64>, u: MatRef<f64>) {
    izip!(x.col_iter_mut(), x0.col_iter(), u.col_iter()).for_each(|(mut x, x0, u)| {
        x[0] = x0[0] + u[0];
        x[1] = x0[1] + u[1];
        x[2] = x0[2] + u[2];
        x.subrows_mut(3, 4)
            .quat_compose(u.subrows(3, 4), x0.subrows(3, 4));
    });
}

#[inline]
fn calc_qp_rr0(rr0: MatMut<f64>, x: MatRef<f64>) {
    let mut m = Mat::<f64>::zeros(3, 3);
    izip!(rr0.col_iter_mut(), x.subrows(3, 4).col_iter()).for_each(|(col, r)| {
        let mut rr0 = col.as_mat_mut(6, 6);
        quat_as_matrix(r, m.as_mut());
        rr0.as_mut().submatrix_mut(0, 0, 3, 3).copy_from(&m);
        rr0.as_mut().submatrix_mut(3, 3, 3, 3).copy_from(&m);
    });
}

#[inline]
fn calc_qp_mat(mat: MatMut<f64>, mat_star: MatRef<f64>, rr0: MatRef<f64>) {
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
fn calc_qp_m_eta_rho(m: ColMut<f64>, eta: MatMut<f64>, rho: MatMut<f64>, muu: MatRef<f64>) {
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
fn calc_qp_strain(
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

#[inline]
fn calc_qp_x0pupss(x0pupss: MatMut<f64>, x0_prime: MatRef<f64>, u_prime: MatRef<f64>) {
    let mut x0pup = Col::<f64>::zeros(3);
    izip!(
        x0pupss.col_iter_mut(),
        x0_prime.col_iter(),
        u_prime.col_iter()
    )
    .for_each(|(x0pupss_col, x0_prime, u_prime)| {
        zipped!(&mut x0pup, x0_prime, u_prime)
            .for_each(|unzipped!(mut x0pup, x0p, up)| *x0pup = *x0p + *up);
        vec_tilde2(x0pup.as_ref(), x0pupss_col.as_mat_mut(3, 3));
    });
}

#[inline]
fn calc_qp_fc(fc: MatMut<f64>, cuu: MatRef<f64>, strain: MatRef<f64>) {
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
fn calc_qp_fd(fd: MatMut<f64>, fc: MatRef<f64>, x0pupss: MatRef<f64>) {
    izip!(fd.col_iter_mut(), fc.col_iter(), x0pupss.col_iter(),).for_each(
        |(fd, fc, x0pupss_col)| {
            matmul(
                fd.subrows_mut(3, 3),
                x0pupss_col.as_mat_ref(3, 3).transpose(),
                fc.subrows(0, 3), // N
                None,
                1.0,
                Parallelism::None,
            );
        },
    );
}

#[inline]
fn calc_qp_fi(
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
        vec_tilde2(eta, eta_tilde.as_mut());
        vec_tilde2(omega, omega_tilde.as_mut());
        vec_tilde2(omega_dot, omega_dot_tilde.as_mut());
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
fn calc_qp_fg(fg: MatMut<f64>, gravity: ColRef<f64>, m: ColRef<f64>, eta: MatRef<f64>) {
    let mut eta_tilde = Mat::<f64>::zeros(3, 3);
    izip!(fg.col_iter_mut(), m.iter(), eta.col_iter(),).for_each(|(mut fg, &m, eta)| {
        vec_tilde2(eta, eta_tilde.as_mut());
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
fn calc_qp_ouu(ouu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
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
        vec_tilde2(fc.subrows(0, 3), ouu12.as_mut()); // n_tilde
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
        vec_tilde2(fc.subrows(3, 3), ouu22.as_mut()); // m_tilde
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
fn calc_qp_puu(puu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
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
        vec_tilde2(fc.subrows(0, 3), puu21.as_mut());
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
fn calc_qp_quu(quu: MatMut<f64>, cuu: MatRef<f64>, x0pupss: MatRef<f64>, fc: MatRef<f64>) {
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
        vec_tilde2(fc.subrows(0, 3), mat.as_mut()); // n_tilde

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
fn calc_qp_guu(
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
        vec_tilde2(eta, eta_tilde.as_mut());
        vec_tilde2(omega, omega_tilde.as_mut());

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
        vec_tilde2(
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
        vec_tilde2(rho_omega.as_ref(), rho_omega_tilde.as_mut());
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

#[inline]
fn calc_qp_kuu(
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
            vec_tilde2(eta, eta_tilde.as_mut());
            vec_tilde2(omega, omega_tilde.as_mut());
            vec_tilde2(omega_dot, omega_dot_tilde.as_mut());
            vec_tilde2(u_ddot, u_ddot_tilde.as_mut());
            vec_tilde2(rho_omega.as_ref(), rho_omega_g_tilde.as_mut());
            vec_tilde2(rho_omega_dot.as_ref(), rho_omega_dot_g_tilde.as_mut());

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
    .for_each(|(mut fe, phi, phi_d)| {
        acc.fill(0.);
        izip!(
            qp_w.iter(),
            qp_j.iter(),
            phi.iter(),
            phi_d.iter(),
            qp_fc.col_iter(),
            qp_fd.col_iter()
        )
        .for_each(|(&w, &j, &phi, &phip, fc, fd)| {
            zipped!(&mut acc, &fc, &fd)
                .for_each(|unzipped!(mut acc, fc, fd)| *acc += w * (*fc * phip + *fd * phi * j));
        });
        fe.copy_from(&acc);
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
    izip!(node_f.col_iter_mut(), shape_interp.col_iter(),).for_each(|(mut node_f, phi)| {
        acc.fill(0.);
        izip!(qp_w.iter(), qp_j.iter(), phi.iter(), qp_f.col_iter(),).for_each(
            |(&w, &j, &phi, qp_f)| {
                zipped!(&mut acc, &qp_f).for_each(|unzipped!(mut acc, f)| *acc += *f * phi * j * w);
            },
        );
        node_f.copy_from(&acc);
    });
}

#[inline]
fn integrate_matrix(
    node_i: usize,
    node_j: usize,
    mut node_mat: ColMut<f64>,
    qp_mat: MatRef<f64>,
    shape_interp: MatRef<f64>,
    qp_w: ColRef<f64>,
    qp_j: ColRef<f64>,
) {
    let mut acc = Col::<f64>::zeros(6 * 6);
    let c = zipped!(
        &qp_w,
        &qp_j,
        &shape_interp.col(node_i),
        &shape_interp.col(node_j)
    )
    .map(|unzipped!(w, j, phi_i, phi_j)| (*w) * (*j) * (*phi_i) * (*phi_j));
    izip!(qp_mat.col_iter(), c.iter()).for_each(|(mat, &c)| {
        zipped!(&mut acc, &mat).for_each(|unzipped!(mut acc, mat)| *acc += *mat * c)
    });
    node_mat.copy_from(&acc);
}

fn integrate_elastic_stiffness_matrix(
    node_i: usize,
    node_j: usize,
    mut node_mat: ColMut<f64>,
    qp_puu: MatRef<f64>,
    qp_quu: MatRef<f64>,
    qp_cuu: MatRef<f64>,
    qp_ouu: MatRef<f64>,
    shape_interp: MatRef<f64>,
    shape_deriv: MatRef<f64>,
    qp_w: ColRef<f64>,
    qp_j: ColRef<f64>,
) {
    let phi_i = shape_interp.col(node_i);
    let phi_j = shape_interp.col(node_j);
    let phip_i = shape_deriv.col(node_i);
    let phip_j = shape_deriv.col(node_j);

    // Matrix to sum quadrature point contributions
    let mut acc = Col::<f64>::zeros(6 * 6);

    // Column of constants
    let mut c = Col::<f64>::zeros(qp_w.nrows());

    // Puu contribution
    zipped!(&mut c, &qp_w, &phi_i, &phip_j)
        .for_each(|unzipped!(mut c, w, phi_i, phip_j)| *c = *w * *phi_i * *phip_j);
    // izip!(c.iter(), qp_puu.col_iter()).for_each(|(&c, puu)| acc += puu * c);
    izip!(c.iter(), qp_puu.col_iter()).for_each(|(&c, puu)| {
        zipped!(&mut acc, &puu).for_each(|unzipped!(mut acc, puu)| *acc += *puu * c)
    });

    // Quu contribution
    zipped!(&mut c, &qp_w, &qp_j, &phi_i, &phi_j)
        .for_each(|unzipped!(mut c, w, j, phi_i, phi_j)| *c = *w * *j * *phi_i * *phi_j);
    // izip!(c.iter(), qp_quu.col_iter()).for_each(|(&c, quu)| acc += quu * c);
    izip!(c.iter(), qp_quu.col_iter()).for_each(|(&c, quu)| {
        zipped!(&mut acc, &quu).for_each(|unzipped!(mut acc, quu)| *acc += *quu * c)
    });

    // Cuu contribution
    zipped!(&mut c, &qp_w, &qp_j, &phip_i, &phip_j)
        .for_each(|unzipped!(mut c, w, j, phip_i, phip_j)| *c = *w * *phip_i * *phip_j / *j);
    // izip!(c.iter(), qp_cuu.col_iter()).for_each(|(&c, cuu)| acc += cuu * c);
    izip!(c.iter(), qp_cuu.col_iter()).for_each(|(&c, cuu)| {
        zipped!(&mut acc, &cuu).for_each(|unzipped!(mut acc, cuu)| *acc += *cuu * c)
    });

    // Ouu contribution
    zipped!(&mut c, &qp_w, &phip_i, &phi_j)
        .for_each(|unzipped!(mut c, w, phip_i, phi_j)| *c = *w * *phip_i * *phi_j);
    // izip!(c.iter(), qp_ouu.col_iter()).for_each(|(&c, ouu)| acc += ouu * c);
    izip!(c.iter(), qp_ouu.col_iter()).for_each(|(&c, ouu)| {
        zipped!(&mut acc, &ouu).for_each(|unzipped!(mut acc, ouu)| *acc += *ouu * c)
    });

    // Copy values into node matrix
    node_mat.copy_from(acc);
}

pub fn vec_tilde(v: ColRef<f64>) -> Mat<f64> {
    mat![[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]]
}

pub fn vec_tilde2(v: ColRef<f64>, mut m: MatMut<f64>) {
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

pub trait ColAsMatMut<'a, T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T>;
    fn as_mat_mut(self, nrows: usize, ncols: usize) -> MatMut<'a, T>;
}

impl<'a, T> ColAsMatMut<'a, T> for Col<T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
    fn as_mat_mut(mut self, nrows: usize, ncols: usize) -> MatMut<'a, T> {
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, ncols, 1, nrows as isize) }
    }
}

impl<'a, T> ColAsMatMut<'a, T> for ColMut<'a, T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
    fn as_mat_mut(self, nrows: usize, ncols: usize) -> MatMut<'a, T> {
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, ncols, 1, nrows as isize) }
    }
}

pub trait ColAsMatRef<'a, T>
where
    T: Entity,
{
    fn as_mat_ref(self, nrows: usize, ncols: usize) -> MatRef<'a, T>;
}

impl<'a, T> ColAsMatRef<'a, T> for ColRef<'a, T>
where
    T: Entity,
{
    fn as_mat_ref(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{interp::gauss_legendre_lobotto_points, node::NodeBuilder, quadrature::Quadrature};
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

        let nodes = vec![
            NodeBuilder::new(0)
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
            NodeBuilder::new(1)
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
            NodeBuilder::new(2)
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
            NodeBuilder::new(3)
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
            NodeBuilder::new(4)
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

        let input = BeamInput {
            gravity: [0., 0., 9.81],
            elements: vec![BeamElement {
                nodes: izip!(node_s.iter(), nodes.iter())
                    .map(|(&si, n)| BeamNode::new(si, n))
                    .collect_vec(),
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

        Beams::new(&input, &nodes)
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
            beams.qp_m_star.col(0).as_mat_ref(6, 6),
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
            beams.qp_c_star.col(0).as_mat_ref(6, 6),
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
            beams.qp_strain.subcols(0, 2).transpose(),
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
            beams.qp_rr0.col(0).as_mat_ref(6, 6),
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
            beams.qp_muu.col(0).as_mat_ref(6, 6),
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
            beams.qp_cuu.col(0).as_mat_ref(6, 6),
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
            beams.qp_fc.subcols(0, 2).transpose(),
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
            beams.qp_fi.subcols(0, 2).transpose(),
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
            beams.qp_fd.subcols(0, 2).transpose(),
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
            beams.qp_fg.subcols(0, 2).transpose(),
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
            beams.qp_ouu.col(0).as_mat_ref(6, 6),
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
            beams.qp_puu.col(0).as_mat_ref(6, 6),
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
            beams.qp_quu.col(0).as_mat_ref(6, 6),
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
            beams.qp_guu.col(0).as_mat_ref(6, 6),
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
            beams.qp_kuu.col(0).as_mat_ref(6, 6),
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

        let nodes = node_s
            .iter()
            .enumerate()
            .map(|(i, &si)| {
                let mut r = Col::<f64>::zeros(4);
                r.as_mut().quat_from_rotation_matrix(rot(si).as_ref());
                NodeBuilder::new(i)
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
        let mut m_star = Mat::<f64>::zeros(6, 6);
        m_star
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(Mat::<f64>::identity(3, 3) * m);
        m_star
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(m * vec_tilde(eta_star.as_ref()).transpose());
        m_star
            .submatrix_mut(3, 0, 3, 3)
            .copy_from(m * vec_tilde(eta_star.as_ref()));
        m_star
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(Mat::from_fn(3, 3, |i, j| ((i + 1) * (j + 1)) as f64));

        let c_star = Mat::from_fn(6, 6, |i, j| ((i + 1) * (j + 1)) as f64);

        // Create quadrature points and weights
        let gq = Quadrature::gauss(7);

        // Create element from nodes
        let sections = vec![
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
        ];

        let input = BeamInput {
            gravity: [0., 0., 9.81],
            elements: vec![BeamElement {
                nodes: izip!(node_s.iter(), nodes.iter())
                    .map(|(&si, n)| BeamNode::new(si, n))
                    .collect_vec(),
                quadrature: gq,
                sections,
            }],
        };

        Beams::new(&input, &nodes)
    }

    #[test]
    fn test_qp_m_star2() {
        let beams = setup_test();
        assert_matrix_eq!(
            beams.qp_m_star.col(0).as_mat_ref(6, 6),
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
            beams.qp_c_star.col(0).as_mat_ref(6, 6),
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
            beams.qp_u.col(0).as_2d(),
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
            beams.qp_u_prime.col(0).as_2d(),
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
            beams.qp_u.col(0).subrows(3, 4),
            beams.qp_x0.col(0).subrows(3, 4),
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
            beams.qp_strain.col(0).as_2d(),
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
            beams.qp_cuu.col(0).as_mat_ref(6, 6),
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
            beams.qp_fc.col(0).as_2d(),
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
            beams.qp_fd.col(0).as_2d(),
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
            beams.qp_fi.col(0).as_2d(),
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

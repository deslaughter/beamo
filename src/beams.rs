use std::iter::zip;
use std::rc::Rc;

use crate::interp::{shape_interp_matrix, unit_vector};
use crate::quadrature::Quadrature;
use crate::quaternion::{Quat, Quaternion};
use crate::{interp::shape_deriv_matrix, node::Node};
use faer::linalg::matmul;
use faer::{mat, Mat, Parallelism, Row};
use itertools::{multiunzip, Itertools};

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
    pub s: f64,                // Distance along centerline from first point
    pub m_star: [[f64; 6]; 6], // Mass matrix
    pub c_star: [[f64; 6]; 6], // Stiffness matrix
}

pub struct ElemIndex {
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
    qp_weight: Row<f64>,   // [n_qps]        Integration weights
    qp_jacobian: Row<f64>, // [n_qps]        Jacobian vector
    qp_m_star: Mat<f64>,   // [n_qps][6][6]  Mass matrix in material frame
    qp_c_star: Mat<f64>,   // [n_qps][6][6]  Stiffness matrix in material frame
    qp_x: Mat<f64>,        // [n_qps][7]     Current position/orientation
    qp_x0: Mat<f64>,       // [n_qps][7]     Initial position
    qp_x0_prime: Mat<f64>, // [n_qps][7]     Initial position derivative
    qp_u: Mat<f64>,        // [n_qps][7]     State: translation displacement
    qp_u_prime: Mat<f64>,  // [n_qps][7]     State: translation displacement derivative
    qp_v: Mat<f64>,        // [n_qps][6]     State: translation velocity
    qp_vd: Mat<f64>,       // [n_qps][6]     State: translation acceleration
    qp_e: Mat<f64>,        // [n_qps][3][4]  Quaternion derivative
    qp_eta: Mat<f64>,      // [n_qps][3]     mass
    qp_rho: Mat<f64>,      // [n_qps][3][3]  mass
    qp_strain: Mat<f64>,   // [n_qps][6]     Strain
    qp_fc: Mat<f64>,       // [n_qps][6]     Elastic force
    qp_fd: Mat<f64>,       // [n_qps][6]     Elastic force
    qp_fi: Mat<f64>,       // [n_qps][6]     Inertial force
    qp_fx: Mat<f64>,       // [n_qps][6]     External force
    qp_fg: Mat<f64>,       // [n_qps][6]     Gravity force
    qp_rr0: Mat<f64>,      // [n_qps][6][6]  Global rotation
    qp_muu: Mat<f64>,      // [n_qps][6][6]  Mass in global frame
    qp_cuu: Mat<f64>,      // [n_qps][6][6]  Stiffness in global frame
    qp_ouu: Mat<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_puu: Mat<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_quu: Mat<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_guu: Mat<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_kuu: Mat<f64>,      // [n_qps][6][6]  Linearization matrices

    residual_vector_terms: Mat<f64>,  // [n_qps][6]
    stiffness_matrix_terms: Mat<f64>, // [n_qps][6][6]
    inertia_matrix_terms: Mat<f64>,   // [n_qps][6][6]

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
        for e in inp.elements.iter() {
            let n_nodes = e.nodes.len();
            let n_qps = e.quadrature.points.len();
            index.push(ElemIndex {
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
                alloc_nodes,
                7,
                |i, j| if i < x0.len() { x0[i][j] } else { 0. },
            ),
            node_u: Mat::from_fn(
                alloc_nodes,
                7,
                |i, j| if i < u.len() { u[i][j] } else { 0. },
            ),
            node_v: Mat::from_fn(
                alloc_nodes,
                6,
                |i, j| if i < v.len() { v[i][j] } else { 0. },
            ),
            node_vd: Mat::from_fn(
                alloc_nodes,
                6,
                |i, j| if i < vd.len() { vd[i][j] } else { 0. },
            ),
            node_fx: Mat::zeros(alloc_nodes, 6),

            // Quadrature points
            qp_weight: Row::from_fn(alloc_qps, |i| {
                if i < qp_weights.len() {
                    qp_weights[0]
                } else {
                    0.
                }
            }),
            qp_jacobian: Row::zeros(alloc_qps),
            qp_m_star: Mat::zeros(alloc_qps, 36), // 6x6
            qp_c_star: Mat::zeros(alloc_qps, 36), // 6x6
            qp_x: Mat::zeros(alloc_qps, 7),
            qp_x0: Mat::zeros(alloc_qps, 7),
            qp_x0_prime: Mat::zeros(alloc_qps, 7),
            qp_u: Mat::zeros(alloc_qps, 7),
            qp_u_prime: Mat::zeros(alloc_qps, 7),
            qp_v: Mat::zeros(alloc_qps, 6),
            qp_vd: Mat::zeros(alloc_qps, 6),
            qp_e: Mat::zeros(alloc_qps, 12), // 3x4
            qp_eta: Mat::zeros(alloc_qps, 3),
            qp_rho: Mat::zeros(alloc_qps, 9), // 3x3
            qp_strain: Mat::zeros(alloc_qps, 6),
            qp_fc: Mat::zeros(alloc_qps, 6),
            qp_fd: Mat::zeros(alloc_qps, 6),
            qp_fi: Mat::zeros(alloc_qps, 6),
            qp_fx: Mat::zeros(alloc_qps, 6),
            qp_fg: Mat::zeros(alloc_qps, 6),
            qp_rr0: Mat::zeros(alloc_qps, 36), // 6x6
            qp_muu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_cuu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_ouu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_puu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_quu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_guu: Mat::zeros(alloc_qps, 36), // 6x6
            qp_kuu: Mat::zeros(alloc_qps, 36), // 6x6

            residual_vector_terms: Mat::zeros(alloc_nodes, 6),
            stiffness_matrix_terms: Mat::zeros(alloc_nodes, 36), //6x6
            inertia_matrix_terms: Mat::zeros(alloc_nodes, 36),   //6x6

            shape_interp: Mat::zeros(alloc_qps, max_elem_nodes),
            shape_deriv: Mat::zeros(alloc_qps, max_elem_nodes),
        };

        //----------------------------------------------------------------------
        // Shape functions
        //----------------------------------------------------------------------

        // Initialize element shape functions for interpolation and derivative
        for (ei, e) in beams.index.iter().zip(inp.elements.iter()) {
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
            let node_x0 = beams.node_x0.subrows(ei.node_range[0], ei.n_nodes);
            let mut qp_x0 = beams.qp_x0.subrows_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_x0.as_mut(),
                shape_interp,
                node_x0,
                None,
                1.,
                Parallelism::None,
            );
            qp_x0.subcols_mut(3, 4).row_iter_mut().for_each(|mut r| {
                let m = r.norm_l2();
                if m != 0. {
                    r /= m;
                }
            });

            // Get shape derivative matrix for this element
            let shape_deriv = beams
                .shape_deriv
                .submatrix(ei.qp_range[0], 0, ei.n_qps, ei.n_nodes);

            // Calculate Jacobian
            let mut qp_jacobian = beams.qp_jacobian.subcols_mut(ei.qp_range[0], ei.n_qps);
            let mut qp_x0_prime = Mat::<f64>::zeros(ei.n_qps, 3);
            matmul::matmul(
                qp_x0_prime.as_mut(),
                shape_deriv,
                node_x0.subcols(0, 3),
                None,
                1.0,
                Parallelism::None,
            );
            qp_jacobian
                .as_mut()
                .iter_mut()
                .zip(qp_x0_prime.row_iter())
                .for_each(|(j, x0_prime)| *j = x0_prime.norm_l2());

            print!("{:?}", qp_jacobian);
        }

        beams
    }

    fn interp_to_qps(mut self) {
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
            let node_u = self.node_u.subrows(ei.qp_range[0], ei.n_qps);
            let mut qp_u = self.qp_u.subrows_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_u.as_mut(),
                shape_interp,
                node_u,
                None,
                1.,
                Parallelism::None,
            );
            qp_u.as_mut()
                .subcols_mut(3, 4)
                .row_iter_mut()
                .for_each(|mut r| {
                    let m = r.norm_l2();
                    if m != 0. {
                        r /= m;
                    }
                });

            // Get Jacobians for this element
            let qp_jacobian = self.qp_jacobian.subcols(ei.qp_range[0], ei.n_qps);

            // Displacement derivative
            let mut qp_u_prime = self.qp_u_prime.subrows_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_u_prime.as_mut(),
                shape_deriv,
                node_u.subcols(0, 3),
                None,
                1.0,
                Parallelism::None,
            );
            qp_u_prime
                .row_iter_mut()
                .zip(qp_jacobian.iter())
                .for_each(|(mut row, &jacobian)| row /= jacobian);

            // Interpolate velocity
            let node_v = self.node_v.subrows(ei.node_range[0], ei.n_nodes);
            let mut qp_v = self.qp_v.subrows_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_v.as_mut(),
                shape_interp,
                node_v.subcols(0, 3),
                None,
                1.0,
                Parallelism::None,
            );

            // Interpolate acceleration
            let node_vd = self.node_vd.subrows(ei.node_range[0], ei.n_nodes);
            let mut qp_vd = self.qp_vd.subrows_mut(ei.qp_range[0], ei.n_qps);
            matmul::matmul(
                qp_vd.as_mut(),
                shape_interp,
                node_vd.subcols(0, 3),
                None,
                1.0,
                Parallelism::None,
            );

            // Calculate current position and rotation
            let qp_x = self.qp_x.subrows_mut(ei.qp_range[0], ei.n_qps);
            let qp_x0 = self.qp_x0.subrows(ei.qp_range[0], ei.n_qps);
            qp_x.row_iter_mut()
                .zip(qp_x0.row_iter())
                .zip(qp_u.row_iter())
                .for_each(|((mut x, x0), u)| {
                    x[0] = x0[0] + u[0];
                    x[1] = x0[1] + u[1];
                    x[2] = x0[2] + u[2];
                    x.subcols_mut(3, 4)
                        .quat_compose(u.subcols(3, 4), x0.subcols(3, 4));
                });
        }
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use approx::assert_relative_eq;
    use faer::row;
    use itertools::Itertools;

    use crate::interp::gauss_legendre_lobotto_points;
    use crate::quadrature::Quadrature;
    use crate::quaternion::Quaternion;

    #[test]
    fn test_me() {
        let xi = gauss_legendre_lobotto_points(5);
        let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

        // Quadrature rule
        let gq = Quadrature::gauss(7);

        // Node initial position and rotation
        let r0 = row![1., 0., 0., 0.];
        let fx = |s: f64| -> f64 { 10. * s + 2. };

        let mut mass = [
            [8.538, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 8.538, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 8.538, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 1.4433, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.40972, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 1.0336],
        ];
        mass.iter_mut()
            .for_each(|row| row.iter_mut().for_each(|v| *v = *v * 1e-2));

        let mut stiffness = [
            [1368.17, 0., 0., 0., 0., 0.],
            [0., 88.56, 0., 0., 0., 0.],
            [0., 0., 38.78, 0., 0., 0.],
            [0., 0., 0., 16.960, 17.610, -0.351],
            [0., 0., 0., 17.610, 59.120, -0.370],
            [0., 0., 0., -0.351, -0.370, 141.47],
        ];
        stiffness
            .iter_mut()
            .for_each(|row| row.iter_mut().for_each(|v| *v = *v * 1e3));

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
                        m_star: mass.clone(),
                        c_star: stiffness.clone(),
                    },
                    BeamSection {
                        s: 1.,
                        m_star: mass.clone(),
                        c_star: stiffness.clone(),
                    },
                ],
            }],
        };

        let _beams = Beams::new(&input);
    }
}

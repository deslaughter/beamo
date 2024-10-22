use std::rc::Rc;

use crate::interp::{shape_interp_matrix, unit_vector};
use crate::quadrature::Quadrature;
use crate::quaternion::Quaternion;
use crate::{interp::shape_deriv_matrix, node::Node};
use itertools::Itertools;
use ndarray::{arr1, arr2, azip, s, Array1, Array2, Array3, Axis, Zip};

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
    gravity: Array1<f64>, // [3] gravity

    // Node-based data
    node_x0: Array2<f64>, // [n_nodes][7] Inital position/rotation
    node_u: Array2<f64>,  // [n_nodes][7] State: translation/rotation displacement
    node_v: Array2<f64>,  // [n_nodes][6] State: translation/rotation velocity
    node_vd: Array2<f64>, // [n_nodes][6] State: translation/rotation acceleration
    node_fx: Array2<f64>, // [n_nodes][6] External forces

    // Quadrature point data
    qp_weight: Array1<f64>,   // [n_qps]        Integration weights
    qp_jacobian: Array1<f64>, // [n_qps]        Jacobian vector
    qp_m_star: Array3<f64>,   // [n_qps][6][6]  Mass matrix in material frame
    qp_c_star: Array3<f64>,   // [n_qps][6][6]  Stiffness matrix in material frame
    qp_x: Array2<f64>,        // [n_qps][7]     Current position/orientation
    qp_x0: Array2<f64>,       // [n_qps][7]     Initial position
    qp_x0_prime: Array2<f64>, // [n_qps][7]     Initial position derivative
    qp_u: Array2<f64>,        // [n_qps][7]     State: translation displacement
    qp_u_prime: Array2<f64>,  // [n_qps][7]     State: translation displacement derivative
    qp_v: Array2<f64>,        // [n_qps][6]     State: translation velocity
    qp_vd: Array2<f64>,       // [n_qps][6]     State: translation acceleration
    qp_e: Array3<f64>,        // [n_qps][3][4]  Quaternion derivative
    qp_eta: Array2<f64>,      // [n_qps][3]     mass
    qp_rho: Array3<f64>,      // [n_qps][3][3]  mass
    qp_strain: Array2<f64>,   // [n_qps][6]     Strain
    qp_fc: Array2<f64>,       // [n_qps][6]     Elastic force
    qp_fd: Array2<f64>,       // [n_qps][6]     Elastic force
    qp_fi: Array2<f64>,       // [n_qps][6]     Inertial force
    qp_fx: Array2<f64>,       // [n_qps][6]     External force
    qp_fg: Array2<f64>,       // [n_qps][6]     Gravity force
    qp_rr0: Array3<f64>,      // [n_qps][6][6]  Global rotation
    qp_muu: Array3<f64>,      // [n_qps][6][6]  Mass in global frame
    qp_cuu: Array3<f64>,      // [n_qps][6][6]  Stiffness in global frame
    qp_ouu: Array3<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_puu: Array3<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_quu: Array3<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_guu: Array3<f64>,      // [n_qps][6][6]  Linearization matrices
    qp_kuu: Array3<f64>,      // [n_qps][6][6]  Linearization matrices

    residual_vector_terms: Array2<f64>,  // [n_qps][6]
    stiffness_matrix_terms: Array3<f64>, // [n_qps][6][6]
    inertia_matrix_terms: Array3<f64>,   // [n_qps][6][6]

    // Shape Function data
    shape_interp: Array2<f64>, // [n_qps][max_nodes] Shape function values
    shape_deriv: Array2<f64>,  // [n_qps][max_nodes] Shape function derivatives
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

        let mut beams = Self {
            index,
            node_ids,
            gravity: arr1(&inp.gravity),

            // Nodes
            node_x0: Array2::zeros((alloc_nodes, 7)),
            node_u: Array2::zeros((alloc_nodes, 7)),
            node_v: Array2::zeros((alloc_nodes, 6)),
            node_vd: Array2::zeros((alloc_nodes, 6)),
            node_fx: Array2::zeros((alloc_nodes, 6)),

            // Quadrature points
            qp_weight: Array1::zeros(alloc_qps),
            qp_jacobian: Array1::zeros(alloc_qps),
            qp_m_star: Array3::zeros((alloc_qps, 6, 6)),
            qp_c_star: Array3::zeros((alloc_qps, 6, 6)),
            qp_x: Array2::zeros((alloc_qps, 7)),
            qp_x0: Array2::zeros((alloc_qps, 7)),
            qp_x0_prime: Array2::zeros((alloc_qps, 7)),
            qp_u: Array2::zeros((alloc_qps, 7)),
            qp_u_prime: Array2::zeros((alloc_qps, 7)),
            qp_v: Array2::zeros((alloc_qps, 6)),
            qp_vd: Array2::zeros((alloc_qps, 6)),
            qp_e: Array3::zeros((alloc_qps, 3, 4)),
            qp_eta: Array2::zeros((alloc_qps, 3)),
            qp_rho: Array3::zeros((alloc_qps, 3, 3)),
            qp_strain: Array2::zeros((alloc_qps, 6)),
            qp_fc: Array2::zeros((alloc_qps, 6)),
            qp_fd: Array2::zeros((alloc_qps, 6)),
            qp_fi: Array2::zeros((alloc_qps, 6)),
            qp_fx: Array2::zeros((alloc_qps, 6)),
            qp_fg: Array2::zeros((alloc_qps, 6)),
            qp_rr0: Array3::zeros((alloc_qps, 6, 6)),
            qp_muu: Array3::zeros((alloc_qps, 6, 6)),
            qp_cuu: Array3::zeros((alloc_qps, 6, 6)),
            qp_ouu: Array3::zeros((alloc_qps, 6, 6)),
            qp_puu: Array3::zeros((alloc_qps, 6, 6)),
            qp_quu: Array3::zeros((alloc_qps, 6, 6)),
            qp_guu: Array3::zeros((alloc_qps, 6, 6)),
            qp_kuu: Array3::zeros((alloc_qps, 6, 6)),

            residual_vector_terms: Array2::zeros((alloc_nodes, 6)),
            stiffness_matrix_terms: Array3::zeros((alloc_nodes, 6, 6)),
            inertia_matrix_terms: Array3::zeros((alloc_nodes, 6, 6)),

            shape_interp: Array2::zeros((alloc_qps, max_elem_nodes)),
            shape_deriv: Array2::zeros((alloc_qps, max_elem_nodes)),
        };

        //----------------------------------------------------------------------
        // Node data
        //----------------------------------------------------------------------

        // Set node initial positions
        beams
            .node_x0
            .slice_mut(s![0..total_nodes, ..])
            .assign(&arr2(
                &inp.elements
                    .iter()
                    .flat_map(|e| e.nodes.iter().map(|n| n.node.x).collect_vec())
                    .collect_vec(),
            ));

        // Set node initial displacement
        beams.node_u.slice_mut(s![0..total_nodes, ..]).assign(&arr2(
            &inp.elements
                .iter()
                .flat_map(|e| e.nodes.iter().map(|n| n.node.u).collect_vec())
                .collect_vec(),
        ));

        // Set node initial velocity
        beams.node_v.slice_mut(s![0..total_nodes, ..]).assign(&arr2(
            &inp.elements
                .iter()
                .flat_map(|e| e.nodes.iter().map(|n| n.node.v).collect_vec())
                .collect_vec(),
        ));

        // Set node initial acceleration
        beams
            .node_vd
            .slice_mut(s![0..total_nodes, ..])
            .assign(&arr2(
                &inp.elements
                    .iter()
                    .flat_map(|e| e.nodes.iter().map(|n| n.node.vd).collect_vec())
                    .collect_vec(),
            ));

        //----------------------------------------------------------------------
        // Shape functions
        //----------------------------------------------------------------------

        // Initialize element shape functions for interpolation and derivative
        for (ei, e) in beams.index.iter().zip(inp.elements.iter()) {
            let node_xi = e.nodes.iter().map(|n| 2. * n.si - 1.).collect_vec();

            // Get shape interpolation matrix for this element
            let mut shape_interp = beams
                .shape_interp
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Create interpolation matrix from quadrature point and node locations
            shape_interp.assign(&shape_interp_matrix(&e.quadrature.points, &node_xi));

            // Get shape derivative matrix for this element
            let mut shape_deriv = beams
                .shape_deriv
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Create derivative matrix from quadrature point and node locations
            shape_deriv.assign(&shape_deriv_matrix(&e.quadrature.points, &node_xi));
        }

        //----------------------------------------------------------------------
        // Quadrature points
        //----------------------------------------------------------------------

        // Set quadrature point weights
        beams.qp_weight.slice_mut(s![0..total_qps]).assign(&arr1(
            &inp.elements
                .iter()
                .flat_map(|e| e.quadrature.points.clone())
                .collect_vec(),
        ));

        for ei in beams.index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp = beams
                .shape_interp
                .slice(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Interpolate initial position
            let node_x0 = beams
                .node_x0
                .slice(s![ei.node_range[0]..ei.node_range[1], ..]);
            let mut qp_x0 = beams
                .qp_x0
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            ndarray::linalg::general_mat_mul(1., &shape_interp, &node_x0, 0., &mut qp_x0);
            for mut row in qp_x0.slice_mut(s![.., 3..7]).rows_mut() {
                let m = row.dot(&row).sqrt();
                if m != 0. {
                    row /= m;
                }
            }

            // Get shape derivative matrix for this element
            let shape_deriv = beams
                .shape_deriv
                .slice(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Calculate Jacobian
            let mut qp_jacobian = beams
                .qp_jacobian
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1]]);
            qp_jacobian.assign(&arr1(
                &shape_deriv
                    .dot(&node_x0.slice(s![.., 0..3]))
                    .rows()
                    .into_iter()
                    .map(|row| row.dot(&row).sqrt())
                    .collect_vec(),
            ));

            print!("{}", qp_jacobian);
        }

        beams
    }

    fn interp_to_qps(mut self) {
        for ei in self.index.iter() {
            // Get shape derivative matrix for this element
            let shape_interp = self
                .shape_interp
                .slice(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Get shape derivative matrix for this element
            let shape_deriv = self
                .shape_deriv
                .slice(s![ei.qp_range[0]..ei.qp_range[1], 0..ei.n_nodes]);

            // Interpolate displacement
            let node_u = self
                .node_u
                .slice(s![ei.node_range[0]..ei.node_range[1], ..]);
            let mut qp_u = self.qp_u.slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            ndarray::linalg::general_mat_mul(1., &shape_interp, &node_u, 0., &mut qp_u);
            for mut row in qp_u.slice_mut(s![.., 3..7]).rows_mut() {
                let m = row.dot(&row).sqrt();
                if m != 0. {
                    row /= m;
                }
            }

            let qp_jacobian = self
                .qp_jacobian
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1]]);

            // Displacement derivative
            let mut qp_u_prime = self
                .qp_u_prime
                .slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            ndarray::linalg::general_mat_mul(1., &shape_deriv, &node_u, 0., &mut qp_u_prime);
            for (mut row, &jacobian) in qp_u_prime.rows_mut().into_iter().zip(qp_jacobian.iter()) {
                row /= jacobian;
            }

            // Interpolate velocity
            let node_v = self
                .node_v
                .slice(s![ei.node_range[0]..ei.node_range[1], ..]);
            let mut qp_v = self.qp_v.slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            ndarray::linalg::general_mat_mul(1., &shape_interp, &node_v, 0., &mut qp_v);

            // Interpolate acceleration
            let node_vd = self
                .node_vd
                .slice(s![ei.node_range[0]..ei.node_range[1], ..]);
            let mut qp_vd = self.qp_vd.slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            ndarray::linalg::general_mat_mul(1., &shape_interp, &node_vd, 0., &mut qp_vd);

            // Calculate current position and rotation
            let mut qp_x = self.qp_x.slice_mut(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            let qp_x0 = self.qp_x0.slice(s![ei.qp_range[0]..ei.qp_range[1], ..]);
            Zip::from(qp_x.rows_mut())
                .and(qp_x0.rows())
                .and(qp_u.rows())
                .for_each(|mut x, x0, u| {
                    let q = Quaternion::from_vec(&[u[3], u[4], u[5], u[6]])
                        .compose(&Quaternion::from_vec(&[x0[3], x0[4], x0[5], x0[6]]))
                        .as_vec();
                    x[0] = x0[0] + u[0];
                    x[1] = x0[1] + u[1];
                    x[2] = x0[2] + u[2];
                    x[3] = q[0];
                    x[4] = q[1];
                    x[5] = q[2];
                    x[6] = q[3];
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
    use itertools::Itertools;
    use ndarray::arr2;

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
        let r0 = Quaternion::identity().as_vec();
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

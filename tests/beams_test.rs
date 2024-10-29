use std::rc::Rc;

use faer::{mat, row, Scale};

use itertools::Itertools;
use ottr::{
    beams::{BeamElement, BeamInput, BeamNode, BeamSection, Beams},
    interp::gauss_legendre_lobotto_points,
    node::Node,
    quadrature::Quadrature,
};

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

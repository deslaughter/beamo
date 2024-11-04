use std::process;

use faer::{mat, Scale};

use itertools::{izip, Itertools};
use ottr::{
    beams::{BeamElement, BeamInput, BeamNode, BeamSection, Beams},
    interp::gauss_legendre_lobotto_points,
    node::NodeBuilder,
    quadrature::Quadrature,
    solver::{Solver, StepParameters},
    state::State,
};

fn main() {
    let xi = gauss_legendre_lobotto_points(4);
    let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Node initial position
    let nodes = s
        .iter()
        .enumerate()
        .map(|(i, &si)| {
            NodeBuilder::new(i)
                .position(10. * si + 2., 0., 0., 1., 0., 0., 0.)
                .build()
        })
        .collect_vec();

    // Mass matrix 6x6
    let m_star = mat![
        [8.538, 0.000, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 8.538, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 8.538, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 1.4433, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.40972, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.00000, 1.0336],
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

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    let input = BeamInput {
        gravity: [0., 0., 0.],
        elements: vec![BeamElement {
            nodes: izip!(s.iter(), nodes.iter())
                .map(|(&s, n)| BeamNode::new(s, n))
                .collect_vec(),
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

    let mut beams = Beams::new(&input, &nodes);
    let time_step = 0.005;

    let step_params = StepParameters::new(time_step, 0., 5);
    let mut solver = Solver::new(step_params, &nodes, &vec![]);

    let mut state = State::new(&nodes);

    for i in 2..10000 {
        solver.fx[(2, solver.n_nodes - 1)] = 100. * (10.0 * (i as f64) * time_step).sin();
        let res = solver.step(&mut state, &mut beams);
        if !res.converged {
            println!("failed!");
            process::exit(1);
        }
        // println!("{} {}", i, state.u[(2, solver.n_nodes - 1)]);
    }

    println!("success")
}

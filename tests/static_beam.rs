use beamo::{
    elements::beams::{BeamSection, Damping},
    model::Model,
    node::Direction,
    quadrature::Quadrature,
};
use equator::assert;
use faer::prelude::*;
use faer::utils::approx::*;
use itertools::Itertools;
use std::process;

#[test]
fn test_static_beam_curl() {
    (6..20).for_each(|n_nodes| {
        run_static_beam_curl(n_nodes, 8 * n_nodes + 1);
    });
}

fn run_static_beam_curl(n_nodes: usize, n_qps: usize) {
    // Create list of Y moments to apply to end of beam
    let my = vec![0., 10920.0, 21840.0, 32761.0, 43681.0, 54600.8803193906];

    // Model
    let mut model = Model::new();
    model.set_rho_inf(1.0);
    model.set_time_step(1.0);
    model.set_max_iter(12);
    model.set_static_solve();
    model.set_solver_tolerance(1e-5, 1e-3);

    //--------------------------------------------------------------------------
    // Initial configuration
    //--------------------------------------------------------------------------

    // Node locations
    let s = (0..n_nodes)
        .into_iter()
        .map(|v| (v as f64) / ((n_nodes - 1) as f64))
        .collect_vec();

    // Quadrature rule
    // let gq = Quadrature::gauss_legendre_lobotto(n_qps);
    // let gq = Quadrature::gauss(n_qps);
    let qp_xi = (0..n_qps)
        .map(|v| -1.0 + 2.0 * (v as f64) / ((n_qps - 1) as f64))
        .collect_vec();
    let gq = Quadrature::simpsons_rule(&qp_xi);

    // Add Nodes
    let node_ids = s
        .iter()
        .map(|&si| {
            model
                .add_node()
                .element_location(si)
                .position(10. * si, 0., 0., 1., 0., 0., 0.)
                .build()
        })
        .collect_vec();

    //--------------------------------------------------------------------------
    // Beam Element
    //--------------------------------------------------------------------------

    // Mass matrix (zeros)
    let m_star = Mat::zeros(6, 6);

    // Stiffness matrix 6x6
    let c_star = mat![
        [1770., 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 1770., 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 1770., 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 8.160, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 86.90, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 215.0],
    ] * Scale(1e3);

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

    //--------------------------------------------------------------------------
    // Create solver
    //--------------------------------------------------------------------------

    // Add constraint to root node
    model.add_prescribed_constraint(node_ids[0]);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Test solve of element with applied load
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    let tip_node_id = *node_ids.last().unwrap();

    //--------------------------------------------------------------------------
    // Loop through moments
    //--------------------------------------------------------------------------

    let mut u_tip = Mat::zeros(3, my.len());

    for (i, &m) in my.iter().enumerate() {
        // Apply moment to tip node about y axis
        state.fx[(Direction::RY as usize, tip_node_id)] = -m;

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Exit if failed to converge
        if !res.converged {
            println!("failed! iter={i}, {:?}", res);
            process::exit(1);
        }

        u_tip[(0, i)] = state.u[(Direction::X as usize, tip_node_id)];
        u_tip[(1, i)] = state.u[(Direction::Y as usize, tip_node_id)];
        u_tip[(2, i)] = state.u[(Direction::Z as usize, tip_node_id)];
        // println!("{:?}", u_tip.col(i))
    }

    println!("{} {}", n_nodes, u_tip[(0, my.len() - 1)]);

    // let approx_eq = CwiseMat(ApproxEq::eps() * 1000.);
    // assert!(
    //     u_tip ~
    //     mat![
    //         [0.0, 0.0, 0.0],
    //         [-2.4521424743945115, 0.0, 5.526921055643609],
    //         [-7.71393432218001, 0.0, 7.215325756355413],
    //         [-11.612495114867894, 0.0, 4.780938463888443],
    //         [-11.912220414483576, 0.0, 1.3458820860655774],
    //         [-10.00069041069354, 0.0, -1.2947392102224953e-5],
    //     ]
    //     .transpose()
    // );
}

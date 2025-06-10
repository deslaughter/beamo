use equator::assert;
use faer::prelude::*;
use faer::utils::approx::*;
use itertools::Itertools;
use ottr::{
    elements::beams::{BeamSection, Damping},
    model::Model,
    node::Direction,
    quadrature::Quadrature,
};
use std::process;

#[test]
fn test_static_beam_curl() {
    // Create list of Y moments to apply to end of beam
    let my = vec![0., 10920.0, 21840.0, 32761.0, 43681.0, 54601.0];

    //--------------------------------------------------------------------------
    // Initial configuration
    //--------------------------------------------------------------------------

    // Node locations
    let num_nodes = 11;
    let s = (0..num_nodes)
        .into_iter()
        .map(|v| (v as f64) / ((num_nodes - 1) as f64))
        .collect_vec();

    // Quadrature rule
    let num_qps = 21;
    let gq = Quadrature::gauss_legendre_lobotto(num_qps - 1);

    // Model
    let mut model = Model::new();

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

    // Set solver parameters
    model.set_rho_inf(1.0);
    model.set_time_step(1.0);
    model.set_max_iter(10);
    model.set_static_solve();

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
        println!("{:?}", u_tip.col(i))
    }

    let approx_eq = CwiseMat(ApproxEq::eps() * 1000.);
    assert!(
        u_tip ~
        mat![
            [0.0, 0.0, 0.0],
            [-2.4521424743945115, 0.0, 5.526921055643609],
            [-7.71393432218001, 0.0, 7.215325756355413],
            [-11.612495114867894, 0.0, 4.780938463888443],
            [-11.912220414483576, 0.0, 1.3458820860655774],
            [-10.00069041069354, 0.0, -1.2947392102224953e-5],
        ]
        .transpose()
    );

    // //--------------------------------------------------------------------------
    // // Numerical Gradient
    // //--------------------------------------------------------------------------

    // return;

    // // Loop through perturbations
    // let delta = 1e-7;

    // let ndof = solver.n_system + solver.n_lambda;

    // // Analytical derivative of residual at reference state.
    // let mut dres_mat = Mat::<f64>::zeros(ndof, ndof);

    // // Memory to ignore when calling with perturbations
    // let mut dres_mat_ignore = Mat::<f64>::zeros(ndof, ndof);

    // // Numerical approximation of 'dres_mat'
    // let mut dres_mat_num = Mat::<f64>::zeros(ndof, ndof);

    // // Initial Calculation for analytical gradient
    // let ref_state = state.clone();
    // let mut res_vec = Col::<f64>::zeros(ndof);
    // let mut xd = Col::<f64>::zeros(ndof);

    // // Do a residual + gradient eval
    // solver.step_res_grad(&mut state, xd.as_ref(), res_vec.as_mut(), dres_mat.as_mut());

    // // Loop through system DOFs
    // (0..ndof).for_each(|i| {
    //     // Positive side of finite difference
    //     let mut state = ref_state.clone();
    //     xd.fill(0.);
    //     xd[i] = delta;

    //     solver.step_res_grad(
    //         &mut state,
    //         xd.as_ref(),
    //         res_vec.as_mut(),
    //         dres_mat_ignore.as_mut(),
    //     );

    //     zip!(&mut dres_mat_num.col_mut(i), &res_vec)
    //         .for_each(|unzip!(col, res)| *col += *res * 0.5 / delta);

    //     // Negative side of finite difference
    //     let mut state = ref_state.clone();
    //     xd.fill(0.);
    //     xd[i] = -delta;

    //     solver.step_res_grad(
    //         &mut state,
    //         xd.as_ref(),
    //         res_vec.as_mut(),
    //         dres_mat_ignore.as_mut(),
    //     );

    //     zip!(&mut dres_mat_num.col_mut(i), &res_vec)
    //         .for_each(|unzip!(col, res)| *col -= *res * 0.5 / delta);
    // });

    // let grad_diff = dres_mat.clone() - dres_mat_num.clone();

    // println!("Grad diff norm: {:?}", grad_diff.norm_l2());
    // println!("Grad (analytical) norm: {:?}", dres_mat.norm_l2());
    // println!(
    //     "Norm ratio (diff/analytical): {:?}",
    //     grad_diff.norm_l2() / dres_mat.norm_l2()
    // );
}

use std::fs;

use faer::{col, mat, Scale};

use itertools::Itertools;
use ottr::{
    elements::beams::{BeamSection, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    quadrature::Quadrature,
    util::cross,
    vtk::beams_nodes_as_vtk,
};

const OUT_DIR: &str = "output/rotating-beam";

#[test]
fn test_rotating_beam() {
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initial rotational velocity
    let omega = col![0., 1.0, 0.];
    let time_step = 0.01;

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    let xi = gauss_legendre_lobotto_points(4);
    let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Model
    let mut model = Model::new();
    let beam_node_ids = s
        .iter()
        .map(|&si| {
            let p = col![10. * si + 2., 0., 0.];
            let mut v = col![0., 0., 0.];
            cross(omega.as_ref(), p.as_ref(), v.as_mut());
            model
                .add_node()
                .element_location(si)
                .position(p[0], p[1], p[2], 1., 0., 0., 0.)
                .velocity(
                    0.5 * v[0],
                    0.5 * v[1],
                    0.5 * v[2],
                    0.5 * omega[0],
                    0.5 * omega[1],
                    0.5 * omega[2],
                )
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
        [1368.17, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 88.56, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 38.78, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 16.960, 17.610, -0.351],
        [0.0000, 0.0000, 0.0000, 17.610, 59.120, -0.370],
        [0.0000, 0.0000, 0.0000, -0.351, -0.370, 141.47],
    ] * Scale(1e3);

    model.add_beam_element(
        &beam_node_ids,
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

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    let hub_node_id = model
        .add_node()
        .position(0., 0., 0., 1., 0., 0., 0.)
        .build();

    // Add constraint to beam node
    let hub_constraint_id = model.add_prescribed_constraint(hub_node_id);

    // Add constraint from hub node to first beam node
    model.add_rigid_constraint(hub_node_id, beam_node_ids[0]);

    //--------------------------------------------------------------------------
    // Create solver
    //--------------------------------------------------------------------------

    // Set solver parameters
    model.set_rho_inf(0.);
    model.set_time_step(time_step);
    model.set_max_iter(5);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    // let tip_node_id = *beam_node_ids.last().unwrap();

    for i in 1..500 {
        // current time
        let t = (i as f64) * time_step;

        if i == 1 {
            beams_nodes_as_vtk(&solver.elements.beams)
                .export_ascii(format!("{OUT_DIR}/step_{:0>3}.vtk", 0))
                .unwrap()
        }

        // Set hub displacement
        solver.constraints.constraints[hub_constraint_id].set_displacement(
            0.,
            0.,
            0.,
            t * omega[0],
            t * omega[1],
            t * omega[2],
        );

        // Take step and get convergence result
        let res = solver.step(&mut state);

        assert_eq!(res.converged, true);

        // state.calculate_x();
        // let x_tip = state.x.col(tip_node_id);
        // println!(
        //     "{}, {}, {}, {}, {}, {}, {}",
        //     x_tip[0], x_tip[1], x_tip[2], x_tip[3], x_tip[4], x_tip[5], x_tip[6]
        // );

        beams_nodes_as_vtk(&solver.elements.beams)
            .export_ascii(format!("{OUT_DIR}/step_{i:0>3}.vtk"))
            .unwrap()
    }
}

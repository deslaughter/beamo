use std::fs;

use faer::prelude::*;

use itertools::Itertools;
use ottr::{
    elements::beams::{BeamSection, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    quadrature::Quadrature,
    util::cross_product,
    vtk::beams_nodes_as_vtk,
};

const OUT_DIR: &str = "output/dynamic-beam";

fn create_model(omega: ColRef<f64>) -> (Model, usize) {
    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    let xi = gauss_legendre_lobotto_points(2);
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
            cross_product(omega.as_ref(), p.as_ref(), v.as_mut());
            model
                .add_node()
                .element_location(si)
                .position(p[0], p[1], p[2], 1., 0., 0., 0.)
                .velocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
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
    ] * 1e-2;

    // Stiffness matrix 6x6
    let c_star = mat![
        [1368.17, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 88.56, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 38.78, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 16.960, 17.610, -0.351],
        [0.0000, 0.0000, 0.0000, 17.610, 59.120, -0.370],
        [0.0000, 0.0000, 0.0000, -0.351, -0.370, 141.47],
    ] * 1e3;

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

    // Add constraint from hub node to first beam node
    model.add_rigid_constraint(hub_node_id, beam_node_ids[0]);

    //--------------------------------------------------------------------------
    // Create solver
    //--------------------------------------------------------------------------

    // Set solver parameters
    model.set_rho_inf(0.);

    model.set_max_iter(5);

    (model, hub_node_id)
}

#[test]
fn test_cantilever_beam() {
    fs::create_dir_all(OUT_DIR).unwrap();

    // Simulation parameters
    let time_step = 0.01;
    let n_steps = 100;

    // Initial rotational velocity about y-axis
    let omega = col![0., 3., 0.];

    // Create model
    let (mut model, hub_node_id) = create_model(omega.rb());
    model.set_time_step(time_step);

    // Add  prescribed constraint to hub node
    model.add_prescribed_constraint(hub_node_id);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    let mut n_iter_sum = 0;
    let mut err_sum = 0.0;

    for i in 1..n_steps + 1 {
        if i == 1 {
            beams_nodes_as_vtk(&solver.elements.beams)
                .export_ascii(format!("{OUT_DIR}/cantilever_{:0>3}.vtk", 0))
                .unwrap()
        }

        solver.fx[1] = 1000.0;

        // Take step and get convergence result
        let res = solver.step(&mut state);

        n_iter_sum += res.iter;
        err_sum += res.err;

        assert_eq!(res.converged, true);

        beams_nodes_as_vtk(&solver.elements.beams)
            .export_ascii(format!("{OUT_DIR}/cantilever_{i:0>3}.vtk"))
            .unwrap()
    }

    println!("Total iterations: {}", n_iter_sum);
    println!("Average iterations: {}", n_iter_sum as f64 / n_steps as f64);
    println!("Average error: {}", err_sum as f64 / n_steps as f64);
}

#[test]
fn test_rotating_beam() {
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initial rotational velocity
    let omega = col![0., 1.0, 0.];
    let time_step = 0.01;
    let n_steps = 500;

    let (mut model, hub_node_id) = create_model(omega.rb());
    model.set_time_step(time_step);

    // Add constraint prescribed constraint to hub node
    let hub_constraint_id = model.add_prescribed_constraint(hub_node_id);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    // let tip_node_id = *beam_node_ids.last().unwrap();

    let mut n_iter_sum = 0;
    let mut err_sum = 0.0;

    for i in 1..n_steps + 1 {
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

        n_iter_sum += res.iter;
        err_sum += res.err;

        assert_eq!(res.converged, true);

        // state.calculate_x();
        // let x_tip = state.x.col(tip_node_id);
        // println!(
        //     "{}, {}, {}, {}, {}, {}, {}",
        //     x_tip[0], x_tip[1], x_tip[2], x_tip[3], x_tip[4], x_tip[5], x_tip[6]
        // );

        // beams_nodes_as_vtk(&solver.elements.beams)
        //     .export_ascii(format!("{OUT_DIR}/step_{i:0>3}.vtk"))
        //     .unwrap()
    }

    println!("Total iterations: {}", n_iter_sum);
    println!("Average iterations: {}", n_iter_sum as f64 / n_steps as f64);
    println!("Average error: {}", err_sum as f64 / n_steps as f64);
}

#[test]
fn test_prescribed_rotating_beam() {
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initial rotational velocity
    let time_step = 0.01;
    let n_steps = 300;

    let (mut model, hub_node_id) = create_model(col![0., 0., 0.].rb());
    model.set_time_step(time_step);

    // Add constraint prescribed constraint to hub node
    let hub_constraint_id = model.add_prescribed_constraint(hub_node_id);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    let mut n_iter_sum = 0;
    let mut err_sum = 0.0;

    for i in 1..n_steps + 1 {
        // current time
        let t = (i as f64) * time_step;

        if i == 1 {
            beams_nodes_as_vtk(&solver.elements.beams)
                .export_ascii(format!("{OUT_DIR}/prescribed_{:0>3}.vtk", 0))
                .unwrap()
        }

        // Set hub displacement
        solver.constraints.constraints[hub_constraint_id].set_displacement(
            0.,
            0.,
            0.,
            0.,
            t * t * 6. / 2.,
            0.,
        );

        // Take step and get convergence result
        let res = solver.step(&mut state);

        n_iter_sum += res.iter;
        err_sum += res.err;

        assert_eq!(res.converged, true);

        beams_nodes_as_vtk(&solver.elements.beams)
            .export_ascii(format!("{OUT_DIR}/prescribed_{i:0>3}.vtk"))
            .unwrap()
    }

    println!("Total iterations: {}", n_iter_sum);
    println!("Average iterations: {}", n_iter_sum as f64 / n_steps as f64);
    println!("Average error: {}", err_sum as f64 / n_steps as f64);
}

#[test]
fn test_revolute_joint() {
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initial rotational velocity
    let omega = col![0., 2., 0.];
    let time_step = 0.1;
    let n_steps = 100;

    let (mut model, hub_node_id) = create_model(omega.rb());
    model.set_time_step(time_step);

    // Give hub node a mass
    model.add_mass_element(hub_node_id, 10. * Mat::<f64>::identity(6, 6));

    // Add shaft base node
    let shaft_base_node_id = model
        .add_node()
        .position(0., -1., 0., 1., 0., 0., 0.)
        .build();

    // Add revolute joint along axis from shaft base to hub
    model.add_revolute_joint(shaft_base_node_id, hub_node_id);

    // Fix shaft base node
    model.add_prescribed_constraint(shaft_base_node_id);

    // Create solver
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    // let tip_node_id = *beam_node_ids.last().unwrap();

    let mut n_iter_sum = 0;
    let mut err_sum = 0.0;

    for i in 0..n_steps {
        beams_nodes_as_vtk(&solver.elements.beams)
            .export_ascii(format!("{OUT_DIR}/revolute_{i:0>3}.vtk"))
            .unwrap();

        // Take step and get convergence result
        let res = solver.step(&mut state);

        n_iter_sum += res.iter;
        err_sum += res.err;

        assert_eq!(res.converged, true);
    }

    println!("Total iterations: {}", n_iter_sum);
    println!("Average iterations: {}", n_iter_sum as f64 / n_steps as f64);
    println!("Average error: {}", err_sum as f64 / n_steps as f64);
}

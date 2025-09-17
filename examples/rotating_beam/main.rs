use std::f64::consts::PI;

use faer::prelude::*;
use itertools::Itertools;
use ottr::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::BeamSection,
    model::Model,
    solver::Solver,
    state::State,
    util::{quat_from_axis_angle_alloc, write_matrix, ColRefReshape},
    vtk::beams_nodes_as_vtk,
};

fn main() {
    let time_step = 0.01;
    // let duration = 2.0;
    // let n_steps = (duration / time_step) as usize;
    let n_revolutions = 1.;

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_solver_tolerance(1e-5, 1e-3);
    model.set_rho_inf(0.);

    let omega = 4. * PI / 3.; // 40 RPM
                              // let omega = 2. * PI / 3.; // 20 RPM
                              // let omega = PI / 4.; // 15 RPM
                              // let omega = PI / 6.; // 10 RPM
                              // let omega = PI / 15.; // 4 RPM
                              // let omega = PI / 30.; // 2 RPM
                              // let omega = 0.; // 0 RPM

    let n_steps = if omega > 0. {
        (n_revolutions * 2. * PI / omega / time_step).ceil() as usize
    } else {
        1
    };

    let omega_v = [omega, 0., 0.];

    let mut blade = build_blade(omega_v, &mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    beams_nodes_as_vtk(&solver.elements.beams)
        .export_ascii(&format!("examples/rotating_beam/vtk/step_{:04}.vtk", 0))
        .unwrap();

    // Loop through steps
    for i in 1..n_steps + 1 {
        // Calculate time
        let t = (i as f64) * time_step;

        let r = [omega_v[0] * t, omega_v[1] * t, omega_v[2] * t];

        solver.constraints.constraints[0].set_displacement(0., 0., 0., r[0], r[1], r[2]);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Get the node motions
        blade.nodes.iter_mut().for_each(|node| {
            node.get_motion(&state);
        });

        if i % 10 == 0 {
            // println!("{:.2},{:?}", t, blade.nodes.last().unwrap().position);
            // println!("{:.2},{:?}", t, blade.nodes.last().unwrap().displacement);
        }

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);

        // beams_nodes_as_vtk(&solver.elements.beams)
        //     .export_ascii(&format!("examples/rotating_beam/vtk/step_{:04}.vtk", i))
        //     .unwrap();
    }

    let mut r = Col::<f64>::zeros(solver.n_dofs);
    solver
        .elements
        .assemble_system(&state, time_step, r.as_mut());

    //--------------------------------------------------------------------------
    // Calculate all the things
    //--------------------------------------------------------------------------

    let mut s = state.clone();
    s.vd.fill(0.);
    s.calc_step_end(time_step);
    s.u_prev.copy_from(&s.u);
    s.u_delta.fill(0.);

    let vd = calc_accel(&mut solver, &s, time_step);

    println!("node_fe = {:?}", solver.elements.beams.node_fe.transpose());
    println!("node_fi = {:?}", solver.elements.beams.node_fi.transpose());
    println!("qp.fe_c = {:?}", solver.elements.beams.qp.fe_c.transpose());
    println!("qp.fe_d = {:?}", solver.elements.beams.qp.fe_d.transpose());
    println!("vd = {:?}", state.vd.transpose());
    println!("reaction = {:?}", solver.lambda);

    let nnm1 = state.n_nodes - 1;

    println!("vd = {:?}", vd.as_ref().reshape(6, nnm1).transpose());

    let cols_disp = (1..s.n_nodes)
        .flat_map(|i| {
            (0..6)
                .map(|j| {
                    let perturb = if j < 3 {
                        0.2_f64.to_radians() * 60.
                    } else {
                        0.2_f64.to_radians()
                    };
                    s.u_delta.fill(0.);
                    s.u_delta[(j, i)] += perturb;
                    s.calc_step_end(1.);
                    let vd_plus = calc_accel(&mut solver, &s, 1.);

                    // println!("state.u = {:?}", s.u.subcols(1, nnm1).transpose());
                    // println!("state.v = {:?}", s.v.subcols(1, nnm1).transpose());
                    // println!("node_x0 = {:?}", solver.elements.beams.node_x0.transpose());
                    // // println!("qp.rr0 = {:?}", solver.elements.beams.qp.rr0.transpose());
                    // println!("qp.u = {:?}", solver.elements.beams.qp.u.transpose());
                    // println!(
                    //     "qp.strain = {:?}",
                    //     solver.elements.beams.qp.strain.transpose()
                    // );
                    // println!("qp.fe_c = {:?}", solver.elements.beams.qp.fe_c.transpose());
                    // println!("qp.fe_d = {:?}", solver.elements.beams.qp.fe_d.transpose());
                    // println!(
                    //     "node_f = {:?}",
                    //     solver.elements.beams.node_f.subcols(1, nnm1).transpose()
                    // );

                    // println!(
                    //     "vd_plus = {:?}",
                    //     vd_plus.as_ref().reshape(6, nnm1).transpose()
                    // );

                    s.u_delta.fill(0.);
                    s.u_delta[(j, i)] -= perturb;
                    s.calc_step_end(1.);
                    let vd_minus = calc_accel(&mut solver, &s, 1.);

                    println!(
                        "vd_minus = {:?}",
                        vd_minus.as_ref().reshape(6, nnm1).transpose()
                    );

                    (vd_plus - vd_minus) / (2. * perturb)
                })
                .collect_vec()
        })
        .collect_vec();

    s.u_delta.fill(0.);
    s.calc_step_end(1.);
    let cols_vel = (1..s.n_nodes)
        .flat_map(|i| {
            (0..6)
                .map(|j| {
                    let perturb = if j < 3 {
                        0.2_f64.to_radians() * 60.
                    } else {
                        0.2_f64.to_radians()
                    };
                    s.v.copy_from(&state.v);
                    s.v[(j, i)] += perturb;
                    let vd_plus = calc_accel(&mut solver, &s, 1.);

                    s.v.copy_from(&state.v);
                    s.v[(j, i)] -= perturb;
                    let vd_minus = calc_accel(&mut solver, &s, 1.);

                    (vd_plus - vd_minus) / (2. * perturb)
                })
                .collect_vec()
        })
        .collect_vec();

    let cols = cols_disp
        .iter()
        .chain(cols_vel.iter())
        .cloned()
        .collect_vec();

    let mut a = Mat::<f64>::zeros(6 * (s.n_nodes - 1), 2 * 6 * (s.n_nodes - 1));
    cols.iter().enumerate().for_each(|(j, col)| {
        a.col_mut(j).copy_from(&col);
    });
    write_matrix(a.as_ref(), "examples/rotating_beam/a.csv").unwrap();
}

fn calc_accel(solver: &mut Solver, state: &State, time_step: f64) -> Col<f64> {
    let mut r = Col::<f64>::zeros(solver.n_dofs);
    solver
        .elements
        .assemble_system(&state, time_step, r.as_mut());

    let m = solver.elements.beams.m_sp.to_dense();
    let m = m.submatrix(6, 6, solver.n_system - 6, solver.n_system - 6);

    let f_shape = solver.elements.beams.node_f.shape();

    let f = Col::from_fn(f_shape.0 * f_shape.1, |i| {
        -solver.elements.beams.node_f[(i % 6, i / 6)]
    });

    let vd_col = m.partial_piv_lu().solve(f.subrows(6, solver.n_system - 6));
    vd_col
}

fn build_blade(omega: [f64; 3], model: &mut Model) -> BeamComponent {
    let blade_length = 60.;

    let c_star = mat![
        [4.7480e+8, 0., 0., 0., 0., 0.],
        [0., 4.7480e+8, 0., 0., 0., 0.],
        [0., 0., 6.5440e+9, 0., 0., 0.],
        [0., 0., 0., 4.2834e+9, 0., 0.],
        [0., 0., 0., 0., 4.2834e+9, 0.],
        [0., 0., 0., 0., 0., 3.2650e+9],
    ];

    let m_star = mat![
        [2.9084e+2, 0., 0., 0., 0., 0.],
        [0., 2.9084e+2, 0., 0., 0., 0.],
        [0., 0., 2.9084e+2, 0., 0., 0.],
        [0., 0., 0., 9.5186e-02, 0., 0.],
        [0., 0., 0., 0., 9.5186e-02, 0.],
        [0., 0., 0., 0., 0., 1.9037e-01],
    ];

    //--------------------------------------------------------------------------
    // Blade
    //--------------------------------------------------------------------------

    let n_blade_nodes = 4;
    let n_qps: usize = 21;

    let s = (0..n_qps)
        .map(|i| i as f64 / (n_qps - 1) as f64)
        .collect_vec();

    let sections = s
        .iter()
        .map(|&s| BeamSection {
            s,
            m_star: m_star.clone(),
            c_star: c_star.clone(),
        })
        .collect_vec();

    let root_orientation =
        quat_from_axis_angle_alloc(-90.0_f64.to_radians(), col![0., 1., 0.].as_ref());

    // Build blade input
    let blade_input = BeamInputBuilder::new()
        .set_element_order(n_blade_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &s,
            &s.iter().map(|s| [0., 0., s * blade_length]).collect_vec(),
            &[0., 1.],
            &[0., 0.],
        )
        .set_sections_z(&sections)
        .set_root_position([
            0.,
            0.,
            0.,
            root_orientation[0],
            root_orientation[1],
            root_orientation[2],
            root_orientation[3],
        ])
        .set_prescribe_root(true)
        .set_root_velocity([0., 0., 0., omega[0], omega[1], omega[2]])
        .build();

    // Build and return the beam component
    BeamComponent::new(&blade_input, model)
}

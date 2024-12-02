use std::f64::consts::PI;

use faer::{assert_matrix_eq, col, mat, Col, Mat};
use ottr::{
    model::Model,
    util::{quat_as_euler_angles, vec_tilde, Quat},
};

#[test]
fn test_precession() {
    // Model
    let mut model = Model::new();

    let mass_node = model
        .new_node()
        .position(0., 0., 0., 1., 0., 0., 0.)
        .velocity(0., 0., 0., 0.5, 0.5, 1.)
        .build();

    // Define mass matrix
    let m = 1.;
    let mut mass_matrix = Mat::<f64>::zeros(6, 6);
    mass_matrix
        .diagonal_mut()
        .column_vector_mut()
        .copy_from(col![m, m, m, 1., 1., 0.5]);

    // Add mass element
    model.add_mass_element(mass_node, mass_matrix);

    //--------------------------------------------------------------------------
    // Create state and solver
    //--------------------------------------------------------------------------

    let mut state = model.create_state();

    let time_step = 0.01;
    model.set_time_step(time_step);
    model.set_rho_inf(1.);
    model.set_max_iter(6);
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    for _ in 0..500 {
        // Output current rotation
        // let q = state.u.col(0).subrows(3, 4);
        // quat_as_euler_angles(q, e.as_mut());
        // println!("{}\t{}\t{}\t{}", (i as f64) * time_step, e[0], e[1], e[2]);

        // Step
        let res = solver.step(&mut state);
        assert_eq!(res.converged, true);
    }

    let mut e = Col::<f64>::zeros(3);
    quat_as_euler_angles(state.u.col(0).subrows(3, 4), e.as_mut());

    assert_matrix_eq!(
        e.as_2d(),
        col![-1.413542763236864, 0.999382175365794, 0.213492011335111].as_2d(),
        comp = float
    )
}

#[test]
fn test_heavy_top() {
    // Model
    let mut model = Model::new();

    let theta = PI / 2.;
    let x0 = col![0., 0., -1.];
    let ui = col![0., 1., 1.];
    let ri = mat![
        [1., 0., 0.],
        [0., theta.cos(), -theta.sin()],
        [0., theta.sin(), theta.cos()],
    ];
    let omega_i_star = col![-4.61538, 0., 150.];
    let omega_i = &ri * omega_i_star;
    let mut omega_i_tilde = Mat::<f64>::zeros(3, 3);
    vec_tilde(omega_i.as_ref(), omega_i_tilde.as_mut());
    let xi = &x0 + &ui;
    let u_dot_i = omega_i_tilde * &xi;
    let mut qi = Col::<f64>::zeros(4);
    qi.as_mut().quat_from_rotation_matrix(ri.as_ref());

    let mass_node_id = model
        .new_node()
        .position(xi[0], xi[1], xi[2], qi[0], qi[1], qi[2], qi[3])
        .velocity(
            u_dot_i[0], u_dot_i[1], u_dot_i[2], omega_i[0], omega_i[1], omega_i[2],
        )
        .build();

    let hub_node_id = model.new_node().position_xyz(0., 0., 0.).build();

    // Add rigid constraint
    model.add_rigid_constraint(mass_node_id, hub_node_id);

    // Define mass matrix
    let mut mass_matrix = Mat::<f64>::zeros(6, 6);
    mass_matrix
        .diagonal_mut()
        .column_vector_mut()
        .copy_from(col![15., 15., 15., 0.234375, 0.234375, 0.46875]);

    // Add mass element
    model.add_mass_element(mass_node_id, mass_matrix);

    //--------------------------------------------------------------------------
    // Create state and solver
    //--------------------------------------------------------------------------

    let mut state = model.create_state();

    let time_step = 0.01;
    model.set_time_step(time_step);
    model.set_rho_inf(1.);
    model.set_max_iter(6);
    let mut solver = model.create_solver();

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    for _ in 0..500 {
        // Output current rotation
        // let q = state.u.col(0).subrows(3, 4);
        // quat_as_euler_angles(q, e.as_mut());
        // println!("{}\t{}\t{}\t{}", (i as f64) * time_step, e[0], e[1], e[2]);

        // Step
        let res = solver.step(&mut state);
        assert_eq!(res.converged, true);
    }

    let mut e = Col::<f64>::zeros(3);
    quat_as_euler_angles(state.u.col(0).subrows(3, 4), e.as_mut());

    assert_matrix_eq!(
        e.as_2d(),
        col![-1.413542763236864, 0.999382175365794, 0.213492011335111].as_2d(),
        comp = float
    )
}

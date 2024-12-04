use std::{
    fs::{self, File},
    io::Write,
};

use faer::{assert_matrix_eq, col, solvers::SpSolver, Col, Mat, Scale};
use ottr::{
    model::Model,
    util::{cross, quat_as_euler_angles, quat_as_rotation_vector, vec_tilde},
};

#[test]
fn test_precession() {
    // Model
    let mut model = Model::new();

    let mass_node = model
        .add_node()
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
    let out_dir = "output";
    let time_step: f64 = 0.002;
    let t_end: f64 = 2.;
    let n_steps = ((t_end / time_step).ceil() as usize) + 1;

    // Model
    let mut model = Model::new();
    model.set_solver_tolerance(1e-5, 1.);
    model.set_time_step(time_step);
    model.set_rho_inf(0.9);
    model.set_max_iter(6);

    let m = 15.;
    let mut j = Mat::<f64>::zeros(3, 3);
    j.diagonal_mut()
        .column_vector_mut()
        .copy_from(col![0.234375, 0.46875, 0.234375]);
    let omega = col![0., 150., -4.61538];
    let gamma = col![0., 0., -9.81];
    let x = col![0., 1., 0.];

    // translational velocity
    let mut x_dot = col![0., 0., 0.];
    cross(omega.as_ref(), x.as_ref(), x_dot.as_mut());

    // angular acceleration
    let mut x_tilde = Mat::<f64>::zeros(3, 3);
    vec_tilde(x.as_ref(), x_tilde.as_mut());
    let j_bar: Mat<f64> = &j - m * &x_tilde * &x_tilde;
    let mut x_cross_m_gamma = col![0., 0., 0.];
    cross(
        x.as_ref(),
        (Scale(m) * &gamma).as_ref(),
        x_cross_m_gamma.as_mut(),
    );
    let mut omega_cross_j_bar_omega = col![0., 0., 0.];
    let j_bar_omega: Col<f64> = &j_bar * &omega;
    cross(
        omega.as_ref(),
        j_bar_omega.as_ref(),
        omega_cross_j_bar_omega.as_mut(),
    );
    let omega_dot = j_bar
        .partial_piv_lu()
        .solve(&x_cross_m_gamma - &omega_cross_j_bar_omega);

    // translational acceleration
    let mut omega_dot_cross_x = col![0., 0., 0.];
    cross(omega_dot.as_ref(), x.as_ref(), omega_dot_cross_x.as_mut());
    let mut omega_cross_x_dot = col![0., 0., 0.];
    cross(omega.as_ref(), x_dot.as_ref(), omega_cross_x_dot.as_mut());
    let x_ddot = omega_dot_cross_x + omega_cross_x_dot;

    // Add mass element
    let mass_node_id = model
        .add_node()
        .position(x[0], x[1], x[2], 1., 0., 0., 0.)
        .velocity(x_dot[0], x_dot[1], x_dot[2], omega[0], omega[1], omega[2])
        .acceleration(
            x_ddot[0],
            x_ddot[1],
            x_ddot[2],
            omega_dot[0],
            omega_dot[1],
            omega_dot[2],
        )
        .build();
    let mut mass_matrix = Mat::<f64>::zeros(6, 6);
    mass_matrix
        .diagonal_mut()
        .column_vector_mut()
        .subrows_mut(0, 3)
        .fill(m);
    mass_matrix.submatrix_mut(3, 3, 3, 3).copy_from(&j);
    model.add_mass_element(mass_node_id, mass_matrix);

    let ground_node_id = model.add_node().position_xyz(0., 0., 0.).build();
    model.add_rigid_constraint(mass_node_id, ground_node_id);
    model.add_prescribed_constraint(ground_node_id);

    // gamma is defined with the z component as positive along negative z axis
    // this took forever to figure out
    model.set_gravity(gamma[0], gamma[1], -gamma[2]);

    //--------------------------------------------------------------------------
    // run simulation
    //--------------------------------------------------------------------------

    // Create output directory
    fs::create_dir_all(out_dir).unwrap();

    // Create solver
    let mut solver = model.create_solver();

    // Create state
    let mut state = model.create_state();

    // Open output file
    let mut file = File::create(format!("{out_dir}/heavy_top.csv")).unwrap();

    // Rotation vector for an
    let mut rv = Col::<f64>::zeros(3);

    // Time step
    for i in 0..n_steps {
        let t = (i as f64) * time_step;

        // Output current position and rotation
        let u = state.u.col(0).subrows(0, 3);
        let q = state.u.col(0).subrows(3, 4);
        quat_as_rotation_vector(q, rv.as_mut());
        file.write_fmt(format_args!(
            "{},{},{},{},{},{},{}\n",
            t, u[0], u[1], u[2], rv[0], rv[1], rv[2]
        ))
        .unwrap();

        if i == 400 {
            assert_matrix_eq!(
                state.u.col(0).as_2d(),
                col![
                    -0.4220299141898183,
                    -0.09451353137427536,
                    -0.04455341442645723,
                    -0.17794086498990777,
                    0.21672292516262048,
                    -0.9597292673920982,
                    -0.016969254156485276,
                ]
                .as_2d()
            );
        }

        // Step
        let res = solver.step(&mut state);

        assert_eq!(res.converged, true);
    }
}

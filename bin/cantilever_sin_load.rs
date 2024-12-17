use std::{
    fs::{self, File},
    io::Write,
    process,
};

use faer::{col, mat, Scale};

use itertools::Itertools;
use ottr::{
    elements::beams::{BeamSection, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    node::Direction,
    quadrature::Quadrature,
};

fn main() {
    // Create output directory
    let out_dir = "output";
    fs::create_dir_all(out_dir).unwrap();

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    let xi = gauss_legendre_lobotto_points(4);
    let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Model
    let mut model = Model::new();
    let node_ids = s
        .iter()
        .map(|&si| {
            model
                .add_node()
                .element_location(si)
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
        [1368.17, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 88.56, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 38.78, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 16.960, 17.610, -0.351],
        [0.0000, 0.0000, 0.0000, 17.610, 59.120, -0.370],
        [0.0000, 0.0000, 0.0000, -0.351, -0.370, 141.47],
    ] * Scale(1e3);

    // let damping=Damping::None;
    let damping = Damping::Mu(col![0.001, 0.001, 0.001, 0.001, 0.001, 0.001]);

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
        damping.clone(),
    );

    //--------------------------------------------------------------------------
    // Create solver
    //--------------------------------------------------------------------------

    // Add constraint to root node
    model.add_prescribed_constraint(node_ids[0]);

    // Set solver parameters
    let time_step = 0.005;
    model.set_rho_inf(0.);
    model.set_time_step(time_step);
    model.set_max_iter(5);

    // Create solver
    let mut solver = model.create_solver();

    let mut file = match damping {
        Damping::None => File::create(format!("{out_dir}/damping-0.csv")).unwrap(),
        Damping::Mu(_) => File::create(format!("{out_dir}/damping-mu.csv")).unwrap(),
        Damping::ModalElement(_) => File::create(format!("{out_dir}/damping-me.csv")).unwrap(),
    };

    //--------------------------------------------------------------------------
    // Run simulation
    //--------------------------------------------------------------------------

    // Create state
    let mut state = model.create_state();

    let tip_node_id = *node_ids.last().unwrap();

    // Get DOF index for beam tip node Z direction
    let tip_z_dof = solver.nfm.get_dof(tip_node_id, Direction::Z).unwrap();

    for i in 0..10000 {
        // current time
        let t = (i as f64) * time_step;

        // Apply sine force on z direction of last node
        solver.fx[tip_z_dof] = 100. * (10.0 * t).sin();

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Exit if failed to converge
        if !res.converged {
            println!("failed! iter={i}, {:?}", res);
            process::exit(1);
        }

        // Write data to file
        file.write_fmt(format_args!(
            "{},{}\n",
            t,
            state.u[(Direction::Z as usize, tip_node_id)]
        ))
        .unwrap();
    }

    println!("success")
}

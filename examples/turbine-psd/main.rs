use std::{f64::consts::PI, fs};

use beamo::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::{BeamSection, Damping},
    model::Model,
    util::{
        quat_compose_alloc, quat_from_axis_angle_alloc, quat_inverse_alloc,
        quat_rotate_vector_alloc, write_matrix,
    },
};
use faer::prelude::*;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

static DATA_DIR: &str = "examples/turbine-psd/data-nm";

fn main() {
    let max_rpm = 40.0; // Maximum rotor speed in rpm
    let d_rpm = 0.25;
    let n_cases = (max_rpm / d_rpm) as usize + 1;

    let signal_scale = 2.0e3;
    let contents = fs::read_to_string("examples/turbine-psd/generated_signal.txt").unwrap();
    let signals = contents
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|s| s.parse::<f64>().unwrap() * signal_scale)
                .collect_vec()
        })
        .collect_vec();

    (0..n_cases).into_par_iter().for_each(|i| {
        let rpm = i as f64 * d_rpm;
        println!("Running case {}, rpm = {}", i + 1, rpm);
        run(rpm, &signals);
    });
}

fn run(rotor_rpm: f64, signals: &[Vec<f64>]) {
    let time_step = 0.01;
    let duration = 1000.0;
    let n_steps = (duration / time_step) as usize;

    // Create model and set basic parameters
    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_solver_tolerance(1e-6, 1e-4);
    model.set_rho_inf(0.);

    // Create directory
    fs::create_dir_all(DATA_DIR).unwrap();

    // Create vector for hub angular velocity
    let omega = [rotor_rpm * 2.0 * PI / 60.0, 0., 0.];

    // build the turbine
    let turbine = build_rotor(omega, &mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    // model.write_mesh_connectivity_file(directory);

    // Create output writer and write initial condition
    // let mut netcdf_file =
    //     netcdf::create(&format!("{}/{:.2}_turbine.nc", directory, rotor_rpm)).unwrap();
    // let mut output_writer = OutputWriter::new(&mut netcdf_file, model.n_nodes());
    // output_writer.write(&state, 0);

    let mut blade_tip_v = Mat::<f64>::zeros(3 * 3, n_steps);
    let mut tower_top_v = Mat::<f64>::zeros(3, n_steps);

    // Apply loads at next to last node of tower and blades
    let signal_node_dofs = [
        turbine.tower.nodes.last().unwrap().id - 1,
        turbine.blades[0].nodes.last().unwrap().id - 1,
        turbine.blades[1].nodes.last().unwrap().id - 1,
        turbine.blades[2].nodes.last().unwrap().id - 1,
    ]
    .iter()
    .flat_map(|&nid| [(nid, 0), (nid, 1), (nid, 2)])
    .collect_vec();

    // Loop through steps
    for i in 0..n_steps {
        // Calculate time
        let t = ((1 + i) as f64) * time_step;

        // Calculate azimuth angle
        let azimuth = omega[0] * t;

        // Set rotation of tower-hub constraint
        solver.constraints.constraints[turbine.tower_hub_constraint_id].set_rotation(azimuth);

        // Set loads on tower top and blade tips
        signal_node_dofs
            .iter()
            .enumerate()
            .for_each(|(j, &(nid, dof))| {
                state.fx[(dof, nid)] = signals[i][j];
            });

        // Take step and get convergence result
        let res = solver.step(&mut state);

        let psi = [azimuth, azimuth + 2.0 * PI / 3.0, azimuth + 4.0 * PI / 3.0];

        // Transform from rotating to non-rotating frame
        let t_mbc = 2. / 3.
            * mat![
                [0.5, 0.5, 0.5],
                [psi[0].sin(), psi[1].sin(), psi[2].sin()],
                [psi[0].cos(), psi[1].cos(), psi[2].cos()],
            ];

        // Get the node motions
        // turbine.blades.iter_mut().for_each(|blade| {
        //     blade.nodes.iter_mut().for_each(|node| {
        //         node.get_motion(&state);
        //     });
        // });

        // get the tower node motions
        // turbine.tower.nodes.iter_mut().for_each(|node| {
        //     node.get_motion(&state);
        // });

        turbine.blades.iter().enumerate().for_each(|(j, blade)| {
            let root_node_id = blade.nodes[0].id;
            let tip_node_id = blade.nodes.last().unwrap().id;

            // Get the blade root rotation displacement
            let q_root = state.x.col(root_node_id).subrows(3, 4);
            let q_root_inv = quat_inverse_alloc(q_root);

            // Blade velocity in rotating frame
            let v = quat_rotate_vector_alloc(
                q_root_inv.as_ref(),
                state.v.col(tip_node_id).subrows(0, 3),
            );

            blade_tip_v[(j + 0, i)] = v[0];
            blade_tip_v[(j + 3, i)] = v[1];
            blade_tip_v[(j + 6, i)] = v[2];
        });

        let vx = &t_mbc * blade_tip_v.col(i).subrows(0, 3);
        let vy = &t_mbc * blade_tip_v.col(i).subrows(3, 3);
        let vz = &t_mbc * blade_tip_v.col(i).subrows(6, 3);
        blade_tip_v.col_mut(i).subrows_mut(0, 3).copy_from(vx);
        blade_tip_v.col_mut(i).subrows_mut(3, 3).copy_from(vy);
        blade_tip_v.col_mut(i).subrows_mut(6, 3).copy_from(vz);

        // Tower top velocity
        tower_top_v.col_mut(i).copy_from(
            &state
                .v
                .col(turbine.tower.nodes.last().unwrap().id)
                .subrows(0, 3),
        );

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);

        // output_writer.write(&state, i + 1);
    }

    write_matrix(
        blade_tip_v.transpose(),
        &format!("{}/{:.2}_blade_tip_v.csv", DATA_DIR, rotor_rpm),
    )
    .unwrap();
    write_matrix(
        tower_top_v.transpose(),
        &format!("{}/{:.2}_tower_top_v.csv", DATA_DIR, rotor_rpm),
    )
    .unwrap();
}

pub struct Turbine {
    pub blades: Vec<BeamComponent>,
    pub tower: BeamComponent,
    pub hub_blade_constraint_ids: Vec<usize>,
    pub tower_hub_constraint_id: usize,
}

fn build_rotor(omega: [f64; 3], model: &mut Model) -> Turbine {
    let blade_length = 60.;
    let tower_height = 120.;

    //--------------------------------------------------------------------------
    // Blade
    //--------------------------------------------------------------------------

    let blade_c_star = mat![
        [4.7480e+8, 0., 0., 0., 0., 0.],
        [0., 4.7480e+8, 0., 0., 0., 0.],
        [0., 0., 6.5440e+9, 0., 0., 0.],
        [0., 0., 0., 4.2834e+9, 0., 0.],
        [0., 0., 0., 0., 4.2834e+9, 0.],
        [0., 0., 0., 0., 0., 3.2650e+9],
    ];

    let blade_m_star = mat![
        [2.9084e+2, 0., 0., 0., 0., 0.],
        [0., 2.9084e+2, 0., 0., 0., 0.],
        [0., 0., 2.9084e+2, 0., 0., 0.],
        [0., 0., 0., 9.5186e-02, 0., 0.],
        [0., 0., 0., 0., 9.5186e-02, 0.],
        [0., 0., 0., 0., 0., 1.9037e-01],
    ];

    let n_blade_nodes = 4;
    let n_qps: usize = 21;

    let s = (0..n_qps)
        .map(|i| i as f64 / (n_qps - 1) as f64)
        .collect_vec();

    let sections = s
        .iter()
        .map(|&s| BeamSection {
            s,
            m_star: blade_m_star.clone(),
            c_star: blade_c_star.clone(),
        })
        .collect_vec();

    let root_orientation =
        quat_from_axis_angle_alloc(-90.0_f64.to_radians(), col![0., 1., 0.].as_ref());

    // Build blade input
    let mut blade_builder = BeamInputBuilder::new();
    blade_builder
        .set_element_order(n_blade_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &s,
            &s.iter().map(|s| [0., 0., s * blade_length]).collect_vec(),
            &[0., 1.],
            &[0., 0.],
        )
        .set_sections_z(&sections)
        .set_damping(Damping::Mu(col![
            1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3
        ]))
        .set_root_velocity([0., 0., 0., omega[0], omega[1], omega[2]]);

    // Build blades at various rotations
    let blades = [0., 120., 240.0_f64]
        .into_iter()
        .map(|angle| {
            let r = quat_from_axis_angle_alloc(angle.to_radians(), col![1., 0., 0.].as_ref());
            let rr0 = quat_compose_alloc(r.as_ref(), root_orientation.as_ref());
            blade_builder.set_root_position([0., 0., tower_height, rr0[0], rr0[1], rr0[2], rr0[3]]);
            BeamComponent::new(&blade_builder.build(), model)
        })
        .collect_vec();

    //--------------------------------------------------------------------------
    // Tower
    //--------------------------------------------------------------------------

    let tower_c_star = mat![
        [21038788069.90780, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 12623272841.94470, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 54700848981.76040, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 108211146223.82200, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 108211146223.82200, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 76923076924.15900],
    ];

    let tower_m_star = mat![
        [2188.033959, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2188.033959, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2188.033959, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.541056, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.541056, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.082111],
    ];

    let n_tower_nodes = 4;
    let n_qps: usize = 21;

    let s = (0..n_qps)
        .map(|i| i as f64 / (n_qps - 1) as f64)
        .collect_vec();

    let root_orientation =
        quat_from_axis_angle_alloc(-90.0_f64.to_radians(), col![0., 1., 0.].as_ref());

    // Build tower input
    let tower_input = BeamInputBuilder::new()
        .set_element_order(n_tower_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &s,
            &s.iter().map(|s| [0., 0., s * tower_height]).collect_vec(),
            &[0., 1.],
            &[0., 0.],
        )
        .set_sections_z(
            &s.iter()
                .map(|&s| BeamSection {
                    s,
                    m_star: tower_m_star.clone(),
                    c_star: tower_c_star.clone(),
                })
                .collect_vec(),
        )
        .set_prescribe_root(true)
        .set_root_position([
            0.,
            0.,
            0.,
            root_orientation[0],
            root_orientation[1],
            root_orientation[2],
            root_orientation[3],
        ])
        // .set_damping(Damping::Mu(col![
        //     1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3
        // ]))
        .build();

    // Build tower
    let tower = BeamComponent::new(&tower_input, model);

    // Get the tower top node id
    let tt_node_id = tower.nodes.last().unwrap().id;

    //--------------------------------------------------------------------------
    // Mass
    //--------------------------------------------------------------------------

    // Build hub node
    let hub_node_id = model
        .add_node()
        .position(0., 0., tower_height, 1., 0., 0., 0.)
        .build();

    // // Hub mass and inertia
    // model.add_mass_element(
    //     hub_node_id,
    //     mat![
    //         [20000., 0., 0., 0., 0., 0.],
    //         [0., 20000., 0., 0., 0., 0.],
    //         [0., 0., 20000., 0., 0., 0.],
    //         [0., 0., 0., 3.0e4, 0., 0.],
    //         [0., 0., 0., 0., 0., 0.],
    //         [0., 0., 0., 0., 0., 0.],
    //     ],
    // );

    // // Add nacelle mass and yaw bearing mass
    // model.add_mass_element(
    //     tt_node_id,
    //     mat![
    //         [100000. + 3000., 0., 0., 0., 0., 0.],
    //         [0., 100000. + 3000., 0., 0., 0., 0.],
    //         [0., 0., 100000. + 3000., 0., 0., 0.],
    //         [0., 0., 0., 0., 0., 0.],
    //         [0., 0., 0., 0., 0., 0.],
    //         [0., 0., 0., 0., 0., 10.0e5],
    //     ],
    // );

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    // Connect blades to hub
    let hub_blade_constraint_ids = blades
        .iter()
        .map(|blade| model.add_rigid_constraint(hub_node_id, blade.nodes[0].id))
        .collect_vec();

    // Add prescribed rotation between tower top node and hub node
    let tower_hub_constraint_id =
        model.add_prescribed_rotation(tt_node_id, hub_node_id, col![1., 0., 0.]);
    // let tower_hub_constraint_id =
    //     model.add_revolute_joint(tt_node_id, hub_node_id, col![1., 0., 0.]);

    Turbine {
        blades,
        tower,
        hub_blade_constraint_ids,
        tower_hub_constraint_id,
    }
}

use std::{f64::consts::PI, fs};

use faer::prelude::*;
use itertools::{izip, Itertools};
use beamo::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::{BeamSection, Damping},
    model::Model,
    output_writer::OutputWriter,
    util::{
        quat_compose_alloc, quat_from_axis_angle_alloc, quat_inverse_alloc, quat_rotate_vector,
        write_matrix,
    }, // output_writer::OutputWriter,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

static OUT_DIR: &str = "examples/blade-reaction";

fn main() {
    [1.0, 20.0, 40.0].par_iter().for_each(|&rotor_rpm| {
        println!("Running for {} rpm", rotor_rpm);
        run(rotor_rpm);
    });
}

fn run(rotor_rpm: f64) {
    let time_step = 0.01;
    let duration = 1000.0;
    let n_steps = (duration / time_step) as usize;

    // Create model and set basic parameters
    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_max_iter(10);
    model.set_solver_tolerance(1e-6, 1e-5);
    model.set_rho_inf(0.95);

    // Create directory
    fs::create_dir_all(OUT_DIR).unwrap();

    // Create vector for hub angular velocity
    let omega = [rotor_rpm * 2.0 * PI / 60.0, 0., 0.];

    // build the turbine
    let turbine = build_rotor(omega, &mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    model.write_mesh_connectivity_file(OUT_DIR);

    // Create output writer and write initial condition
    // let mut netcdf_file =
    //     netcdf::create(&format!("{}/{:.2}_turbine.nc", directory, rotor_rpm)).unwrap();
    // let mut output_writer = OutputWriter::new(&mut netcdf_file, model.n_nodes());
    // output_writer.write(&state, 0);

    let root_orientation =
        quat_from_axis_angle_alloc(-90.0_f64.to_radians(), col![0., 1., 0.].as_ref());

    let mut blade_root_force = Mat::<f64>::zeros(3 * 3, n_steps);
    let mut blade_root_moment = Mat::<f64>::zeros(3 * 3, n_steps);

    let force_constraint_indices = turbine
        .hub_blade_constraint_ids
        .iter()
        .map(|&cid| {
            let i = &solver.constraints.constraints[cid].first_row_index;
            [i + 0, i + 1, i + 2]
        })
        .collect_vec();

    let moment_constraint_indices = turbine
        .hub_blade_constraint_ids
        .iter()
        .map(|&cid| {
            let i = &solver.constraints.constraints[cid].first_row_index;
            [i + 3, i + 4, i + 5]
        })
        .collect_vec();

    // Loop through steps
    for i in 0..n_steps {
        // Calculate time
        let t = ((1 + i) as f64) * time_step;

        // Calculate azimuth angle
        let azimuth = omega[0] * t;

        // Set rotation of tower-hub constraint
        solver.constraints.constraints[turbine.tower_hub_constraint_id].set_rotation(azimuth);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        izip!(
            turbine.blades.iter(),
            force_constraint_indices.iter(),
            moment_constraint_indices.iter(),
        )
        .enumerate()
        .for_each(|(j, (blade, force_indices, moment_indices))| {
            let root_node_id = blade.nodes[0].id;

            // Get the blade root rotation displacement
            let q_root = state.x.col(root_node_id).subrows(3, 4);
            let q_root_inv = quat_inverse_alloc(q_root);
            let rr0 = quat_compose_alloc(root_orientation.as_ref(), q_root_inv.as_ref());

            // Blade Force in rotating frame
            quat_rotate_vector(
                rr0.as_ref(),
                solver.lambda.subrows(force_indices[0], 3),
                blade_root_force.col_mut(i).subrows_mut(j * 3, 3),
            );

            // Blade moment in rotating frame
            quat_rotate_vector(
                rr0.as_ref(),
                solver.lambda.subrows(moment_indices[0], 3),
                blade_root_moment.col_mut(i).subrows_mut(j * 3, 3),
            );
        });

        // Exit if failed to converge
        if !res.converged {
            println!("{}RPM, failed, t={}, err={}", rotor_rpm, t, res.err);
            break;
        }

        assert_eq!(res.converged, true);

        // output_writer.write(&state, i + 1);
    }

    write_matrix(
        blade_root_force.transpose(),
        &format!("{}/{:.2}_blade_root_force.csv", OUT_DIR, rotor_rpm),
    )
    .unwrap();
    write_matrix(
        blade_root_moment.transpose(),
        &format!("{}/{:.2}_blade_root_moment.csv", OUT_DIR, rotor_rpm),
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

    // Hub mass and inertia
    model.add_mass_element(
        hub_node_id,
        mat![
            [20000., 0., 0., 0., 0., 0.],
            [0., 20000., 0., 0., 0., 0.],
            [0., 0., 20000., 0., 0., 0.],
            [0., 0., 0., 3.0e4, 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
        ],
    );

    // Add nacelle mass and yaw bearing mass
    model.add_mass_element(
        tt_node_id,
        mat![
            [100000. + 3000., 0., 0., 0., 0., 0.],
            [0., 100000. + 3000., 0., 0., 0., 0.],
            [0., 0., 100000. + 3000., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 10.0e5],
        ],
    );

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

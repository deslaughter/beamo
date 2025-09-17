use std::{collections::HashMap, f64::consts::PI, fs, io::Write};

use faer::prelude::*;
use itertools::Itertools;
use ottr::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::BeamSection,
    model::Model,
    output_writer::OutputWriter,
    util::{quat_as_rotation_vector, quat_compose_alloc, quat_from_axis_angle_alloc},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() {
    (0..11_usize).into_par_iter().for_each(|i| {
        let rpm = i as f64 * 4.0;
        let omega = rpm * 2.0 * PI / 60.0;
        println!("Running case {}, rpm = {}, omega = {}", i + 1, rpm, omega);
        run(i + 1, omega);
    });
}

#[rustfmt::skip]
fn run(op_num: usize, omega: f64) {
    let time_step = 0.01;
    // let duration = 2.0;
    // let n_steps = (duration / time_step) as usize;
    let n_revolutions = 1.;

    let rotor_rpm = omega * 60.0 / (2.0 * PI);

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_solver_tolerance(1e-5, 1e-3);
    model.set_rho_inf(0.);

    let n_steps = if omega > 0. {
        (n_revolutions * 2. * PI / omega / time_step).ceil() as usize
    } else {
        1
    };

    // Create directory
    let directory = format!("examples/rotor/lin").to_string();
    fs::create_dir_all(&directory).unwrap();

    let omega_v = [omega, 0., 0.];

    let mut turbine = build_rotor(omega_v, &mut model);
    model.write_mesh_connectivity_file(&directory);

    let mut netcdf_file = netcdf::create(format!(
        "{}/{:02}_turbine_{:04.2}.nc",
        directory,
        op_num,
        rotor_rpm.abs()
    ))
    .unwrap();
    let mut output_writer = OutputWriter::new(&mut netcdf_file, model.n_nodes());

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    output_writer.write(&state, 0);

    // Loop through steps
    for i in 1..n_steps + 1 {
        // Calculate time
        let t = (i as f64) * time_step;

        // Set rotation
        solver.constraints.constraints[turbine.tower_hub_constraint_id].set_rotation(omega * t);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Get the node motions
        turbine.blades.iter_mut().for_each(|blade| {
            blade.nodes.iter_mut().for_each(|node| {
                node.get_motion(&state);
            });
        });

        // get the tower node motions
        turbine.tower.nodes.iter_mut().for_each(|node| {
            node.get_motion(&state);
        });

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);

        output_writer.write(&state, i);
    }

    //--------------------------------------------------------------------------
    // Calculate all the things
    //--------------------------------------------------------------------------

    let t = time_step * n_steps as f64;
    let azimuth: f64 = 0.0;

    println!(
        "linearizing RPM {:04.2}, OP {}, t={} s",
        rotor_rpm, op_num, t
    );

    struct Desc {
        module: String,
        label: String,
        rot_frame: String,
    }

    let mut desc_map = HashMap::new();
    let mut node_data = Vec::new();
    turbine
        .tower
        .nodes
        .iter()
        .enumerate()
        .skip(1)
        .for_each(|(i, node)| {
            desc_map.insert(
                node.id,
                Desc {
                    module: "ED".to_string(),
                    label: format!("Tower N{}", i + 1),
                    rot_frame: "F".to_string(),
                },
            );
            node_data.push(node);
        });
    turbine.blades.iter().enumerate().for_each(|(i, blade)| {
        blade
            .nodes
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(j, node)| {
                desc_map.insert(
                    node.id,
                    Desc {
                        module: format!("BD_{}", i + 1),
                        label: format!("Blade N{}", j + 1),
                        rot_frame: "T".to_string(),
                    },
                );
                node_data.push(node);
            });
    });

    // Get the dof indices in the state
    let dof_indices = node_data
        .iter()
        .flat_map(|node| {
            let dofs = &solver.nfm.node_dofs[node.id];
            (dofs.first_dof_index..dofs.first_dof_index + dofs.count()).collect_vec()
        })
        .collect_vec();

    let n = dof_indices.len();

    let a_ss = solver.linearize(&state);

    let mut rv = col![0.0, 0.0, 0.0];

    let file_path = format!("{}/{:02}_turbine_{:04.2}.lin", directory, op_num, rotor_rpm.abs());
    let mut file = fs::File::create(file_path).expect("Failed to create output file");

    writeln!(file, "Simulation information:").unwrap();
    writeln!(file, "Simulation time:                    {} s", t).unwrap();
    writeln!(file, "Rotor Speed:                        {} rad/s", omega.abs()).unwrap();
    writeln!(file, "Azimuth:                            {} rad", azimuth.abs() % (2.0 * PI)).unwrap();
    writeln!(file, "Wind Speed:                         0.0000 m/s").unwrap();
    writeln!(file, "Number of continuous states:        {}", 2 * n).unwrap();
    writeln!(file, "Number of discrete states:          0").unwrap();
    writeln!(file, "Number of constraint states:        0").unwrap();
    writeln!(file, "Number of inputs:                   0").unwrap();
    writeln!(file, "Number of outputs:                  0").unwrap();
    writeln!(file, "Jacobians included in this file?    No").unwrap();

    writeln!(file, "\nOrder of continuous states:").unwrap();
    writeln!(file, "   Row/Column Operating Point    Rotating Frame? Derivative Order Description:").unwrap();
    writeln!(file, "   ---------- ---------------    --------------- ---------------- -----------").unwrap();

    let mut k: usize = 0;
    node_data.iter().for_each(|n| {
        let d = desc_map.get(&n.id).unwrap();
        quat_as_rotation_vector(n.position.subrows(3, 4), rv.as_mut());
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} trans-disp in X, m", k + 1, n.position[0], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} trans-disp in Y, m", k + 2, n.position[1], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} trans-disp in Z, m", k + 3, n.position[2], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} rot-disp in X, rad", k + 4, rv[0], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} rot-disp in Y, rad", k + 5, rv[1], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} {} rot-disp in Z, rad", k + 6, rv[2], d.rot_frame, d.module, d.label).unwrap();
        k += 6;
    });
    node_data.iter().for_each(|n| {
        let d = desc_map.get(&n.id).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in X, m/s", k + 1, n.velocity[0], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in Y, m/s", k + 2, n.velocity[1], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in Z, m/s", k + 3, n.velocity[2], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in X, rad/s", k + 4, n.velocity[3], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in Y, rad/s", k + 5, n.velocity[4], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in Z, rad/s", k + 6, n.velocity[5], d.rot_frame, d.module, d.label).unwrap();
        k += 6;
    });

    writeln!(file, "\nOrder of continuous state derivatives:").unwrap();
    writeln!(file, "   Row/Column Operating Point    Rotating Frame? Derivative Order Description:").unwrap();
    writeln!(file, "   ---------- ---------------    --------------- ---------------- -----------").unwrap();
    let mut k: usize = 0;
    node_data.iter().for_each(|n| {
        let d = desc_map.get(&n.id).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in X, m/s", k + 1, n.velocity[0], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in Y, m/s", k + 2, n.velocity[1], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} trans-disp in Z, m/s", k + 3, n.velocity[2], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in X, rad/s", k + 4, n.velocity[3], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in Y, rad/s", k + 5, n.velocity[4], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of {} rot-disp in Z, rad/s", k + 6, n.velocity[5], d.rot_frame, d.module, d.label).unwrap();
        k += 6;
    });
    node_data.iter().for_each(|n|{
        let d = desc_map.get(&n.id).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} trans-disp in X, m/s/s", k+1, n.acceleration[0], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} trans-disp in Y, m/s/s", k+2, n.acceleration[1], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} trans-disp in Z, m/s/s", k+3, n.acceleration[2], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} rot-disp in X, rad/s/s", k+4, n.acceleration[3], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} rot-disp in Y, rad/s/s", k+5, n.acceleration[4], d.rot_frame, d.module, d.label).unwrap();
        writeln!(file, "{:>5}  {:<32.16e} {} 2 {} First time derivative of First time derivative of {} rot-disp in Z, rad/s/s", k+6, n.acceleration[5], d.rot_frame, d.module, d.label).unwrap();
        k += 6;
    });

    let a_indices = dof_indices
        .iter()
        .cloned()
        .chain(dof_indices.iter().map(|i| *i + a_ss.ncols() / 2))
        .collect_vec();
    writeln!(file, "\nLinearized state matrices:").unwrap();
    writeln!(file, "\nA: {} x {}", a_indices.len(), a_indices.len()).unwrap();
    a_indices.iter().for_each(|i| {
        a_indices.iter().for_each(|j| {
            write!(file, "{:>24.16e}", a_ss[(*i, *j)]).unwrap();
        });
        writeln!(file).unwrap();
    });

    file.flush().unwrap();
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
        .set_root_velocity([0., 0., 0., omega[0], omega[1], omega[2]]);

    // Build blades at various rotations
    let blades = [0., 120., 240.0_f64]
        .into_iter()
        .map(|angle| {
            let r0 = quat_from_axis_angle_alloc(-90.0_f64.to_radians(), col![0., 1., 0.].as_ref());
            let r = quat_from_axis_angle_alloc(angle.to_radians(), col![1., 0., 0.].as_ref());
            let rr0 = quat_compose_alloc(r.as_ref(), r0.as_ref());
            blade_builder.set_root_position([0., 0., tower_height, rr0[0], rr0[1], rr0[2], rr0[3]]);
            BeamComponent::new(&blade_builder.build(), model)
        })
        .collect_vec();

    //--------------------------------------------------------------------------
    // Tower
    //--------------------------------------------------------------------------

    let n_tower_nodes = 4;
    let n_qps: usize = 21;

    let s = (0..n_qps)
        .map(|i| i as f64 / (n_qps - 1) as f64)
        .collect_vec();

    let sections = s
        .iter()
        .map(|&s| BeamSection {
            s,
            m_star: tower_m_star.clone(),
            c_star: tower_c_star.clone(),
        })
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
        .set_sections_z(&sections)
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

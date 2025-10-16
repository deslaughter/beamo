use std::io::Write;
use std::{f64::consts::PI, fs::File};

use faer::prelude::*;
use itertools::{izip, Itertools};

use beamo::util::{quat_rotate_vector, quat_rotate_vector_alloc};
use beamo::{
    components::{
        aero::{AeroBodyInput, AeroComponent, AeroSection},
        beam::BeamInputBuilder,
        inflow::Inflow,
        turbine::{Turbine, TurbineBuilder},
    },
    elements::beams::BeamSection,
    model::Model,
    output_writer::OutputWriter,
    util::annular_section,
};
use serde_yaml::Value;

fn main() {
    let time_step = 0.01;
    let duration = 100.0;
    let n_steps = (duration / time_step) as usize;

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, -9.81);
    model.set_solver_tolerance(1e-6, 1e-4);
    model.set_rho_inf(0.);

    let (mut turbine, mut aero) = build_turbine(&mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    model.write_mesh_connectivity_file("examples/iea15-aero");

    // Create netcdf output file
    let mut netcdf_file = netcdf::create("examples/iea15-aero/turbine.nc").unwrap();
    let mut ow = OutputWriter::new(&mut netcdf_file, state.n_nodes);
    ow.write(&state, 0);

    // Create inflow definition
    let fluid_density = 1.225; // kg/m^3
    let vel_h = 10.6; // m/s
    let h_ref = 150.0; // Reference height
    let pl_exp = 0.12; // Power law exponent
    let flow_angle = 0.0_f64.to_radians(); // Flow angle in radians
    let inflow = Inflow::steady_wind(vel_h, h_ref, pl_exp, flow_angle);

    // Create output file in OpenFAST format
    let mut outfile = File::create("output/output.out").unwrap();
    write!(outfile, "\n\n\n\n\n\n").unwrap();
    write!(outfile, "Time\tAzimuth\tRotSpeed\n").unwrap();
    write!(outfile, "(s)\t(deg)\t(rpm)\n").unwrap();

    // Torque at rated wind speed
    turbine.torque = -21.03e6;

    // Loop through steps
    let mut n_iter = 0;
    for i in 1..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        // Calculate torque load on the hub
        quat_rotate_vector(
            state.x.col(turbine.hub_node.id).subrows(3, 4),
            col![turbine.torque, 0., 0.].as_ref(),
            turbine.hub_node.loads.subrows_mut(3, 3),
        );

        // Copy loads from nodes to state
        turbine.set_loads(&mut state);

        // Calculate motion of aerodynamic centers
        aero.calculate_motion(&state);

        // Calculate inflow velocities by calling inflow.velocity at each aerodynamic center
        aero.set_inflow_from_function(|pos| inflow.velocity(t, pos));

        // Calculate aerodynamic loads
        aero.calculate_aerodynamic_loads(fluid_density);

        // Add function to apply aerodynamic loads directly

        // Calculate the nodal loads from the aerodynamic loads
        aero.calculate_nodal_loads();

        // Add the nodal loads to the state
        aero.add_nodal_loads_to_state(&mut state);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Get rotor info
        let (azimuth, rotor_speed, rotor_acc) = solver.constraints.constraints
            [turbine.shaft_base_azimuth_constraint_id]
            .calculate_revolute_output(&state);

        write!(
            outfile,
            "{}\t{}\t{}\n",
            t,
            if azimuth > 0. {
                azimuth
            } else {
                azimuth + 2.0 * std::f64::consts::PI
            }
            .to_degrees(),
            rotor_speed * 60.0 / (2.0 * std::f64::consts::PI)
        )
        .unwrap();

        turbine.torque -= (rotor_speed - 7.56 * 0.104719755) * 1e5;

        let aero_loads = aero
            .bodies
            .iter()
            .map(|b| {
                [
                    b.loads.row(0).sum(),
                    b.loads.row(1).sum(),
                    b.loads.row(2).sum(),
                ]
            })
            .collect_vec();

        let aero_loads = col![
            aero_loads.iter().map(|x| x[0]).sum::<f64>(),
            aero_loads.iter().map(|x| x[1]).sum::<f64>(),
            aero_loads.iter().map(|x| x[2]).sum::<f64>(),
        ];

        if i % 10 == 0 {
            println!(
                "t: {:6.2}, azimuth: {:.4}, rotor_speed: {:.3},  torque: {:.2}", //, tt_loads: {:?}, blade_loads: {:?}",
                t,
                azimuth,
                rotor_speed * 9.549297,
                turbine.torque //solver.constraint_loads(turbine.yaw_bearing_shaft_base_constraint_id).subrows(0, 3)/1000., aero_loads/1000.
            );
        }

        n_iter += res.iter;

        // Copy motion from nodes to state
        turbine.get_motion(&state);

        // Write nodal output
        ow.write(&state, i);

        // Write aerodynamic output
        aero.bodies.iter().enumerate().for_each(|(j, body)| {
            body.as_vtk()
                .export_ascii(format!(
                    "examples/iea15-aero/vtk_output/aero_b{}.{:0>3}.vtk",
                    j + 1,
                    i
                ))
                .unwrap()
        });

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);
    }

    println!("num nonlinear iterations: {}", n_iter);
}

fn build_turbine(model: &mut Model) -> (Turbine, AeroComponent) {
    let windio_path = "examples/IEA-15-240-RWT-aero.yaml";
    let yaml_file = std::fs::read_to_string(windio_path).expect("Unable to read file");
    let wio: Value = serde_yaml::from_str(&yaml_file).expect("Unable to parse YAML");

    //--------------------------------------------------------------------------
    // Blade
    //--------------------------------------------------------------------------

    let n_blade_nodes = 11;

    let blade = &wio["components"]["blade"];
    let ref_axis = &blade["reference_axis"];
    let blade_twist = &blade["outer_shape"]["twist"];
    let inertia_matrix = &blade["structure"]["elastic_properties"]["inertia_matrix"];
    let stiffness_matrix = &blade["structure"]["elastic_properties"]["stiffness_matrix"];

    // Inertia matrix components
    let ic = ["mass", "cm_x", "cm_y", "i_cp", "i_edge", "i_flap", "i_plr"]
        .iter()
        .map(|key| serde_yaml::from_value::<Vec<f64>>(inertia_matrix[key].clone()).unwrap())
        .collect_vec();

    // Stiffness matrix components
    let sc = [
        "K11", "K12", "K13", "K14", "K15", "K16", "K22", "K23", "K24", "K25", "K26", "K33", "K34",
        "K35", "K36", "K44", "K45", "K46", "K55", "K56", "K66",
    ]
    .iter()
    .map(|key| serde_yaml::from_value::<Vec<f64>>(stiffness_matrix[key].clone()).unwrap())
    .collect_vec();

    // Grid
    let blade_matrix_grid =
        serde_yaml::from_value::<Vec<f64>>(inertia_matrix["grid"].clone()).unwrap();

    let blade_sections = (0..blade_matrix_grid.len())
        .map(|i| {
            let (mass, cm_x, cm_y, i_cp, i_edge, i_flap, i_plr) =
                ic.iter().map(|arr| arr[i]).collect_tuple().unwrap();
            let m = mat![
                [mass, 0., 0., 0., 0., -mass * cm_y],
                [0., mass, 0., 0., 0., mass * cm_x],
                [0., 0., mass, mass * cm_y, -mass * cm_x, 0.],
                [0., 0., mass * cm_y, i_edge, -i_cp, 0.],
                [0., 0., -mass * cm_x, -i_cp, i_flap, 0.],
                [-mass * cm_y, mass * cm_x, 0., 0., 0., i_plr],
            ];
            let k = mat![
                [sc[0][i], sc[1][i], sc[2][i], sc[3][i], sc[4][i], sc[5][i]],
                [sc[1][i], sc[6][i], sc[7][i], sc[8][i], sc[9][i], sc[10][i]],
                [sc[2][i], sc[7][i], sc[11][i], sc[12][i], sc[13][i], sc[14][i]],
                [sc[3][i], sc[8][i], sc[12][i], sc[15][i], sc[16][i], sc[17][i]],
                [sc[4][i], sc[9][i], sc[13][i], sc[16][i], sc[18][i], sc[19][i]],
                [sc[5][i], sc[10][i], sc[14][i], sc[17][i], sc[19][i], sc[20][i]],
            ];
            BeamSection {
                s: blade_matrix_grid[i],
                m_star: m,
                c_star: k,
            }
        })
        .collect_vec();

    // Build blade input
    let blade_input = BeamInputBuilder::new()
        .set_element_order(n_blade_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &serde_yaml::from_value::<Vec<f64>>(ref_axis["x"]["grid"].clone()).unwrap(),
            &itertools::izip!(
                serde_yaml::from_value::<Vec<f64>>(ref_axis["x"]["values"].clone()).unwrap(),
                serde_yaml::from_value::<Vec<f64>>(ref_axis["y"]["values"].clone()).unwrap(),
                serde_yaml::from_value::<Vec<f64>>(ref_axis["z"]["values"].clone()).unwrap(),
            )
            .map(|(x, y, z)| [x, y, z])
            .collect_vec(),
            &serde_yaml::from_value::<Vec<f64>>(blade_twist["grid"].clone()).unwrap(),
            &serde_yaml::from_value::<Vec<f64>>(blade_twist["values"].clone()).unwrap(),
        )
        .set_sections_z(&blade_sections)
        .build();

    //--------------------------------------------------------------------------
    // Tower
    //--------------------------------------------------------------------------

    let n_tower_nodes = 7;

    let tower = &wio["components"]["tower"];
    let tower_diameter = &tower["outer_shape"]["outer_diameter"];
    let tower_material_thickness = &tower["structure"]["layers"][0]["thickness"];
    let tower_material_name = &tower["structure"]["layers"][0]["material"];
    let ref_axis = &tower["reference_axis"];

    let tower_material = wio["materials"]
        .as_sequence()
        .unwrap()
        .iter()
        .find(|m| m["name"] == *tower_material_name)
        .expect("Tower material not found");

    // Build tower input
    let elastic_modulus = tower_material["E"].as_f64().unwrap();
    let shear_modulus = tower_material["G"].as_f64().unwrap();
    let poisson_ratio = tower_material["nu"].as_f64().unwrap();
    let density = tower_material["rho"].as_f64().unwrap();
    let tower_input = BeamInputBuilder::new()
        .set_element_order(n_tower_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &ref_axis["x"]["grid"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
            &itertools::izip!(
                ref_axis["x"]["values"]
                    .as_sequence()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect_vec(),
                ref_axis["y"]["values"]
                    .as_sequence()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect_vec(),
                ref_axis["z"]["values"]
                    .as_sequence()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect_vec()
            )
            .map(|(x, y, z)| [x, y, z])
            .collect_vec(),
            &[0., 1.], // twist grid
            &[0., 0.], // twist values
        )
        .set_sections_z(
            izip!(
                tower_material_thickness["grid"]
                    .as_sequence()
                    .unwrap()
                    .iter(),
                tower_diameter["values"].as_sequence().unwrap().iter(),
                tower_material_thickness["values"]
                    .as_sequence()
                    .unwrap()
                    .iter()
            )
            .map(|(grid_location, diameter, thickness)| {
                let (m_star, c_star) = annular_section(
                    diameter.as_f64().unwrap(),
                    thickness.as_f64().unwrap(),
                    elastic_modulus,
                    shear_modulus,
                    poisson_ratio,
                    density,
                );
                BeamSection {
                    s: grid_location.as_f64().unwrap(),
                    m_star,
                    c_star,
                }
            })
            .collect_vec()
            .as_slice(),
        )
        .build();

    //--------------------------------------------------------------------------
    // Turbine
    //--------------------------------------------------------------------------

    let hub = &wio["components"]["hub"];
    let drivetrain = &wio["components"]["drivetrain"];

    // Construct turbine
    let turbine = TurbineBuilder::new()
        .set_blade_input(blade_input)
        .set_tower_input(tower_input)
        .set_n_blades(3)
        .set_cone_angle(hub["cone_angle"].as_f64().unwrap().to_radians())
        .set_hub_diameter(hub["diameter"].as_f64().unwrap())
        .set_rotor_apex_to_hub(0.)
        .set_tower_axis_to_rotor_apex(drivetrain["outer_shape"]["overhang"].as_f64().unwrap())
        .set_tower_top_to_rotor_apex(
            drivetrain["outer_shape"]["distance_tt_hub"]
                .as_f64()
                .unwrap(),
        )
        .set_shaft_tilt_angle(
            drivetrain["outer_shape"]["uptilt"]
                .as_f64()
                .unwrap()
                .to_radians(),
        )
        // .set_shaft_tilt_angle(0.)
        .set_rotor_speed(7.56 * 0.104719755) // 7.56 rpm
        .build(model)
        .unwrap();

    // Add hub mass
    let hub_mass = hub["elastic_properties"]["mass"].as_f64().unwrap();
    let hub_inertia =
        serde_yaml::from_value::<Vec<f64>>(hub["elastic_properties"]["inertia"].clone()).unwrap();
    // .as_mapping()
    // .unwrap()
    // .get(&Value::from("values"))
    // .unwrap()
    // .as_sequence()
    // .unwrap()
    // .iter()
    // .map(|v| v.as_f64().unwrap())
    // .collect_vec();
    model.add_mass_element(
        turbine.hub_node.id,
        mat![
            [hub_mass, 0., 0., 0., 0., 0.],
            [0., hub_mass, 0., 0., 0., 0.],
            [0., 0., hub_mass, 0., 0., 0.],
            [0., 0., 0., hub_inertia[0], hub_inertia[3], hub_inertia[4]],
            [0., 0., 0., hub_inertia[3], hub_inertia[1], hub_inertia[5]],
            [0., 0., 0., hub_inertia[4], hub_inertia[5], hub_inertia[2]]
        ],
    );

    //--------------------------------------------------------------------------
    // Aero
    //--------------------------------------------------------------------------

    // Build aero sections, same for all blades
    let aero_sections = wio["airfoils"]
        .as_sequence()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, af)| AeroSection {
            id: i,
            s: af["spanwise_position"].as_f64().unwrap(),
            chord: af["chord"].as_f64().unwrap(),
            twist: af["twist"].as_f64().unwrap().to_radians(),
            section_offset_x: af["section_offset_x"].as_f64().unwrap(),
            section_offset_y: af["section_offset_y"].as_f64().unwrap(),
            aerodynamic_center: af["aerodynamic_center"].as_f64().unwrap(),
            aoa: serde_yaml::from_value::<Vec<f64>>(
                af["polars"][0]["re_sets"][0]["cl"]["grid"].clone(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_radians)
            .collect_vec(),
            cl: serde_yaml::from_value::<Vec<f64>>(
                af["polars"][0]["re_sets"][0]["cl"]["values"].clone(),
            )
            .unwrap(),
            cd: serde_yaml::from_value::<Vec<f64>>(
                af["polars"][0]["re_sets"][0]["cd"]["values"].clone(),
            )
            .unwrap(),
            cm: serde_yaml::from_value::<Vec<f64>>(
                af["polars"][0]["re_sets"][0]["cm"]["values"].clone(),
            )
            .unwrap(),
        })
        .collect_vec();

    // Create actuator line coupling
    let aero = AeroComponent::new(
        &turbine
            .blades
            .iter()
            .map(|b| AeroBodyInput {
                id: 0,
                beam_node_ids: b.nodes.iter().map(|n| n.id).collect(),
                aero_sections: aero_sections.clone(),
            })
            .collect_vec(),
        &model.nodes,
    );

    (turbine, aero)
}

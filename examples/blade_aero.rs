use faer::prelude::*;
use itertools::Itertools;
use ottr::{
    components::{
        aero::{AeroBodyInput, AeroComponent, AeroSection},
        beam::{BeamComponent, BeamInputBuilder},
        inflow::Inflow,
    },
    elements::beams::BeamSection,
    model::Model,
    output_writer::OutputWriter,
    util::quat_from_rotation_vector_alloc,
};
use serde_yaml::Value;

fn main() {
    let time_step = 0.01;
    let duration = 10.0;
    let n_steps = (duration / time_step) as usize;

    let fluid_density = 1.225; // kg/m^3
    let vel_h = 50.0; // m/s
    let h_ref = 100.0; // Reference height
    let pl_exp = 0.0; // Power law exponent
    let flow_angle = 0.0; // Flow angle (radians)

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.);
    model.set_solver_tolerance(1e-5, 1e-3);
    model.set_rho_inf(0.);

    let (_blade, mut aero) = build_blade("examples/IEA-15-240-RWT-aero.yaml", &mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    model.write_mesh_connectivity_file("output");

    // Create netcdf output file
    let mut netcdf_file = netcdf::create("output/blade_aero.nc").unwrap();
    let mut ow = OutputWriter::new(&mut netcdf_file, state.n_nodes);
    ow.write(&state, 0);

    // Create inflow
    let inflow = Inflow::steady_wind(vel_h, h_ref, pl_exp, flow_angle);

    // Loop through steps
    for i in 1..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        // Calculate motion of aerodynamic centers
        aero.calculate_motion(&state);

        // Calculate inflow velocities by calling inflow.velocity at each aerodynamic center
        aero.set_inflow_from_function(|pos| inflow.velocity(t, pos));

        // Calculate aerodynamic loads
        aero.calculate_aerodynamic_loads(fluid_density);

        // Calculate the nodal loads from the aerodynamic loads
        aero.calculate_nodal_loads();

        // Clear external loads
        state.fx.fill(0.);

        // Add the nodal loads to the state
        aero.add_nodal_loads_to_state(&mut state);

        // println!("t = {}, node loads = {:?}", t, state.fx);
        // println!("t = {}, node loads = {:?}", t, aero.bodies[0].node_f);
        // println!("t = {}, aero loads = {:?}", t, aero.bodies[0].loads);

        // println!(
        //     "t = {}, node_f sum = {:?}, {:?}, {:?}, loads sum = {:?}, {:?}, {:?}",
        //     t,
        //     aero.bodies[0].node_f.row(0).sum(),
        //     aero.bodies[0].node_f.row(1).sum(),
        //     aero.bodies[0].node_f.row(2).sum(),
        //     aero.bodies[0].loads.row(0).sum(),
        //     aero.bodies[0].loads.row(1).sum(),
        //     aero.bodies[0].loads.row(2).sum()
        // );

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Write output
        ow.write(&state, i);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);
    }
}

fn build_blade(wio_path: &str, model: &mut Model) -> (BeamComponent, AeroComponent) {
    let n_blade_nodes = 11;

    // Parse the YAML file
    // let wio = read_windio_from_file(wio_path);

    let yaml_file = std::fs::read_to_string(wio_path).expect("Unable to read file");
    let wio: Value = serde_yaml::from_str(&yaml_file).expect("Unable to parse YAML");

    //--------------------------------------------------------------------------
    // Blade
    //--------------------------------------------------------------------------

    let blade = &wio["components"]["blade"];
    let ref_axis = &blade["reference_axis"];
    let blade_twist = &blade["outer_shape"]["twist"];
    let inertia_matrix = &blade["structure"]["elastic_properties"]["inertia_matrix"];
    let stiffness_matrix = &blade["structure"]["elastic_properties"]["stiffness_matrix"];

    // Inertia matrix components
    let ic = ["mass", "cm_x", "cm_y", "i_cp", "i_edge", "i_flap", "i_plr"]
        .iter()
        .map(|key| {
            inertia_matrix[key]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec()
        })
        .collect_vec();

    // Stiffness matrix components
    let sc = [
        "K11", "K12", "K13", "K14", "K15", "K16", "K22", "K23", "K24", "K25", "K26", "K33", "K34",
        "K35", "K36", "K44", "K45", "K46", "K55", "K56", "K66",
    ]
    .iter()
    .map(|key| {
        stiffness_matrix[key]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect_vec()
    })
    .collect_vec();

    // Grid
    let matrix_grid = inertia_matrix["grid"]
        .as_sequence()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect_vec();

    let sections = (0..matrix_grid.len())
        .map(|i| {
            let mass = ic[0][i];
            let cm_x = ic[1][i];
            let cm_y = ic[2][i];
            let i_cp = ic[3][i];
            let i_edge = ic[4][i];
            let i_flap = ic[5][i];
            let i_plr = ic[6][i];
            let m = mat![
                [mass, 0., 0., 0., 0., -mass * cm_y],
                [0., mass, 0., 0., 0., mass * cm_x],
                [0., 0., mass, mass * cm_y, -mass * cm_x, 0.],
                [0., 0., mass * cm_y, i_edge, -i_cp, 0.],
                [0., 0., -mass * cm_x, -i_cp, i_flap, 0.],
                [-mass * cm_y, mass * cm_x, 0., 0., 0., i_plr],
            ];
            let k11 = sc[0][i];
            let k12 = sc[1][i];
            let k13 = sc[2][i];
            let k14 = sc[3][i];
            let k15 = sc[4][i];
            let k16 = sc[5][i];
            let k22 = sc[6][i];
            let k23 = sc[7][i];
            let k24 = sc[8][i];
            let k25 = sc[9][i];
            let k26 = sc[10][i];
            let k33 = sc[11][i];
            let k34 = sc[12][i];
            let k35 = sc[13][i];
            let k36 = sc[14][i];
            let k44 = sc[15][i];
            let k45 = sc[16][i];
            let k46 = sc[17][i];
            let k55 = sc[18][i];
            let k56 = sc[19][i];
            let k66 = sc[20][i];
            let k = mat![
                [k11, k12, k13, k14, k15, k16],
                [k12, k22, k23, k24, k25, k26],
                [k13, k23, k33, k34, k35, k36],
                [k14, k24, k34, k44, k45, k46],
                [k15, k25, k35, k45, k55, k56],
                [k16, k26, k36, k46, k56, k66],
            ];
            BeamSection {
                s: matrix_grid[i],
                m_star: m,
                c_star: k,
            }
        })
        .collect_vec();

    // Determine root orientation such that leading edge is aligned with -x-axis
    // let root_orientation = quat_compose_alloc(
    //     quat_from_rotation_vector_alloc(col![0., 0., -90.0_f64.to_radians()].as_ref()).as_ref(),
    //     quat_from_rotation_vector_alloc(col![180.0_f64.to_radians(), 0., 0.].as_ref()).as_ref(),
    // );
    let root_orientation =
        quat_from_rotation_vector_alloc(col![0., 0., -90.0_f64.to_radians()].as_ref());

    // Build blade input
    let blade_input = BeamInputBuilder::new()
        .set_element_order(n_blade_nodes - 1)
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
            &blade_twist["grid"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
            &blade_twist["values"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
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
        .build();

    // Build and return the beam component
    let blade = BeamComponent::new(&blade_input, model);

    //--------------------------------------------------------------------------
    // Aero
    //--------------------------------------------------------------------------

    let aero_points = wio["airfoils"]
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
            aoa: af["polars"][0]["re_sets"][0]["cl"]["grid"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap().to_radians())
                .collect_vec(),
            cl: af["polars"][0]["re_sets"][0]["cl"]["values"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
            cd: af["polars"][0]["re_sets"][0]["cd"]["values"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
            cm: af["polars"][0]["re_sets"][0]["cm"]["values"]
                .as_sequence()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect_vec(),
        })
        .collect_vec();

    // Create actuator line coupling
    let aero = AeroComponent::new(
        &[AeroBodyInput {
            id: 0,
            beam_node_ids: blade.nodes.iter().map(|n| n.id).collect(),
            aero_sections: aero_points,
        }],
        &model.nodes,
    );

    (blade, aero)
}

use rayon::prelude::*;
use std::f64::consts::PI;

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
    util::{quat_compose_alloc, quat_from_rotation_vector_alloc},
};
use serde_yaml::Value;

fn main() {
    let n_angles: usize = 72;
    let angle_step = 2.0 * PI / n_angles as f64;
    (0..n_angles).into_par_iter().for_each(|i| {
        let flow_angle = i as f64 * angle_step;
        run_simulation(flow_angle);
    });
}
fn run_simulation(flow_angle: f64) {
    let time_step = 0.01;
    let duration = 5.0;
    let n_steps = (duration / time_step) as usize;

    let fluid_density = 1.225; // kg/m^3
    let vel_h = 30.0; // m/s
    let h_ref = 100.0; // Reference height
    let pl_exp = 0.0; // Power law exponent

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_solver_tolerance(1e-5, 1e-3);
    model.set_rho_inf(0.);

    let (_blade, mut aero) = build_blade(&mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    model.write_mesh_connectivity_file("output");

    // Create netcdf output file
    let mut netcdf_file = netcdf::create(format!(
        "output/blade_aero_{:.0}.nc",
        flow_angle.to_degrees()
    ))
    .unwrap();
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

fn build_blade(model: &mut Model) -> (BeamComponent, AeroComponent) {
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
                s: matrix_grid[i],
                m_star: m,
                c_star: k,
            }
        })
        .collect_vec();

    // Orient blade so reference axis is along z-axis
    let root_orientation = quat_compose_alloc(
        quat_from_rotation_vector_alloc(col![0., 0., -90.0_f64.to_radians()].as_ref()).as_ref(),
        quat_from_rotation_vector_alloc(col![0., -90.0_f64.to_radians(), 0.].as_ref()).as_ref(),
    );

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

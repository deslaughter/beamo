use faer::prelude::*;
use itertools::{izip, Itertools};
use beamo::{
    components::{
        beam::BeamInputBuilder,
        turbine::{Turbine, TurbineBuilder},
    },
    elements::beams::BeamSection,
    model::Model,
    output_writer::OutputWriter,
    util::{annular_section, quat_rotate_vector},
    windio::{read_windio_from_file, ScalarOrVector},
};

fn main() {
    let time_step = 0.01;
    let duration = 100.0;
    let n_steps = (duration / time_step) as usize;

    faer::set_global_parallelism(Par::Seq);

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, -9.81);
    model.set_solver_tolerance(1e-6, 1e-4);
    model.set_rho_inf(0.);

    let mut turbine = build_turbine("examples/IEA-15-240-RWT.yaml", &mut model);

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Write mesh connectivity file
    model.write_mesh_connectivity_file("output");

    // Create netcdf output file
    let mut netcdf_file = netcdf::create("output/turbine.nc").unwrap();
    let mut ow = OutputWriter::new(&mut netcdf_file, state.n_nodes);
    ow.write(&state, 0);

    // Set tower top force
    turbine.tower.nodes.last_mut().unwrap().loads[0] = 1e5;

    // Set torque on the hub
    turbine.torque = 1e8;

    let mut torque_vector = Col::<f64>::zeros(3);

    turbine.azimuth_node.get_motion(&state);

    let mut n_iter = 0;

    // Loop through steps
    for i in 1..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        // Calculate torque and apply to the azimuth node
        if i < 500 {
            quat_rotate_vector(
                turbine.azimuth_node.displacement.subrows(3, 4),
                col![turbine.torque, 0., 0.].as_ref(),
                torque_vector.as_mut(),
            );
            // println!("t={}, torque_vector={:?}", t, torque_vector);
            turbine
                .azimuth_node
                .loads
                .subrows_mut(3, 3)
                .copy_from(&torque_vector);
        } else if i == 500 {
            torque_vector.fill(0.);
            turbine
                .azimuth_node
                .loads
                .subrows_mut(3, 3)
                .copy_from(&torque_vector);
        }

        // Set blade 3 pitch constraint rotation
        solver.constraints.constraints[turbine.pitch_constraint_ids[2]].set_rotation(t * 0.5);

        // Set the yaw constraint rotation
        solver.constraints.constraints[turbine.yaw_constraint_id].set_rotation(t * 0.3);

        // Copy loads from nodes to state
        turbine.set_loads(&mut state);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        n_iter += res.iter;

        // Copy motion from nodes to state
        turbine.get_motion(&state);

        // Write output
        ow.write(&state, i);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
            break;
        }

        assert_eq!(res.converged, true);
    }

    println!("num nonlinear iterations: {}", n_iter);
}

fn build_turbine(wio_path: &str, model: &mut Model) -> Turbine {
    let n_blade_nodes = 11;
    let n_tower_nodes = 11;

    // Parse the YAML file
    let wio = read_windio_from_file(wio_path);

    //--------------------------------------------------------------------------
    // Blade
    //--------------------------------------------------------------------------

    let blade = &wio.components.blade;
    let ref_axis = &blade.elastic_properties_mb.six_x_six.reference_axis;
    let blade_twist = &blade.elastic_properties_mb.six_x_six.twist;
    let blade_matrices = &blade.elastic_properties_mb.six_x_six;

    // Build blade input
    let blade_input = BeamInputBuilder::new()
        .set_element_order(n_blade_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &ref_axis.x.grid,
            &itertools::izip!(
                ref_axis.x.values.iter(),
                ref_axis.y.values.iter(),
                ref_axis.z.values.iter()
            )
            .map(|(&x, &y, &z)| [x, y, z])
            .collect_vec(),
            &blade_twist.grid,
            &blade_twist.values,
        )
        .set_sections_z(
            izip!(
                blade_matrices.stiff_matrix.grid.iter(),
                blade_matrices.stiff_matrix.values.iter(),
                blade_matrices.inertia_matrix.values.iter()
            )
            .map(|(&grid_location, stiff_matrix, inertia_matrix)| {
                let m_star = inertia_matrix.as_matrix();
                let c_star = stiff_matrix.as_matrix();
                BeamSection {
                    s: grid_location,
                    m_star,
                    c_star,
                }
            })
            .collect_vec()
            .as_slice(),
        )
        .build();

    //--------------------------------------------------------------------------
    // Tower
    //--------------------------------------------------------------------------

    let tower = &wio.components.tower;
    let tower_layers = &tower.internal_structure_2d_fem.layers;
    let ref_axis = &tower.outer_shape_bem.reference_axis;
    let tower_diameter = &tower.outer_shape_bem.outer_diameter;

    let tower_material = wio
        .materials
        .iter()
        .find(|m| m.name == tower_layers[0].material)
        .expect("Tower material not found");
    let elastic_modulus = *match &tower_material.elastic_modulus {
        ScalarOrVector::Scalar(v) => v,
        ScalarOrVector::Vector(v) => &v[0],
    };
    let shear_modulus = *match &tower_material.shear_modulus {
        ScalarOrVector::Scalar(v) => v,
        ScalarOrVector::Vector(v) => &v[0],
    };
    let poisson_ratio = *match &tower_material.poisson_ratio {
        ScalarOrVector::Scalar(v) => v,
        ScalarOrVector::Vector(v) => &v[0],
    };

    // Build tower input
    let tower_input = BeamInputBuilder::new()
        .set_element_order(n_tower_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis_z(
            &ref_axis.x.grid,
            &itertools::izip!(
                ref_axis.x.values.iter(),
                ref_axis.y.values.iter(),
                ref_axis.z.values.iter()
            )
            .map(|(&x, &y, &z)| [x, y, z])
            .collect_vec(),
            &[0., 1.],
            &[0., 0.],
        )
        .set_sections_z(
            izip!(
                tower_layers[0].thickness.grid.iter(),
                tower_diameter.values.iter(),
                tower_layers[0].thickness.values.iter()
            )
            .map(|(&grid_location, &diameter, &thickness)| {
                let (m_star, c_star) = annular_section(
                    diameter,
                    thickness,
                    elastic_modulus,
                    shear_modulus,
                    poisson_ratio,
                    tower_material.density,
                );
                BeamSection {
                    s: grid_location,
                    m_star,
                    c_star,
                }
            })
            .collect_vec()
            .as_slice(),
        )
        .build();

    // Construct turbine
    TurbineBuilder::new()
        .set_blade_input(blade_input)
        .set_tower_input(tower_input)
        .set_n_blades(3)
        .set_cone_angle(wio.components.hub.cone_angle)
        .set_hub_diameter(wio.components.hub.diameter)
        .set_rotor_apex_to_hub(0.)
        .set_tower_axis_to_rotor_apex(wio.components.nacelle.drivetrain.overhang)
        .set_tower_top_to_rotor_apex(wio.components.nacelle.drivetrain.distance_tt_hub)
        .set_shaft_tilt_angle(wio.components.nacelle.drivetrain.uptilt)
        .build(model)
        .unwrap()
}

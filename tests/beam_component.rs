use faer::prelude::*;
use itertools::{izip, Itertools};
use ottr::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::BeamSection,
    model::Model,
    output_writer::OutputWriter,
    util::write_matrix,
    windio::read_windio_from_file,
};

#[test]
fn test_beam_component_simple() {
    let time_step = 0.01;
    let duration = 1.0;
    let n_steps = (duration / time_step) as usize;

    let n_nodes = 4;

    // Mass matrix 6x6
    let m_star = mat![
        [8.538, 0.000, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 8.538, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 8.538, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 1.4433, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.40972, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.00000, 1.0336],
    ] * 1e-2;

    // Stiffness matrix 6x6
    let c_star = mat![
        [1368.17, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 88.56, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 38.78, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 16.960, 17.610, -0.351],
        [0.0000, 0.0000, 0.0000, 17.610, 59.120, -0.370],
        [0.0000, 0.0000, 0.0000, -0.351, -0.370, 141.47],
    ] * 1e3;

    let n_kps = 12;
    let n_sections = 21;

    let beam_input = BeamInputBuilder::new()
        .set_element_order(n_nodes - 1)
        .set_section_refinement(0)
        .set_reference_axis(
            (0..n_kps)
                .map(|i| i as f64 / (n_kps - 1) as f64)
                .collect_vec()
                .as_slice(),
            (0..n_kps)
                .map(|i| [10. * i as f64 / (n_kps - 1) as f64, 0., 0.])
                .collect_vec()
                .as_slice(),
            &[0., 1.],
            &[0., 0.],
        )
        .set_sections(
            (0..n_sections)
                .map(|i| BeamSection {
                    s: i as f64 / (n_sections - 1) as f64,
                    m_star: m_star.clone(),
                    c_star: c_star.clone(),
                })
                .collect_vec()
                .as_slice(),
        )
        .build();

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_rho_inf(0.0);
    model.set_max_iter(1);
    model.set_gravity(0.0, 0.0, -9.81);

    let beam_1 = BeamComponent::new(&beam_input, &mut model);
    let beam_2 = BeamComponent::new(&beam_input, &mut model);
    beam_2.nodes.iter().for_each(|node| {
        model.nodes[node.id].translate([10., 0., 0.]);
    });

    model.add_prescribed_constraint(beam_1.nodes.first().unwrap().id);
    model.add_rigid_constraint(
        beam_1.nodes.last().unwrap().id,
        beam_2.nodes.first().unwrap().id,
    );

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Create netcdf output file
    let mut netcdf_file = netcdf::create("output/beam_component.nc").unwrap();
    let mut ow = OutputWriter::new(&mut netcdf_file, state.n_nodes);
    ow.write(&state, 0);
    model.write_mesh_connectivity_file("output");

    let res = solver.step(&mut state);
    write_matrix(solver.st.as_ref(), "output/st.csv").unwrap();
    solver.rhs.iter().enumerate().for_each(|(i, v)| {
        println!("rhs[{}] = {}", i, v);
    });
    // assert_eq!(res.converged, true);

    // // Loop through steps
    // for i in 1..n_steps {
    //     // Take step and get convergence result
    //     let res = solver.step(&mut state);

    //     ow.write(&state, i);

    //     // Exit if failed to converge
    //     if !res.converged {
    //         println!("failed, i={}, err={}", i, res.err);
    //     }

    //     // assert_eq!(res.converged, true);
    // }
}

#[test]
fn test_beam_component() {
    let time_step = 0.01;
    let duration = 10.0;
    let n_steps = (duration / time_step) as usize;

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_rho_inf(0.0);
    model.set_max_iter(0);
    model.set_gravity(0.0, 0.0, -9.81);

    let n_blade_nodes = 11;

    // Parse the YAML file
    let wio = read_windio_from_file("examples/IEA-15-240-RWT.yaml");

    let blade = &wio.components.blade;
    let ref_axis = &blade.elastic_properties_mb.six_x_six.reference_axis;
    let blade_twist = &blade.elastic_properties_mb.six_x_six.twist;
    let blade_matrices = &blade.elastic_properties_mb.six_x_six;

    // Build blade input
    let blade_input = BeamInputBuilder::new()
        .set_element_order(n_blade_nodes - 1)
        .set_section_refinement(1)
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

    let blade1 = BeamComponent::new(&blade_input, &mut model);
    let blade2 = BeamComponent::new(&blade_input, &mut model);

    model.add_prescribed_constraint(blade1.nodes[0].id);
    model.add_rigid_constraint(
        blade1.nodes.last().unwrap().id,
        blade2.nodes.first().unwrap().id,
    );

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Create netcdf output file
    let mut netcdf_file = netcdf::create("output/beam_component.nc").unwrap();
    let mut ow = OutputWriter::new(&mut netcdf_file, state.n_nodes);
    ow.write(&state, 0);

    // Loop through steps
    for i in 1..n_steps {
        // Take step and get convergence result
        let res = solver.step(&mut state);

        ow.write(&state, i);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, i={}, err={}", i, res.err);
        }

        assert_eq!(res.converged, true);
    }
}

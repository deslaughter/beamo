use std::f64::consts::PI;

use faer::prelude::*;
use itertools::Itertools;
use ottr::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::BeamSection,
    model::Model,
    util::{quat_from_axis_angle_alloc, write_matrix},
};
use std::ops::AddAssign;

static OUT_DIR: &str = "examples/tower-modes";

fn main() {
    let time_step = 0.01;

    let mut model = Model::new();
    model.set_time_step(time_step);
    model.set_gravity(0.0, 0.0, 0.0);
    model.set_solver_tolerance(1e-5, 1e-3);
    model.set_rho_inf(0.);

    let n_tower_nodes = 11;
    let tower_height = 120.;
    let n_qps: usize = 21;

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
    let tower = BeamComponent::new(&tower_input, &mut model);

    let tt_node_id = tower.nodes.last().unwrap().id;

    // Hub + nacelle + yaw bearing mass and inertia
    model.add_mass_element(
        tt_node_id,
        mat![
            [20000. + 3.0 * 13699.980, 0., 0., 0., 0., 0.],
            [0., 20000. + 3.0 * 13699.980, 0., 0., 0., 0.],
            [0., 0., 20000. + 3.0 * 13699.980, 0., 0., 0.],
            [0., 0., 0., 3.0e4, 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
        ] + mat![
            [100000. + 3000., 0., 0., 0., 0., 0.],
            [0., 100000. + 3000., 0., 0., 0., 0.],
            [0., 0., 100000. + 3000., 0., 0., 0.],
            [0., 0., 0., 3.0 * 1.81477E+07, 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 10.0e5],
        ],
    );

    let mut solver = model.create_solver();
    let state = model.create_state();

    tower.nodes.iter().for_each(|n| {
        println!("Tower node {} pos: {:?}", n.id, state.x.col(n.id));
    });

    solver
        .elements
        .assemble_system(&state, time_step, solver.r.as_mut());

    // Beam mass and stiffness
    let mut m = solver.elements.beams.m_sp.to_dense();
    let mut k = solver.elements.beams.k_sp.to_dense();

    // System mass and stiffness
    m.add_assign(&solver.elements.masses.m_sp.to_dense());
    k.add_assign(&solver.elements.masses.k_sp.to_dense());

    let ndof_bc = solver.n_system - 6;
    let lu = m.submatrix(6, 6, ndof_bc, ndof_bc).partial_piv_lu();
    let a = lu.solve(k.submatrix(6, 6, ndof_bc, ndof_bc));

    let eig = a.eigen().unwrap();
    let eig_val_raw = eig.S().column_vector();
    let eig_vec_raw = eig.U();

    let mut eig_order: Vec<_> = (0..eig_val_raw.nrows()).collect();
    eig_order.sort_by(|&i, &j| {
        eig_val_raw
            .get(i)
            .re
            .partial_cmp(&eig_val_raw.get(j).re)
            .unwrap()
    });

    let eig_val = Col::<f64>::from_fn(eig_val_raw.nrows(), |i| eig_val_raw[eig_order[i]].re);
    let mut eig_vec = Mat::<f64>::from_fn(solver.n_system, eig_vec_raw.ncols(), |i, j| {
        if i < 6 {
            0.
        } else {
            eig_vec_raw[(i - 6, eig_order[j])].re
        }
    });
    // normalize eigen vectors
    eig_vec.as_mut().col_iter_mut().for_each(|mut c| {
        let max = *c
            .as_ref()
            .iter()
            .reduce(|acc, e| if e.abs() > acc.abs() { e } else { acc })
            .unwrap();
        zip!(&mut c).for_each(|unzip!(c)| *c /= max);
    });

    println!("Natural frequencies (Hz):");
    eig_val
        .iter()
        .enumerate()
        .take(10)
        .for_each(|(i, &lambda)| println!("Mode {i}: {:.3}", lambda.sqrt() / (2. * PI)));

    write_matrix(eig_vec.as_ref(), &format!("{OUT_DIR}/tower-modes.csv")).unwrap();
}

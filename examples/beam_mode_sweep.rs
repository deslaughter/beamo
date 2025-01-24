use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::{
    col,
    complex_native::c64,
    linalg::solvers::{Eigendecomposition, SpSolver},
    mat, unzipped, zipped, Col, ColRef, Mat, Scale,
};

use itertools::{izip, Itertools};
use ottr::{
    elements::beams::{BeamSection, Damping},
    // external::parse_beamdyn_sections,
    interp::gauss_legendre_lobotto_points,
    model::Model,
    quadrature::Quadrature,
    util::{quat_as_rotation_vector, ColAsMatRef},
};

const V_SCALE: f64 = 1.0;

const OUT_DIR: &str = "output/beam_mode_sweep";

fn main() {
    // Damping ratio for modes 1-6
    let zeta = col![0.01, 0.02, 0.03, 0.04, 0.05, 0.06];

    // Select damping type
    // let damping = Damping::None;
    // let damping = Damping::Mu(col![0., 0., 0., 0., 0., 0.]);
    let damping = Damping::ModalElement(zeta.clone());

    // Settings
    let n_cycles = 3.5; // Number of oscillations to simulate
    let rho_inf = 1.; // Numerical damping
    let max_iter = 6; // Max convergence iterations
    let time_step = 0.0001; // Time step

    // Create output directory
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initialize model
    let mut model = setup_model(damping.clone());
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

    // Perform modal analysis
    let (eig_val, eig_vec) = modal_analysis(&model);

    // Calculate omega from eigenvalues
    let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    // Additional initialization for mu damping
    match damping {
        Damping::Mu(_) => {
            // Get index of maximum value
            let i_max = eig_vec
                .col_iter()
                .map(|psi| {
                    psi.iter()
                        .enumerate()
                        .max_by(|(_, &a), (_, &b)| a.abs().total_cmp(&b.abs()))
                        .map(|(index, _)| index)
                        .unwrap()
                        % 3
                })
                .collect_vec();
            let i_max_x = i_max.iter().position(|&i| i == 0).unwrap();
            let i_max_y = i_max.iter().position(|&i| i == 1).unwrap();
            let i_max_z = i_max.iter().position(|&i| i == 2).unwrap();

            let mu_x = 2. * zeta[i_max_x] / omega[i_max_x];
            let mu_y = 2. * zeta[i_max_y] / omega[i_max_y];
            let mu_z = 2. * zeta[i_max_z] / omega[i_max_z];

            let mu = col![mu_x, mu_y, mu_z, mu_x, mu_z, mu_y];
            println!("mu={:?}", mu);
            println!(
                "modes: x={}, y={}, z={}",
                i_max_x + 1,
                i_max_y + 1,
                i_max_z + 1
            );
            model
                .beam_elements
                .iter_mut()
                .for_each(|e| e.damping = Damping::Mu(mu.clone()));
        }
        _ => (),
    }

    // Loop through modes and run simulation
    izip!(omega.iter(), eig_vec.col_iter())
        .take(6)
        .enumerate()
        .for_each(|(i, (&omega, shape))| {
            let t_end = 2. * PI / omega;
            let n_steps = (n_cycles * t_end / time_step) as usize;
            run_simulation(i + 1, time_step, n_steps, shape, &model);
        });
}

fn run_simulation(mode: usize, time_step: f64, n_steps: usize, shape: ColRef<f64>, model: &Model) {
    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Apply scaled mode shape to state as velocity
    let v = shape * Scale(V_SCALE);
    state.v.copy_from(v.as_ref().as_mat_ref(6, state.n_nodes));

    // Create output file
    let mut file = File::create(format!("{OUT_DIR}/sweep_{:02}.csv", mode)).unwrap();

    // Cartesian rotation vector
    let mut rv = Col::<f64>::zeros(3);

    // Loop through times and run simulation
    for i in 0..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        file.write_fmt(format_args!("{t}")).unwrap();
        state.u.col_iter().for_each(|c| {
            quat_as_rotation_vector(c.subrows(3, 4), rv.as_mut());
            file.write_fmt(format_args!(
                ",{},{},{},{},{},{}",
                c[0], c[1], c[2], rv[0], rv[1], rv[2]
            ))
            .unwrap();
        });
        file.write(b"\n").unwrap();

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
        }

        assert_eq!(res.converged, true);
    }
}

fn modal_analysis(model: &Model) -> (Col<f64>, Mat<f64>) {
    // Create solver and state from model
    let mut solver = model.create_solver();
    let state = model.create_state();

    // Calculate system based on initial state
    solver.elements.beams.calculate_system(&state);

    // Get matrices
    solver.elements.beams.assemble_system(
        &solver.nfm,
        solver.m.as_mut(),
        solver.ct.as_mut(),
        solver.kt.as_mut(),
        solver.r.as_mut(),
    );

    let ndof_bc = solver.n_system - 6;
    let lu = solver.m.submatrix(6, 6, ndof_bc, ndof_bc).partial_piv_lu();
    let a = lu.solve(solver.kt.submatrix(6, 6, ndof_bc, ndof_bc));

    let eig: Eigendecomposition<c64> = a.eigendecomposition();
    let eig_val_raw = eig.s().column_vector();
    let eig_vec_raw = eig.u();

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
        zipped!(&mut c).for_each(|unzipped!(c)| *c /= max);
    });

    // Write mode shapes to output file
    let mut file = File::create(format!("{OUT_DIR}/sweep_modes.csv")).unwrap();
    izip!(eig_val.iter(), eig_vec.col_iter()).for_each(|(&lambda, c)| {
        file.write_fmt(format_args!("{}", lambda.sqrt() / (2. * PI)))
            .unwrap();
        for &v in c.iter() {
            file.write_fmt(format_args!(",{v}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });

    (eig_val, eig_vec)
}

fn setup_model(damping: Damping) -> Model {
    let beam_length = 10.;
    let xi = gauss_legendre_lobotto_points(6);
    let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    // Quadrature rule
    let gq = Quadrature::gauss(12);

    // Model
    let mut model = Model::new();
    let node_ids = s
        .iter()
        .map(|&si| {
            model
                .add_node()
                .element_location(si)
                .position(beam_length * si + 2., 0., 0., 1., 0., 0., 0.)
                .build()
        })
        .collect_vec();

    let m_star = mat![
        [8.538, 0.000, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 8.538, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 8.538, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 1.4433, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.40972, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.00000, 1.0336],
    ] * Scale(1e-2);

    let c_star = mat![
        [1368.17, 0., 0., 0., 0., 0.],
        [0., 88.56, 0., 0., 0., 0.],
        [0., 0., 38.78, 0., 0., 0.],
        [0., 0., 0., 16.960, 17.610, -0.351],
        [0., 0., 0., 17.610, 59.120, -0.370],
        [0., 0., 0., -0.351, -0.370, 141.47],
    ] * Scale(1e3);

    let sections = vec![
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
    ];

    //--------------------------------------------------------------------------
    // Add beam element
    //--------------------------------------------------------------------------

    model.add_beam_element(&node_ids, &gq, &sections, damping);

    //--------------------------------------------------------------------------
    // Add constraint element
    //--------------------------------------------------------------------------

    // Prescribed constraint to first node of beam
    model.add_prescribed_constraint(node_ids[0]);

    model
}

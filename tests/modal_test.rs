use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::{
    col,
    complex_native::c64,
    linalg::solvers::{Eigendecomposition, SpSolver},
    mat, unzipped, zipped, Col, Mat, Scale,
};

use itertools::{izip, Itertools};
use ottr::{
    elements::beams::{BeamSection, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    quadrature::Quadrature,
    util::ColAsMatMut,
};

#[test]
#[ignore]
fn test_damping() {
    // Damping ratio for modes 1-4
    let zeta = col![0.1];

    // Select damping type
    // let damping = Damping::None;
    let damping = Damping::ModalElement(zeta.clone());
    // let damping = Damping::Mu(col![0., 0., 0., 0., 0., 0.]);

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 1.; // Velocity scaling factor
    let t_end = 0.7; // Simulation length
    let time_step = 0.001; // Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 6; // Max convergence iterations
    let n_steps = (t_end / time_step) as usize;
    // let n_steps = 1;

    // Create output directory
    let out_dir = "output/modal";
    fs::create_dir_all(out_dir).unwrap();

    // Initialize model
    let mut model = setup_model(damping.clone());
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

    // Perform modal analysis
    let (eig_val, eig_vec) = modal_analysis(&out_dir, &model);
    let omega_n = eig_val[i_mode].sqrt();
    let f_n = omega_n / (2. * PI);

    println!(
        "mode={}, omega_n={omega_n}, fn={f_n}, zeta={}",
        i_mode, zeta[i_mode]
    );

    // Additional initialization for mu damping
    match damping {
        Damping::Mu(_) => {
            // Get index of maximum value
            let i_max = eig_vec
                .col(i_mode)
                .iter()
                .enumerate()
                .max_by(|(_, &a), (_, &b)| a.abs().total_cmp(&b.abs()))
                .map(|(index, _)| index)
                .unwrap()
                % 3;
            let mu = match i_max {
                0 => [2. * zeta[i_mode] / omega_n, 0., 0.],
                1 => [0., 2. * zeta[i_mode] / omega_n, 0.],
                2 => [0., 0., 2. * zeta[i_mode] / omega_n],
                _ => [0., 0., 0.],
            };
            model.beam_elements.iter_mut().for_each(|e| {
                e.damping = Damping::Mu(col![mu[0], mu[1], mu[2], mu[0], mu[2], mu[1]])
            });
        }
        _ => (),
    }

    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Apply scaled mode shape to state as velocity
    let v = eig_vec.col(i_mode) * Scale(v_scale);
    state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

    // Initialize output storage
    let ts = Col::<f64>::from_fn(n_steps, |i| (i as f64) * time_step);
    let mut u = Mat::<f64>::zeros(model.n_nodes() * 3, n_steps);

    // Loop through times and run simulation
    for (t, u_col) in izip!(ts.iter(), u.col_iter_mut()) {
        let u = u_col.as_mat_mut(3, model.n_nodes());
        zipped!(&mut u.subrows_mut(0, 3), state.u.subrows(0, 3))
            .for_each(|unzipped!(u, us)| *u = *us);

        // Take step and get convergence result
        let res = solver.step(&mut state);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, t={}, err={}", t, res.err);
        }

        assert_eq!(res.converged, true);

        // println!("g={:?}", solver.ct);
    }

    // Output results
    let mut file = File::create(format!("{out_dir}/displacement.csv")).unwrap();
    izip!(ts.iter(), u.col_iter()).for_each(|(&t, tv)| {
        file.write_fmt(format_args!("{t}")).unwrap();
        for &v in tv.iter() {
            file.write_fmt(format_args!(",{v}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });
}

fn modal_analysis(out_dir: &str, model: &Model) -> (Col<f64>, Mat<f64>) {
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
    let mut file = File::create(format!("{out_dir}/shapes.csv")).unwrap();
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
                .position(10. * si + 2., 0., 0., 1., 0., 0., 0.)
                .build()
        })
        .collect_vec();

    // Mass matrix 6x6
    let m_star = mat![
        [8.538, 0.000, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 8.538, 0.000, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 8.538, 0.0000, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 1.4433, 0.00000, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.40972, 0.0000],
        [0.000, 0.000, 0.000, 0.0000, 0.00000, 1.0336],
    ] * Scale(1e-2);

    // Stiffness matrix 6x6
    let c_star = mat![
        [1368.17, 0., 0., 0., 0., 0.],
        [0., 88.56, 0., 0., 0., 0.],
        [0., 0., 38.78, 0., 0., 0.],
        [0., 0., 0., 16.960, 0., 0.],
        [0., 0., 0., 0., 59.120, 0.],
        [0., 0., 0., 0., 0., 141.47],
        // [0., 0., 0., 16.960, 17.610, -0.351],
        // [0., 0., 0., 17.610, 59.120, -0.370],
        // [0., 0., 0., -0.351, -0.370, 141.47],
    ] * Scale(1e3);

    //--------------------------------------------------------------------------
    // Add beam element
    //--------------------------------------------------------------------------

    model.add_beam_element(
        &node_ids,
        &gq,
        &[
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
        ],
        damping,
    );

    //--------------------------------------------------------------------------
    // Add constraint element
    //--------------------------------------------------------------------------

    // Prescribed constraint to first node of beam
    model.add_prescribed_constraint(node_ids[0]);

    model
}

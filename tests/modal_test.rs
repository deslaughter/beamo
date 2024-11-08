use std::{f64::consts::PI, fs::File, io::Write, process};

use faer::{
    col, mat,
    solvers::{Eigendecomposition, SpSolver},
    unzipped, zipped, Col, Mat, Scale,
};

use itertools::{izip, Itertools};
use num_complex::Complex64;
use ottr::{
    beams::{BeamElement, BeamInput, BeamNode, BeamSection, Beams, Damping},
    interp::gauss_legendre_lobotto_points,
    node::{Node, NodeBuilder},
    quadrature::Quadrature,
    solver::{Solver, StepParameters},
    state::State,
};

fn setup_test() -> (Vec<Node>, Beams, State) {
    let xi = gauss_legendre_lobotto_points(6);
    let s = xi.iter().map(|v| (v + 1.) / 2.).collect_vec();

    println!("{:?}", s);

    // Quadrature rule
    let gq = Quadrature::gauss(12);

    // Node initial position
    let nodes = s
        .iter()
        .enumerate()
        .map(|(i, &si)| {
            NodeBuilder::new(i)
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
    // Create element
    //--------------------------------------------------------------------------

    let input = BeamInput {
        gravity: [0., 0., 0.],
        damping: Damping::None,
        // damping: Damping::Mu(faer::col![0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        elements: vec![BeamElement {
            nodes: izip!(s.iter(), nodes.iter())
                .map(|(&s, n)| BeamNode::new(s, n))
                .collect_vec(),
            quadrature: gq,
            sections: vec![
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
        }],
    };

    let beams = Beams::new(&input, &nodes);
    let state = State::new(&nodes);

    (nodes, beams, state)
}

#[test]
fn test_modal_frequency() {
    // Initialize system
    let (nodes, mut beams, mut state) = setup_test();

    // Perform modal analysis
    let (eig_val, eig_vec) = modal_analysis(&mut beams, &state, state.u.ncols() * 6);
    let mut file = File::create(format!("shapes.csv")).unwrap();
    izip!(eig_val.iter(), eig_vec.col_iter()).for_each(|(&lambda, c)| {
        file.write_fmt(format_args!("{}", lambda.sqrt() / (2. * PI)))
            .unwrap();
        for &v in c.iter() {
            file.write_fmt(format_args!(",{v}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });

    // Apply mode shape as displacement
    let i_mode = 0;
    let scale = 1.;
    let omega_n = eig_val[i_mode].sqrt();
    let f_n = omega_n / (2. * PI);

    // Apply 10% damping, pay attention to the order of the mu vector
    let zeta = 0.1;
    let mu_x = 0.;
    let mu_y = 0.;
    let mu_z = 2. * zeta / omega_n;
    beams.damping = Damping::Mu(col![mu_x, mu_y, mu_z, mu_x, mu_z, mu_y]);
    println!("fn={f_n}, zeta={zeta}, mu={mu_z}");

    // Apply scaled mode shape to state as a displacement
    let v = eig_vec.col(i_mode) * Scale(scale);
    state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

    let rho_inf = 1.0;
    let max_iter = 6;
    let time_step = 0.001;
    let n_steps = 600;
    let mut solver = Solver::new(
        StepParameters::new(time_step, rho_inf, max_iter),
        &nodes,
        &vec![],
    );

    let ts = Col::<f64>::from_fn(n_steps, |i| (i as f64) * time_step);
    let mut tv = Mat::<f64>::zeros(nodes.len() * 3, n_steps);

    for mut tv_col in tv.col_iter_mut() {
        tv_col.copy_from(Col::<f64>::from_fn(3 * nodes.len(), |i| {
            state.u[(i % 3, i / 3)]
        }));

        // Take step and get convergence result
        let res = solver.step(&mut state, &mut beams);

        // Exit if failed to converge
        if !res.converged {
            println!("failed, err={}", res.err);
            process::exit(1);
        }
    }

    let mut file = File::create(format!("mode.csv")).unwrap();
    izip!(ts.iter(), tv.col_iter()).for_each(|(&t, tv)| {
        file.write_fmt(format_args!("{t}")).unwrap();
        for &v in tv.iter() {
            file.write_fmt(format_args!(",{v}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });
}

fn modal_analysis(beams: &mut Beams, state: &State, n_dofs: usize) -> (Col<f64>, Mat<f64>) {
    // Calculate system based on initial state
    beams.calculate_system(&state);

    let mut m = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut c = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut k = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut r = Col::<f64>::zeros(n_dofs);

    // Get matrices
    beams.assemble_system(m.as_mut(), c.as_mut(), k.as_mut(), r.as_mut());

    let ndof_bc = n_dofs - 6;
    let lu = m.submatrix(6, 6, ndof_bc, ndof_bc).partial_piv_lu();
    let a = lu.solve(k.submatrix(6, 6, ndof_bc, ndof_bc));

    let eig: Eigendecomposition<Complex64> = a.eigendecomposition();
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

    let eig_val = Col::<f64>::from_fn(eig_val_raw.nrows(), |i| *eig_val_raw.get(eig_order[i]).re);
    let mut eig_vec = Mat::<f64>::from_fn(n_dofs, eig_vec_raw.ncols(), |i, j| {
        if i < 6 {
            0.
        } else {
            *eig_vec_raw.get(i - 6, eig_order[j]).re
        }
    });
    // normalize eigen vectors
    eig_vec.as_mut().col_iter_mut().for_each(|mut c| {
        let max = *c
            .as_ref()
            .iter()
            .reduce(|acc, e| if e.abs() > acc.abs() { e } else { acc })
            .unwrap();
        zipped!(&mut c).for_each(|unzipped!(mut c)| *c /= max);
    });

    (eig_val, eig_vec)
}

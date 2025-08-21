use faer::{linalg::matmul::matmul, prelude::*, Accum};
use itertools::{izip, Itertools};
use ottr::{
    elements::beams::{BeamSection, Beams, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    node::NodeFreedomMap,
    quadrature::Quadrature,
    state::State,
    util::{
        quat_as_matrix, quat_compose, quat_from_axis_angle, quat_from_rotation_vector,
        quat_rotate_vector,
    },
};
use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

fn dump_matrix(file_name: &str, mat: MatRef<f64>) {
    let mut file = File::create(file_name).unwrap();
    mat.row_iter().for_each(|r| {
        for (j, &v) in r.iter().enumerate() {
            if j > 0 {
                file.write(b",").unwrap();
            }
            file.write_fmt(format_args!("{v:e}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });
}

const OUT_DIR: &str = "output/iea10_modal_analysis";

fn main() {
    // Create output directory
    fs::create_dir_all(OUT_DIR).unwrap();

    // Initialize system
    let model = setup_test();

    let nfm = model.create_node_freedom_map();
    let mut elements = model.create_elements(&nfm);
    let mut state = model.create_state();

    // Perform modal analysis
    let (eig_val, eig_vec) = modal_analysis(&nfm, &mut elements.beams, &state, state.u.ncols() * 6);
    let mut file = File::create(format!("{OUT_DIR}/shapes.csv")).unwrap();
    izip!(eig_val.iter(), eig_vec.col_iter()).for_each(|(&lambda, c)| {
        file.write_fmt(format_args!("{}", lambda.sqrt() / (2. * PI)))
            .unwrap();
        for &v in c.iter() {
            file.write_fmt(format_args!(",{v}")).unwrap();
        }
        file.write(b"\n").unwrap();
    });

    // Apply scaled mode shape to state as initial velocity
    let i_mode = 1;
    let omega_n = eig_val[i_mode].sqrt();
    let f_n = omega_n / (2. * PI);
    let period = 1. / f_n;
    println!("fn={f_n}");

    let a_tip = 1.;
    let dt = 0.01;
    let n = (period / dt) as usize + 1;
    let t = Col::<f64>::from_fn(n, |i| (i as f64) * dt);
    let mut u = Mat::<f64>::zeros(eig_vec.nrows(), n);
    let mut v = Mat::<f64>::zeros(eig_vec.nrows(), n);
    let mut vd = Mat::<f64>::zeros(eig_vec.nrows(), n);
    let mut energy = Col::<f64>::zeros(n);
    println!("{period} {}", dt * (n as f64));
    izip!(
        t.iter(),
        u.col_iter_mut(),
        v.col_iter_mut(),
        vd.col_iter_mut()
    )
    .for_each(|(t, mut u, mut v, mut vd)| {
        u.copy_from(eig_vec.col(i_mode) * Scale(a_tip * omega_n.powi(0) * (omega_n * t).sin()));
        v.copy_from(eig_vec.col(i_mode) * Scale(a_tip * omega_n.powi(1) * (omega_n * t).cos()));
        vd.copy_from(eig_vec.col(i_mode) * Scale(-a_tip * omega_n.powi(2) * (omega_n * t).sin()));
    });

    let mut m = Mat::<f64>::zeros(nfm.n_system_dofs, nfm.n_system_dofs);
    let mut c = Mat::<f64>::zeros(nfm.n_system_dofs, nfm.n_system_dofs);
    let mut k = Mat::<f64>::zeros(nfm.n_system_dofs, nfm.n_system_dofs);
    let mut r = Col::<f64>::zeros(nfm.n_system_dofs);

    //Only matters for viscoelastic material, but needs to be passed to create_beams
    let h = 0.001;

    elements.beams.calculate_system(&state, h);
    elements.beams.assemble_system(r.as_mut());

    dump_matrix(&format!("{OUT_DIR}/m.csv"), m.as_ref());
    dump_matrix(&format!("{OUT_DIR}/c.csv"), c.as_ref());
    dump_matrix(&format!("{OUT_DIR}/k.csv"), k.as_ref());
    dump_matrix(&format!("{OUT_DIR}/u.csv"), u.transpose());
    dump_matrix(&format!("{OUT_DIR}/v.csv"), v.transpose());
    dump_matrix(&format!("{OUT_DIR}/vd.csv"), vd.transpose());

    let mut force = Col::<f64>::zeros(nfm.n_system_dofs);

    (0..n).for_each(|i| {
        let mut q = col![0., 0., 0., 0.];
        state
            .u
            .col_iter_mut()
            .zip(u.col_iter())
            .for_each(|(mut us, u)| {
                quat_from_rotation_vector(u.subrows(3, 3), q.as_mut());
                us[0] = u[0];
                us[1] = u[1];
                us[2] = u[2];
                us[3] = q[0];
                us[4] = q[1];
                us[5] = q[2];
                us[6] = q[3];
            });
        state
            .v
            .col_iter_mut()
            .enumerate()
            .for_each(|(j, mut c)| c.copy_from(v.col(i).subrows(j * 6, 6)));
        state
            .vd
            .col_iter_mut()
            .enumerate()
            .for_each(|(j, mut c)| c.copy_from(vd.col(i).subrows(j * 6, 6)));

        m.fill(0.);
        c.fill(0.);
        k.fill(0.);
        r.fill(0.);

        //Only matters for viscoelastic material, but needs to be passed to create_beams
        let h = 0.001;

        elements.beams.calculate_system(&state, h);
        elements.beams.assemble_system(r.as_mut());

        // Calculate energy dissipation
        matmul(
            force.as_mut(),
            Accum::Replace,
            c.as_ref(),
            v.col(i),
            1.,
            Par::Seq,
        );
        energy[i] = &force.transpose() * &v.col(i);
    });

    let total_energy = (1..energy.nrows())
        .map(|i| dt * (energy[i - 1] + energy[i]) / 2.)
        .sum::<f64>()
        / (dt * (n as f64));

    println!("energy = {total_energy}");
}

// fn modal_analysis_complex(beams: &mut Beams, state: &State, n_dofs: usize) -> (Col<f64>, Mat<f64>) {
//     // Calculate system based on initial state
//     beams.calculate_system(&state);

//     let mut m = Mat::<f64>::zeros(n_dofs, n_dofs);
//     let mut c = Mat::<f64>::zeros(n_dofs, n_dofs);
//     let mut k = Mat::<f64>::zeros(n_dofs, n_dofs);
//     let mut r = Col::<f64>::zeros(n_dofs);

//     // Get matrices
//     beams.assemble_system(m.as_mut(), c.as_mut(), k.as_mut(), r.as_mut());

//     let n_reduced = n_dofs - 6;

//     let m_lu = m.submatrix(6, 6, n_reduced, n_reduced).full_piv_lu();
//     let m_inv_k = m_lu.solve(k.submatrix(6, 6, n_reduced, n_reduced));
//     let m_inv_c = m_lu.solve(c.submatrix(6, 6, n_reduced, n_reduced));

//     let mut a = Mat::<f64>::zeros(2 * n_reduced, 2 * n_reduced);
//     a.submatrix_mut(0, n_reduced, n_reduced, n_reduced)
//         .copy_from(&Mat::<f64>::identity(n_reduced, n_reduced));
//     a.submatrix_mut(n_reduced, 0, n_reduced, n_reduced)
//         .copy_from(&m_inv_k);
//     a.submatrix_mut(n_reduced, n_reduced, n_reduced, n_reduced)
//         .copy_from(&m_inv_c);

//     let eig: Eigendecomposition<c64> = a.eigendecomposition();
//     let eig_val_raw = eig.s().column_vector().to_owned();
//     let eig_vec_raw = eig.u();

//     println!("{:?}", eig_val_raw);

//     let eig_val = eig_val_raw.iter().map(|c| c.norm()).collect_vec();

//     let mut eig_order: Vec<_> = (0..eig_val_raw.nrows()).collect();
//     eig_order.sort_by(|&i, &j| eig_val[i].partial_cmp(&eig_val[j]).unwrap());
//     // let eig_order = eig_order
//     //     .iter()
//     //     .filter_map(|&i| if eig_val_raw[i].re >= 0. {Some})
//     //     .collect_vec();

//     let eig_val = Col::<f64>::from_fn(eig_order.nrows(), |i| *eig_val_raw.get(eig_order[i]).re);
//     let mut eig_vec = Mat::<f64>::from_fn(n_dofs, eig_vec_raw.ncols(), |i, j| {
//         if i < 6 {
//             0.
//         } else {
//             *eig_vec_raw.get(i - 6, eig_order[j]).re
//         }
//     });
//     // normalize eigen vectors
//     eig_vec.as_mut().col_iter_mut().for_each(|mut c| {
//         let max = *c
//             .as_ref()
//             .iter()
//             .reduce(|acc, e| if e.abs() > acc.abs() { e } else { acc })
//             .unwrap();
//         zip!(&mut c).for_each(|unzip!(c)| *c /= max);
//     });

//     (eig_val, eig_vec)
// }

fn modal_analysis(
    nfm: &NodeFreedomMap,
    beams: &mut Beams,
    state: &State,
    n_dofs: usize,
) -> (Col<f64>, Mat<f64>) {
    //Only matters for viscoelastic material, but needs to be passed to create_beams
    let h = 0.001;

    // Calculate system based on initial state
    beams.calculate_system(&state, h);

    let mut m = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut c = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut k = Mat::<f64>::zeros(n_dofs, n_dofs);
    let mut r = Col::<f64>::zeros(n_dofs);

    // Get matrices
    beams.assemble_system(r.as_mut());

    let ndof_bc = n_dofs - 6;
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
    let mut eig_vec = Mat::<f64>::from_fn(n_dofs, eig_vec_raw.ncols(), |i, j| {
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
        c /= max;
    });

    (eig_val, eig_vec)
}

fn setup_test() -> Model {
    let node_position_raw = mat![
        [
            0.,
            0.,
            0.,
            0.017725637555812964,
            0.0092508320270342295,
            -0.2534694562770049
        ],
        [
            0.014818947853545678,
            0.021400034496237862,
            3.1745311973723513,
            -0.028324841537410668,
            -0.00081289683581318756,
            -0.24760592184866853
        ],
        [
            0.011624629077729556,
            0.39724464534400378,
            10.366344916802753,
            -0.062633763571766157,
            -0.0095394185653552531,
            -0.21093999008791323
        ],
        [
            -0.04469242526421606,
            0.91918504842943005,
            20.912180771482543,
            -0.026305258076704537,
            -0.0095340956981474247,
            -0.14071229848232455
        ],
        [
            -0.1881001198737024,
            0.86255623666034409,
            33.874033678268233,
            0.03047622559764512,
            -0.013861447998489334,
            -0.074983893212325015
        ],
        [
            -0.47979587513362798,
            0.31957410487560789,
            48.100000000000009,
            0.036724727761501756,
            -0.02608333694140139,
            -0.033442471879058175
        ],
        [
            -0.98677048072598006,
            0.00015934334154722832,
            62.325966321731798,
            0.0076764107408473304,
            -0.047325420916998026,
            0.0046361679714946339
        ],
        [
            -1.8657141499489684,
            -0.0066222613370174521,
            75.287819228517421,
            -0.0039658579199553894,
            -0.096065138293530875,
            0.045431523897588699
        ],
        [
            -3.3246070644460817,
            -0.024488075667257969,
            85.833655083197229,
            -0.00071711447911382428,
            -0.19034902259837524,
            0.058266025105385311
        ],
        [
            -5.1044525454549117,
            -0.038186034626468766,
            93.025468802627671,
            -0.010028603215331944,
            -0.30278117233637347,
            0.028512694405219231
        ],
        [
            -6.2062205400000003,
            0.0051506032000000002,
            96.200000000000003,
            -0.021634402366878018,
            -0.36819680783353054,
            -0.0033590571482629747
        ],
    ];

    let xi = gauss_legendre_lobotto_points(node_position_raw.nrows() - 1);
    let s = xi.iter().map(|&xi| (xi + 1.) / 2.).collect_vec();

    let mut r = Col::<f64>::zeros(4);
    quat_from_axis_angle(PI / 2., col![0., 1., 0.].as_ref(), r.as_mut());
    let mut ru = Col::<f64>::zeros(4);
    quat_from_axis_angle(-PI / 2., col![0., 1., 0.].as_ref(), ru.as_mut());

    let mut model = Model::new();
    let node_ids = node_position_raw
        .row_iter()
        .zip(s)
        .map(|(p, s)| {
            // Rotate positions and wm parameters
            let mut pr = col![0., 0., 0.];
            quat_rotate_vector(r.as_ref(), col![p[0], p[1], p[2]].as_ref(), pr.as_mut());
            let mut wm = col![0., 0., 0.];
            quat_rotate_vector(r.as_ref(), col![p[3], p[4], p[5]].as_ref(), wm.as_mut());
            // Convert WM rotation to quaternion
            let c0 = 2. - (wm[0] * wm[0] + wm[1] * wm[1] + wm[2] * wm[2]) / 8.;
            let q = col![c0, wm[0], wm[1], wm[2]] * Scale(1. / (4. - c0));
            let mut rt = col![0., 0., 0.];
            let mut rq = col![0., 0., 0., 0.];
            quat_rotate_vector(ru.as_ref(), pr.as_ref(), rt.as_mut());
            quat_compose(q.as_ref(), ru.as_ref(), rq.as_mut());
            model
                .add_node()
                .element_location(s)
                .position(rt[0], rt[1], rt[2], rq[0], rq[1], rq[2], rq[3])
                .build()
        })
        .collect_vec();

    let quadrature = Quadrature {
        points: vec![
            -1., -0.989934, -0.979868, -0.959744, -0.93962, -0.91951, -0.8994, -0.849163,
            -0.798926, -0.748732, -0.698538, -0.648308, -0.598078, -0.547777, -0.497476, -0.422025,
            -0.346574, -0.271155, -0.195736, -0.120325, -0.044914, 0.030503, 0.10592, 0.181326,
            0.256732, 0.332085, 0.407438, 0.482659, 0.55788, 0.632802, 0.707724, 0.781913,
            0.856102, 0.909299, 0.962496, 0.981248, 1.,
        ],
        weights: vec![
            0.005033, 0.010066, 0.015095, 0.020124, 0.020117, 0.02011, 0.0351735, 0.050237,
            0.0502155, 0.0501940, 0.050212, 0.05023, 0.0502655, 0.050301, 0.062876, 0.075451,
            0.075435, 0.075419, 0.075415, 0.075411, 0.075414, 0.075417, 0.0754115, 0.075406,
            0.0753795, 0.075353, 0.075287, 0.075221, 0.0750715, 0.074922, 0.0745555, 0.074189,
            0.063693, 0.053197, 0.0359745, 0.018752, 0.009376,
        ],
    };

    // Section information
    let mut sections = vec![
        BeamSection {
            s: 0.000000,
            c_star: mat![
                [3.593168E+09, -2.727900E+02, 0., 0., 0., -4.905916E+07],
                [-2.727900E+02, 3.581695E+09, 0., 0., 0., -2.965258E+03],
                [0., 0., 4.673179E+10, 6.583941E+08, 2.686685E+04, 0.],
                [0., 0., 6.583941E+08, 1.200300E+11, -3.701091E+04, 0.],
                [0., 0., 2.686685E+04, -3.701091E+04, 1.184482E+11, 0.],
                [-4.905916E+07, -2.965258E+03, 0., 0., 0., 3.658683E+10],
            ],
            m_star: mat![
                [2.333405E+03, 0., 0., 0., 0., -3.051225E+01],
                [0., 2.333405E+03, 0., 0., 0., -1.345602E-03],
                [0., 0., 2.333405E+03, 3.051225E+01, 1.345602E-03, 0.],
                [0., 0., 3.051225E+01, 5.990830E+03, -1.722840E-03, 0.],
                [0., 0., 1.345602E-03, -1.722840E-03, 5.917230E+03, 0.],
                [-3.051225E+01, -1.345602E-03, 0., 0., 0., 1.190806E+04],
            ],
        },
        BeamSection {
            s: 0.010066,
            c_star: mat![
                [3.552259E+09, -2.897388E+04, 0., 0., 0., -4.259331E+07],
                [-2.897388E+04, 3.551248E+09, 0., 0., 0., 8.975824E+05],
                [0., 0., 4.633311E+10, 6.527299E+08, -4.163426E+06, 0.],
                [0., 0., 6.527299E+08, 1.188062E+11, -5.263855E+07, 0.],
                [0., 0., -4.163426E+06, -5.263855E+07, 1.169630E+11, 0.],
                [-4.259331E+07, 8.975824E+05, 0., 0., 0., 3.611730E+10],
            ],
            m_star: mat![
                [2.312620E+03, 0., 0., 0., 0., -2.884787E+01],
                [0., 2.312620E+03, 0., 0., 0., 4.448364E-01],
                [0., 0., 2.312620E+03, 2.884787E+01, -4.448364E-01, 0.],
                [0., 0., 2.884787E+01, 5.928401E+03, -2.538168E+00, 0.],
                [0., 0., -4.448364E-01, -2.538168E+00, 5.839705E+03, 0.],
                [-2.884787E+01, 4.448364E-01, 0., 0., 0., 1.1768106E+04],
            ],
        },
        BeamSection {
            s: 0.030190,
            c_star: mat![
                [2.114404E+09, 2.089309E+07, 0., 0., 0., -2.359093E+08],
                [2.089309E+07, 1.964826E+09, 0., 0., 0., 2.082296E+07],
                [0., 0., 2.328865E+10, 2.169527E+09, -4.107831E+07, 0.],
                [0., 0., 2.169527E+09, 4.312038E+10, -4.079381E+09, 0.],
                [0., 0., -4.107831E+07, -4.079381E+09, 7.209608E+10, 0.],
                [-2.359093E+08, 2.082296E+07, 0., 0., 0., 1.894314E+10],
            ],
            m_star: mat![
                [1.330201E+03, 0., 0., 0., 0., -8.821310E+01],
                [0., 1.330201E+03, 0., 0., 0., 2.784873E+00],
                [0., 0., 1.330201E+03, 8.821310E+01, -2.784873E+00, 0.],
                [0., 0., 8.821310E+01, 2.495230E+03, -1.959879E+02, 0.],
                [0., 0., -2.784873E+00, -1.959879E+02, 3.891182E+03, 0.],
                [-8.821310E+01, 2.784873E+00, 0., 0., 0., 6.386412E+03],
            ],
        },
        BeamSection {
            s: 0.050300,
            c_star: mat![
                [2.055530E+09, 5.472490E+06, 0., 0., 0., -1.061586E+08],
                [5.472490E+06, 2.023017E+09, 0., 0., 0., 4.768980E+07],
                [0., 0., 2.200028E+10, 1.442828E+09, -1.809307E+08, 0.],
                [0., 0., 1.442828E+09, 4.080950E+10, -3.413067E+09, 0.],
                [0., 0., -1.809307E+08, -3.413067E+09, 6.092364E+10, 0.],
                [-1.061586E+08, 4.768980E+07, 0., 0., 0., 1.705358E+10],
            ],
            m_star: mat![
                [1.301072E+03, 0., 0., 0., 0., -3.994667E+01],
                [0., 1.301072E+03, 0., 0., 0., 1.110306E+01],
                [0., 0., 1.301072E+03, 3.994667E+01, -1.110306E+01, 0.],
                [0., 0., 3.994667E+01, 2.370558E+03, -1.672315E+02, 0.],
                [0., 0., -1.110306E+01, -1.672315E+02, 3.360960E+03, 0.],
                [-3.994667E+01, 1.110306E+01, 0., 0., 0., 5.731518E+03],
            ],
        },
        BeamSection {
            s: 0.100537,
            c_star: mat![
                [1.294164E+09, 2.407344E+08, 0., 0., 0., 4.300925E+08],
                [2.407344E+08, 1.651678E+09, 0., 0., 0., 1.988240E+08],
                [0., 0., 1.779158E+10, -2.137477E+09, -6.402857E+08, 0.],
                [0., 0., -2.137477E+09, 3.239020E+10, -2.071066E+09, 0.],
                [0., 0., -6.402857E+08, -2.071066E+09, 2.896647E+10, 0.],
                [4.300925E+08, 1.988240E+08, 0., 0., 0., 8.498183E+09],
            ],
            m_star: mat![
                [1.049375E+03, 0., 0., 0., 0., 1.402242E+02],
                [0., 1.049375E+03, 0., 0., 0., 3.886787E+01],
                [0., 0., 1.049375E+03, -1.402242E+02, -3.886787E+01, 0.],
                [0., 0., -1.402242E+02, 1.874229E+03, -1.517100E+02, 0.],
                [0., 0., -3.886787E+01, -1.517100E+02, 1.623914E+03, 0.],
                [1.402242E+02, 3.886787E+01, 0., 0., 0., 3.498143E+03],
            ],
        },
        BeamSection {
            s: 0.150731,
            c_star: mat![
                [6.720711E+08, 5.462951E+07, 0., 0., 0., 5.844542E+08],
                [5.462951E+07, 1.369341E+09, 0., 0., 0., 1.862520E+08],
                [0., 0., 1.543672E+10, -4.957117E+09, -7.182875E+08, 0.],
                [0., 0., -4.957117E+09, 3.275607E+10, -1.072217E+09, 0.],
                [0., 0., -7.182875E+08, -1.072217E+09, 1.456821E+10, 0.],
                [5.844542E+08, 1.862520E+08, 0., 0., 0., 4.089474E+09],
            ],
            m_star: mat![
                [8.965193E+02, 0., 0., 0., 0., 2.794410E+02],
                [0., 8.965193E+02, 0., 0., 0., 4.488537E+01],
                [0., 0., 8.965193E+02, -2.794410E+02, -4.488537E+01, 0.],
                [0., 0., -2.794410E+02, 1.949947E+03, -7.235794E+01, 0.],
                [0., 0., -4.488537E+01, -7.235794E+01, 7.629744E+02, 0.],
                [2.794410E+02, 4.488537E+01, 0., 0., 0., 2.712921E+03],
            ],
        },
        BeamSection {
            s: 0.200961,
            c_star: mat![
                [5.167014E+08, 1.160958E+07, 0., 0., 0., 5.558835E+08],
                [1.160958E+07, 9.478617E+08, 0., 0., 0., 8.900176E+07],
                [0., 0., 1.297089E+10, -6.087417E+09, -5.097146E+08, 0.],
                [0., 0., -6.087417E+09, 3.276638E+10, -3.155725E+08, 0.],
                [0., 0., -5.097146E+08, -3.155725E+08, 9.325592E+09, 0.],
                [5.558835E+08, 8.900176E+07, 0., 0., 0., 2.696128E+09],
            ],
            m_star: mat![
                [7.733115E+02, 0., 0., 0., 0., 3.314804E+02],
                [0., 7.733115E+02, 0., 0., 0., 3.349157E+01],
                [0., 0., 7.733115E+02, -3.314804E+02, -3.349157E+01, 0.],
                [0., 0., -3.314804E+02, 1.957488E+03, -2.133386E+01, 0.],
                [0., 0., -3.349157E+01, -2.133386E+01, 4.913814E+02, 0.],
                [3.314804E+02, 3.349157E+01, 0., 0., 0., 2.448869E+03],
            ],
        },
        BeamSection {
            s: 0.251262,
            c_star: mat![
                [4.554783E+08, 1.289969E+06, 0., 0., 0., 5.316506E+08],
                [1.289969E+06, 5.505680E+08, 0., 0., 0., 4.060399E+07],
                [0., 0., 1.067550E+10, -6.005907E+09, -2.394961E+08, 0.],
                [0., 0., -6.005907E+09, 2.740248E+10, -9.754347E+07, 0.],
                [0., 0., -2.394961E+08, -9.754347E+07, 6.906459E+09, 0.],
                [5.316506E+08, 4.060399E+07, 0., 0., 0., 1.857584E+09],
            ],
            m_star: mat![
                [6.495623E+02, 0., 0., 0., 0., 3.308640E+02],
                [0., 6.495623E+02, 0., 0., 0., 1.819585E+01],
                [0., 0., 6.495623E+02, -3.308640E+02, -1.819585E+01, 0.],
                [0., 0., -3.308640E+02, 1.636901E+03, -5.697434E+00, 0.],
                [0., 0., -1.819585E+01, -5.697434E+00, 3.656839E+02, 0.],
                [3.308640E+02, 1.819585E+01, 0., 0., 0., 2.002585E+03],
            ],
        },
        BeamSection {
            s: 0.326713,
            c_star: mat![
                [4.237675E+08, -1.742459E+05, 0., 0., 0., 4.743670E+08],
                [-1.742459E+05, 3.042461E+08, 0., 0., 0., 1.995675E+07],
                [0., 0., 8.863854E+09, -5.268581E+09, -1.388932E+08, 0.],
                [0., 0., -5.268581E+09, 1.915149E+10, 6.640658E+07, 0.],
                [0., 0., -1.388932E+08, 6.640658E+07, 4.944189E+09, 0.],
                [4.743670E+08, 1.995675E+07, 0., 0., 0., 1.233042E+09],
            ],
            m_star: mat![
                [5.414273E+02, 0., 0., 0., 0., 2.912441E+02],
                [0., 5.414273E+02, 0., 0., 0., 1.177993E+01],
                [0., 0., 5.414273E+02, -2.912441E+02, -1.177993E+01, 0.],
                [0., 0., -2.912441E+02, 1.133652E+03, 5.292839E+00, 0.],
                [0., 0., -1.177993E+01, 5.292839E+00, 2.612561E+02, 0.],
                [2.912441E+02, 1.177993E+01, 0., 0., 0., 1.394908E+03],
            ],
        },
        BeamSection {
            s: 0.402132,
            c_star: mat![
                [3.762319E+08, 4.689784E+05, 0., 0., 0., 3.545832E+08],
                [4.689784E+05, 2.823743E+08, 0., 0., 0., 1.583895E+07],
                [0., 0., 8.001166E+09, -4.240327E+09, -1.001758E+08, 0.],
                [0., 0., -4.240327E+09, 1.284839E+10, 8.906198E+07, 0.],
                [0., 0., -1.001758E+08, 8.906198E+07, 3.403184E+09, 0.],
                [3.545832E+08, 1.583895E+07, 0., 0., 0., 8.319830E+08],
            ],
            m_star: mat![
                [4.756747E+02, 0., 0., 0., 0., 2.285770E+02],
                [0., 4.756747E+02, 0., 0., 0., 8.451265E+00],
                [0., 0., 4.756747E+02, -2.285770E+02, -8.451265E+00, 0.],
                [0., 0., -2.285770E+02, 7.437815E+02, 6.343820E+00, 0.],
                [0., 0., -8.451265E+00, 6.343820E+00, 1.772484E+02, 0.],
                [2.285770E+02, 8.451265E+00, 0., 0., 0., 9.210299E+02],
            ],
        },
        BeamSection {
            s: 0.477543,
            c_star: mat![
                [2.966093E+08, 4.686077E+04, 0., 0., 0., 2.216594E+08],
                [4.686077E+04, 2.903540E+08, 0., 0., 0., 1.131860E+07],
                [0., 0., 7.162242E+09, -3.161722E+09, -5.107110E+07, 0.],
                [0., 0., -3.161722E+09, 7.808167E+09, 5.561212E+07, 0.],
                [0., 0., -5.107110E+07, 5.561212E+07, 1.998753E+09, 0.],
                [2.216594E+08, 1.131860E+07, 0., 0., 0., 4.787575E+08],
            ],
            m_star: mat![
                [4.101871E+02, 0., 0., 0., 0., 1.661885E+02],
                [0., 4.101871E+02, 0., 0., 0., 4.618106E+00],
                [0., 0., 4.101871E+02, -1.661885E+02, -4.618106E+00, 0.],
                [0., 0., -1.661885E+02, 4.413304E+02, 3.908546E+00, 0.],
                [0., 0., -4.618106E+00, 3.908546E+00, 1.020685E+02, 0.],
                [1.661885E+02, 4.618106E+00, 0., 0., 0., 5.433990E+02],
            ],
        },
        BeamSection {
            s: 0.552960,
            c_star: mat![
                [2.173494E+08, -9.573195E+05, 0., 0., 0., 1.287152E+08],
                [-9.573195E+05, 3.124895E+08, 0., 0., 0., 9.016861E+06],
                [0., 0., 6.287240E+09, -2.300603E+09, -5.315862E+07, 0.],
                [0., 0., -2.300603E+09, 4.504791E+09, 4.589388E+07, 0.],
                [0., 0., -5.315862E+07, 4.589388E+07, 1.035534E+09, 0.],
                [1.287152E+08, 9.016861E+06, 0., 0., 0., 2.542686E+08],
            ],
            m_star: mat![
                [3.459320E+02, 0., 0., 0., 0., 1.190096E+02],
                [0., 3.459320E+02, 0., 0., 0., 3.905333E+00],
                [0., 0., 3.459320E+02, -1.190096E+02, -3.905333E+00, 0.],
                [0., 0., -1.190096E+02, 2.468134E+02, 2.893793E+00, 0.],
                [0., 0., -3.905333E+00, 2.893793E+00, 5.184792E+01, 0.],
                [1.190096E+02, 3.905333E+00, 0., 0., 0., 2.986613E+02],
            ],
        },
        BeamSection {
            s: 0.628366,
            c_star: mat![
                [1.538347E+08, -2.260034E+06, 0., 0., 0., 7.262009E+07],
                [-2.260034E+06, 3.341170E+08, 0., 0., 0., 8.867805E+06],
                [0., 0., 5.424404E+09, -1.662074E+09, -7.638210E+07, 0.],
                [0., 0., -1.662074E+09, 2.539909E+09, 4.260924E+07, 0.],
                [0., 0., -7.638210E+07, 4.260924E+07, 4.997168E+08, 0.],
                [7.262009E+07, 8.867805E+06, 0., 0., 0., 1.296271E+08],
            ],
            m_star: mat![
                [2.884928E+02, 0., 0., 0., 0., 8.488493E+01],
                [0., 2.884928E+02, 0., 0., 0., 4.571619E+00],
                [0., 0., 2.884928E+02, -8.488493E+01, -4.571619E+00, 0.],
                [0., 0., -8.488493E+01, 1.359517E+02, 2.428770E+00, 0.],
                [0., 0., -4.571619E+00, 2.428770E+00, 2.460657E+01, 0.],
                [8.488493E+01, 4.571619E+00, 0., 0., 0., 1.605582E+02],
            ],
        },
        BeamSection {
            s: 0.703719,
            c_star: mat![
                [1.180478E+08, -3.224140E+06, 0., 0., 0., 4.150381E+07],
                [-3.224140E+06, 3.291902E+08, 0., 0., 0., 8.910988E+06],
                [0., 0., 4.499312E+09, -1.165206E+09, -8.358631E+07, 0.],
                [0., 0., -1.165206E+09, 1.396322E+09, 3.474065E+07, 0.],
                [0., 0., -8.358631E+07, 3.474065E+07, 2.386197E+08, 0.],
                [4.150381E+07, 8.910988E+06, 0., 0., 0., 6.568776E+07],
            ],
            m_star: mat![
                [2.336307E+02, 0., 0., 0., 0., 5.921963E+01],
                [0., 2.336307E+02, 0., 0., 0., 4.577084E+00],
                [0., 0., 2.336307E+02, -5.921963E+01, -4.577084E+00, 0.],
                [0., 0., -5.921963E+01, 7.368700E+01, 1.880106E+00, 0.],
                [0., 0., -4.577084E+00, 1.880106E+00, 1.161930E+01, 0.],
                [5.921963E+01, 4.577084E+00, 0., 0., 0., 8.530631E+01],
            ],
        },
        BeamSection {
            s: 0.778940,
            c_star: mat![
                [7.660383E+07, -3.238791E+06, 0., 0., 0., 2.750210E+07],
                [-3.238791E+06, 2.949071E+08, 0., 0., 0., 7.265498E+06],
                [0., 0., 3.476552E+09, -7.741769E+08, -6.663891E+07, 0.],
                [0., 0., -7.741769E+08, 7.502292E+08, 2.173123E+07, 0.],
                [0., 0., -6.663891E+07, 2.173123E+07, 1.145876E+08, 0.],
                [2.750210E+07, 7.265498E+06, 0., 0., 0., 3.638148E+07],
            ],
            m_star: mat![
                [1.788020E+02, 0., 0., 0., 0., 3.943396E+01],
                [0., 1.788020E+02, 0., 0., 0., 3.537851E+00],
                [0., 0., 1.788020E+02, -3.943396E+01, -3.537851E+00, 0.],
                [0., 0., -3.943396E+01, 3.966949E+01, 1.158440E+00, 0.],
                [0., 0., -3.537851E+00, 1.158440E+00, 5.551981E+00, 0.],
                [3.943396E+01, 3.537851E+00, 0., 0., 0., 4.522147E+01],
            ],
        },
        BeamSection {
            s: 0.853862,
            c_star: mat![
                [5.719970E+07, -2.044932E+06, 0., 0., 0., 1.856795E+07],
                [-2.044932E+06, 2.393425E+08, 0., 0., 0., 5.220612E+06],
                [0., 0., 2.446578E+09, -4.675623E+08, -4.160578E+07, 0.],
                [0., 0., -4.675623E+08, 3.759393E+08, 1.056054E+07, 0.],
                [0., 0., -4.160578E+07, 1.056054E+07, 5.487856E+07, 0.],
                [1.856795E+07, 5.220612E+06, 0., 0., 0., 2.089154E+07],
            ],
            m_star: mat![
                [1.260219E+02, 0., 0., 0., 0., 2.401343E+01],
                [0., 1.260219E+02, 0., 0., 0., 2.196570E+00],
                [0., 0., 1.260219E+02, -2.401343E+01, -2.196570E+00, 0.],
                [0., 0., -2.401343E+01, 2.028161E+01, 5.655258E-01, 0.],
                [0., 0., -2.196570E+00, 5.655258E-01, 2.653477E+00, 0.],
                [2.401343E+01, 2.196570E+00, 0., 0., 0., 2.293508E+01],
            ],
        },
        BeamSection {
            s: 0.928051,
            c_star: mat![
                [3.548064E+07, -5.377792E+05, 0., 0., 0., 9.724373E+06],
                [-5.377792E+05, 1.624377E+08, 0., 0., 0., 3.608388E+06],
                [0., 0., 1.395799E+09, -2.040705E+08, -2.347224E+07, 0.],
                [0., 0., -2.040705E+08, 1.452700E+08, 3.839627E+06, 0.],
                [0., 0., -2.347224E+07, 3.839627E+06, 1.953123E+07, 0.],
                [9.724373E+06, 3.608388E+06, 0., 0., 0., 9.311543E+06],
            ],
            m_star: mat![
                [7.361383E+01, 0., 0., 0., 0., 1.068846E+01],
                [0., 7.361383E+01, 0., 0., 0., 1.250748E+00],
                [0., 0., 7.361383E+01, -1.068846E+01, -1.250748E+00, 0.],
                [0., 0., -1.068846E+01, 8.341949E+00, 2.064802E-01, 0.],
                [0., 0., -1.250748E+00, 2.064802E-01, 9.385748E-01, 0.],
                [1.068846E+01, 1.250748E+00, 0., 0., 0., 9.280523E+00],
            ],
        },
        BeamSection {
            s: 0.981248,
            c_star: mat![
                [1.247964E+07, 4.390599E+05, 0., 0., 0., 2.316181E+06],
                [4.390599E+05, 7.323157E+07, 0., 0., 0., 1.295006E+06],
                [0., 0., 5.772860E+08, -4.053886E+07, -7.682647E+06, 0.],
                [0., 0., -4.053886E+07, 3.361686E+07, 3.419533E+05, 0.],
                [0., 0., -7.682647E+06, 3.419533E+05, 3.538161E+06, 0.],
                [2.316181E+06, 1.295006E+06, 0., 0., 0., 1.947648E+06],
            ],
            m_star: mat![
                [3.192010E+01, 0., 0., 0., 0., 2.047015E+00],
                [0., 3.192010E+01, 0., 0., 0., 4.194599E-01],
                [0., 0., 3.192010E+01, -2.047015E+00, -4.194599E-01, 0.],
                [0., 0., -2.047015E+00, 1.970347E+00, 1.462185E-02, 0.],
                [0., 0., -4.194599E-01, 1.462185E-02, 1.457259E-01, 0.],
                [2.047015E+00, 4.194599E-01, 0., 0., 0., 2.116073E+00],
            ],
        },
        BeamSection {
            s: 1.000000,
            c_star: mat![
                [2.804532E+06, 8.437883E+04, 0., 0., 0., 1.672230E+05],
                [8.437883E+04, 2.332482E+07, 0., 0., 0., 1.457651E+05],
                [0., 0., 1.379757E+08, -2.663047E+06, -6.455480E+05, 0.],
                [0., 0., -2.663047E+06, 5.559955E+06, -8.100379E+03, 0.],
                [0., 0., -6.455480E+05, -8.100379E+03, 5.115436E+05, 0.],
                [1.672230E+05, 1.457651E+05, 0., 0., 0., 3.883041E+05],
            ],
            m_star: mat![
                [9.547910E+00, 0., 0., 0., 0., 1.706336E-01],
                [0., 9.547910E+00, 0., 0., 0., 4.399876E-02],
                [0., 0., 9.547910E+00, -1.706336E-01, -4.399876E-02, 0.],
                [0., 0., -1.706336E-01, 7.462437E-02, 5.111898E-04, 0.],
                [0., 0., -4.399876E-02, 5.111898E-04, 4.869354E-03, 0.],
                [1.706336E-01, 4.399876E-02, 0., 0., 0., 7.949373E-02],
            ],
        },
    ];

    // Original [2.55e-3, 1.53e-3, 3.3e-4, 1.53e-3, 2.55e-3, 3.3e-4];
    let mu = col![3.3e-4, 1.53e-3, 2.55e-3, 3.3e-4, 2.55e-3, 1.53e-3];

    // Rotate sections so blade is along X axis
    {
        let mut rr = Mat::<f64>::zeros(6, 6);
        let mut m6 = Mat::<f64>::zeros(6, 6);
        let mut m3 = Mat::<f64>::zeros(3, 3);
        quat_as_matrix(r.as_ref(), m3.as_mut());
        rr.as_mut().submatrix_mut(0, 0, 3, 3).copy_from(&m3);
        rr.as_mut().submatrix_mut(3, 3, 3, 3).copy_from(&m3);

        sections.iter_mut().for_each(|s| {
            // Rotate mass matrix
            matmul(
                m6.as_mut(),
                Accum::Replace,
                rr.as_ref(),
                s.m_star.as_ref(),
                1.,
                Par::Seq,
            );
            matmul(
                s.m_star.as_mut(),
                Accum::Replace,
                m6.as_ref(),
                rr.transpose(),
                1.,
                Par::Seq,
            );

            // Rotate stiffness matrix
            matmul(
                m6.as_mut(),
                Accum::Replace,
                rr.as_ref(),
                s.c_star.as_ref(),
                1.,
                Par::Seq,
            );
            matmul(
                s.c_star.as_mut(),
                Accum::Replace,
                m6.as_ref(),
                rr.transpose(),
                1.,
                Par::Seq,
            );
        });
    }

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    model.set_gravity(0., 0., -9.81);

    model.add_beam_element(&node_ids, &quadrature, &sections, &Damping::Mu(mu));

    model
}

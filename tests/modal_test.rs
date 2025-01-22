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


#[test]
#[ignore]
fn test_viscoelastic() {

    // Target damping value
    let zeta = col![0.01];

    // Viscoelastic is run seperately with prescribed mode shapes
    // Because using stiffness is frequency dependent, so not
    // trivial to calculate these in general.

    // 6x6 mass matrix
    let m_star = mat![
        [ 8.1639955821658532e+01,  0.0000000000000000e+00, 0.0000000000000000e+00,
            0.0000000000000000e+00,  0.0000000000000000e+00, -1.0438698516690437e-15],
        [ 0.0000000000000000e+00,  8.1639955821658532e+01, 0.0000000000000000e+00,
            0.0000000000000000e+00,  0.0000000000000000e+00, -2.2078120857317492e-06],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00, 8.1639955821658532e+01,
            1.0438698516690437e-15,  2.2078120857317492e-06,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00, 1.0438698516690437e-15,
            3.6256489177087214e-01, -6.7154254365978626e-17,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00, 2.2078120857317492e-06,
            -6.7154254365978626e-17,  1.0822522299792561e-01,  0.0000000000000000e+00],
        [-1.0438698516690437e-15, -2.2078120857317492e-06, 0.0000000000000000e+00,
            0.0000000000000000e+00,  0.0000000000000000e+00,  4.7079011476879862e-01],
    ];

    // 6x6 stiffness with time scale of infinity
    let c_star = mat![
        [ 1.9673154323503646e+08,  6.3080848185613104e+03,  0.0000000000000000e+00,
            0.0000000000000000e+00, 0.0000000000000000e+00, -3.4183556229693512e+00],
        [ 6.3080848218442034e+03,  5.6376577109183133e+08,  0.0000000000000000e+00,
            0.0000000000000000e+00, 0.0000000000000000e+00, -8.1579043049956724e+01],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,  2.1839988181592059e+09,
            -2.2882083986558970e-07, 5.9062488886340653e+01,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00, -2.1273647125393454e-07,
            9.6994472729496751e+06, 1.8311899365084852e+01,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,  5.9062489064719855e+01,
            1.8311899368561253e+01, 2.8954872824363140e+06,  0.0000000000000000e+00],
        [-3.4183556439358025e+00, -8.1579042993851402e+01,  0.0000000000000000e+00,
            0.0000000000000000e+00, 0.0000000000000000e+00,  2.8197682819547984e+06],
    ];

    // Constant 6x6 stiffness for viscoelastic material
    let c_star_inf = mat![
        [ 2.0931738091201134e+09,  4.5647518044212236e-13,  4.8169745840647056e-13,
            -7.8678624079238851e-14, -5.6606283012184477e+01, -2.5812117918812244e-07],
        [ 1.5597295420533168e-11,  5.4032068914879858e+08, -6.0457532385914647e+03,
            7.8186450812608697e+01,  7.7263353572661315e-13,  2.2742071455094342e-12],
        [ 0.0000000000000000e+00, -6.0457532417315051e+03,  1.8855015410424042e+08,
            -3.2761979737973199e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [-7.7192698264130860e-13,  7.8039682453767483e+01, -3.2384246480523711e+00,
            2.7022137206988120e+06, -3.8235769218915541e-14, -1.1194819068444191e-13],
        [-5.6606282839448653e+01,  2.2574578250343211e-14,  2.3708013850364261e-14,
            -3.8922335362808024e-15,  2.7745148325549518e+06,  7.3617417129062318e+01],
        [-2.5007516606586434e-07,  6.1243992921363272e-14,  3.1330546484050769e-14,
            -1.0911633334478454e-14,  7.3617417125147128e+01,  9.2956243628904670e+06],
        ];

    // Select viscoelastic stiffness at time scale tau_i
    // This should be a list of matrices for later expansion to multiple term Prony series
    let c_star_tau_i = mat![
        [ 9.2013916060891056e+08,  2.3243668428067584e-13,  3.5887979073606163e-13,
            -3.8849556459231303e-14, -2.4883579905579833e+01, -1.1429820668884585e-07],
        [ 2.3776680985842247e-12,  2.3751980041358393e+08, -2.6576552247924187e+03,
            3.4370015001289872e+01,  1.1718608966659488e-13,  2.1012697621527337e-13],
        [-7.4466807599400762e-26, -2.6576552261742540e+03,  8.2884842039532781e+07,
            -1.4401852541502962e+00, -2.1480933678900110e-16, -1.0694834858691245e-15],
        [-1.1721351506632148e-13,  3.4305497038520144e+01, -1.4235804608947455e+00,
            1.1878672730935828e+06, -5.7984110361332067e-15, -1.0480709558414323e-14],
        [-2.4883579832645903e+01,  1.1538970382917394e-14,  1.7632023079758250e-14,
            -1.9306377148308136e-15,  1.2196501494528179e+06,  3.2361511551526746e+01],
        [-1.0888534133232360e-07,  4.4099192119082034e-14,  1.4262375493332581e-14,
            -7.9456524155848800e-15,  3.2361511550051176e+01,  4.0862674477095800e+06],
        ];

    let tau_i = col![0.05];

    let undamped_damping=Damping::None;
    let damping = Damping::Viscoelastic(c_star_tau_i.clone(), tau_i.clone());

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 1.; // Velocity scaling factor
    let t_end = 0.04; //3.1; // Simulation length
    let time_step = 0.005; // Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 6; // Max convergence iterations
    let n_steps = (t_end / time_step) as usize;
    // let n_steps = 1;

    // Create output directory
    let out_dir = "output/modal";
    fs::create_dir_all(out_dir).unwrap();

    // Initialize model without damping for modal analysis
    let mut undamped_model = setup_model_custom(undamped_damping.clone(), m_star.clone(), c_star.clone());
    undamped_model.set_rho_inf(rho_inf);
    undamped_model.set_max_iter(max_iter);
    undamped_model.set_time_step(time_step);


    // Perform modal analysis
    let (eig_val, eig_vec) = modal_analysis(&out_dir, &undamped_model);
    let omega_n = eig_val[i_mode].sqrt();
    let f_n = omega_n / (2. * PI);

    println!(
        "mode={}, omega_n={omega_n}, fn={f_n}, zeta={}",
        i_mode, zeta[i_mode]
    );

    // New model with viscoelastic damping
    let mut model = setup_model_custom(damping.clone(), m_star.clone(), c_star_inf.clone());
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);


    // Additional initialization for mu damping
    /*
    // Test only intended for viscoelastic material case
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
    */

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

    let model = setup_model_custom(damping, m_star, c_star);

    model
}

fn setup_model_custom(damping: Damping, m_star: Mat<f64>, c_star: Mat<f64>) -> Model{

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

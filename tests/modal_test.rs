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
    // let damping = Damping::ModalElement(zeta.clone());
    let damping = Damping::Mu(col![0., 0., 0., 0., 0., 0.]);

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
    // Viscoelastic test uses mode shapes calculated based on an
    // undamped model with a different stiffness matrix that
    // should recreate equivalent mode shapes.

    // Target damping value
    let zeta = col![0.01, 0.0];

    // 6x6 mass matrix
    let m_star = mat![
        [ 8.1639955821658532e+01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.2078120857317492e-06,  1.0438698516690437e-15],
        [ 0.0000000000000000e+00,  8.1639955821658532e+01,  0.0000000000000000e+00, -2.2078120857317492e-06,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,  8.1639955821658532e+01, -1.0438698516690437e-15,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00, -2.2078120857317492e-06, -1.0438698516690437e-15,  4.7079011476879862e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 2.2078120857317492e-06,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.0822522299792561e-01, -6.7154254365978626e-17],
        [ 1.0438698516690437e-15,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -6.7154254365978626e-17,  3.6256489177087214e-01],
    ];

    // 6x6 stiffness for reference solution (mode shape calculation)
    let c_star = mat![
        [ 2.1839988181592059e+09,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  5.9062488886340653e+01, -2.2882083986558970e-07],
        [ 0.0000000000000000e+00,  5.6376577109183133e+08,  6.3080848218442034e+03, -8.1579043049956724e+01,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00,  6.3080848185613104e+03,  1.9673154323503646e+08, -3.4183556229693512e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 0.0000000000000000e+00, -8.1579042993851402e+01, -3.4183556439358025e+00,  2.8197682819547984e+06,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 5.9062489064719855e+01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.8954872824363140e+06,  1.8311899368561253e+01],
        [-2.1273647125393454e-07,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.8311899365084852e+01,  9.6994472729496751e+06],
    ];

    // Constant 6x6 stiffness for viscoelastic material
    let c_star_inf = mat![
        [ 2.1695435690146260e+09, -1.1609894041035092e-13, -2.2297438864313538e-13,  1.8937786174491672e-14, -5.8671571734817554e+01, -2.6242497175010508e-07],
        [-3.4285616462604358e-12,  5.6003437040933549e+08, -6.2663334508912903e+03,  8.1039095179536545e+01, -1.7219803593426008e-13, -1.0414524361099484e-12],
        [ 0.0000000000000000e+00, -6.2663334541576742e+03,  1.9542943471348783e+08, -3.3957305473889132e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 1.7150682265730904e-13,  8.0886971955846079e+01, -3.3565790547830106e+00,  2.8008043929746081e+06,  8.6096413194164738e-15,  5.1142944433815467e-14],
        [-5.8671571560838629e+01, -5.7804934801834121e-15, -1.0946687029310310e-14,  9.4459025717049081e-16,  2.8757434217255106e+06,  7.6303359617750402e+01],
        [-2.6151779835359987e-07, -2.6996811636488128e-14, -6.4730343074106203e-15,  4.8883261356043323e-15,  7.6303359614075504e+01,  9.6347766098612845e+06],
    ];

    // Select viscoelastic stiffness at time scale tau_i
    // TODO : expand to a list of matrices for later expansion to multiple term Prony series
    let c_star_tau_i = mat![
        [ 1.4644469574323347e+08, -1.0188771691870348e-13, -5.1994012431391355e-14,  1.8154602253786624e-14, -3.9603447520475354e+00, -1.7940072085357155e-08],
        [-6.2263602975713452e-14,  3.7802450317781217e+07, -4.2297896605674765e+02,  5.4701577808431656e+00, -3.6104088455590133e-15, -1.2982574802947295e-13],
        [ 0.0000000000000000e+00, -4.2297896627679188e+02,  1.3191532317898544e+07, -2.2921260211961439e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
        [ 3.4880988682707595e-15,  5.4598894285510555e+00, -2.2656986732261966e-01,  1.8905494824872792e+05,  1.9828648143279700e-16,  6.3633316992209696e-15],
        [-3.9603447396536526e+00, -5.0172485853446039e-15, -2.5742691000235807e-15,  8.9383261389056998e-16,  1.9411335012804990e+05,  5.1504945293381743e+00],
        [-1.7617514621622548e-08, -7.3581919932964069e-15, -7.8196914513452354e-15,  1.2676553728314626e-15,  5.1504945290922839e+00,  6.5034966309802630e+05],
    ];

    let tau_i = col![0.05];

    let undamped_damping=Damping::None;
    let damping = Damping::Viscoelastic(c_star_tau_i.clone(), tau_i.clone());

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 1.; // Velocity scaling factor
    let t_end = 3.1; //3.1; // Simulation length
    let time_step = 0.001; // 0.001, Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
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


    // Perform modal analysis (on undamped model)
    let (eig_val, eig_vec) = modal_analysis(&out_dir, &undamped_model);
    let omega_n = eig_val[i_mode].sqrt();
    let f_n = omega_n / (2. * PI);

    println!("eigvals: {:?}", eig_val);

    println!(
        "mode={}, omega_n={omega_n}, fn={f_n}, zeta={}",
        i_mode, zeta[i_mode]
    );

    // New model with viscoelastic damping
    let mut model = setup_model_custom(damping.clone(), m_star.clone(), c_star_inf.clone());
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

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
fn test_viscoelastic_grad() {

    // Named viscoelastic, but can also be used for Mu damping gradient checks.
    // Has separate undamped model to get mode shapes prior to
    // Looking at damped model.

    // Finite difference size
    let delta = 1e-9; //1e-9 to 1e-6 are reasonable here

    // Target damping value
    let zeta = col![0.01, 0.0];

    // ----------- Reference Values for mass, stiffness, prony series --------
    // // 6x6 mass matrix
    // let m_star = mat![
    //     [ 8.1639955821658532e+01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.2078120857317492e-06,  1.0438698516690437e-15],
    //     [ 0.0000000000000000e+00,  8.1639955821658532e+01,  0.0000000000000000e+00, -2.2078120857317492e-06,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 0.0000000000000000e+00,  0.0000000000000000e+00,  8.1639955821658532e+01, -1.0438698516690437e-15,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 0.0000000000000000e+00, -2.2078120857317492e-06, -1.0438698516690437e-15,  4.7079011476879862e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 2.2078120857317492e-06,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.0822522299792561e-01, -6.7154254365978626e-17],
    //     [ 1.0438698516690437e-15,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -6.7154254365978626e-17,  3.6256489177087214e-01],
    // ];

    // // 6x6 stiffness for reference solution
    // let c_star = mat![
    //     [ 2.1839988181592059e+09,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  5.9062488886340653e+01, -2.2882083986558970e-07],
    //     [ 0.0000000000000000e+00,  5.6376577109183133e+08,  6.3080848218442034e+03, -8.1579043049956724e+01,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 0.0000000000000000e+00,  6.3080848185613104e+03,  1.9673154323503646e+08, -3.4183556229693512e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 0.0000000000000000e+00, -8.1579042993851402e+01, -3.4183556439358025e+00,  2.8197682819547984e+06,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 5.9062489064719855e+01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.8954872824363140e+06,  1.8311899368561253e+01],
    //     [-2.1273647125393454e-07,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.8311899365084852e+01,  9.6994472729496751e+06],
    // ];

    // // Constant 6x6 stiffness for viscoelastic material
    // let c_star_inf = mat![
    //     [ 2.1695435690146260e+09, -1.1609894041035092e-13, -2.2297438864313538e-13,  1.8937786174491672e-14, -5.8671571734817554e+01, -2.6242497175010508e-07],
    //     [-3.4285616462604358e-12,  5.6003437040933549e+08, -6.2663334508912903e+03,  8.1039095179536545e+01, -1.7219803593426008e-13, -1.0414524361099484e-12],
    //     [ 0.0000000000000000e+00, -6.2663334541576742e+03,  1.9542943471348783e+08, -3.3957305473889132e+00,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 1.7150682265730904e-13,  8.0886971955846079e+01, -3.3565790547830106e+00,  2.8008043929746081e+06,  8.6096413194164738e-15,  5.1142944433815467e-14],
    //     [-5.8671571560838629e+01, -5.7804934801834121e-15, -1.0946687029310310e-14,  9.4459025717049081e-16,  2.8757434217255106e+06,  7.6303359617750402e+01],
    //     [-2.6151779835359987e-07, -2.6996811636488128e-14, -6.4730343074106203e-15,  4.8883261356043323e-15,  7.6303359614075504e+01,  9.6347766098612845e+06],
    // ];


    // // Select viscoelastic stiffness at time scale tau_i
    // // This should be a list of matrices for later expansion to multiple term Prony series
    // let c_star_tau_i = mat![
    //     [ 1.4644469574323347e+08, -1.0188771691870348e-13, -5.1994012431391355e-14,  1.8154602253786624e-14, -3.9603447520475354e+00, -1.7940072085357155e-08],
    //     [-6.2263602975713452e-14,  3.7802450317781217e+07, -4.2297896605674765e+02,  5.4701577808431656e+00, -3.6104088455590133e-15, -1.2982574802947295e-13],
    //     [ 0.0000000000000000e+00, -4.2297896627679188e+02,  1.3191532317898544e+07, -2.2921260211961439e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 3.4880988682707595e-15,  5.4598894285510555e+00, -2.2656986732261966e-01,  1.8905494824872792e+05,  1.9828648143279700e-16,  6.3633316992209696e-15],
    //     [-3.9603447396536526e+00, -5.0172485853446039e-15, -2.5742691000235807e-15,  8.9383261389056998e-16,  1.9411335012804990e+05,  5.1504945293381743e+00],
    //     [-1.7617514621622548e-08, -7.3581919932964069e-15, -7.8196914513452354e-15,  1.2676553728314626e-15,  5.1504945290922839e+00,  6.5034966309802630e+05],
    // ];


    // Reducing for debugging.
    // let c_star_tau_i = mat![
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 3.4880988682707595e-15,  5.4598894285510555e+00, -2.2656986732261966e-01,  1.8905494824872792e+05,  1.9828648143279700e-16,  6.3633316992209696e-15],
    //     [-3.9603447396536526e+00, -5.0172485853446039e-15, -2.5742691000235807e-15,  8.9383261389056998e-16,  1.9411335012804990e+05,  5.1504945293381743e+00],
    //     [-1.7617514621622548e-08, -7.3581919932964069e-15, -7.8196914513452354e-15,  1.2676553728314626e-15,  5.1504945290922839e+00,  6.5034966309802630e+05],
    // ];


    // Reducing for debugging.
    // let c_star_tau_i = mat![
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.0, 0.0,  1.8905494824872792e+05,  1.9828648143279700e-16,  6.3633316992209696e-15],
    //     [0.0, 0.0, 0.0,  8.9383261389056998e-16,  1.9411335012804990e+05,  5.1504945293381743e+00],
    //     [0.0, 0.0, 0.0,  1.2676553728314626e-15,  5.1504945290922839e+00,  6.5034966309802630e+05],
    // ];


    // Reducing for debugging.
    // let c_star_tau_i = mat![
    //     [ 1.4644469574323347e+08, -1.0188771691870348e-13, -5.1994012431391355e-14,  0.0, 0.0, 0.0],
    //     [-6.2263602975713452e-14,  3.7802450317781217e+07, -4.2297896605674765e+02,  0.0, 0.0, 0.0],
    //     [ 0.0000000000000000e+00, -4.2297896627679188e+02,  1.3191532317898544e+07,  0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // ];

    /*
    // Reducing for debugging.
    let c_star_tau_i = mat![
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    */

    // ------- Making Everything Diagonal -------------------

    let m_star = mat![
        [ 8.1639955821658532e+01,  0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 8.1639955821658532e+01,  0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0,  8.1639955821658532e+01, 0.0,  0.0,  0.0],
        [ 0.0, 0.0, 0.0,  4.7079011476879862e-01,  0.0,  0.0],
        [ 0.0, 0.0,  0.0,  0.0,  1.0822522299792561e-01, 0.0],
        [ 0.0, 0.0,  0.0,  0.0, 0.0,  3.6256489177087214e-01],
    ];

    // 6x6 stiffness for reference solution
    let c_star = mat![
        [ 2.1839988181592059e+09,  0.0,  0.0, 0.0, 0.0, 0.0],
        [ 0.0,  5.6376577109183133e+08,  0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0,  1.9673154323503646e+08,  0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0,  2.8197682819547984e+06,  0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0,  2.8954872824363140e+06,  0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0,  9.6994472729496751e+06],
    ];

    // Constant 6x6 stiffness for viscoelastic material
    let c_star_inf = mat![
        [ 2.1695435690146260e+09, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,  5.6003437040933549e+08, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,  1.9542943471348783e+08, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  2.8008043929746081e+06,  0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0,  2.8757434217255106e+06,  0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0,  9.6347766098612845e+06],
    ];


    let c_star_tau_i = mat![
        [ 1.4644469574323347e+08, 0.0, 0.0,  0.0, 0.0, 0.0],
        [ 0.0,  3.7802450317781217e+07, 0.0,  0.0, 0.0, 0.0],
        [ 0.0, 0.0,  1.3191532317898544e+07,  0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let tau_i = col![0.05];

    let undamped_damping=Damping::None;

    // Choose one of these damping models to check gradients of
    let damping = Damping::Viscoelastic(c_star_tau_i.clone(), tau_i.clone());
    // let damping = Damping::Mu(col![0.016, 0.016, 0.016, 0.016, 0.016, 0.016]); // 3.9003076417101866e-5
    // let damping = Damping::Mu(col![0.5, 0.2, 0.1, 0.3, 0.7, 0.6]); // 3.7663479228680903e-5
    // let damping = Damping::Mu(col![0.0, 0.0, 0.0, 0.16, 0.16, 0.16]); // 6.138906379474862e-5
    // let damping = Damping::Mu(col![0.5, 0.5, 0.5, 0.0, 0.0, 0.0]); // 3.7756168437499526e-5
    // let damping = Damping::Mu(col![0.5, 0.0, 0.0, 0.0, 0.0, 0.0]); // 0.0003942734860216614 -> now 7.3e-6
    // let damping = Damping::Mu(col![0.0, 0.5, 0.0, 0.0, 0.0, 0.0]); // 4.0333723924300566e-5
    // let damping = Damping::Mu(col![0.0, 0.0, 0.5, 0.0, 0.0, 0.0]); // 3.718334599605073e-7
    // let damping=Damping::None;

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 1.; // Velocity scaling factor
    // let t_end = 3.1; //3.1; // Simulation length - no simulation
    let time_step = 0.001; // 0.001, Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
    // let n_steps = (t_end / time_step) as usize; - no simulation
    // let n_steps = 1; - no simulation

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

    // New model with viscoelastic damping (or damped Mu model to check)
    let mut model = setup_model_custom(damping.clone(), m_star.clone(), c_star_inf.clone());
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

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

    //------------------------------------------------------------------
    // Numerical Gradient Calculation
    //------------------------------------------------------------------
    // Loop through perturbations

    let ndof = solver.n_system + solver.n_lambda;

    // Analytical derivative of residual at reference state.
    let mut dres_mat = Mat::<f64>::zeros(ndof, ndof);

    // Memory to ignore when calling with perturbations
    let mut dres_mat_ignore = Mat::<f64>::zeros(ndof, ndof);

    // Numerical approximation of 'dres_mat'
    let mut dres_mat_num = Mat::<f64>::zeros(ndof, ndof);

    // Initial Calculation for analytical gradient
    let mut state = model.create_state();
    let mut res_vec = Col::<f64>::zeros(ndof);
    let xd = Col::<f64>::zeros(ndof);


    state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

    // Do a residual + gradient eval
    solver.step_res_grad(&mut state, xd.as_ref(), res_vec.as_mut(), dres_mat.as_mut());


    for i in 0..ndof {

        // Positive side of finite difference
        let mut state = model.create_state();
        let mut res_vec = Col::<f64>::zeros(ndof);
        let mut xd = Col::<f64>::zeros(ndof);

        state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

        xd[i] = delta;

        solver.step_res_grad(&mut state, xd.as_ref(), res_vec.as_mut(), dres_mat_ignore.as_mut());

        let tmp = dres_mat_num.col(i) + res_vec*Scale(0.5/delta);
        dres_mat_num.col_mut(i).copy_from(tmp.clone());

        // Negative side of finite difference
        let mut state = model.create_state();
        let mut res_vec = Col::<f64>::zeros(ndof);
        let mut xd = Col::<f64>::zeros(ndof);

        state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

        xd[i] = -delta;

        solver.step_res_grad(&mut state, xd.as_ref(), res_vec.as_mut(), dres_mat_ignore.as_mut());

        let tmp = dres_mat_num.col(i) - res_vec*Scale(0.5/delta);
        dres_mat_num.col_mut(i).copy_from(tmp);

    }

    // Optional output of portions of derivative matrix
    // println!("Analytical:");
    // println!("{:?}", dres_mat);
    // println!("Numerical:");
    // println!("{:?}", dres_mat_num);

    // println!("Analytical:");
    // println!("{:?}", dres_mat.submatrix(0,0,6,6));
    // println!("Numerical:");
    // println!("{:?}", dres_mat_num.submatrix(0,0,6,6));

    let grad_diff = dres_mat.clone() - dres_mat_num.clone();

    println!("Grad diff norm: {:?}", grad_diff.norm_l2());
    println!("Grad (analytical) norm: {:?}", dres_mat.norm_l2());
    println!("Norm ratio (diff/analytical): {:?}", grad_diff.norm_l2() / dres_mat.norm_l2());

}


fn modal_analysis(out_dir: &str, model: &Model) -> (Col<f64>, Mat<f64>) {
    // Create solver and state from model
    let mut solver = model.create_solver();
    let state = model.create_state();

    // time step does not matter here for the modal analysis
    let h = 1.0;

    // Calculate system based on initial state
    solver.elements.beams.calculate_system(&state, h);

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

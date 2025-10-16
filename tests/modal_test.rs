use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

use faer::prelude::*;

use itertools::{izip, Itertools};
use beamo::{
    elements::beams::{BeamSection, Damping},
    interp::gauss_legendre_lobotto_points,
    model::Model,
    quadrature::Quadrature,
    util::ColMutReshape,
};

#[test]
#[ignore]
fn test_damping() {
    // Damping ratio for modes 1-4
    let zeta = col![0.01];

    // Select damping type
    // let damping = Damping::None;
    // let damping = Damping::ModalElement(zeta.clone());
    let damping = Damping::Mu(col![0., 0., 0., 0., 0., 0.]);

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 80.; // Velocity scaling factor
    let t_end = 0.7; // Simulation length
    let time_step = 0.01; // Time step
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
        let u = u_col.reshape_mut(3, model.n_nodes());
        zip!(&mut u.subrows_mut(0, 3), state.u.subrows(0, 3)).for_each(|unzip!(u, us)| *u = *us);

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
    let zeta = col![0.01, 0.015];

    // 6x6 mass matrix
    let m_star = mat![
        [
            8.1639955821658532e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            2.2078120857317492e-06,
            1.0438698516690437e-15
        ],
        [
            0.0000000000000000e+00,
            8.1639955821658532e+01,
            0.0000000000000000e+00,
            -2.2078120857317492e-06,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            8.1639955821658532e+01,
            -1.0438698516690437e-15,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            -2.2078120857317492e-06,
            -1.0438698516690437e-15,
            4.7079011476879862e-01,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            2.2078120857317492e-06,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            1.0822522299792561e-01,
            -6.7154254365978626e-17
        ],
        [
            1.0438698516690437e-15,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            -6.7154254365978626e-17,
            3.6256489177087214e-01
        ],
    ];

    // 6x6 stiffness for reference solution (mode shape calculation)
    let c_star = mat![
        [
            2.1839988181592059e+09,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            5.9062488886340653e+01,
            -2.2882083986558970e-07
        ],
        [
            0.0000000000000000e+00,
            5.6376577109183133e+08,
            6.3080848218442034e+03,
            -8.1579043049956724e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            6.3080848185613104e+03,
            1.9673154323503646e+08,
            -3.4183556229693512e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            -8.1579042993851402e+01,
            -3.4183556439358025e+00,
            2.8197682819547984e+06,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            5.9062489064719855e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            2.8954872824363140e+06,
            1.8311899368561253e+01
        ],
        [
            -2.1273647125393454e-07,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            1.8311899365084852e+01,
            9.6994472729496751e+06
        ],
    ];

    // // ------------ Single Term Prony Series
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
    // // TODO : expand to a list of matrices for later expansion to multiple term Prony series
    // let c_star_tau_i = mat![
    //     [ 1.4644469574323347e+08, -1.0188771691870348e-13, -5.1994012431391355e-14,  1.8154602253786624e-14, -3.9603447520475354e+00, -1.7940072085357155e-08],
    //     [-6.2263602975713452e-14,  3.7802450317781217e+07, -4.2297896605674765e+02,  5.4701577808431656e+00, -3.6104088455590133e-15, -1.2982574802947295e-13],
    //     [ 0.0000000000000000e+00, -4.2297896627679188e+02,  1.3191532317898544e+07, -2.2921260211961439e-01,  0.0000000000000000e+00,  0.0000000000000000e+00],
    //     [ 3.4880988682707595e-15,  5.4598894285510555e+00, -2.2656986732261966e-01,  1.8905494824872792e+05,  1.9828648143279700e-16,  6.3633316992209696e-15],
    //     [-3.9603447396536526e+00, -5.0172485853446039e-15, -2.5742691000235807e-15,  8.9383261389056998e-16,  1.9411335012804990e+05,  5.1504945293381743e+00],
    //     [-1.7617514621622548e-08, -7.3581919932964069e-15, -7.8196914513452354e-15,  1.2676553728314626e-15,  5.1504945290922839e+00,  6.5034966309802630e+05],
    // ];

    // let tau_i = col![0.05];
    // let mut c_star_all = Mat::<f64>::zeros(36, 1);
    // let c_star_col = c_star_all.col_mut(0);
    // let mut c_single_mat = c_star_col.reshape_mut(6, 6);
    // c_single_mat.copy_from(c_star_tau_i.clone());

    // ------------ Prony series stiffnesses for multiple terms
    let c_star_tau_0 = mat![
        [
            2.1831356082262930e+08,
            -3.6957008766891517e-14,
            -2.7424783318537356e-14,
            7.0795168782312469e-15,
            -5.9039144660167171e+00,
            2.9321663591980022e-08
        ],
        [
            -3.7987822508967875e-13,
            5.6354294669792451e+07,
            6.3055916071755360e+02,
            8.1546799526975047e+00,
            -1.8580671993194649e-14,
            9.6698959240565326e-16
        ],
        [
            1.0596446510373938e-25,
            6.3055916065999202e+02,
            1.9665378649788186e+07,
            3.4170045171919944e-01,
            -2.6736786846400837e-16,
            -5.3669926558797432e-17
        ],
        [
            1.8617318932166703e-14,
            8.1393723459371863e+00,
            3.3776077839723290e-01,
            2.8183512372423348e+05,
            8.8429149001247509e-16,
            -6.5218805541439177e-17
        ],
        [
            -5.9039144688736513e+00,
            -1.8019369730409791e-15,
            -1.3315433037532005e-15,
            3.4510814436121627e-16,
            2.8937597537821624e+05,
            -7.6781394840564658e+00
        ],
        [
            2.1829374998076511e-08,
            -2.5925105343049129e-15,
            -3.5253919275946727e-15,
            5.1373376454589595e-16,
            -7.6781394842223527e+00,
            9.6951378136403335e+05
        ],
    ];
    let c_star_tau_1 = mat![
        [
            2.2067067170711074e+07,
            6.0539779983968755e-15,
            -8.0382623699156603e-17,
            -1.1108563242162073e-15,
            -5.9676584731645699e-01,
            2.9923746593763171e-09
        ],
        [
            8.5803581086749989e-14,
            5.6962746663580090e+06,
            6.3736724838417707e+01,
            8.2427252612755020e-01,
            4.2676123407343990e-15,
            -1.6461601493249896e-14
        ],
        [
            1.5210739576036056e-26,
            6.3736724832657686e+01,
            1.9877703884593605e+06,
            3.4538975912377548e-02,
            -6.7127917733124537e-18,
            3.3421358933305893e-17
        ],
        [
            -4.2598158454429705e-15,
            8.2272523797781327e-01,
            3.4140754953416147e-02,
            2.8487807092025898e+04,
            -2.1116883601880089e-16,
            8.0610976760451514e-16
        ],
        [
            -5.9676584751106998e-01,
            2.9694973772141666e-16,
            -5.6884403445063805e-18,
            -5.4468262809423448e-17,
            2.9250034043690212e+04,
            -7.7610396304579643e-01
        ],
        [
            2.1771827691696946e-09,
            -9.5152479761638393e-17,
            5.0942521786175066e-16,
            1.2030352847209010e-17,
            -7.7610396305948370e-01,
            9.7998153003753119e+04
        ],
    ];

    let c_star_inf = mat![
        [
            2.1597420367286029e+09,
            3.8412385045559106e-13,
            2.3187276108286278e-13,
            -7.3015066675758679e-14,
            -5.8406505793673830e+01,
            2.9677403251737724e-07
        ],
        [
            -3.2821558390801757e-12,
            5.5750425530105424e+08,
            6.2380235149459631e+03,
            8.0672977998093472e+01,
            -1.5638489920510812e-13,
            -9.4470036142524460e-13
        ],
        [
            0.0000000000000000e+00,
            6.2380235143867185e+03,
            1.9454652646447238e+08,
            3.3803893205160702e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            1.5764465703252059e-13,
            8.0521542237040322e+01,
            3.3414147493328206e+00,
            2.7881509597489857e+06,
            7.5078957846561722e-15,
            4.6180516801102872e-14
        ],
        [
            -5.8406505817779205e+01,
            1.8749582309920044e-14,
            1.1237250733948351e-14,
            -3.5629806261856335e-15,
            2.8627514300471852e+06,
            -7.5958637407269663e+01
        ],
        [
            2.2433543331337642e-07,
            2.0901212269692965e-14,
            3.5850638033526972e-14,
            -4.2211644389389901e-15,
            -7.5958637408598292e+01,
            9.5912487566487044e+06
        ],
    ];

    let tau_i = col![0.02299974, 0.14994091];
    let n_tau = tau_i.nrows();

    // Make c_star_tau_i into a column and insert as columns
    let mut c_star_all = Mat::<f64>::zeros(36, n_tau);
    let c_star_col = c_star_all.col_mut(0);
    let mut c_single_mat = c_star_col.reshape_mut(6, 6);
    c_single_mat.copy_from(c_star_tau_0.clone());

    let c_star_col = c_star_all.col_mut(1);
    let mut c_single_mat = c_star_col.reshape_mut(6, 6);
    c_single_mat.copy_from(c_star_tau_1.clone());

    let undamped_damping = Damping::None;
    let damping = Damping::Viscoelastic(c_star_all, tau_i.clone());

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 1.; // Velocity scaling factor
    let t_end = 3.1; //3.1; // Simulation length
    let time_step = 0.01; // 0.001, Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
    let n_steps = (t_end / time_step) as usize;
    // let n_steps = 1;

    // Create output directory
    let out_dir = "output/modal";
    fs::create_dir_all(out_dir).unwrap();

    // Initialize model without damping for modal analysis
    let mut undamped_model =
        setup_model_custom(undamped_damping.clone(), m_star.clone(), c_star.clone());
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
        let u = u_col.reshape_mut(3, model.n_nodes());
        zip!(&mut u.subrows_mut(0, 3), state.u.subrows(0, 3)).for_each(|unzip!(u, us)| *u = *us);

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
    let delta = 1e-7; //1e-9 to 1e-6 are reasonable here

    // Target damping value
    let zeta = col![0.01, 0.0];

    // ----------- Reference Values for mass, stiffness, prony series --------
    // 6x6 mass matrix
    let m_star = mat![
        [
            8.1639955821658532e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            2.2078120857317492e-06,
            1.0438698516690437e-15
        ],
        [
            0.0000000000000000e+00,
            8.1639955821658532e+01,
            0.0000000000000000e+00,
            -2.2078120857317492e-06,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            8.1639955821658532e+01,
            -1.0438698516690437e-15,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            -2.2078120857317492e-06,
            -1.0438698516690437e-15,
            4.7079011476879862e-01,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            2.2078120857317492e-06,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            1.0822522299792561e-01,
            -6.7154254365978626e-17
        ],
        [
            1.0438698516690437e-15,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            -6.7154254365978626e-17,
            3.6256489177087214e-01
        ],
    ];

    // 6x6 stiffness for reference solution
    let c_star = mat![
        [
            2.1839988181592059e+09,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            5.9062488886340653e+01,
            -2.2882083986558970e-07
        ],
        [
            0.0000000000000000e+00,
            5.6376577109183133e+08,
            6.3080848218442034e+03,
            -8.1579043049956724e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            6.3080848185613104e+03,
            1.9673154323503646e+08,
            -3.4183556229693512e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            0.0000000000000000e+00,
            -8.1579042993851402e+01,
            -3.4183556439358025e+00,
            2.8197682819547984e+06,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            5.9062489064719855e+01,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            2.8954872824363140e+06,
            1.8311899368561253e+01
        ],
        [
            -2.1273647125393454e-07,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            1.8311899365084852e+01,
            9.6994472729496751e+06
        ],
    ];

    // Constant 6x6 stiffness for viscoelastic material
    let c_star_inf = mat![
        [
            2.1695435690146260e+09,
            -1.1609894041035092e-13,
            -2.2297438864313538e-13,
            1.8937786174491672e-14,
            -5.8671571734817554e+01,
            -2.6242497175010508e-07
        ],
        [
            -3.4285616462604358e-12,
            5.6003437040933549e+08,
            -6.2663334508912903e+03,
            8.1039095179536545e+01,
            -1.7219803593426008e-13,
            -1.0414524361099484e-12
        ],
        [
            0.0000000000000000e+00,
            -6.2663334541576742e+03,
            1.9542943471348783e+08,
            -3.3957305473889132e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            1.7150682265730904e-13,
            8.0886971955846079e+01,
            -3.3565790547830106e+00,
            2.8008043929746081e+06,
            8.6096413194164738e-15,
            5.1142944433815467e-14
        ],
        [
            -5.8671571560838629e+01,
            -5.7804934801834121e-15,
            -1.0946687029310310e-14,
            9.4459025717049081e-16,
            2.8757434217255106e+06,
            7.6303359617750402e+01
        ],
        [
            -2.6151779835359987e-07,
            -2.6996811636488128e-14,
            -6.4730343074106203e-15,
            4.8883261356043323e-15,
            7.6303359614075504e+01,
            9.6347766098612845e+06
        ],
    ];

    // Viscoelastic stiffness at time scale tau_i
    // This should be a list of matrices for later expansion to multiple term Prony series
    let _c_star_tau_i = mat![
        [
            1.4644469574323347e+08,
            -1.0188771691870348e-13,
            -5.1994012431391355e-14,
            1.8154602253786624e-14,
            -3.9603447520475354e+00,
            -1.7940072085357155e-08
        ],
        [
            -6.2263602975713452e-14,
            3.7802450317781217e+07,
            -4.2297896605674765e+02,
            5.4701577808431656e+00,
            -3.6104088455590133e-15,
            -1.2982574802947295e-13
        ],
        [
            0.0000000000000000e+00,
            -4.2297896627679188e+02,
            1.3191532317898544e+07,
            -2.2921260211961439e-01,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ],
        [
            3.4880988682707595e-15,
            5.4598894285510555e+00,
            -2.2656986732261966e-01,
            1.8905494824872792e+05,
            1.9828648143279700e-16,
            6.3633316992209696e-15
        ],
        [
            -3.9603447396536526e+00,
            -5.0172485853446039e-15,
            -2.5742691000235807e-15,
            8.9383261389056998e-16,
            1.9411335012804990e+05,
            5.1504945293381743e+00
        ],
        [
            -1.7617514621622548e-08,
            -7.3581919932964069e-15,
            -7.8196914513452354e-15,
            1.2676553728314626e-15,
            5.1504945290922839e+00,
            6.5034966309802630e+05
        ],
    ];

    let _tau_i = col![0.05];

    let undamped_damping = Damping::None;

    // Choose one of these damping models to check gradients of
    let damping = Damping::None;
    // let damping = Damping::Mu(col![0.016, 0.016, 0.016, 0.016, 0.016, 0.016]);
    // let damping = Damping::Viscoelastic(c_star_tau_i.clone(), tau_i.clone());

    // Settings
    let i_mode = 0; // Mode to simulate
    let v_scale = 70.; // Velocity scaling factor
    let t_end = 0.1; //3.1; // Simulation length, gradient is checked at this time (approx)
    let time_step = 0.01; // 0.001, Time step
    let rho_inf = 1.; // Numerical damping
    let max_iter = 20; // Max convergence iterations
    let n_steps = (t_end / time_step) as usize;

    // Create output directory
    let out_dir = "output/modal";
    fs::create_dir_all(out_dir).unwrap();

    // Initialize model without damping for modal analysis
    let mut undamped_model =
        setup_model_custom(undamped_damping.clone(), m_star.clone(), c_star.clone());
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
    let mut ref_state = model.create_state();

    // Apply scaled mode shape to state as velocity
    let v = eig_vec.col(i_mode) * Scale(v_scale);
    ref_state
        .v
        .col_iter_mut()
        .enumerate()
        .for_each(|(i_node, mut node_v)| {
            node_v.copy_from(v.subrows(i_node * 6, 6));
        });

    // Do a number of time integration steps so that gradient can be
    // checked starting from nonzero displacements
    for _iter in 0..n_steps {
        solver.step(&mut ref_state);
    }
    // println!("u: {:?}", ref_state.u.clone().subrows(0, 3));

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

    // Copy all of state from the reference
    state.n_nodes = ref_state.n_nodes;
    state.xr.copy_from(ref_state.xr.clone());
    state.x.copy_from(ref_state.x.clone());
    state.u_delta.copy_from(ref_state.u_delta.clone());
    state.u_prev.copy_from(ref_state.u_prev.clone());
    state.u.copy_from(ref_state.u.clone());
    state.v.copy_from(ref_state.v.clone());
    state.vd.copy_from(ref_state.vd.clone());
    state.a.copy_from(ref_state.a.clone());
    state.visco_hist.copy_from(ref_state.visco_hist.clone());
    state.strain_dot_n.copy_from(ref_state.strain_dot_n.clone());
    print!("copy v {:?}", state.v.clone());

    // Do a residual + gradient eval
    solver.step_res_grad(&mut state, xd.as_ref(), res_vec.as_mut(), dres_mat.as_mut());

    for i in 0..ndof {
        // Positive side of finite difference
        let mut state = model.create_state();
        let mut res_vec = Col::<f64>::zeros(ndof);
        let mut xd = Col::<f64>::zeros(ndof);

        // Copy all of state from the reference
        state.n_nodes = ref_state.n_nodes;
        state.xr.copy_from(ref_state.xr.clone());
        state.x.copy_from(ref_state.x.clone());
        state.u_delta.copy_from(ref_state.u_delta.clone());
        state.u_prev.copy_from(ref_state.u_prev.clone());
        state.u.copy_from(ref_state.u.clone());
        state.v.copy_from(ref_state.v.clone());
        state.vd.copy_from(ref_state.vd.clone());
        state.a.copy_from(ref_state.a.clone());
        state.visco_hist.copy_from(ref_state.visco_hist.clone());
        state.strain_dot_n.copy_from(ref_state.strain_dot_n.clone());

        xd[i] = delta;

        solver.step_res_grad(
            &mut state,
            xd.as_ref(),
            res_vec.as_mut(),
            dres_mat_ignore.as_mut(),
        );

        let tmp = dres_mat_num.col(i) + res_vec * Scale(0.5 / delta);
        dres_mat_num.col_mut(i).copy_from(tmp.clone());

        // Negative side of finite difference
        let mut state = model.create_state();
        let mut res_vec = Col::<f64>::zeros(ndof);
        let mut xd = Col::<f64>::zeros(ndof);

        // Copy all of state from the reference
        state.n_nodes = ref_state.n_nodes;
        state.xr.copy_from(ref_state.xr.clone());
        state.x.copy_from(ref_state.x.clone());
        state.u_delta.copy_from(ref_state.u_delta.clone());
        state.u_prev.copy_from(ref_state.u_prev.clone());
        state.u.copy_from(ref_state.u.clone());
        state.v.copy_from(ref_state.v.clone());
        state.vd.copy_from(ref_state.vd.clone());
        state.a.copy_from(ref_state.a.clone());
        state.visco_hist.copy_from(ref_state.visco_hist.clone());
        state.strain_dot_n.copy_from(ref_state.strain_dot_n.clone());

        xd[i] = -delta;

        solver.step_res_grad(
            &mut state,
            xd.as_ref(),
            res_vec.as_mut(),
            dres_mat_ignore.as_mut(),
        );

        let tmp = dres_mat_num.col(i) - res_vec * Scale(0.5 / delta);
        dres_mat_num.col_mut(i).copy_from(tmp);
    }

    // Optional output of portions of derivative matrix
    // println!("Analytical:");
    // println!("{:?}", dres_mat);
    // println!("Numerical:");
    // println!("{:?}", dres_mat_num);

    // println!("Analytical:");
    // println!("{:?}", dres_mat.submatrix(6,6,6,6));
    // println!("Numerical:");
    // println!("{:?}", dres_mat_num.submatrix(6,6,6,6));

    let grad_diff = dres_mat.clone() - dres_mat_num.clone();

    println!("Grad diff norm: {:?}", grad_diff.norm_l2());
    println!("Grad (analytical) norm: {:?}", dres_mat.norm_l2());
    println!(
        "Norm ratio (diff/analytical): {:?}",
        grad_diff.norm_l2() / dres_mat.norm_l2()
    );
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
    solver.elements.beams.assemble_system(solver.r.as_mut());

    let ndof_bc = solver.n_system - 6;
    let lu = solver
        .elements
        .beams
        .m_sp
        .to_dense()
        .submatrix(6, 6, ndof_bc, ndof_bc)
        .partial_piv_lu();
    let a = lu.solve(
        solver
            .elements
            .beams
            .k_sp
            .to_dense()
            .submatrix(6, 6, ndof_bc, ndof_bc),
    );

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

fn setup_model_custom(damping: Damping, m_star: Mat<f64>, c_star: Mat<f64>) -> Model {
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
        &damping,
    );

    //--------------------------------------------------------------------------
    // Add constraint element
    //--------------------------------------------------------------------------

    // Prescribed constraint to first node of beam
    model.add_prescribed_constraint(node_ids[0]);

    model
}

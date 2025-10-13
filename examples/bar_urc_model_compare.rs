use faer::prelude::*;
use itertools::izip;
use ottr::{
    elements::beams::Damping,
    external::add_beamdyn_blade,
    model::Model,
    util::{quat_as_rotation_vector, quat_from_rotation_vector, ColRefReshape},
    vtk::beams_qps_as_vtk,
};
use std::{
    f64::consts::PI,
    fs::{self, File},
    io::Write,
};

fn main() {
    // Damping ratio for modes 1-6
    let zeta = col![0.01, 0.02, 0.03, 0.04, 0.05, 0.06];

    // Select damping type
    // let damping = Damping::None;
    // let damping = Damping::Mu(col![0., 0., 0., 0., 0., 0.]);
    let damping = Damping::ModalElement(zeta.clone());

    // Settings
    let inp_dir = "examples";
    let out_dir = "output/bar_urc";
    let _n_cycles = 3.5; // Number of oscillations to simulate
    let rho_inf = 1.; // Numerical damping
    let max_iter = 6; // Max convergence iterations
    let time_step = 0.0001; // Time step

    // Create output directory
    fs::create_dir_all(out_dir).unwrap();

    // Create model and set solver parameters
    let mut model = Model::new();
    model.set_rho_inf(rho_inf);
    model.set_max_iter(max_iter);
    model.set_time_step(time_step);

    // Add BeamDyn blade to model
    let beam = add_beamdyn_blade(
        &mut model,
        &format!("{inp_dir}/bar_urc_bd_primary.inp"),
        &format!("{inp_dir}/bar_urc_bd_props.inp"),
        &[0., 0., 0., 0., 0., 0.],
    );

    // Prescribed constraint to first node of beam
    model.add_prescribed_constraint(beam.nodes[0].id);

    // Perform modal analysis
    let (_eig_val, _eig_vec) = modal_analysis(&out_dir, &model);

    // let omega = Col::<f64>::from_fn(eig_val.nrows(), |i| eig_val[i].sqrt());

    // // Additional initialization for mu damping
    // match damping {
    //     Damping::Mu(_) => {
    //         // Get index of maximum value
    //         let i_max = eig_vec
    //             .col_iter()
    //             .map(|psi| {
    //                 psi.iter()
    //                     .enumerate()
    //                     .max_by(|(_, &a), (_, &b)| a.abs().total_cmp(&b.abs()))
    //                     .map(|(index, _)| index)
    //                     .unwrap()
    //                     % 3
    //             })
    //             .collect_vec();
    //         let i_max_x = i_max.iter().position(|&i| i == 0).unwrap();
    //         let i_max_y = i_max.iter().position(|&i| i == 1).unwrap();
    //         let i_max_z = i_max.iter().position(|&i| i == 2).unwrap();

    //         let mu_x = 2. * zeta[i_max_x] / omega[i_max_x];
    //         let mu_y = 2. * zeta[i_max_y] / omega[i_max_y];
    //         let mu_z = 2. * zeta[i_max_z] / omega[i_max_z];

    //         let mu = col![mu_x, mu_y, mu_z, mu_x, mu_z, mu_y];
    //         println!("mu={:?}", mu);
    //         println!(
    //             "modes: x={}, y={}, z={}",
    //             i_max_x + 1,
    //             i_max_y + 1,
    //             i_max_z + 1
    //         );
    //         model
    //             .beam_elements
    //             .iter_mut()
    //             .for_each(|e| e.damping = Damping::Mu(mu.clone()));
    //     }
    //     _ => (),
    // }

    // // Loop through modes and run simulation
    // izip!(omega.iter(), eig_vec.col_iter())
    //     .take(6)
    //     .enumerate()
    //     .for_each(|(i, (&omega, shape))| {
    //         let t_end = 2. * PI / omega;
    //         let n_steps = (n_cycles * t_end / time_step) as usize;
    //         run_simulation(i + 1, time_step, n_steps, shape, out_dir, &model);
    //     });
}

#[allow(dead_code)]
fn run_simulation(
    mode: usize,
    time_step: f64,
    n_steps: usize,
    shape: ColRef<f64>,
    out_dir: &str,
    model: &Model,
) {
    // Create new solver where beam elements have damping
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // Apply scaled mode shape to state as velocity
    let v = shape;
    state.v.copy_from(v.as_ref().reshape(6, state.n_nodes));

    // Create output file
    let mut file = File::create(format!("{out_dir}/displacement_{:02}.csv", mode)).unwrap();

    // Cartesian rotation vector
    let mut rv = Col::<f64>::zeros(3);

    // Loop through times and run simulation
    for i in 0..n_steps {
        // Calculate time
        let t = (i as f64) * time_step;

        write!(file, "{t}").unwrap();
        state.u.col_iter().for_each(|c| {
            quat_as_rotation_vector(c.subrows(3, 4), rv.as_mut());
            write!(
                file,
                ",{},{},{},{},{},{}",
                c[0], c[1], c[2], rv[0], rv[1], rv[2]
            )
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

fn modal_analysis(out_dir: &str, model: &Model) -> (Col<f64>, Mat<f64>) {
    // Create solver and state from model
    let mut solver = model.create_solver();
    let mut state = model.create_state();

    // should not matter in modal analysis, argument needed for viscoelastic
    let h = 0.0;

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

    // Write eigenanalysis results to output file
    let mut file = File::create(format!("{out_dir}/compare_eigenanalysis.csv")).unwrap();
    write!(file, "freq").unwrap();
    for i in 1..=state.n_nodes {
        for d in ["u1", "u2", "u3", "r1", "r2", "r3"] {
            write!(file, ",N{i}_{d}").unwrap();
        }
    }
    izip!(eig_val.iter(), eig_vec.col_iter()).for_each(|(&lambda, c)| {
        write!(file, "\n{}", lambda.sqrt() / (2. * PI)).unwrap();
        for &v in c.iter() {
            write!(file, ",{v}").unwrap();
        }
    });

    // Write mode shapes to output file
    let mut file = File::create(format!("{out_dir}/compare_modes.csv")).unwrap();
    write!(file, "freq").unwrap();
    for i in 1..=solver.elements.beams.qp.x.ncols() {
        for d in ["u1", "u2", "u3", "r1", "r2", "r3"] {
            write!(file, ",qp{i}_{d}").unwrap();
        }
    }

    fs::create_dir_all(format!("{out_dir}/vtk")).unwrap();
    let mut q = col![0., 0., 0., 0.];
    let mut rv = col![0., 0., 0.];
    izip!(0..eig_val.nrows(), eig_val.iter(), eig_vec.col_iter()).for_each(
        |(i, &lambda, phi_col)| {
            // Apply eigvector displacements to state
            let phi = phi_col.reshape(6, state.u.ncols());

            izip!(state.u.col_iter_mut(), phi.col_iter()).for_each(|(mut u, phi)| {
                let phi = phi * Scale(1.);
                quat_from_rotation_vector(phi.subrows(3, 3), q.as_mut());
                u[0] = phi[0];
                u[1] = phi[1];
                u[2] = phi[2];
                u[3] = q[0];
                u[4] = q[1];
                u[5] = q[2];
                u[6] = q[3];
            });

            // should not matter in modal analysis, argument needed for viscoelastic
            let h = 0.0;

            // Update beam elements from state
            solver.elements.beams.calculate_system(&state, h);

            // Write frequency to file
            write!(file, "\n{}", lambda.sqrt() / (2. * PI)).unwrap();

            // Loop through element quadrature points and output position and rotation
            for c in solver.elements.beams.qp.u.col_iter() {
                quat_as_rotation_vector(c.subrows(3, 4), rv.as_mut());
                write!(
                    file,
                    ",{},{},{},{},{},{}",
                    c[0], c[1], c[2], rv[0], rv[1], rv[2]
                )
                .unwrap();
            }

            beams_qps_as_vtk(&solver.elements.beams)
                .export_ascii(format!("{out_dir}/vtk/mode.{:0>3}.vtk", i + 1))
                .unwrap()
        },
    );

    (eig_val, eig_vec)
}

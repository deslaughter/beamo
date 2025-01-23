use core::panic;
use std::f64::consts::PI;

use faer::{col, Mat};
use itertools::Itertools;

use crate::{
    elements::beams::BeamSection,
    util::{quat_as_matrix, Quat},
};

pub fn parse_beamdyn_keypoints(file_data: &str) -> Vec<[f64; 4]> {
    let lines = file_data.lines().collect_vec();

    let member_total_line = lines.get(19).unwrap();
    if !member_total_line.contains("member_total") {
        panic!("line 20 doesn't contain member_total")
    }
    let member_total: usize = member_total_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    if member_total != 1 {
        panic!("member_total must equal 1")
    }

    // Get the number of key points
    let kp_total_line = lines.get(20).unwrap();
    if !kp_total_line.contains("kp_total") {
        panic!("line 21 doesn't contain kp_total")
    }
    let kp_total = kp_total_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    // Get key point coordinates and twist (swap x and z so length is along x)
    lines
        .iter()
        .skip(24)
        .take(kp_total)
        .map(|l| {
            let v = l
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect_vec();
            [v[2], v[1], v[0], v[3]]
        })
        .collect_vec()
}

pub fn parse_beamdyn_sections(file_data: &str) -> Vec<BeamSection> {
    let mut m = Mat::<f64>::zeros(3, 3);
    let mut q_rot = col![0., 0., 0., 0.];
    q_rot
        .as_mut()
        .quat_from_rotation_vector(col![0., -PI / 2., 0.].as_ref());
    quat_as_matrix(q_rot.as_ref(), m.as_mut());
    let mut m_rot = Mat::<f64>::zeros(6, 6);
    m_rot.submatrix_mut(0, 0, 3, 3).copy_from(&m);
    m_rot.submatrix_mut(3, 3, 3, 3).copy_from(&m);
    let lines = file_data.lines().skip(10).collect_vec();
    lines
        .iter()
        .chunks(15)
        .into_iter()
        .map(|chunks| {
            let ls = chunks.collect_vec();
            let s = ls[0].trim().parse::<f64>().unwrap();
            let c = (1..7)
                .map(|i| {
                    ls[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            let m = (8..14)
                .map(|i| {
                    ls[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            BeamSection {
                s,
                m_star: &m_rot * Mat::<f64>::from_fn(6, 6, |i, j| m[i][j]) * &m_rot.transpose(),
                c_star: &m_rot * Mat::<f64>::from_fn(6, 6, |i, j| c[i][j]) * &m_rot.transpose(),
            }
        })
        .collect_vec()
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_beamdyn_keypoints() {
        let keypoints = parse_beamdyn_keypoints(BD_INPUT1);
        println!("{:?}", keypoints);
    }

    #[test]
    fn test_parse_beamdyn_sections() {
        let sections = parse_beamdyn_sections(BD_INPUT2);
        println!("{:?}", sections);
    }

    const BD_INPUT1: &str = r#"--------- BEAMDYN with OpenFAST INPUT FILE -------------------------------------------
bar_urc blade
---------------------- SIMULATION CONTROL --------------------------------------
True          Echo             - Echo input data to "<RootName>.ech"? (flag)
False         QuasiStaticInit  - Use quasi-static pre-conditioning with centripetal accelerations in initialization? (flag) [dynamic solve only]
          0   rhoinf           - Numerical damping parameter for generalized-alpha integrator
          1   quadrature       - Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)
          1   refine           - Refinement factor for trapezoidal quadrature (-) [DEFAULT = 1; used only when quadrature=2]
"DEFAULT"     n_fact           - Factorization frequency for the Jacobian in N-R iteration(-) [DEFAULT = 5]
"DEFAULT"     DTBeam           - Time step size (s)
50     load_retries     - Number of factored load retries before quitting the simulation [DEFAULT = 20]
"DEFAULT"     NRMax            - Max number of iterations in Newton-Raphson algorithm (-) [DEFAULT = 10]
"DEFAULT"     stop_tol         - Tolerance for stopping criterion (-) [DEFAULT = 1E-5]
"DEFAULT"     tngt_stf_fd      - Use finite differenced tangent stiffness matrix? (flag)
"DEFAULT"     tngt_stf_comp    - Compare analytical finite differenced tangent stiffness matrix? (flag)
"DEFAULT"     tngt_stf_pert    - Perturbation size for finite differencing (-) [DEFAULT = 1E-6]
"DEFAULT"     tngt_stf_difftol - Maximum allowable relative difference between analytical and fd tangent stiffness (-); [DEFAULT = 0.1]
True          RotStates        - Orient states in the rotating frame during linearization? (flag) [used only when linearizing]
---------------------- GEOMETRY PARAMETER --------------------------------------
          1   member_total    - Total number of members (-)
          5   kp_total        - Total number of key points (-) [must be at least 3]
     1     5                  - Member number; Number of key points in this member
		 kp_xr 			 kp_yr 			 kp_zr 		 initial_twist
		  (m)  			  (m)  			  (m)  		   (deg)
	 0.00000e+00 	 0.00000e+00 	 0.00000e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 2.04082e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 4.08163e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 6.12245e+00 	 0.00000e+00
	 0.00000e+00 	 0.00000e+00 	 8.16327e+00 	 0.00000e+00 "#;

    const BD_INPUT2: &str =
        "------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------
! NACA 0012 airfoil with chord 0.1 - Written using beamdyn.py
---------------------- BLADE PARAMETERS --------------------------------------
   21  station_total    - Number of blade input stations (-)
    1  damp_type        - Damping type (switch): 0: no damping; 1: viscous damping
---------------------- DAMPING COEFFICIENT------------------------------------
   mu1        mu2        mu3        mu4        mu5        mu6
   (s)        (s)        (s)        (s)        (s)        (s)
 0.01 0.01 0.01 0.01 0.01 0.01
---------------------- DISTRIBUTED PROPERTIES---------------------------------
0.000000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03

0.050000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03

0.100000
   6.804663E-18 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 -1.549113E-28
   0.000000E+00 5.440283E-18 0.000000E+00 0.000000E+00 0.000000E+00 6.234980E-20
   0.000000E+00 0.000000E+00 1.715755E+08 -8.494318E-11 -2.880672E+06 0.000000E+00
   0.000000E+00 0.000000E+00 -8.494318E-11 1.134880E+05 -5.616727E+04 0.000000E+00
   0.000000E+00 0.000000E+00 -2.880672E+06 -5.616727E+04 7.793935E+04 0.000000E+00
   -1.549113E-28 6.234980E-20 0.000000E+00 0.000000E+00 0.000000E+00 2.120621E+03

   6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 3.175257E-18
   0.000000E+00 6.413657E+00 0.000000E+00 0.000000E+00 0.000000E+00 1.076823E-01
   0.000000E+00 0.000000E+00 6.413657E+00 -3.175257E-18 -1.076823E-01 0.000000E+00
   0.000000E+00 0.000000E+00 -3.175257E-18 6.776322E-09 5.331106E-20 0.000000E+00
   0.000000E+00 0.000000E+00 -1.076823E-01 5.331106E-20 1.808609E-03 0.000000E+00
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03";
}

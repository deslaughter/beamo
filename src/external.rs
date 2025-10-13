use core::panic;
use faer::prelude::*;
use itertools::Itertools;
use std::fs;

use crate::{
    components::beam::{BeamComponent, BeamInputBuilder},
    elements::beams::{BeamSection, Damping},
    model::Model,
    util::quat_from_rotation_vector_alloc,
};

pub fn add_beamdyn_blade(
    model: &mut Model,
    bd_primary_file_path: &str,
    bd_blade_file_path: &str,
    root_position: &[f64; 6],
) -> BeamComponent {
    // Read key points
    let bd_input = parse_beamdyn_primary_file(&fs::read_to_string(bd_primary_file_path).unwrap());

    // Read sections
    let bd_blade_input = parse_beamdyn_blade_file(&fs::read_to_string(bd_blade_file_path).unwrap());

    // Calculate key point position on range of [0,1]
    let distances = bd_input
        .key_points
        .iter()
        .tuple_windows()
        .map(|(kp0, kp1)| {
            ((kp1[0] - kp0[0]).powi(2) + (kp1[1] - kp0[1]).powi(2) + (kp1[2] - kp0[2]).powi(2))
                .sqrt()
        })
        .collect_vec();
    let kp_range = distances.iter().sum::<f64>();
    let mut kp_cumdist = vec![0.];
    for d in distances {
        kp_cumdist.push(kp_cumdist.last().unwrap() + d);
    }
    let kp_grid = kp_cumdist.iter().map(|&d| d / kp_range).collect_vec();

    let root_orientation = quat_from_rotation_vector_alloc(
        col![root_position[3], root_position[4], root_position[5]].as_ref(),
    );

    let input = BeamInputBuilder::new()
        .set_element_order(bd_input.elem_order)
        .set_damping(bd_blade_input.damping)
        .set_section_refinement(bd_input.refinement - 1)
        .set_reference_axis_z(
            &kp_grid,
            &bd_input
                .key_points
                .iter()
                .map(|kp| [kp[0], kp[1], kp[2]])
                .collect_vec(),
            &kp_grid,
            &bd_input.key_points.iter().map(|kp| kp[3]).collect_vec(),
        )
        .set_sections_z(&bd_blade_input.sections)
        .set_root_position([
            root_position[0],
            root_position[1],
            root_position[2],
            root_orientation[0],
            root_orientation[1],
            root_orientation[2],
            root_orientation[3],
        ])
        .build();

    BeamComponent::new(&input, model)
}

#[derive(Debug)]
pub struct BeamDynInput {
    pub key_points: Vec<[f64; 4]>,
    pub refinement: usize,
    pub elem_order: usize,
}

pub fn parse_beamdyn_primary_file(file_data: &str) -> BeamDynInput {
    let lines = file_data.lines().collect_vec();

    let refinement_line = lines.get(7).unwrap();
    let refinement: usize = refinement_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

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
    let key_points = lines
        .iter()
        .skip(24)
        .take(kp_total)
        .map(|l| {
            let v = l
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect_vec();
            [v[0], v[1], v[2], v[3]]
        })
        .collect_vec();

    let elem_order_line = lines.get(25 + kp_total).unwrap();
    let elem_order: usize = elem_order_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    BeamDynInput {
        key_points,
        refinement,
        elem_order,
    }
}

#[derive(Debug)]
pub struct BeamDynBladeInput {
    pub sections: Vec<BeamSection>,
    pub damping: Damping,
}

pub fn parse_beamdyn_blade_file(file_data: &str) -> BeamDynBladeInput {
    let damping_type_line = file_data.lines().nth(4).unwrap();
    let damping_type: usize = damping_type_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    let damping_coefficients_line = file_data.lines().nth(8).unwrap();
    let damping_coefficients: [f64; 6] = damping_coefficients_line
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect_vec()
        .try_into()
        .unwrap();

    let station_total_line = file_data.lines().nth(3).unwrap();
    let n_stations: usize = station_total_line
        .split_whitespace()
        .collect_vec()
        .first()
        .unwrap()
        .parse()
        .unwrap();

    let section_lines = file_data
        .lines()
        .skip(10)
        .take(n_stations * 15)
        .collect_vec();
    let sections = section_lines
        .chunks(15)
        .map(|chunk| {
            let s = chunk[0].trim().parse::<f64>().unwrap();
            let c = (1..7)
                .map(|i| {
                    chunk[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            let m = (8..14)
                .map(|i| {
                    chunk[i]
                        .split_ascii_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect_vec()
                })
                .collect_vec();
            BeamSection {
                s,
                m_star: Mat::<f64>::from_fn(6, 6, |i, j| m[i][j]),
                c_star: Mat::<f64>::from_fn(6, 6, |i, j| c[i][j]),
            }
        })
        .collect_vec();

    BeamDynBladeInput {
        sections: sections,
        damping: match damping_type {
            0 => Damping::None,
            1 => Damping::Mu(col![
                damping_coefficients[0],
                damping_coefficients[1],
                damping_coefficients[2],
                damping_coefficients[3],
                damping_coefficients[4],
                damping_coefficients[5]
            ]),
            _ => panic!("Unsupported damping type"),
        },
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_beamdyn_keypoints() {
        let inp = parse_beamdyn_primary_file(BD_INPUT1);
        assert_eq!(inp.key_points.len(), 5);
        assert_eq!(inp.refinement, 1);
    }

    #[test]
    fn test_parse_beamdyn_blade_file() {
        let inp = parse_beamdyn_blade_file(BD_INPUT2);
        assert_eq!(inp.sections.len(), 3);
        assert!(matches!(inp.damping, Damping::Mu(_)));
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
	 0.00000e+00 	 0.00000e+00 	 8.16327e+00 	 0.00000e+00
---------------------- MESH PARAMETER ------------------------------------------
          10   order_elem     - Order of interpolation (basis) function (-)
---------------------- MATERIAL PARAMETER -------------------------------------- "#;

    const BD_INPUT2: &str =
        "------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------
! NACA 0012 airfoil with chord 0.1 - Written using beamdyn.py
---------------------- BLADE PARAMETERS --------------------------------------
    3  station_total    - Number of blade input stations (-)
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
   3.175257E-18 1.076823E-01 0.000000E+00 0.000000E+00 0.000000E+00 1.808616E-03

";
}

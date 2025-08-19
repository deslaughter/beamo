use std::f64::consts::PI;

use faer::{linalg::matmul::matmul, prelude::*, Accum};
use itertools::{multizip, Itertools};

use crate::{
    interp::{shape_deriv_matrix, shape_interp_matrix},
    node::Node,
    state::State,
    util::{
        cross_product, quat_compose, quat_from_tangent_twist_alloc, quat_inverse,
        quat_rotate_vector, quat_rotate_vector_alloc,
    },
};

pub struct AeroComponent {
    inflow: Inflow, // Inflow velocity vector
    elems: Vec<AeroElement>,
}

impl AeroComponent {
    pub fn new(elems: &[AeroElementInput], inflow: Inflow, nodes: &[Node]) -> Self {
        AeroComponent {
            inflow,
            elems: elems
                .iter()
                .map(|elem| {
                    // Number of motion points in element
                    let n_motion = elem.aero_points.len();

                    // Number of force points in the element
                    // TODO: decouple from motion points
                    let n_force = n_motion;

                    // Number of nodes in beam element
                    let n_nodes = elem.beam_node_ids.len();

                    //----------------------------------------------------------
                    // Beam node and aero point locations on beam reference axis
                    //----------------------------------------------------------

                    // Get motion point location along beam (0-1)
                    let motion_point_xi = elem
                        .aero_points
                        .iter()
                        .map(|point| 2. * point.s - 1.)
                        .collect_vec();

                    // Get location of node along beam (0-1)
                    let beam_node_xi = elem
                        .beam_node_ids
                        .iter()
                        .map(|&id| 2. * nodes[id].s - 1.)
                        .collect_vec();

                    //----------------------------------------------------------
                    // Motion point locations on beam reference axis
                    //----------------------------------------------------------

                    // Calculate shape function interpolation matrix to map from beam to aero points
                    let mut motion_interp =
                        Mat::<f64>::zeros(beam_node_xi.len(), motion_point_xi.len());
                    shape_interp_matrix(&beam_node_xi, &motion_point_xi, motion_interp.as_mut());

                    // Get node reference position (translation only)
                    let node_x =
                        Mat::from_fn(3, n_nodes, |i, j| nodes[elem.beam_node_ids[j]].xr[i]);

                    // Interpolate node positions to aerodynamic locations on the beam reference axis
                    let mut xr = Mat::zeros(3, n_motion);
                    matmul(
                        xr.transpose_mut(),
                        Accum::Replace,
                        &motion_interp,
                        node_x.transpose(),
                        1.,
                        Par::Seq,
                    );

                    //----------------------------------------------------------
                    // Motion point tangents on beam reference axis
                    //----------------------------------------------------------

                    // Calculate shape function derivative matrix to map from beam to motion points
                    let mut shape_deriv_node =
                        Mat::<f64>::zeros(beam_node_xi.len(), motion_point_xi.len());
                    shape_deriv_matrix(&beam_node_xi, &motion_point_xi, shape_deriv_node.as_mut());

                    // Calculate spatial derivative of reference axis at motion locations
                    let mut x_tan = Mat::zeros(7, n_motion);
                    matmul(
                        x_tan.transpose_mut(),
                        Accum::Replace,
                        &shape_deriv_node,
                        node_x.transpose(),
                        1.,
                        Par::Seq,
                    );
                    x_tan.col_iter_mut().for_each(|mut col| {
                        let m = col.norm_l2();
                        if m > f64::EPSILON {
                            col /= m;
                        }
                    });

                    //----------------------------------------------------------
                    // Vector from motion point on beam reference axis to aerodynamic center
                    // The beam is initialized with:
                    //   - root at (0, 0, 0)
                    //   - reference axis aligned with the x-axis
                    //   - the trailing edge is towards +y-axis
                    //   - the suction side of the airfoil is towards +z-axis
                    // x_r translates and rotates the beam into the reference position
                    //----------------------------------------------------------

                    // Calculate vector from beam reference to aerodynamic center
                    let ac_vec = elem
                        .aero_points
                        .iter()
                        .zip(x_tan.col_iter())
                        .map(|(node, tan)| {
                            calculate_ac_vector(
                                node.section_offset_y - node.aerodynamic_center,
                                node.section_offset_x,
                                node.twist,
                                [tan[0], tan[1], tan[2]],
                            )
                        })
                        .collect_vec();

                    //----------------------------------------------------------
                    // Aero point widths
                    //----------------------------------------------------------

                    // Calculate jacobian xi locations for width integration
                    let jacobian_xi = calculate_jacobian_xi(&motion_point_xi);

                    // Calculate shape derivative matrix to calculate jacobians
                    let mut jacobian_integration_matrix =
                        Mat::<f64>::zeros(jacobian_xi.len(), beam_node_xi.len());
                    shape_deriv_matrix(
                        &jacobian_xi,
                        &beam_node_xi,
                        jacobian_integration_matrix.as_mut(),
                    );

                    // Calculate aero point widths
                    let ds = calculate_aero_node_widths(
                        &jacobian_xi,
                        jacobian_integration_matrix.as_ref(),
                        node_x.as_ref(),
                    );

                    //----------------------------------------------------------
                    // Force point locations on beam reference axis
                    //----------------------------------------------------------

                    // For now, assume that they're the same as the aero points,
                    // this can be changed for blade resolved
                    let force_point_xi = motion_point_xi.clone();

                    // Calculate shape function interpolation matrix to distribute
                    // loads from force points to beam nodes
                    let mut force_interp =
                        Mat::<f64>::zeros(force_point_xi.len(), beam_node_xi.len());
                    shape_interp_matrix(&force_point_xi, &beam_node_xi, force_interp.as_mut());

                    //----------------------------------------------------------
                    // Polars
                    //----------------------------------------------------------

                    // Max number of polar points in any aero point.
                    // All polars at the same point share the same angle of attack grid
                    let n_polar_points_max = elem
                        .aero_points
                        .iter()
                        .map(|p| p.aoa.len())
                        .max()
                        .unwrap_or_default();

                    //----------------------------------------------------------
                    // Populate element structure
                    //----------------------------------------------------------

                    AeroElement {
                        id: elem.id,
                        node_ids: elem.beam_node_ids.clone(),
                        node_u: Mat::zeros(7, n_nodes),
                        node_v: Mat::zeros(6, n_nodes),
                        node_f: Mat::zeros(6, n_nodes),

                        xr_motion_map: xr.cloned(),
                        u_motion_map: Mat::zeros(7, n_motion),
                        v_motion_map: Mat::zeros(6, n_motion),
                        qqr_motion_map: Mat::zeros(4, n_motion),
                        con_motion: Mat::from_fn(3, n_motion, |i, j| ac_vec[j][i]),
                        x_motion: Mat::zeros(3, n_motion),
                        v_motion: Mat::zeros(3, n_motion),

                        v_rel: Mat::zeros(3, n_motion),
                        v_inflow: Mat::zeros(3, n_motion),
                        twist: Col::from_fn(n_motion, |i| elem.aero_points[i].twist),
                        chord: Col::from_fn(n_motion, |i| elem.aero_points[i].chord),
                        ds: Col::from_iter(ds.into_iter()),
                        n_polar_points: elem.aero_points.iter().map(|p| p.aoa.len()).collect(),
                        aoa: Mat::from_fn(n_polar_points_max, n_motion, |i, j| {
                            if i < elem.aero_points[j].aoa.len() {
                                elem.aero_points[j].aoa[i].to_radians()
                            } else {
                                0. // Fill with zero if not enough points
                            }
                        }),
                        cl: Mat::from_fn(n_polar_points_max, n_motion, |i, j| {
                            if i < elem.aero_points[j].cl.len() {
                                elem.aero_points[j].cl[i]
                            } else {
                                0. // Fill with zero if not enough points
                            }
                        }),
                        cd: Mat::from_fn(n_polar_points_max, n_motion, |i, j| {
                            if i < elem.aero_points[j].cd.len() {
                                elem.aero_points[j].cd[i]
                            } else {
                                0. // Fill with zero if not enough points
                            }
                        }),
                        cm: Mat::from_fn(n_polar_points_max, n_motion, |i, j| {
                            if i < elem.aero_points[j].cm.len() {
                                elem.aero_points[j].cm[i]
                            } else {
                                0. // Fill with zero if not enough points
                            }
                        }),

                        load: Mat::zeros(6, n_force),

                        motion_interp,
                        force_interp: force_interp,
                        shape_deriv_jac: jacobian_integration_matrix,
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Inflow {
    typ: InflowType,
    uniform_flow: UniformFlow,
}

#[derive(Debug, Clone)]
pub enum InflowType {
    Uniform = 1,
}

#[derive(Debug, Clone)]
pub struct UniformFlow {
    pub time: Vec<f64>, // Time vector for uniform flow parameters
    pub data: Vec<UniformFlowParameters>,
}

impl UniformFlow {
    pub fn velocity(&self, t: f64, position: [f64; 3]) -> [f64; 3] {
        match self.time.len() {
            1 => self.data[0].velocity(position),
            _ => unreachable!("Time-dependent uniform flow not implemented"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniformFlowParameters {
    pub velocity_horizontal: f64,   // Horizontal inflow velocity (m/s)
    pub height_reference: f64,      // Reference height (m)
    pub shear_vertical: f64,        // Vertical shear exponent
    pub flow_angle_horizontal: f64, // Flow angle relative to x axis (radians)
}

impl UniformFlowParameters {
    pub fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        // Calculate horizontal velocity
        let vh = self.velocity_horizontal
            * (position[2] / self.height_reference).powf(self.shear_vertical);

        // Get sin and cos of flow angle
        let (sin_flow_angle, cos_flow_angle) = self.flow_angle_horizontal.sin_cos();

        // Apply horizontal direction
        [vh * cos_flow_angle, -vh * sin_flow_angle, 0.]
    }
}

impl Inflow {
    pub fn steady_wind(
        velocity_horizontal: f64,
        height_reference: f64,
        shear_vertical: f64,
        flow_angle_horizontal: f64,
    ) -> Self {
        Inflow {
            typ: InflowType::Uniform,
            uniform_flow: UniformFlow {
                time: vec![0.],
                data: vec![UniformFlowParameters {
                    velocity_horizontal,
                    height_reference,
                    shear_vertical,
                    flow_angle_horizontal,
                }],
            },
        }
    }
    pub fn velocity(&self, t: f64, position: [f64; 3]) -> [f64; 3] {
        match self.typ {
            InflowType::Uniform => self.uniform_flow.velocity(t, position),
        }
    }
}

pub struct AeroElementInput {
    pub id: usize,                   // Element ID
    pub beam_node_ids: Vec<usize>,   // Node IDs for the beam element
    pub aero_points: Vec<AeroPoint>, // Aero points along beam axis
}

#[derive(Debug, Clone)]
pub struct AeroPoint {
    pub id: usize,
    pub s: f64,                  // Position along beam element length [0-1]
    pub chord: f64,              // Chord length at each node
    pub section_offset_x: f64,   // Section offset in x-direction
    pub section_offset_y: f64,   // Section offset in y-direction
    pub twist: f64,              // Twist angle at each node
    pub aerodynamic_center: f64, // Aerodynamic center distance from the leading edge
    pub aoa: Vec<f64>,           // Angle of attack for all polars
    pub cl: Vec<f64>,            // Lift coefficient polar
    pub cd: Vec<f64>,            // Drag coefficient polar
    pub cm: Vec<f64>,            // Moment coefficient polar
}

pub struct AeroElement {
    pub id: usize, // Element ID

    // Beam
    pub node_ids: Vec<usize>, // Beam node IDs for this aero element
    pub node_u: Mat<f64>,     // Beam node displacements `[7][n_nodes]`
    pub node_v: Mat<f64>,     // Beam node velocities `[6][n_nodes]`
    pub node_f: Mat<f64>,     // Beam node forces `[6][n_nodes]`

    // Motion
    pub xr_motion_map: Mat<f64>, // Motion map reference position `[7][n_motion]`
    pub u_motion_map: Mat<f64>,  // Motion map displacements `[7][n_motion]`
    pub v_motion_map: Mat<f64>,  // Motion map velocities `[6][n_motion]`
    pub qqr_motion_map: Mat<f64>, // Motion map rotation position `[4][n_motion]`
    pub con_motion: Mat<f64>,    // Motion connectivity vector `[3][n_motion]`
    pub x_motion: Mat<f64>,      // Motion aerodynamic center position `[3][n_motion]`
    pub v_motion: Mat<f64>,      // Motion aerodynamic center velocity `[3][n_motion]`

    // Blade Element theory
    pub v_inflow: Mat<f64>, // Fluid velocity at each motion point `[3][n_motion]`
    pub v_rel: Mat<f64>,    // Relative wind velocity at AC
    pub twist: Col<f64>,    // Twist angle at each aero point (radians)
    pub chord: Col<f64>,    // Chord length at each aero point (meters)
    pub ds: Col<f64>,       // Width of aerodynamic node
    pub n_polar_points: Vec<usize>, // Number of polar points at each aero point
    pub aoa: Mat<f64>,      // Angle of attack for all polars
    pub cl: Mat<f64>,       // Lift coefficient polar
    pub cd: Mat<f64>,       // Drag coefficient polar
    pub cm: Mat<f64>,       // Moment coefficient polar

    // Forces
    pub load: Mat<f64>, // Aerodynamic loads at force points `[3][n_force]`

    // Interpolation and derivative matrices
    pub motion_interp: Mat<f64>, // Shape function matrix `[n_motion][n_nodes]`
    pub force_interp: Mat<f64>,  // Shape function matrix `[n_nodes][n_force]`
    pub shape_deriv_jac: Mat<f64>, // Shape function derivative matrix `[n_jac][n_nodes]`
}

impl AeroElement {
    /// Calculate the aerodynamic center motion at each motion point
    pub fn calculate_motion(&mut self, state: &State) {
        // Copy beam node displacements from state
        self.node_ids.iter().enumerate().for_each(|(i, &id)| {
            self.node_u.col_mut(i).copy_from(&state.x.col(id));
        });

        // Copy beam node velocities from state
        self.node_ids.iter().enumerate().for_each(|(i, &id)| {
            self.node_v.col_mut(i).copy_from(&state.v.col(id));
        });

        // Interpolate beam node displacement to motion locations on the reference axis
        matmul(
            self.u_motion_map.transpose_mut(),
            Accum::Replace,
            &self.motion_interp,
            self.node_u.transpose(),
            1.,
            Par::Seq,
        );

        // Normalize displacement rotation quaternions
        self.u_motion_map
            .subrows_mut(3, 4)
            .col_iter_mut()
            .for_each(|mut c| {
                let m = c.norm_l2();
                if m < f64::EPSILON {
                    c.fill(0.);
                } else {
                    c /= m;
                }
            });

        // Interpolate beam node velocities to motion locations on the reference axis
        matmul(
            self.v_motion_map.transpose_mut(),
            Accum::Replace,
            &self.motion_interp,
            self.node_v.transpose(),
            1.,
            Par::Seq,
        );

        // Calculate total rotation from material coordinates
        multizip((
            self.qqr_motion_map.col_iter_mut(),
            self.xr_motion_map.subrows(3, 4).col_iter(),
            self.u_motion_map.subrows(3, 4).col_iter(),
        ))
        .for_each(|(mut qqr, qr, q)| {
            quat_compose(q.as_ref(), qr.as_ref(), qqr.as_mut());
        });

        // Calculate motion of aerodynamic centers in global coordinates
        multizip((
            self.x_motion.subrows_mut(0, 3).col_iter_mut(), // AC position
            self.v_motion.subrows_mut(0, 3).col_iter_mut(), // AC position
            self.xr_motion_map.subrows(0, 3).col_iter(),    // Axis reference position
            self.qqr_motion_map.col_iter(),                 // Axis rotation displacement
            self.u_motion_map.subrows(0, 3).col_iter(),     // Axis translation displacement
            self.v_motion_map.subrows(0, 3).col_iter(),     // Axis translation velocity
            self.v_motion_map.subrows(3, 4).col_iter(),     // Axis rotation velocity
            self.con_motion.col_iter(),                     // Vector from axis to AC
        ))
        .for_each(|(mut x, mut v, xr, qqr, u, v_mm, omega, con)| {
            // Rotate connection vector from material coordinates
            let mut qqr_con = col![0., 0., 0.];
            quat_rotate_vector(qqr.as_ref(), con.as_ref(), qqr_con.as_mut());

            // Calculate current position of aerodynamic center
            x.copy_from(&xr + &u + &qqr_con);

            // Calculate current velocity of aerodynamic center
            let mut omega_qqr_con = col![0., 0., 0.];
            cross_product(omega, qqr_con.as_ref(), omega_qqr_con.as_mut());
            v.copy_from(&v_mm + &omega_qqr_con);
        });
    }

    pub fn calculate_inflow(&mut self, t: f64, inflow: &Inflow) {
        multizip((self.v_inflow.col_iter_mut(), self.x_motion.col_iter())).for_each(
            |(mut v_inflow, x)| {
                let v = inflow.velocity(t, [x[0], x[1], x[2]]);
                v_inflow.copy_from(&col![v[0], v[1], v[2]]);
            },
        );
    }

    fn calculate_blade_element_force(&mut self, fluid_density: f64) {
        // Calculate relative velocity for each motion point
        multizip((
            self.v_rel.col_iter_mut(),                    // Relative velocity
            self.v_inflow.col_iter(),                     // Fluid velocity (inflow)
            self.v_motion.subrows_mut(0, 3).col_iter(),   // AC velocity
            self.qqr_motion_map.subrows(3, 4).col_iter(), // Axis reference rotation
        ))
        .for_each(|(v_rel, v_fl, v_ac, qqr)| {
            calculate_relative_velocity(v_rel, v_fl, v_ac, qqr);
        });

        multizip((
            self.load.col_iter_mut(),
            self.v_rel.col_iter(),
            self.n_polar_points.iter(),
            self.aoa.col_iter(),
            self.cl.col_iter(),
            self.cd.col_iter(),
            self.cm.col_iter(),
            self.chord.iter(),
            self.twist.iter(),
            self.ds.iter(),
            self.qqr_motion_map.col_iter(),
        ))
        .for_each(
            |(
                load,
                v_rel,
                &n_polar_points,
                aoa_polar,
                cl_polar,
                cd_polar,
                cm_polar,
                &twist,
                &chord,
                &delta_s,
                qqr,
            )| {
                calculate_aerodynamic_load(
                    load,
                    v_rel,
                    n_polar_points,
                    aoa_polar,
                    cl_polar,
                    cd_polar,
                    cm_polar,
                    twist,
                    chord,
                    delta_s,
                    fluid_density,
                    qqr,
                )
            },
        );
    }
}

fn calculate_relative_velocity(
    mut v_rel: ColMut<f64>,
    v_inflow: ColRef<f64>,
    v_motion: ColRef<f64>,
    qqr: ColRef<f64>,
) {
    // Inverse of qqr
    let mut qqr_inv = col![0., 0., 0., 0.];
    quat_inverse(qqr.as_ref(), qqr_inv.as_mut());

    // Calculate difference between inflow velocity and aerodynamic center velocity
    let v_diff = &v_inflow - &v_motion;
    let mut v_diff_rot = col![0., 0., 0.];
    quat_rotate_vector(qqr_inv.as_ref(), v_diff.as_ref(), v_diff_rot.as_mut());

    // Relative velocity only considers flow in the y-z plane
    v_rel[1] = v_diff_rot[1];
    v_rel[2] = v_diff_rot[2];
}

fn calculate_aerodynamic_load(
    mut load: ColMut<f64>,
    v_rel: ColRef<f64>,
    n_polar_points: usize,
    aoa_polar: ColRef<f64>,
    cl_polar: ColRef<f64>,
    cd_polar: ColRef<f64>,
    cm_polar: ColRef<f64>,
    twist: f64,
    chord: f64,
    delta_s: f64,
    fluid_density: f64,
    qqr: ColRef<f64>,
) {
    // Calculate beta
    let beta = if v_rel[1] < 0. {
        2. * PI - (v_rel[1] / v_rel.norm_l2()).acos()
    } else {
        (v_rel[1] / v_rel.norm_l2()).acos()
    };

    // Calculate angle of attack
    let aoa = beta - twist;

    // Find index of angle of attack in aoa vector
    let i_aoa = (0..n_polar_points - 1)
        .position(|i| aoa > aoa_polar[i] && aoa <= aoa_polar[i + 1])
        .unwrap();

    // Calculate blending term between values above and below
    let alpha = (aoa - aoa_polar[i_aoa]) / (aoa_polar[i_aoa + 1] - aoa_polar[i_aoa]);

    // Calculate lift and drag coefficients using linear interpolation
    let cl = (1. - alpha) * cl_polar[i_aoa] + alpha * cl_polar[i_aoa + 1];
    let cd = (1. - alpha) * cd_polar[i_aoa] + alpha * cd_polar[i_aoa + 1];
    let cm = (1. - alpha) * cm_polar[i_aoa] + alpha * cm_polar[i_aoa + 1];

    // Calculate force and moment in local coordinates
    let dynamic_pressure = 0.5 * fluid_density * v_rel.norm_l2().powi(2);
    let (twist_sin, twist_cos) = twist.sin_cos();
    let force_local = col![
        0.,
        (cd * twist_cos - cl * twist_sin) * dynamic_pressure * chord * delta_s,
        (cd * twist_sin - cl * twist_cos) * dynamic_pressure * chord * delta_s,
    ];
    let moment_local = col![cm * dynamic_pressure * chord.powi(2) * delta_s, 0., 0.];

    // Rotate force and moment into global coordinates
    quat_rotate_vector(
        qqr.as_ref(),
        force_local.as_ref(),
        load.rb_mut().subrows_mut(0, 3),
    );
    quat_rotate_vector(
        qqr.as_ref(),
        moment_local.as_ref(),
        load.rb_mut().subrows_mut(3, 3),
    );
}

/// Returns a vector of positions [0-1] where the jacobian is calculated for
/// integration of the reference axis length. The integration is done using
/// Simpson's rule, which requires at least three points per segment. Therefore,
/// points are added at the midpoints between each aero point and an additional
/// point is added after the first point and before the last point and the
/// next respective midpoint.
fn calculate_jacobian_xi(aero_node_xi: &[f64]) -> Vec<f64> {
    // Number of aero nodes
    let n_aero = aero_node_xi.len();

    // Number of xi locations for integration
    let n_jacobian = 2 * n_aero + 1;

    // Calculate xi locations for integration
    let mut jacobian_xi = vec![0.; n_jacobian];
    aero_node_xi.windows(2).enumerate().for_each(|(i, w)| {
        jacobian_xi[2 * i + 2] = 0.5 * (w[0] + w[1]);
        jacobian_xi[2 * i + 3] = w[1];
    });
    jacobian_xi[0] = aero_node_xi[0];
    jacobian_xi[1] = (3. * aero_node_xi[0] + aero_node_xi[1]) / 4.;
    jacobian_xi[n_jacobian - 2] = (aero_node_xi[n_aero - 2] + 3. * aero_node_xi[n_aero - 1]) / 4.;
    jacobian_xi[n_jacobian - 1] = aero_node_xi[n_aero - 1];

    jacobian_xi
}

/// Return a vector of widths for each aero node calculated using the
/// Jacobian vector and Simpson's Rule.
fn calculate_aero_node_widths(
    xi: &[f64],
    jacobian_integration_matrix: MatRef<f64>,
    node_x: MatRef<f64>,
) -> Vec<f64> {
    // Calculate un-normalized derivative vectors
    let tan = jacobian_integration_matrix * node_x.transpose();

    // Calculate jacobian values as magnitude of derivative vectors
    let j = tan
        .transpose()
        .col_iter()
        .map(|col| col.norm_l2())
        .collect_vec();

    // Calculate widths using Simpson's rule
    // https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    (0..j.len())
        .tuple_windows::<(_, _, _)>()
        .enumerate()
        .filter_map(|(i, (j0, j1, j2))| {
            if i % 2 == 0 {
                let h1 = xi[j1] - xi[j0];
                let h2 = xi[j2] - xi[j1];
                let width = (h1 + h2) / 6.
                    * ((2. - h2 / h1) * j[j0]
                        + ((h1 + h2).powi(2) / (h1 * h2)) * j[j1]
                        + (2. - h1 / h2) * j[j2]);
                Some(width)
            } else {
                None
            }
        })
        .collect()
}

/// Calculates vector from aero point on beam reference line to aerodynamic center.
///
/// # Arguments
///
/// * `ac_to_ref_axis_horizontal` - Horizontal distance from leading edge to reference axis (+ towards leading edge)
/// * `chord_to_ref_axis_vertical` - Vertical distance from chord line to reference axis (+ towards suction side)
/// * `twist` - Rotation about (0, 0)
/// * `tangent` - Tangent vector of reference line at the aero point (orientation of blade in 3D space)
fn calculate_ac_vector(
    ac_to_ref_axis_horizontal: f64,
    chord_to_ref_axis_vertical: f64,
    twist: f64,
    tangent: [f64; 3],
) -> [f64; 3] {
    let q = quat_from_tangent_twist_alloc(col![tangent[0], tangent[1], tangent[2]].as_ref(), twist);
    let v_base = col![0., -ac_to_ref_axis_horizontal, chord_to_ref_axis_vertical];
    let v_rot = quat_rotate_vector_alloc(q.as_ref(), v_base.as_ref());
    [v_rot[0], v_rot[1], v_rot[2]]
}

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use std::vec;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_calculate_aerodynamic_load() {
        // Polars
        let n_polar_points = 3;
        let aoa_polar = col![-1.0, 1.0]; // Angle of attack points
        let cl_polar = col![0.0, 1.0]; // Lift coefficient values
        let cd_polar = col![0.5, 0.0]; // Drag coefficient values
        let cm_polar = col![0.01, 0.03]; // Moment coefficient values

        struct Data {
            v_rel: Col<f64>,
            twist: f64,
            chord: f64,
            delta_s: f64,
            fluid_density: f64,
            qqr: Col<f64>,
            load: Col<f64>,
        }

        let test_cases = vec![
            // AoA = 0
            // twist = 0
            // cl = 0.5
            // cd = 0.25
            // cm = 0.02
            // dynamic pressure = 0.5 * 1.225 * 10 * 10
            // force  = [0., 0.5 * 1.225 * 10 * 10 * 0.25 * 2. * 1.5, -0.5 * 1.225 * 10 * 10 * 0.5 * 2. * 1.5]
            //        = [0., 45.9375, -91.875]
            // moment = [0.5 * 1.225 * 10 * 10 * 0.02 * 2. * 2. * 1.5, 0., 0.]
            //        = [7.35, 0., 0.]
            // No rotation from local to global
            // Data {
            //     v_rel: col![0.0, 10.0, 0.0],
            //     twist: 0.0,
            //     chord: 2.0,
            //     delta_s: 1.5,
            //     fluid_density: 1.225,
            //     qqr: col![1.0, 0.0, 0.0, 0.0],
            //     load: col![0.0, 45.9375, -91.875, 7.35, 0.0, 0.0],
            // },
            // twist = 0.0
            // v_rel = [0.0, 10*cos(0.1), 10*sin(0.1)] = [0.0, 9.950041652780259, 0.9983341664682815]
            // AoA = 0.1
            // cl = 0.55
            // cd = 0.225
            // cm = 0.021
            // dynamic pressure = 0.5 * 1.225 * 10 * 10
            // force  = [0., 0.5 * 1.225 * 10 * 10 * 0.225 * 2. * 1.5, -0.5 * 1.225 * 10 * 10 * 0.55 * 2. * 1.5]
            //        = [0., 41.34375, -101.0625]
            // moment = [0.5 * 1.225 * 10 * 10 * 0.021 * 2. * 2. * 1.5, 0., 0.]
            //        = [7.7175, 0., 0.]
            // No rotation from local to global
            Data {
                v_rel: col![0.0, 9.950041652780259, 0.9983341664682815],
                twist: 0.0,
                chord: 2.0,
                delta_s: 1.5,
                fluid_density: 1.225,
                qqr: col![1.0, 0.0, 0.0, 0.0],
                load: col![0., 41.34375, -101.0625, 7.7175, 0.0, 0.0],
            },
        ];

        for case in test_cases {
            let mut load = col![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Force and moment output

            calculate_aerodynamic_load(
                load.as_mut(),
                case.v_rel.as_ref(),
                n_polar_points,
                aoa_polar.as_ref(),
                cl_polar.as_ref(),
                cd_polar.as_ref(),
                cm_polar.as_ref(),
                case.twist,
                case.chord,
                case.delta_s,
                case.fluid_density,
                case.qqr.as_ref(),
            );

            assert_relative_eq!(load[0], case.load[0], epsilon = 1e-10);
            assert_relative_eq!(load[1], case.load[1], epsilon = 1e-10);
            assert_relative_eq!(load[2], case.load[2], epsilon = 1e-10);
            assert_relative_eq!(load[3], case.load[3], epsilon = 1e-10);
            assert_relative_eq!(load[4], case.load[4], epsilon = 1e-10);
            assert_relative_eq!(load[5], case.load[5], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_calculate_ac_vector() {
        struct Data {
            ac_to_ref_axis_horizontal: f64,
            chord_to_ref_axis_vertical: f64,
            twist: f64,
            tangent: [f64; 3],
            ac_vec_exp: [f64; 3],
        }

        let test_cases = vec![
            // AC 1m from straight reference axis, 0 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 0.0_f64.to_radians(),
                tangent: [1.0, 0.0, 0.0],
                ac_vec_exp: [0.0, -1.0, 0.0],
            },
            // AC 1m from straight reference axis, 90 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 90.0_f64.to_radians(),
                tangent: [1.0, 0.0, 0.0],
                ac_vec_exp: [0.0, 0.0, -1.0],
            },
            // AC 0.5m from straight reference axis, -90 twist
            Data {
                ac_to_ref_axis_horizontal: 0.5,
                chord_to_ref_axis_vertical: 0.0,
                twist: -90.0_f64.to_radians(),
                tangent: [1.0, 0.0, 0.0],
                ac_vec_exp: [0.0, 0.0, 0.5],
            },
            // AC 1m from reference axis curved in x-z plane toward inflow, 0 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 0.0_f64.to_radians(),
                tangent: [0.8, 0.0, 0.6],
                ac_vec_exp: [0.0, -1.0, 0.0],
            },
            // AC 1m from reference axis curved in x-y plane toward TE, 0 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 0.0_f64.to_radians(),
                tangent: [0.8, 0.6, 0.0],
                ac_vec_exp: [0.6, -0.8, 0.0],
            },
            // AC 1m from reference axis curved in x-y plane toward TE, 90 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 90.0_f64.to_radians(),
                tangent: [0.8, 0.6, 0.0],
                ac_vec_exp: [0.0, 0.0, -1.0],
            },
            // AC 1m from reference axis curved in x-y plane toward TE, 30 twist
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                twist: 30.0_f64.to_radians(),
                tangent: [0.8, 0.6, 0.0],
                ac_vec_exp: [0.51961524227066314, -0.69282032302755081, -0.5],
            },
        ];

        for case in test_cases {
            let ac_vec = calculate_ac_vector(
                case.ac_to_ref_axis_horizontal,
                case.chord_to_ref_axis_vertical,
                case.twist,
                case.tangent,
            );
            ac_vec
                .into_iter()
                .zip(case.ac_vec_exp.iter())
                .for_each(|(ac, &exp)| {
                    assert_relative_eq!(ac, exp, epsilon = 1e-12);
                });
        }
    }

    #[test]
    fn test_calculate_aero_node_widths_straight() {
        let beam_node_xi = vec![-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let node_x: Mat<f64> = mat![
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0., 0.],
            [3., 0., 0.],
            [4., 0., 0.],
            [5., 0., 0.],
            [6., 0., 0.],
            [7., 0., 0.],
            [8., 0., 0.]
        ]
        .transpose()
        .to_owned();

        // Calculate the jacobian xi locations
        let aero_node_xi = vec![-1.0, 0., 1.0];
        let jacobian_xi = calculate_jacobian_xi(&aero_node_xi);

        // Calculate the jacobian integration matrix (shape function derivative)
        // Rows are from jacobian xi and columns are from beam xi
        let mut jacobian_integration_matrix =
            Mat::<f64>::zeros(jacobian_xi.len(), beam_node_xi.len());
        shape_deriv_matrix(
            &jacobian_xi,
            &beam_node_xi,
            jacobian_integration_matrix.as_mut(),
        );

        // Calculate the aerodynamic node widths
        let widths = calculate_aero_node_widths(
            &jacobian_xi,
            jacobian_integration_matrix.as_ref(),
            node_x.as_ref(),
        );

        // Verify the widths are correct
        widths
            .iter()
            .zip(vec![0.25 * 8., 0.5 * 8., 0.25 * 8.].iter())
            .for_each(|(act, exp)| {
                assert_relative_eq!(act, exp, epsilon = 1e-12);
            });
    }

    // import numpy as np
    // from numpy.polynomial.polynomial import Polynomial
    // x = [0, 1, 2]
    // y = [0, 1, 0.5]
    // c = np.polyfit(x, y, 2)
    // p = Polynomial(c[::-1])
    // for bounds in [(0, 0.25), (0.25, 0.75), (0.75, 1.25), (1.25, 1.75), (1.75, 2)]:
    //     xx = np.linspace(bounds[0], bounds[1], 1000)
    //     yy = p(xx)
    //     curve_length = np.sum(np.sqrt(np.diff(xx)**2 + np.diff(yy)**2))
    //     print(curve_length)
    #[test]
    fn test_calculate_aero_node_widths_curved() {
        let beam_node_xi = vec![-1.0, 0.0, 1.0];
        let node_x: Mat<f64> = mat![[0., 0., 0.], [1., 1., 0.], [2., 0.5, 0.],]
            .transpose()
            .to_owned();

        // Calculate the jacobian xi locations
        let aero_node_xi = vec![-1.0, -0.5, 0., 0.5, 1.0];
        let jacobian_xi = calculate_jacobian_xi(&aero_node_xi);

        // Calculate the jacobian integration matrix (shape function derivative)
        // Rows are from jacobian xi and columns are from beam xi
        let mut jacobian_integration_matrix =
            Mat::<f64>::zeros(jacobian_xi.len(), beam_node_xi.len());
        shape_deriv_matrix(
            &jacobian_xi,
            &beam_node_xi,
            jacobian_integration_matrix.as_mut(),
        );

        // Calculate the aerodynamic node widths
        let widths = calculate_aero_node_widths(
            &jacobian_xi,
            jacobian_integration_matrix.as_ref(),
            node_x.as_ref(),
        );

        // Verify the widths are correct
        widths
            .iter()
            .zip(
                vec![
                    0.46400663710393675,
                    0.71135714204928224,
                    0.52584462380517116,
                    0.56739053334406075,
                    0.36524408545476744,
                ]
                .iter(),
            )
            .for_each(|(act, exp)| {
                assert_relative_eq!(act, exp, epsilon = 1e-12);
            });
    }

    #[test]
    fn test_calculate_jacobian_xi() {
        let aero_node_xi = vec![-1.0, 0.0, 1.0];
        let result = calculate_jacobian_xi(&aero_node_xi);
        assert_eq!(result, vec![-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0]);

        let aero_node_xi = vec![-1.0, 0.2, 1.0];
        let result = calculate_jacobian_xi(&aero_node_xi);
        assert_eq!(result, vec![-1.0, -0.7, -0.4, 0.2, 0.6, 0.8, 1.0]);

        let aero_node_xi = vec![-1.0, -0.2, 0.2, 1.0];
        let result = calculate_jacobian_xi(&aero_node_xi);
        assert_eq!(result, vec![-1.0, -0.8, -0.6, -0.2, 0., 0.2, 0.6, 0.8, 1.0]);
    }
}

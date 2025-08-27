use std::ops::AddAssign;

use faer::{linalg::matmul::matmul, prelude::*, Accum};
use itertools::{multizip, Itertools};
use vtkio::model::*;

use crate::{
    interp::{shape_deriv_matrix, shape_interp_matrix},
    node::Node,
    state::State,
    util::{
        cross_product, quat_compose, quat_compose_alloc, quat_from_axis_angle_alloc, quat_inverse,
        quat_rotate_vector, quat_rotate_vector_alloc,
    },
};

pub struct AeroComponent {
    pub bodies: Vec<AeroBody>,
}

impl AeroComponent {
    /// Create a new AeroComponent from element input, inflow, and nodes
    pub fn new(bodies: &[AeroBodyInput], nodes: &[Node]) -> Self {
        AeroComponent {
            bodies: bodies
                .iter()
                .map(|body| AeroBody::new(body, nodes))
                .collect(),
        }
    }

    /// Calculate the aerodynamic center motion at each section for each body:
    /// body.x_motion contains position and body.v_motion contains velocity
    pub fn calculate_motion(&mut self, state: &State) {
        self.bodies.iter_mut().for_each(|body| {
            body.calculate_motion(state);
        });
    }

    /// Set the inflow velocity at each section using a function which takes the
    /// aerodynamic center position as input and returns a XYZ velocity vector
    pub fn set_inflow_from_function(&mut self, inflow_velocity: impl Fn([f64; 3]) -> [f64; 3]) {
        self.bodies.iter_mut().for_each(|body| {
            body.set_inflow_from_function(&inflow_velocity);
        });
    }

    /// Set the inflow velocity at each section from vectors of XYZ velocity values per body.
    pub fn set_inflow_from_vector(&mut self, body_inflow_velocities: &[&[[f64; 3]]]) {
        multizip((self.bodies.iter_mut(), body_inflow_velocities.iter())).for_each(|(body, &v)| {
            body.set_inflow_from_vector(v);
        });
    }

    /// Calculate aerodynamic loads for each body from inflow
    pub fn calculate_aerodynamic_loads(&mut self, fluid_density: f64) {
        self.bodies.iter_mut().for_each(|body| {
            body.calculate_aerodynamic_loads(fluid_density);
        });
    }

    /// Calculate nodal loads for each body from the aerodynamic loads
    pub fn calculate_nodal_loads(&mut self) {
        self.bodies.iter_mut().for_each(|body| {
            body.calculate_nodal_loads();
        });
    }

    /// Add nodal loads to state determined by `calculate_nodal_loads` to `state`.
    pub fn add_nodal_loads_to_state(&self, state: &mut State) {
        self.bodies.iter().for_each(|body| {
            body.add_nodal_loads_to_state(state);
        });
    }
}

#[derive(Debug, Clone)]
pub struct AeroBodyInput {
    pub id: usize,                       // Element ID
    pub beam_node_ids: Vec<usize>,       // Node IDs for the beam element
    pub aero_sections: Vec<AeroSection>, // Aerodynamic sections along beam axis
}

#[derive(Debug, Clone)]
pub struct AeroSection {
    pub id: usize,
    pub s: f64,                  // Position along beam element length [0-1]
    pub chord: f64,              // Chord length
    pub section_offset_x: f64,   // Section offset in x-direction
    pub section_offset_y: f64,   // Section offset in y-direction
    pub aerodynamic_center: f64, // Distance from leading edge to aerodynamic center
    pub twist: f64,              // Twist angle (radians)
    pub aoa: Vec<f64>,           // Angle of attack for all polars (radians)
    pub cl: Vec<f64>,            // Lift coefficient polar
    pub cd: Vec<f64>,            // Drag coefficient polar
    pub cm: Vec<f64>,            // Moment coefficient polar
}

pub struct AeroBody {
    pub id: usize, // Element ID

    // Beam
    pub node_ids: Vec<usize>, // Beam node IDs for this aero element
    pub node_u: Mat<f64>,     // Beam node displacements `[7][n_nodes]`
    pub node_v: Mat<f64>,     // Beam node velocities `[6][n_nodes]`
    pub node_f: Mat<f64>,     // Beam node forces `[6][n_nodes]`

    // Motion
    pub xr_motion_map: Mat<f64>, // Motion map reference position `[7][n_sections]`
    pub u_motion_map: Mat<f64>,  // Motion map displacements `[7][n_sections]`
    pub v_motion_map: Mat<f64>,  // Motion map velocities `[6][n_sections]`
    pub qqr_motion_map: Mat<f64>, // Motion map rotation position `[4][n_sections]`
    pub con_motion: Mat<f64>,    // Motion connectivity vector `[3][n_sections]`
    pub x_motion: Mat<f64>,      // Motion aerodynamic center position `[3][n_sections]`
    pub v_motion: Mat<f64>,      // Motion aerodynamic center velocity `[3][n_sections]`

    // Loads
    pub con_loads: Mat<f64>, // Load connectivity vector `[3][n_loads]`
    pub loads: Mat<f64>,     // Forces and moments at aerodynamic centers `[3][n_loads]`
    pub moment: Mat<f64>,    // Moment due to aerodynamic center moment arm `[3][n_loads]`

    // Blade Element Theory
    pub jacobian_xi: Col<f64>, // Jacobian xi locations for width integration
    pub v_inflow: Mat<f64>,    // Fluid velocity at each motion point `[3][n_sections]`
    pub twist: Col<f64>,       // Twist angle of each aerodynamic section (radians)
    pub chord: Col<f64>,       // Chord length of each aerodynamic section
    pub delta_s: Col<f64>,     // Width of aerodynamic section for load calculation
    pub polar_size: Vec<usize>, // Number of polar points in each aerodynamic section
    pub aoa: Mat<f64>,         // Angle of attack grid for all polars
    pub cl: Mat<f64>,          // Lift coefficient polar
    pub cd: Mat<f64>,          // Drag coefficient polar
    pub cm: Mat<f64>,          // Moment coefficient polar

    // Interpolation and derivative matrices
    pub motion_interp: Mat<f64>, // Shape function matrix interpolating node motion to motion map `[n_sections][n_nodes]`
    pub shape_deriv_jac: Mat<f64>, // Shape function derivative matrix `[n_jac][n_nodes]`
}

impl AeroBody {
    pub fn new(input: &AeroBodyInput, nodes: &[Node]) -> Self {
        // Number of aerodynamic sections in body
        let n_sections = input.aero_sections.len();

        // Number of force points in the body
        let n_loads = n_sections;

        // Number of nodes in beam element
        let n_nodes = input.beam_node_ids.len();

        //----------------------------------------------------------------------
        // Beam node and aero point locations on beam reference axis
        //----------------------------------------------------------------------

        // Get aerodynamic section location along beam (-1,1)
        let section_xi = input
            .aero_sections
            .iter()
            .map(|point| 2. * point.s - 1.)
            .collect_vec();

        // Get location of node along beam (-1,1)
        let beam_node_xi = input
            .beam_node_ids
            .iter()
            .map(|&id| 2. * nodes[id].s - 1.)
            .collect_vec();

        //----------------------------------------------------------------------
        // Section reference positions
        //----------------------------------------------------------------------

        // Calculate shape function interpolation matrix to map from beam to aero points
        let mut motion_interp = Mat::<f64>::zeros(section_xi.len(), beam_node_xi.len());
        shape_interp_matrix(&section_xi, &beam_node_xi, motion_interp.as_mut());

        // Get node reference position (translation only)
        let node_x = Mat::from_fn(7, n_nodes, |i, j| nodes[input.beam_node_ids[j]].xr[i]);

        // Interpolate node positions to aerodynamic sections on the beam reference axis
        let mut xr = Mat::zeros(7, n_sections);
        matmul(
            xr.transpose_mut(),
            Accum::Replace,
            &motion_interp,
            node_x.transpose(),
            1.,
            Par::Seq,
        );

        // Normalize rotation quaternions so they have unit length
        xr.subrows_mut(3, 4).col_iter_mut().for_each(|mut qr| {
            let m = qr.norm_l2();
            if m > f64::EPSILON {
                qr /= m;
            }
        });

        //----------------------------------------------------------------------
        // Section location tangents on beam reference axis
        //----------------------------------------------------------------------

        // Calculate shape function derivative matrix to map from beam nodes to sections
        let mut shape_deriv_node = Mat::<f64>::zeros(section_xi.len(), beam_node_xi.len());
        shape_deriv_matrix(&section_xi, &beam_node_xi, shape_deriv_node.as_mut());

        // Calculate spatial derivative of reference axis at section locations
        let mut x_tan = Mat::zeros(3, n_sections);
        matmul(
            x_tan.transpose_mut(),
            Accum::Replace,
            &shape_deriv_node,
            node_x.subrows(0, 3).transpose(),
            1.,
            Par::Seq,
        );
        x_tan.col_iter_mut().for_each(|mut col| {
            let m = col.norm_l2();
            if m > f64::EPSILON {
                col /= m;
            }
        });

        //----------------------------------------------------------------------
        // Add twist to reference rotation
        //----------------------------------------------------------------------

        multizip((
            xr.subrows_mut(3, 4).col_iter_mut(),
            x_tan.col_iter(),
            input.aero_sections.iter(),
        ))
        .for_each(|(mut qr, tan, section)| {
            // Calculate twist about current tangent
            let q_twist = quat_from_axis_angle_alloc(-section.twist, tan);

            // Compose twist quaternion with reference rotation quaternion
            let q = quat_compose_alloc(q_twist.as_ref(), qr.as_ref());

            // Update reference rotation quaternion
            qr.copy_from(&q);
        });

        //----------------------------------------------------------------------
        // Vector from section location on beam reference axis to aerodynamic center
        // The beam is initialized with:
        //   - root at (0, 0, 0)
        //   - reference axis aligned with the x-axis
        //   - the trailing edge is towards +y-axis
        //   - the suction side of the airfoil is towards +z-axis
        // x_r translates and rotates the beam into the reference position
        //----------------------------------------------------------------------

        // Calculate vector from beam reference to aerodynamic center in the body coordinates
        let con_motion = input
            .aero_sections
            .iter()
            .map(|section| {
                calculate_con_motion_vector(
                    section.section_offset_y - section.aerodynamic_center,
                    section.section_offset_x,
                )
            })
            .collect_vec();

        //----------------------------------------------------------------------
        // Aero point widths
        //----------------------------------------------------------------------

        // Calculate jacobian xi locations for width integration
        let jacobian_xi = calculate_jacobian_xi(&section_xi);

        // Calculate shape derivative matrix to calculate jacobians
        let mut jacobian_integration_matrix =
            Mat::<f64>::zeros(jacobian_xi.len(), beam_node_xi.len());
        shape_deriv_matrix(
            &jacobian_xi,
            &beam_node_xi,
            jacobian_integration_matrix.as_mut(),
        );

        // Calculate aero point widths
        let jacobian_xi = Col::from_iter(jacobian_xi.into_iter());
        let delta_s = calculate_aero_point_widths(
            jacobian_xi.as_ref(),
            jacobian_integration_matrix.as_ref(),
            node_x.as_ref(),
        );

        //----------------------------------------------------------------------
        // Polar data
        //----------------------------------------------------------------------

        // Max number of polar points in any aero point.
        // All polars in the section share the same angle of attack grid
        let n_polar_points_max = input
            .aero_sections
            .iter()
            .map(|p| p.aoa.len())
            .max()
            .unwrap_or_default();

        let aoa = Mat::from_fn(n_polar_points_max, n_sections, |i, j| {
            if i < input.aero_sections[j].aoa.len() {
                input.aero_sections[j].aoa[i]
            } else {
                0. // Fill with zero if not enough points
            }
        });
        let cl = Mat::from_fn(n_polar_points_max, n_sections, |i, j| {
            if i < input.aero_sections[j].cl.len() {
                input.aero_sections[j].cl[i]
            } else {
                0. // Fill with zero if not enough points
            }
        });
        let cd = Mat::from_fn(n_polar_points_max, n_sections, |i, j| {
            if i < input.aero_sections[j].cd.len() {
                input.aero_sections[j].cd[i]
            } else {
                0. // Fill with zero if not enough points
            }
        });
        let cm = Mat::from_fn(n_polar_points_max, n_sections, |i, j| {
            if i < input.aero_sections[j].cm.len() {
                input.aero_sections[j].cm[i]
            } else {
                0. // Fill with zero if not enough points
            }
        });

        //----------------------------------------------------------------------
        // Populate element structure
        //----------------------------------------------------------------------

        Self {
            id: input.id,
            node_ids: input.beam_node_ids.clone(),
            node_u: Mat::zeros(7, n_nodes),
            node_v: Mat::zeros(6, n_nodes),
            node_f: Mat::zeros(6, n_nodes),

            xr_motion_map: xr.cloned(),
            u_motion_map: Mat::zeros(7, n_sections),
            v_motion_map: Mat::zeros(6, n_sections),
            qqr_motion_map: Mat::zeros(4, n_sections),
            con_motion: Mat::from_fn(3, n_sections, |i, j| con_motion[j][i]),
            x_motion: Mat::zeros(3, n_sections),
            v_motion: Mat::zeros(3, n_sections),

            con_loads: Mat::from_fn(3, n_sections, |i, j| -con_motion[j][i]),
            loads: Mat::zeros(6, n_loads),
            moment: Mat::zeros(3, n_loads),

            jacobian_xi,
            v_inflow: Mat::zeros(3, n_sections),
            twist: Col::from_fn(n_sections, |i| input.aero_sections[i].twist),
            chord: Col::from_fn(n_sections, |i| input.aero_sections[i].chord),
            delta_s,
            polar_size: input.aero_sections.iter().map(|p| p.aoa.len()).collect(),
            aoa,
            cl,
            cd,
            cm,

            motion_interp,
            shape_deriv_jac: jacobian_integration_matrix,
        }
    }

    /// Calculate the aerodynamic center motion at each aerodynamic section:
    /// self.x_motion contains position and self.v_motion contains velocity
    fn calculate_motion(&mut self, state: &State) {
        // Copy beam node displacements from state
        self.node_ids.iter().enumerate().for_each(|(i, &id)| {
            self.node_u.col_mut(i).copy_from(&state.u.col(id));
        });

        // Copy beam node velocities from state
        self.node_ids.iter().enumerate().for_each(|(i, &id)| {
            self.node_v.col_mut(i).copy_from(&state.v.col(id));
        });

        // Interpolate beam node displacement to aerodynamic sections on the reference axis
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

        // Interpolate beam node velocities to aerodynamic sections on the reference axis
        matmul(
            self.v_motion_map.transpose_mut(),
            Accum::Replace,
            &self.motion_interp,
            self.node_v.transpose(),
            1.,
            Par::Seq,
        );

        // Calculate global rotation of each section
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
            self.v_motion_map.subrows(3, 3).col_iter(),     // Axis rotation velocity
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

        let node_x = Mat::from_fn(7, self.node_ids.len(), |i, j| {
            state.x[(i, self.node_ids[j])]
        });

        // Update element widths
        self.delta_s = calculate_aero_point_widths(
            self.jacobian_xi.as_ref(),
            self.shape_deriv_jac.as_ref(),
            node_x.as_ref(),
        );
    }

    /// Set the inflow velocity at each section using a function which takes the
    /// aerodynamic center position as input and returns a XYZ velocity vector
    fn set_inflow_from_function(&mut self, inflow_velocity: impl Fn([f64; 3]) -> [f64; 3]) {
        multizip((self.v_inflow.col_iter_mut(), self.x_motion.col_iter())).for_each(
            |(mut v, x)| {
                let inflow_velocity = inflow_velocity([x[0], x[1], x[2]]);
                v[0] = inflow_velocity[0];
                v[1] = inflow_velocity[1];
                v[2] = inflow_velocity[2];
            },
        );
    }

    /// Set the inflow velocity at each section from a vector of XYZ velocity values
    fn set_inflow_from_vector(&mut self, inflow_velocity: &[[f64; 3]]) {
        multizip((self.v_inflow.col_iter_mut(), inflow_velocity.iter())).for_each(
            |(mut v, &v_inflow)| {
                v[0] = v_inflow[0];
                v[1] = v_inflow[1];
                v[2] = v_inflow[2];
            },
        );
    }

    /// Calculate the aerodynamic loads from the inflow velocities at each section.
    /// Uses inflow values set by `set_inflow_from_function` or `set_inflow_from_vector`.
    fn calculate_aerodynamic_loads(&mut self, fluid_density: f64) {
        multizip((
            self.loads.col_iter_mut(),
            self.v_inflow.col_iter(), // Fluid velocity (inflow)
            self.v_motion.subrows_mut(0, 3).col_iter(), // AC velocity
            self.polar_size.iter(),
            self.aoa.col_iter(),
            self.cl.col_iter(),
            self.cd.col_iter(),
            self.cm.col_iter(),
            self.chord.iter(),
            self.delta_s.iter(),
            self.qqr_motion_map.col_iter(),
        ))
        .for_each(
            |(
                loads,
                v_inflow,
                v_motion,
                &polar_size,
                aoa_polar,
                cl_polar,
                cd_polar,
                cm_polar,
                &chord,
                &delta_s,
                qqr,
            )| {
                calculate_aerodynamic_load(
                    loads,
                    v_inflow,
                    v_motion,
                    polar_size,
                    aoa_polar,
                    cl_polar,
                    cd_polar,
                    cm_polar,
                    chord,
                    delta_s,
                    fluid_density,
                    qqr,
                )
            },
        );
    }

    /// Calculate nodal loads from aerodynamic loads stored in `self.loads`.
    fn calculate_nodal_loads(&mut self) {
        // Calculate additional moment from aerodynamic center moment arm
        multizip((
            self.moment.col_iter_mut(),
            self.loads.subrows(0, 3).col_iter(),
            self.con_loads.col_iter(),
        ))
        .for_each(|(m, f, con)| {
            cross_product(f, con, m);
        });

        // Distribute aerodynamic loads (force and moments) to nodes
        matmul(
            self.node_f.transpose_mut(),
            Accum::Replace,
            self.motion_interp.transpose(),
            self.loads.transpose(),
            1.0,
            Par::Seq,
        );

        // Add additional moments to nodes
        matmul(
            self.node_f.subrows_mut(3, 3).transpose_mut(),
            Accum::Add,
            self.motion_interp.transpose(),
            self.moment.transpose(),
            1.0,
            Par::Seq,
        );
    }

    /// Add nodal loads to state determined by `calculate_nodal_loads` to `state`.
    fn add_nodal_loads_to_state(&self, state: &mut State) {
        self.node_ids.iter().enumerate().for_each(|(i, &id)| {
            let node_f = self.node_f.col(i);
            state.fx.col_mut(id).add_assign(&node_f);
        });
    }

    pub fn as_vtk(&self) -> Vtk {
        // let rotations = multizip((
        //     self.u_motion_map.subrows(3, 4).col_iter(),
        //     self.xr_motion_map.subrows(3, 4).col_iter(),
        // ))
        // .map(|(r, r0)| {
        //     let mut q = Col::<f64>::zeros(4);
        //     quat_compose(r, r0, q.as_mut());
        //     let mut m = Mat::<f64>::zeros(3, 3);
        //     quat_as_matrix(q.as_ref(), m.as_mut());
        //     m
        // })
        // .collect_vec();
        // let orientations = vec!["OrientationX", "OrientationY", "OrientationZ"];
        let n_sections = self.delta_s.nrows();

        Vtk {
            version: Version { major: 4, minor: 2 },
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(
                    self.x_motion
                        .col_iter()
                        .flat_map(|x| [x[0], x[1], x[2]])
                        .collect_vec(),
                ),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity: {
                            let mut a = vec![0, n_sections - 1];
                            let b = (1..n_sections - 1).collect_vec();
                            a.extend(b);
                            a.iter().map(|&i| i as u64).collect_vec()
                        },
                        offsets: vec![n_sections as u64],
                    },
                    types: vec![CellType::LagrangeCurve],
                },
                data: Attributes {
                    point: vec![
                        // orientations
                        // .iter()
                        // .enumerate()
                        // .map(|(i, &orientation)| {
                        //     Attribute::DataArray(DataArrayBase {
                        //         name: orientation.to_string(),
                        //         elem: ElementType::Vectors,
                        //         data: IOBuffer::F32(
                        //             rotations
                        //                 .iter()
                        //                 .flat_map(|r| {
                        //                     r.col(i).iter().map(|&v| v as f32).collect_vec()
                        //                 })
                        //                 .collect_vec(),
                        //         ),
                        //     })
                        // })
                        Attribute::DataArray(DataArrayBase {
                            name: "TranslationalVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.v_motion
                                    .subrows(0, 3)
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                        // Attribute::DataArray(DataArrayBase {
                        //     name: "AngularVelocity".to_string(),
                        //     elem: ElementType::Vectors,
                        //     data: IOBuffer::F32(
                        //         self.v_motion
                        //             .subrows(3, 3)
                        //             .col_iter()
                        //             .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                        //             .collect_vec(),
                        //     ),
                        // }),
                        Attribute::DataArray(DataArrayBase {
                            name: "Force".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.loads
                                    .subrows(0, 3)
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "Moment".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                self.loads
                                    .subrows(0, 3)
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                    ],
                    ..Default::default()
                },
            }),
        }
    }
}

//------------------------------------------------------------------------------
// Computation kernels
//------------------------------------------------------------------------------

/// Calculate aerodynamic loads in the global coordinate system.
fn calculate_aerodynamic_load(
    mut loads: ColMut<f64>,
    v_inflow: ColRef<f64>,
    v_motion: ColRef<f64>,
    polar_size: usize,
    aoa_polar: ColRef<f64>,
    cl_polar: ColRef<f64>,
    cd_polar: ColRef<f64>,
    cm_polar: ColRef<f64>,
    chord: f64,
    delta_s: f64,
    fluid_density: f64,
    qqr: ColRef<f64>,
) {
    // Calculate difference between inflow velocity and aerodynamic center velocity
    // in the global coordinate system
    let v_rel_global = &v_inflow - &v_motion;

    // Transform the relative velocity to the local coordinate system with
    // +Y to the trailing edge and +Z to the pressure side of the airfoil.
    // This is done by rotating the relative velocity vector by the inverse of
    // the total rotation quaternion (rotation displacement + initial rotation)
    let mut qqr_inv = col![0., 0., 0., 0.];
    quat_inverse(qqr.as_ref(), qqr_inv.as_mut());
    let mut v_rel = quat_rotate_vector_alloc(qqr_inv.as_ref(), v_rel_global.as_ref());

    // Relative velocity only considers flow in the y-z plane
    v_rel[0] = 0.;

    // Calculate angle of attack
    let aoa = calculate_angle_of_attack(v_rel.as_ref());

    // Find index of angle of attack in aoa vector for interpolation
    let i_aoa = (0..polar_size - 1)
        .position(|i| aoa >= aoa_polar[i] && aoa <= aoa_polar[i + 1])
        .unwrap_or_else(|| {
            panic!(
                "Angle of attack {} not in range {} - {}",
                aoa,
                aoa_polar[0],
                aoa_polar[polar_size - 1]
            )
        });

    // Calculate blending term between values above and below
    let alpha = (aoa - aoa_polar[i_aoa]) / (aoa_polar[i_aoa + 1] - aoa_polar[i_aoa]);

    // Calculate lift and drag coefficients using linear interpolation
    let cl = (1. - alpha) * cl_polar[i_aoa] + alpha * cl_polar[i_aoa + 1];
    let cd = (1. - alpha) * cd_polar[i_aoa] + alpha * cd_polar[i_aoa + 1];
    let cm = (1. - alpha) * cm_polar[i_aoa] + alpha * cm_polar[i_aoa + 1];

    // Calculate force and moment in local coordinates
    let dynamic_pressure = fluid_density * v_rel.norm_l2().powi(2) / 2.;

    // Calculate drag direction vector as normalized negative relative flow velocity vector
    let drag_vector = &v_rel / v_rel.norm_l2();

    // Calculate lift vector perpendicular to the drag vector
    let mut lift_vector = col![0., 0., 0.];
    cross_product(
        col![-1., 0., 0.].as_ref(),
        drag_vector.as_ref(),
        lift_vector.as_mut(),
    );

    // Calculate force and moment in local coordinates
    let force_local = dynamic_pressure * chord * delta_s * (cl * &lift_vector + cd * &drag_vector);
    let moment_local = col![cm * dynamic_pressure * chord.powi(2) * delta_s, 0., 0.];

    // Rotate force and moment into global coordinates
    quat_rotate_vector(
        qqr.as_ref(),
        force_local.as_ref(),
        loads.rb_mut().subrows_mut(0, 3),
    );
    quat_rotate_vector(
        qqr.as_ref(),
        moment_local.as_ref(),
        loads.rb_mut().subrows_mut(3, 3),
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
fn calculate_aero_point_widths(
    jacobian_xi: ColRef<f64>,
    jacobian_integration_matrix: MatRef<f64>,
    node_x: MatRef<f64>,
) -> Col<f64> {
    // Calculate un-normalized derivative vectors
    let tan = jacobian_integration_matrix * node_x.subrows(0, 3).transpose();

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
                let h1 = jacobian_xi[j1] - jacobian_xi[j0];
                let h2 = jacobian_xi[j2] - jacobian_xi[j1];
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
fn calculate_con_motion_vector(
    ac_to_ref_axis_horizontal: f64,
    chord_to_ref_axis_vertical: f64,
) -> [f64; 3] {
    [0., -ac_to_ref_axis_horizontal, chord_to_ref_axis_vertical]
}

/// Calculate angle of attack based on relative velocity. The airfoil
/// coordinate system has +X pointing out of the page, +Y pointing
/// to the trailing edge, and +Z pointing to the pressure side.
fn calculate_angle_of_attack(v_rel: ColRef<f64>) -> f64 {
    (-v_rel[2]).atan2(v_rel[1])
}

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_calculate_angle_of_attack() {
        struct Case {
            flow_angle: f64,
            expected_aoa: f64,
        }

        vec![
            Case {
                flow_angle: 0.0,
                expected_aoa: 0.0,
            },
            Case {
                flow_angle: -0.1,
                expected_aoa: 0.1,
            },
            Case {
                flow_angle: 0.2,
                expected_aoa: -0.2,
            },
            Case {
                flow_angle: 0.1 - PI,
                expected_aoa: PI - 0.1,
            },
        ]
        .iter()
        .for_each(|c| {
            let v_rel = col![0.0, c.flow_angle.cos(), c.flow_angle.sin()];
            let aoa = calculate_angle_of_attack(v_rel.as_ref());
            assert_relative_eq!(aoa, c.expected_aoa, epsilon = 1e-12);
        });
    }

    #[test]
    fn test_calculate_aerodynamic_load() {
        // Polars
        let n_polar_points = 3;
        let aoa_polar = col![-1.0, 1.0]; // Angle of attack points
        let cl_polar = col![0.0, 1.0]; // Lift coefficient values
        let cd_polar = col![0.5, 0.0]; // Drag coefficient values
        let cm_polar = col![0.01, 0.03]; // Moment coefficient values

        struct Data {
            v_inflow: Col<f64>,
            v_motion: Col<f64>,
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
            Data {
                v_inflow: col![0.0, 10.0, 0.0],
                v_motion: col![0.0, 0.0, 0.0],
                chord: 2.0,
                delta_s: 1.5,
                fluid_density: 1.225,
                qqr: col![1.0, 0.0, 0.0, 0.0],
                load: col![0.0, 45.9375, -91.875, 7.35, 0.0, 0.0],
            },
            // twist = 0.0
            // v_rel = [0.0, 10*cos(0.1), -10*sin(0.1)] = [0.0, 9.950041652780259, -0.9983341664682815]
            // AoA = 0.1
            // cl = 0.55
            // cd = 0.225
            // cm = 0.021
            // dynamic pressure = 0.5 * 1.225 * 10 * 10
            // No rotation from local to global
            Data {
                v_inflow: col![0.0, 9.950041652780259, -0.9983341664682815],
                v_motion: col![0.0, 0.0, 0.0],
                chord: 2.0,
                delta_s: 1.5,
                fluid_density: 1.225,
                qqr: col![1.0, 0.0, 0.0, 0.0],
                load: col![0., 31.04778878834331, -104.68509627290281, 7.7175, 0.0, 0.0],
            },
        ];

        for case in test_cases {
            let mut load = col![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Force and moment output

            calculate_aerodynamic_load(
                load.as_mut(),
                case.v_inflow.as_ref(),
                case.v_motion.as_ref(),
                n_polar_points,
                aoa_polar.as_ref(),
                cl_polar.as_ref(),
                cd_polar.as_ref(),
                cm_polar.as_ref(),
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
            ac_vec_exp: Col<f64>,
        }

        let test_cases = vec![
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.0,
                ac_vec_exp: col![0.0, -1.0, 0.0],
            },
            Data {
                ac_to_ref_axis_horizontal: 1.0,
                chord_to_ref_axis_vertical: 0.5,
                ac_vec_exp: col![0.0, -1.0, 0.5],
            },
        ];

        for case in test_cases {
            let ac_vec = calculate_con_motion_vector(
                case.ac_to_ref_axis_horizontal,
                case.chord_to_ref_axis_vertical,
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
        let jacobian_xi = Col::from_iter(jacobian_xi.into_iter());
        let widths = calculate_aero_point_widths(
            jacobian_xi.as_ref(),
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
        let jacobian_xi = Col::from_iter(jacobian_xi.into_iter());
        let widths = calculate_aero_point_widths(
            jacobian_xi.as_ref(),
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

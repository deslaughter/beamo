use crate::{
    components::node_data::NodeData,
    elements::beams::{BeamSection, Damping},
    interp::{gauss_legendre_lobotto_points, shape_deriv_matrix, shape_interp_matrix},
    model::Model,
    quadrature::Quadrature,
    util::{
        quat_as_rotation_vector_alloc, quat_from_rotation_vector_alloc, quat_from_tangent_twist,
        quat_rotate_vector_alloc, rotate_section_matrix,
    },
};
use faer::prelude::*;
use interp::{interp, InterpMode};
use itertools::{izip, Itertools};
use std::f64::consts::PI;

//------------------------------------------------------------------------------
// Beam Component
//------------------------------------------------------------------------------

pub struct BeamComponent {
    pub elem_id: usize,                    // Unique identifier for the beam element
    pub nodes: Vec<NodeData>,              // Node identifiers for the beam element
    pub node_xi: Vec<f64>,                 // Local coordinates of nodes in the beam element
    pub root_constraint_id: Option<usize>, // Optional constraint ID for the root node
}

impl BeamComponent {
    pub fn new(input: &BeamInput, model: &mut Model) -> Self {
        // Get node locations [-1, 1]
        let node_xi = gauss_legendre_lobotto_points(input.element_order);
        let node_s = node_xi
            .iter()
            .map(|&xi| 0.5 * (xi + 1.)) // Convert from [-1, 1] to [0, 1]
            .collect_vec();

        // Curve fit node coordinates
        let (node_xyz, node_tan) = get_node_coordinates(&input.reference_axis, &node_xi);

        // Add nodes to model
        let mut q = Col::zeros(4);
        let node_ids = izip!(
            node_s.iter(),
            node_xyz.transpose().col_iter(),
            node_tan.transpose().col_iter()
        )
        .map(|(&si, c, tan)| {
            quat_from_tangent_twist(tan.as_ref(), 0., q.as_mut()); // Calculate twist about tangent
            model
                .add_node()
                .element_location(si)
                .position(c[0], c[1], c[2], q[0], q[1], q[2], q[3])
                .build()
        })
        .collect_vec();

        // Translate and rotate nodes based on root position
        node_ids.iter().for_each(|&id| {
            model.nodes[id]
                .rotate(
                    quat_as_rotation_vector_alloc(
                        col![
                            input.root.position[3],
                            input.root.position[4],
                            input.root.position[5],
                            input.root.position[6]
                        ]
                        .as_ref(),
                    )
                    .as_ref(),
                    col![0., 0., 0.].as_ref(),
                )
                .translate([
                    input.root.position[0],
                    input.root.position[1],
                    input.root.position[2],
                ]);
        });

        let mut sections: Vec<BeamSection> =
            Vec::with_capacity(input.sections.len() * (input.section_refinement + 1) + 1);

        // Add first section after rotating matrices to account for twist
        let twist = interp(
            &input.reference_axis.twist_grid,
            &input.reference_axis.twist,
            input.sections[0].s,
            &InterpMode::default(),
        );
        sections.push(BeamSection {
            s: input.sections[0].s,
            m_star: rotate_section_matrix(&input.sections[0].m_star, &col![twist, 0., 0.]),
            c_star: rotate_section_matrix(&input.sections[0].c_star, &col![twist, 0., 0.]),
        });

        // Loop through remaining section locations
        for i in 1..input.sections.len() {
            // Add refinement sections if requested
            for j in 0..input.section_refinement {
                // Calculate interpolation ratio between bounding sections
                let alpha = (j + 1) as f64 / (input.section_refinement + 1) as f64;

                // Interpolate grid location
                let grid_value =
                    (1. - alpha) * input.sections[i - 1].s + alpha * input.sections[i].s;

                // Interpolate mass and stiffness matrices from bounding sections
                let m = (1. - alpha) * &input.sections[i - 1].m_star
                    + alpha * &input.sections[i].m_star;
                let k = (1. - alpha) * &input.sections[i - 1].c_star
                    + alpha * &input.sections[i].c_star;

                // Calculate twist at current section location via linear interpolation
                let twist = interp(
                    &input.reference_axis.twist_grid,
                    &input.reference_axis.twist,
                    grid_value,
                    &InterpMode::default(),
                );

                // Add refinement section
                sections.push(BeamSection {
                    s: grid_value,
                    m_star: rotate_section_matrix(&m, &col![twist, 0., 0.]),
                    c_star: rotate_section_matrix(&k, &col![twist, 0., 0.]),
                });
            }

            // Add ending section
            let twist = interp(
                &input.reference_axis.twist_grid,
                &input.reference_axis.twist,
                input.sections[i].s,
                &InterpMode::default(),
            );
            sections.push(BeamSection {
                s: input.sections[i].s,
                m_star: rotate_section_matrix(&input.sections[i].m_star, &col![twist, 0., 0.]),
                c_star: rotate_section_matrix(&input.sections[i].c_star, &col![twist, 0., 0.]),
            });
        }

        let q = Quadrature::trapezoidal(&sections.iter().map(|s| s.s).collect_vec());

        BeamComponent {
            elem_id: model.add_beam_element(&node_ids, &q, &sections, &input.damping),
            nodes: node_ids.iter().map(|&id| NodeData::new(id)).collect_vec(),
            node_xi,
            root_constraint_id: if input.prescribe_root {
                Some(model.add_prescribed_constraint(node_ids[0]))
            } else {
                None
            },
        }
    }
}

//------------------------------------------------------------------------------
// Beam Input
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BeamInput {
    element_order: usize,
    section_refinement: usize,
    reference_axis: ReferenceAxis,
    sections: Vec<BeamSection>,
    root: Root,
    damping: Damping,
    prescribe_root: bool,
}

#[derive(Debug, Clone)]
pub struct Root {
    position: [f64; 7],
    velocity: [f64; 6],
    acceleration: [f64; 6],
}

#[derive(Debug, Clone)]
pub struct ReferenceAxis {
    coordinate_grid: Vec<f64>,
    coordinates: Vec<[f64; 3]>,
    twist_grid: Vec<f64>,
    twist: Vec<f64>,
}

//------------------------------------------------------------------------------
// Beam Builder
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BeamInputBuilder {
    beam_input: BeamInput,
}

impl BeamInputBuilder {
    pub fn new() -> Self {
        BeamInputBuilder {
            beam_input: BeamInput {
                element_order: 1,
                section_refinement: 0,
                reference_axis: ReferenceAxis {
                    coordinate_grid: vec![],
                    coordinates: vec![],
                    twist_grid: vec![],
                    twist: vec![],
                },
                sections: vec![],
                root: Root {
                    position: [0., 0., 0., 1., 0., 0., 0.],
                    velocity: [0.0; 6],
                    acceleration: [0.0; 6],
                },
                damping: Damping::None,
                prescribe_root: false,
            },
        }
    }

    pub fn set_element_order(&mut self, order: usize) -> &mut Self {
        self.beam_input.element_order = order;
        self
    }

    pub fn set_section_refinement(&mut self, refinement: usize) -> &mut Self {
        self.beam_input.section_refinement = refinement;
        self
    }

    pub fn add_section(
        &mut self,
        location: f64,
        mass_matrix: Mat<f64>,
        stiffness_matrix: Mat<f64>,
    ) -> &mut Self {
        self.beam_input.sections.push(BeamSection {
            s: location,
            m_star: mass_matrix,
            c_star: stiffness_matrix,
        });
        self
    }

    pub fn set_reference_axis(
        &mut self,
        coordinate_grid: &[f64],
        coordinates: &[[f64; 3]],
        twist_grid: &[f64],
        twist: &[f64],
    ) -> &mut Self {
        self.beam_input.reference_axis = ReferenceAxis {
            coordinate_grid: coordinate_grid.to_vec(),
            coordinates: coordinates.to_vec(),
            twist_grid: twist_grid.to_vec(),
            twist: twist.to_vec(),
        };
        self
    }

    pub fn set_reference_axis_z(
        &mut self,
        coordinate_grid: &[f64],
        coordinates: &[[f64; 3]],
        twist_grid: &[f64],
        twist: &[f64],
    ) -> &mut Self {
        let q = quat_from_rotation_vector_alloc(col![0., PI / 2., 0.].as_ref());
        self.beam_input.reference_axis = ReferenceAxis {
            coordinate_grid: coordinate_grid.to_vec(),
            coordinates: coordinates
                .iter()
                .map(|c| {
                    let rotated =
                        quat_rotate_vector_alloc(q.as_ref(), col![c[0], c[1], c[2]].as_ref());
                    [rotated[0], rotated[1], rotated[2]]
                })
                .collect_vec(),
            twist_grid: twist_grid.to_vec(),
            twist: twist.to_vec(),
        };
        self
    }

    pub fn add_reference_axis_point(
        &mut self,
        grid_location: f64,
        coordinates: [f64; 3],
    ) -> &mut Self {
        self.beam_input
            .reference_axis
            .coordinate_grid
            .push(grid_location);
        self.beam_input.reference_axis.coordinates.push(coordinates);
        self
    }

    pub fn add_reference_axis_twist(&mut self, grid_location: f64, twist: f64) -> &mut Self {
        self.beam_input
            .reference_axis
            .twist_grid
            .push(grid_location);
        self.beam_input.reference_axis.twist.push(twist);
        self
    }

    pub fn set_root_position(&mut self, position: [f64; 7]) -> &mut Self {
        self.beam_input.root.position = position;
        self
    }

    pub fn set_root_velocity(&mut self, velocity: [f64; 6]) -> &mut Self {
        self.beam_input.root.velocity = velocity;
        self
    }

    pub fn set_root_acceleration(&mut self, acceleration: [f64; 6]) -> &mut Self {
        self.beam_input.root.acceleration = acceleration;
        self
    }

    pub fn clear_sections(&mut self) -> &mut Self {
        self.beam_input.sections.clear();
        self
    }

    pub fn add_section_x(
        &mut self,
        grid_location: f64,
        mass_matrix: Mat<f64>,
        stiffness_matrix: Mat<f64>,
    ) -> &mut Self {
        self.beam_input.sections.push(BeamSection {
            s: grid_location,
            m_star: mass_matrix,
            c_star: stiffness_matrix,
        });
        self
    }

    pub fn set_sections(&mut self, sections: &[BeamSection]) -> &mut Self {
        self.beam_input.sections = sections.to_vec();
        self
    }

    pub fn add_section_z(
        &mut self,
        grid_location: f64,
        mass_matrix: &Mat<f64>,
        stiffness_matrix: &Mat<f64>,
    ) -> &mut Self {
        let rv = col![0., PI / 2., 0.];
        self.beam_input.sections.push(BeamSection {
            s: grid_location,
            m_star: rotate_section_matrix(mass_matrix, &rv),
            c_star: rotate_section_matrix(stiffness_matrix, &rv),
        });
        self
    }

    pub fn set_sections_z(&mut self, sections: &[BeamSection]) -> &mut Self {
        let rv = col![0., PI / 2., 0.];
        self.beam_input.sections = sections
            .iter()
            .map(|s| BeamSection {
                s: s.s,
                m_star: rotate_section_matrix(&s.m_star, &rv),
                c_star: rotate_section_matrix(&s.c_star, &rv),
            })
            .collect();
        self
    }

    pub fn set_damping(&mut self, damping: Damping) -> &mut Self {
        self.beam_input.damping = damping;
        self
    }

    pub fn set_prescribe_root(&mut self, prescribe: bool) -> &mut Self {
        self.beam_input.prescribe_root = prescribe;
        self
    }

    pub fn build(&self) -> BeamInput {
        self.beam_input.clone()
    }
}

fn get_node_coordinates(reference_axis: &ReferenceAxis, node_xi: &[f64]) -> (Mat<f64>, Mat<f64>) {
    let n_nodes = node_xi.len();
    let n_kps = reference_axis.coordinate_grid.len();

    // Get the reference axis points on [-1, 1] instead of [0, 1]
    let kp_xi = &reference_axis
        .coordinate_grid
        .iter()
        .map(|&x| 2. * x - 1.)
        .collect::<Vec<_>>();

    // Build interpolation matrix to go from key points to node
    let mut shape_interp = Mat::<f64>::zeros(kp_xi.len(), node_xi.len());
    shape_interp_matrix(&kp_xi, &node_xi, shape_interp.as_mut());

    // Build A matrix for fitting key points to element nodes
    let mut a_matrix = Mat::<f64>::zeros(n_nodes, n_nodes);
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            for k in 0..n_kps {
                a_matrix[(i, j)] += shape_interp[(k, i)] * shape_interp[(k, j)];
            }
        }
    }
    a_matrix.row_mut(0).fill(0.);
    a_matrix.row_mut(n_nodes - 1).fill(0.);
    a_matrix[(0, 0)] = 1.;
    a_matrix[(n_nodes - 1, n_nodes - 1)] = 1.;

    // Build B matrix for fitting key points to element nodes
    let kp_matrix = Mat::from_fn(n_kps, 3, |i, j| reference_axis.coordinates[i][j]);
    let mut b_matrix = shape_interp.transpose() * &kp_matrix;
    b_matrix.row_mut(0).copy_from(&kp_matrix.row(0));
    b_matrix
        .row_mut(n_nodes - 1)
        .copy_from(&kp_matrix.row(n_kps - 1));

    // Solve for node locations using least squares
    let node_xyz = a_matrix.full_piv_lu().solve(&b_matrix);

    // Calculate tangents at nodes
    let mut shape_deriv = Mat::<f64>::zeros(node_xi.len(), node_xi.len());
    shape_deriv_matrix(&node_xi, &node_xi, shape_deriv.as_mut());
    let deriv = shape_deriv * &node_xyz;
    let mut tan = Mat::<f64>::zeros(deriv.nrows(), deriv.ncols());
    izip!(deriv.row_iter(), tan.row_iter_mut()).for_each(|(deriv_col, mut tan_col)| {
        let m = deriv_col.norm_l2();
        tan_col.copy_from(deriv_col / m);
    });

    (node_xyz, tan)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_get_node_coordinates() {
        let reference_axis = ReferenceAxis {
            coordinate_grid: vec![0., 0.25, 0.5, 0.75, 1.],
            coordinates: vec![
                [0., 0., 0.],
                [1., 0., 0.],
                [2., 0., 0.],
                [3., 0., 0.],
                [4., 0., 0.],
            ],
            twist_grid: vec![0., 1.],
            twist: vec![0., 0.],
        };
        let node_xi = vec![-1.0, 0.0, 1.0];

        let (node_xyz, node_tan) = get_node_coordinates(&reference_axis, &node_xi);

        println!("Node XYZ: {:?}", node_xyz);
        println!("Node Tangents: {:?}", node_tan);
    }
}

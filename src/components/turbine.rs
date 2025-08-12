use std::f64::consts::PI;

use crate::{
    components::{
        beam::{BeamComponent, BeamInput},
        node_data::NodeData,
    },
    model::Model,
    state::State,
    util::cross_product,
};
use faer::prelude::*;
use itertools::Itertools;

pub struct Turbine {
    pub blades: Vec<BeamComponent>,
    pub tower: BeamComponent,

    pub hub_node: NodeData,
    pub azimuth_node: NodeData,
    pub shaft_base_node: NodeData,
    pub yaw_bearing_node: NodeData,
    pub apex_nodes: Vec<NodeData>,

    pub pitch_constraint_ids: Vec<usize>,
    pub hub_apex_constraint_ids: Vec<usize>,
    pub azimuth_hub_constraint_id: usize,
    pub shaft_base_azimuth_constraint_id: usize,
    pub yaw_bearing_shaft_base_constraint_id: usize,
    pub yaw_constraint_id: usize,
    pub tower_base_constraint_id: usize,

    pub torque: f64, // Torque applied to the hub
}

impl Turbine {
    pub fn new(input: TurbineInput, model: &mut Model) -> Self {
        //----------------------------------------------------------------------
        // Create blade and tower components
        //----------------------------------------------------------------------

        let blades = (0..input.n_blades)
            .map(|_| BeamComponent::new(input.blade.as_ref().unwrap(), model))
            .collect_vec();

        let tower = BeamComponent::new(&input.tower.unwrap(), model);

        //----------------------------------------------------------------------
        // Position and orient blades and tower, add rotor apex nodes
        //----------------------------------------------------------------------

        let origin = col![0., 0., 0.];

        // Calculate angle between blades
        let blade_angle_delta = 2. * PI / input.n_blades as f64;

        // Rotation for to orient beam axes along global z-axis
        let rv_x_to_z = col![0., -PI / 2., 0.];

        // Orient tower in global frame
        tower.nodes.iter().for_each(|node| {
            model.nodes[node.id].rotate(rv_x_to_z.as_ref(), origin.as_ref());
        });

        //----------------------------------------------------------------------
        // Additional nodes
        //----------------------------------------------------------------------

        // Calculate shaft length from tower top to hub and overhang
        let shaft_length = input.tower_axis_to_rotor_apex / input.shaft_tilt_angle.cos();

        // Get the tower top position
        let tt_node_id = tower.nodes.last().unwrap().id;
        let tt_position = Col::from_iter(model.nodes[tt_node_id].x.iter().cloned());

        // Calculate rotor apex position based on tower top and hub offset
        let apex_position = [
            tt_position[0] - input.tower_axis_to_rotor_apex,
            tt_position[1],
            tt_position[2] + input.tower_top_to_rotor_apex,
        ];

        // Create hub node at origin
        let hub_node_id = model
            .add_node()
            .position_xyz(-input.rotor_apex_to_hub, 0., 0.)
            .default_orientation()
            .build();

        // Create azimuth node at shaft base node
        let azimuth_node_id = model
            .add_node()
            .position_xyz(shaft_length, 0., 0.)
            .default_orientation()
            .build();

        // Create shaft base node relative to hub
        let shaft_base_node_id = model
            .add_node()
            .position_xyz(shaft_length, 0., 0.)
            .default_orientation()
            .build();

        // Create yaw bearing node at tower top position
        let yaw_bearing_node_id = model
            .add_node()
            .position_xyz(tt_position[0], tt_position[1], tt_position[2])
            .default_orientation()
            .build();

        // Loop through blades
        let rv_cone = col![0., -input.cone_angle, 0.];
        let apex_node_ids = blades
            .iter()
            .enumerate()
            .map(|(i, blade)| {
                // Calculate azimuth rotation for each blade
                let rv_azimuth = col![i as f64 * blade_angle_delta, 0., 0.];

                // Add rotor apex node
                let apex_node_id = model
                    .add_node()
                    .position_xyz(0., 0., 0.)
                    .default_orientation()
                    .build();

                // Rotate apex node for cone angle and blade azimuth
                model.nodes[apex_node_id]
                    .rotate(rv_cone.as_ref(), origin.as_ref())
                    .rotate(rv_azimuth.as_ref(), origin.as_ref());

                // Rotate blade to point along global z-axis, translate for hub radius,
                // apply cone angle, and adjust for blade azimuth
                blade.nodes.iter().for_each(|node| {
                    model.nodes[node.id]
                        .rotate(rv_x_to_z.as_ref(), origin.as_ref())
                        .translate([0., 0., input.hub_diameter / 2.])
                        .rotate(rv_cone.as_ref(), origin.as_ref())
                        .rotate(rv_azimuth.as_ref(), origin.as_ref());
                });

                apex_node_id
            })
            .collect_vec();

        // Get all node IDs for rotor components
        // Collect all rotor node IDs including hub, azimuth, shaft base, yaw bearing, and apex nodes
        let rotor_node_ids = vec![hub_node_id, azimuth_node_id, shaft_base_node_id]
            .iter()
            .chain(blades.iter().flat_map(|b| b.nodes.iter().map(|n| &n.id)))
            .chain(apex_node_ids.iter())
            .cloned()
            .collect_vec();

        // Apply shaft tilt to all rotor nodes and translate to apex position
        let rv_shaft_tilt = col![0., input.shaft_tilt_angle, 0.];
        rotor_node_ids.iter().for_each(|&node_id| {
            model.nodes[node_id]
                .rotate(rv_shaft_tilt.as_ref(), origin.as_ref())
                .translate(apex_position);
        });

        //----------------------------------------------------------------------
        // Constraints
        //----------------------------------------------------------------------

        let (pitch_constraint_ids, hub_apex_constraint_ids) = apex_node_ids
            .iter()
            .zip(&blades)
            .map(|(&apex_node_id, blade)| {
                let root_node_id = blade.nodes.first().unwrap().id;
                let axis = Col::from_fn(3, |i| {
                    model.nodes[apex_node_id].x[i] - model.nodes[root_node_id].x[i]
                });
                (
                    model.add_prescribed_rotation(apex_node_id, root_node_id, axis),
                    model.add_rigid_constraint(hub_node_id, apex_node_id),
                )
            })
            .unzip();

        // Connect azimuth node to hub node to with a rigid constraint
        let azimuth_hub_constraint_id = model.add_rigid_constraint(azimuth_node_id, hub_node_id);

        // Connect azimuth node to shaft base node with revolute joint
        let shaft_axis = col![
            input.shaft_tilt_angle.cos(),
            0.,
            -input.shaft_tilt_angle.sin()
        ];
        let shaft_base_azimuth_constraint_id = if input.prescribed_azimuth {
            model.add_prescribed_rotation(shaft_base_node_id, azimuth_node_id, shaft_axis)
        } else {
            model.add_revolute_joint(shaft_base_node_id, azimuth_node_id, shaft_axis)
        };

        // Connect yaw bearing to shaft base node with a rigid constraint
        let yaw_bearing_shaft_base_constraint_id =
            model.add_rigid_constraint(yaw_bearing_node_id, shaft_base_node_id);

        // Add yaw constraint for yaw bearing node
        let yaw_constraint_id =
            model.add_prescribed_rotation(tt_node_id, yaw_bearing_node_id, col![0., 0., 1.]);

        // Add prescribed constraint for tower base node
        let tower_base_constraint_id =
            model.add_prescribed_constraint(tower.nodes.first().unwrap().id);

        //----------------------------------------------------------------------
        // Initial displacements
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // Initial rotor velocity
        //----------------------------------------------------------------------

        let hub_node = &model.nodes[hub_node_id];
        let hub_node_pos = col![
            hub_node.x[0] + hub_node.u[0],
            hub_node.x[1] + hub_node.u[1],
            hub_node.x[2] + hub_node.u[2],
        ];

        let azimuth_node = &model.nodes[azimuth_node_id];
        let azimuth_node_pos = col![
            azimuth_node.x[0] + azimuth_node.u[0],
            azimuth_node.x[1] + azimuth_node.u[1],
            azimuth_node.x[2] + azimuth_node.u[2],
        ];

        // Get the shaft axis unit vector
        let shaft_axis =
            (&hub_node_pos - &azimuth_node_pos) / (&hub_node_pos - &azimuth_node_pos).norm_l2();

        let omega = input.rotor_speed * shaft_axis;
        let mut v = Col::<f64>::zeros(3);

        rotor_node_ids.iter().for_each(|&node_id| {
            let node = &model.nodes[node_id];
            let node_pos = col![
                node.x[0] + node.u[0],
                node.x[1] + node.u[1],
                node.x[2] + node.u[2],
            ];

            let r = node_pos - &hub_node_pos;
            cross_product(omega.rb(), r.rb(), v.as_mut());

            // Set initial velocities for rotor nodes
            model.nodes[node_id].v[0] = v[0];
            model.nodes[node_id].v[1] = v[1];
            model.nodes[node_id].v[2] = v[2];
            model.nodes[node_id].v[3] = omega[0];
            model.nodes[node_id].v[4] = omega[1];
            model.nodes[node_id].v[5] = omega[2];
        });

        //----------------------------------------------------------------------
        // Populate and return the Turbine struct
        //----------------------------------------------------------------------

        Turbine {
            blades,
            tower,
            hub_node: NodeData::new(hub_node_id),
            azimuth_node: NodeData::new(azimuth_node_id),
            shaft_base_node: NodeData::new(shaft_base_node_id),
            yaw_bearing_node: NodeData::new(yaw_bearing_node_id),
            apex_nodes: apex_node_ids
                .iter()
                .map(|&id| NodeData::new(id))
                .collect_vec(),
            pitch_constraint_ids,
            hub_apex_constraint_ids,
            azimuth_hub_constraint_id,
            shaft_base_azimuth_constraint_id,
            yaw_bearing_shaft_base_constraint_id,
            yaw_constraint_id,
            tower_base_constraint_id,
            torque: 0.0, // Initialize torque to zero
        }
    }

    pub fn set_loads(&mut self, state: &mut State) {
        // Set loads for each blade node
        self.blades.iter().for_each(|b| {
            b.nodes.iter().for_each(|n| {
                n.set_loads(state);
            });
        });

        // Set loads for tower nodes
        self.tower.nodes.iter().for_each(|n| {
            n.set_loads(state);
        });

        // Set loads for hub and azimuth nodes
        self.hub_node.set_loads(state);
        self.azimuth_node.set_loads(state);
    }

    pub fn get_motion(&mut self, state: &State) {
        // Get motion for each blade node
        self.blades.iter_mut().for_each(|b| {
            b.nodes.iter_mut().for_each(|n| {
                n.get_motion(state);
            });
        });

        // Get motion for tower nodes
        self.tower.nodes.iter_mut().for_each(|n| {
            n.get_motion(state);
        });

        // Get motion for hub and azimuth nodes
        self.hub_node.get_motion(state);
        self.azimuth_node.get_motion(state);
    }
}

//------------------------------------------------------------------------------
// TurbineInput
//------------------------------------------------------------------------------

pub struct TurbineInput {
    blade: Option<BeamInput>,      // Blade input data
    tower: Option<BeamInput>,      // Tower input data
    n_blades: usize,               // Number of blades
    tower_top_to_rotor_apex: f64,  // Distance from tower top to rotor apex (meters)
    tower_axis_to_rotor_apex: f64, // Distance from tower centerline to rotor apex (meters)
    rotor_apex_to_hub: f64,        // Distance from rotor apex to hub center of mass (meters)
    shaft_tilt_angle: f64,         // Shaft tilt angle (radians)
    hub_diameter: f64,             // Hub diameter (meters)
    cone_angle: f64,               // Define blade cone angle (radians)
    rotor_speed: f64,              // Initial rotor speed (rad/s)
    prescribed_azimuth: bool,      // Whether to prescribe azimuth angle
}

#[derive(Default)]
pub struct TurbineBuilder {
    blade: Option<BeamInput>,
    tower: Option<BeamInput>,
    n_blades: Option<usize>,
    tower_top_to_rotor_apex: Option<f64>,
    tower_axis_to_rotor_apex: Option<f64>,
    rotor_apex_to_hub: Option<f64>,
    shaft_tilt_angle: Option<f64>,
    azimuth_angle: Option<f64>,
    nacelle_yaw_angle: Option<f64>,
    blade_pitch_angle: Option<f64>,
    rotor_speed: Option<f64>,
    hub_diameter: Option<f64>,
    cone_angle: Option<f64>,
    prescribed_azimuth: Option<bool>,
}

impl TurbineBuilder {
    pub fn new() -> Self {
        TurbineBuilder {
            blade: None,
            tower: None,
            n_blades: None,
            tower_top_to_rotor_apex: None,
            tower_axis_to_rotor_apex: None,
            rotor_apex_to_hub: None,
            shaft_tilt_angle: None,
            azimuth_angle: None,
            nacelle_yaw_angle: None,
            blade_pitch_angle: None,
            rotor_speed: None,
            hub_diameter: None,
            cone_angle: None,
            prescribed_azimuth: None,
        }
    }
    pub fn set_blade_input(&mut self, blade: BeamInput) -> &mut Self {
        self.blade = Some(blade);
        self
    }

    pub fn set_tower_input(&mut self, tower: BeamInput) -> &mut Self {
        self.tower = Some(tower);
        self
    }

    pub fn set_n_blades(&mut self, n_blades: usize) -> &mut Self {
        self.n_blades = Some(n_blades);
        self
    }

    pub fn set_tower_top_to_rotor_apex(&mut self, distance: f64) -> &mut Self {
        self.tower_top_to_rotor_apex = Some(distance);
        self
    }

    pub fn set_tower_axis_to_rotor_apex(&mut self, distance: f64) -> &mut Self {
        self.tower_axis_to_rotor_apex = Some(distance);
        self
    }

    pub fn set_rotor_apex_to_hub(&mut self, distance: f64) -> &mut Self {
        self.rotor_apex_to_hub = Some(distance);
        self
    }

    /// Set the shaft tilt angle (radians).
    pub fn set_shaft_tilt_angle(&mut self, angle: f64) -> &mut Self {
        self.shaft_tilt_angle = Some(angle);
        self
    }

    /// Set the initial azimuth angle (radians).
    pub fn set_azimuth_angle(&mut self, angle: f64) -> &mut Self {
        self.azimuth_angle = Some(angle);
        self
    }

    /// Set the initial nacelle yaw angle (radians).
    pub fn set_nacelle_yaw_angle(&mut self, angle: f64) -> &mut Self {
        self.nacelle_yaw_angle = Some(angle);
        self
    }

    /// Set the initial blade pitch angle (radians).
    pub fn set_blade_pitch_angle(&mut self, angle: f64) -> &mut Self {
        self.blade_pitch_angle = Some(angle);
        self
    }

    /// Set the initial rotor speed (rad/s).
    pub fn set_rotor_speed(&mut self, speed: f64) -> &mut Self {
        self.rotor_speed = Some(speed);
        self
    }

    /// Set the hub diameter (meters).
    pub fn set_hub_diameter(&mut self, diameter: f64) -> &mut Self {
        self.hub_diameter = Some(diameter);
        self
    }

    /// Set the cone angle for the turbine blades (radians).
    pub fn set_cone_angle(&mut self, angle: f64) -> &mut Self {
        self.cone_angle = Some(angle);
        self
    }

    pub fn set_prescribed_azimuth(&mut self, prescribed: bool) -> &mut Self {
        self.prescribed_azimuth = Some(prescribed);
        self
    }

    pub fn build(&self, model: &mut Model) -> Result<Turbine, String> {
        Ok(Turbine::new(
            TurbineInput {
                blade: self.blade.clone(),
                tower: self.tower.clone(),
                n_blades: self.n_blades.ok_or("n_blades is required")?,
                tower_top_to_rotor_apex: self
                    .tower_top_to_rotor_apex
                    .ok_or("tower_top_to_rotor_apex is required")?,
                tower_axis_to_rotor_apex: self
                    .tower_axis_to_rotor_apex
                    .ok_or("tower_axis_to_rotor_apex is required")?,
                rotor_apex_to_hub: self
                    .rotor_apex_to_hub
                    .ok_or("rotor_apex_to_hub is required")?,
                shaft_tilt_angle: self
                    .shaft_tilt_angle
                    .ok_or("shaft_tilt_angle is required")?,
                hub_diameter: self.hub_diameter.unwrap_or_default(),
                cone_angle: self.cone_angle.unwrap_or_default(),
                rotor_speed: self.rotor_speed.unwrap_or(0.0),
                prescribed_azimuth: self.prescribed_azimuth.unwrap_or(false),
            },
            model,
        ))
    }
}

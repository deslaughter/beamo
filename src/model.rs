pub struct Model {
    pub nodes: Vec<Node>,
}

impl Model {
    pub fn new() -> Model {
        Model { nodes: vec![] }
    }
    // Creates and returns a node builder for adding a new node to the model
    pub fn new_node(&mut self) -> NodeBuilder {
        self.nodes.push(Node {
            id: self.nodes.len(),
            s: 0.,
            x: [0., 0., 0., 1., 0., 0., 0.],
            u: [0., 0., 0., 1., 0., 0., 0.],
            v: [0., 0., 0., 0., 0., 0.],
            vd: [0., 0., 0., 0., 0., 0.],
            dofs: 0,
        });

        // Return builder
        NodeBuilder {
            node: self.nodes.last_mut().unwrap(),
            built: false,
        }
    }
}

//------------------------------------------------------------------------------
// Node
//------------------------------------------------------------------------------

const TRANSLATION_DOFS: u8 = 7;
const ROTATION_DOFS: u8 = 56;

pub struct Node {
    /// Node identifier
    pub id: usize,
    /// Position along beam element length [0-1]
    pub s: f64,
    /// Initial position
    pub x: [f64; 7],
    /// Initial displacement
    pub u: [f64; 7],
    /// Initial velocity
    pub v: [f64; 6],
    /// Initial acceleration
    pub vd: [f64; 6],
    /// Packed active degrees of freedom
    pub dofs: u8,
}

pub struct NodeBuilder<'a> {
    node: &'a mut Node,
    built: bool,
}

impl<'a> NodeBuilder<'a> {
    /// Sets position of node within beam element
    pub fn element_location(self, s: f64) -> Self {
        self.node.s = s;
        self.node.dofs |= TRANSLATION_DOFS | ROTATION_DOFS;
        self
    }

    /// Sets initial position
    pub fn position(self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.x[3] = w;
        self.node.x[4] = i;
        self.node.x[5] = j;
        self.node.x[6] = k;
        self.node.dofs |= TRANSLATION_DOFS | ROTATION_DOFS;
        self
    }

    /// Sets initial displacement
    pub fn displacement(self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.u[3] = w;
        self.node.u[4] = i;
        self.node.u[5] = j;
        self.node.u[6] = k;
        self.node.dofs |= TRANSLATION_DOFS | ROTATION_DOFS;
        self
    }

    /// Sets initial velocity
    pub fn velocity(self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.v[3] = i;
        self.node.v[4] = j;
        self.node.v[5] = k;
        self.node.dofs |= TRANSLATION_DOFS | ROTATION_DOFS;
        self
    }

    /// Sets initial acceleration
    pub fn acceleration(self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.vd[3] = i;
        self.node.vd[4] = j;
        self.node.vd[5] = k;
        self.node.dofs |= TRANSLATION_DOFS | ROTATION_DOFS;
        self
    }

    /// Sets initial position
    pub fn position_xyz(self, x: f64, y: f64, z: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.dofs |= TRANSLATION_DOFS;
        self
    }

    /// Sets initial orientation from quaternion
    pub fn orientation(self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.x[3] = w;
        self.node.x[4] = x;
        self.node.x[5] = y;
        self.node.x[6] = z;
        self.node.dofs |= ROTATION_DOFS;
        self
    }

    /// Sets translational displacement
    pub fn translation_displacement(self, x: f64, y: f64, z: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.dofs |= TRANSLATION_DOFS;
        self
    }

    /// Sets angular displacement with quaternion
    pub fn angular_displacement(self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.u[3] = w;
        self.node.u[4] = x;
        self.node.u[5] = y;
        self.node.u[6] = z;
        self.node.dofs |= ROTATION_DOFS;
        self
    }

    /// Sets translational velocity
    pub fn translation_velocity(self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.dofs |= TRANSLATION_DOFS;
        self
    }

    /// Sets angular velocity
    pub fn angular_velocity(self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[3] = x;
        self.node.v[4] = y;
        self.node.v[5] = z;
        self.node.dofs |= ROTATION_DOFS;
        self
    }

    /// Sets translational acceleration
    pub fn translation_acceleration(self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.dofs |= TRANSLATION_DOFS;
        self
    }

    /// Sets angular acceleration
    pub fn angular_acceleration(self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[3] = x;
        self.node.vd[4] = y;
        self.node.vd[5] = z;
        self.node.dofs |= ROTATION_DOFS;
        self
    }

    pub fn build(mut self) -> usize {
        self.built = true;
        self.node.id
    }
}

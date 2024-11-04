pub struct Node {
    pub id: usize,
    pub x: [f64; 7],  // Initial position
    pub u: [f64; 7],  // Initial displacement
    pub v: [f64; 6],  // Initial velocity
    pub vd: [f64; 6], // Initial acceleration
    pub dofs: u8,     // Packed active degrees of freedom
}

pub struct NodeBuilder {
    node: Node,
}

impl NodeBuilder {
    pub fn new(id: usize) -> Self {
        Self {
            node: Node {
                id,
                x: [0., 0., 0., 1., 0., 0., 0.],
                u: [0., 0., 0., 1., 0., 0., 0.],
                v: [0., 0., 0., 0., 0., 0.],
                vd: [0., 0., 0., 0., 0., 0.],
                dofs: 0,
            },
        }
    }

    // Sets initial position
    pub fn position(mut self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.x[3] = w;
        self.node.x[4] = i;
        self.node.x[5] = j;
        self.node.x[6] = k;
        self.node.dofs |= 63;
        self
    }

    // Sets initial displacement
    pub fn displacement(mut self, x: f64, y: f64, z: f64, w: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.u[3] = w;
        self.node.u[4] = i;
        self.node.u[5] = j;
        self.node.u[6] = k;
        self.node.dofs |= 63;
        self
    }

    // Sets initial velocity
    pub fn velocity(mut self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.v[3] = i;
        self.node.v[4] = j;
        self.node.v[5] = k;
        self.node.dofs |= 63;
        self
    }

    // Sets initial acceleration
    pub fn acceleration(mut self, x: f64, y: f64, z: f64, i: f64, j: f64, k: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.vd[3] = i;
        self.node.vd[4] = j;
        self.node.vd[5] = k;
        self.node.dofs |= 63;
        self
    }

    // Sets initial position
    pub fn position_xyz(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.x[0] = x;
        self.node.x[1] = y;
        self.node.x[2] = z;
        self.node.dofs |= 7;
        self
    }

    // Sets initial orientation from quaternion
    pub fn orientation(mut self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.x[3] = w;
        self.node.x[4] = x;
        self.node.x[5] = y;
        self.node.x[6] = z;
        self.node.dofs |= 56;
        self
    }

    // Sets translational displacement
    pub fn translation_displacement(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.u[0] = x;
        self.node.u[1] = y;
        self.node.u[2] = z;
        self.node.dofs |= 7;
        self
    }

    // Sets angular displacement with quaternion
    pub fn angular_displacement(mut self, w: f64, x: f64, y: f64, z: f64) -> Self {
        self.node.u[3] = w;
        self.node.u[4] = x;
        self.node.u[5] = y;
        self.node.u[6] = z;
        self.node.dofs |= 56;
        self
    }

    // Sets translational velocity
    pub fn translation_velocity(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[0] = x;
        self.node.v[1] = y;
        self.node.v[2] = z;
        self.node.dofs |= 7;
        self
    }

    // Sets angular velocity
    pub fn angular_velocity(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.v[3] = x;
        self.node.v[4] = y;
        self.node.v[5] = z;
        self.node.dofs |= 56;
        self
    }

    // Sets translational acceleration
    pub fn translation_acceleration(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[0] = x;
        self.node.vd[1] = y;
        self.node.vd[2] = z;
        self.node.dofs |= 7;
        self
    }

    // Sets angular acceleration
    pub fn angular_acceleration(mut self, x: f64, y: f64, z: f64) -> Self {
        self.node.vd[3] = x;
        self.node.vd[4] = y;
        self.node.vd[5] = z;
        self.node.dofs |= 56;
        self
    }

    pub fn build(self) -> Node {
        self.node
    }
}

use ndarray::{array, Array2};

#[derive(Clone, Debug, Copy)]
pub struct Quaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quaternion {
    pub fn identity() -> Self {
        Quaternion {
            w: 1.,
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    pub fn from_vec(a: &[f64; 4]) -> Self {
        return Self {
            w: a[0],
            x: a[1],
            y: a[2],
            z: a[3],
        };
    }

    pub fn as_vec(self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }

    pub fn compose(self, q2: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * q2.w - self.x * q2.x - self.y * q2.y - self.z * q2.z,
            x: self.w * q2.x + self.x * q2.w + self.y * q2.z - self.z * q2.y,
            y: self.w * q2.y - self.x * q2.z + self.y * q2.w + self.z * q2.x,
            z: self.w * q2.z + self.x * q2.y - self.y * q2.x + self.z * q2.w,
        }
    }

    pub fn as_matrix(self) -> [[f64; 3]; 3] {
        [
            [
                self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z,
                2. * (self.x * self.y - self.w * self.z),
                2. * (self.x * self.z + self.w * self.y),
            ],
            [
                2. * (self.x * self.y + self.w * self.z),
                self.w * self.w - self.x * self.x + self.y * self.y - self.z * self.z,
                2. * (self.y * self.z - self.w * self.x),
            ],
            [
                2. * (self.x * self.z - self.w * self.y),
                2. * (self.y * self.z + self.w * self.x),
                self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z,
            ],
        ]
    }

    pub fn from_matrix(m: &[[f64; 3]; 3]) -> Self {
        let m22_p_m33 = m[1][1] + m[2][2];
        let m22_m_m33 = m[1][1] - m[2][2];
        let vals = vec![
            m[0][0] + m22_p_m33,
            m[0][0] - m22_p_m33,
            -m[0][0] + m22_m_m33,
            -m[0][0] - m22_m_m33,
        ];
        let (max_idx, max_num) =
            vals.iter()
                .enumerate()
                .fold((0, vals[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });

        let half = 0.5;
        let tmp = (max_num + 1.).sqrt();
        let coef = half / tmp;

        match max_idx {
            0 => Self {
                w: half * tmp,
                x: (m[2][1] - m[1][2]) * coef,
                y: (m[0][2] - m[2][0]) * coef,
                z: (m[1][0] - m[0][1]) * coef,
            },
            1 => Self {
                w: (m[2][1] - m[1][2]) * coef,
                x: half * tmp,
                y: (m[0][1] + m[1][0]) * coef,
                z: (m[0][2] + m[2][0]) * coef,
            },
            2 => Self {
                w: (m[0][2] - m[2][0]) * coef,
                x: (m[0][1] + m[1][0]) * coef,
                y: half * tmp,
                z: (m[1][2] + m[2][1]) * coef,
            },
            3 => Self {
                w: (m[1][0] - m[0][1]) * coef,
                x: (m[0][2] + m[2][0]) * coef,
                y: (m[1][2] + m[2][1]) * coef,
                z: half * tmp,
            },
            _ => unreachable!(),
        }
    }

    pub fn from_axis_angle(angle: f64, axis: &[f64; 3]) -> Self {
        if angle < 1e-12 {
            Quaternion {
                w: 1.,
                x: 0.,
                y: 0.,
                z: 0.,
            }
        } else {
            let (sin, cos) = (angle / 2.).sin_cos();
            let factor = sin / angle;
            Quaternion {
                w: cos,
                x: angle * axis[0] * factor,
                y: angle * axis[1] * factor,
                z: angle * axis[2] * factor,
            }
        }
    }

    pub fn from_scaled_axis(v: &[f64; 3]) -> Self {
        return Self {
            w: a[0],
            x: a[1],
            y: a[2],
            z: a[3],
        };
    }

    pub fn rotate_vector(self, v: &[f64; 3]) -> [f64; 3] {
        [
            (self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z) * v[0]
                + 2. * (self.x * self.y - self.w * self.z) * v[1]
                + 2. * (self.x * self.z + self.w * self.y) * v[2],
            2. * (self.x * self.y + self.w * self.z) * v[0]
                + (self.w * self.w - self.x * self.x + self.y * self.y - self.z * self.z) * v[1]
                + 2. * (self.y * self.z - self.w * self.x) * v[2],
            2. * (self.x * self.z - self.w * self.y) * v[0]
                + 2. * (self.y * self.z + self.w * self.x) * v[1]
                + (self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z) * v[2],
        ]
    }

    pub fn derivative(self) -> Array2<f64> {
        array![
            [-self.x, self.w, -self.y, self.z],
            [-self.z, self.y, self.w, -self.x],
            [-self.y, -self.z, self.x, self.w],
        ]
    }
}

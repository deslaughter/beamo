use std::f64::consts::PI;

use faer::{col, mat, row, Col, ColMut, ColRef, MatMut, MatRef, Row, RowMut};

#[derive(Clone, Debug, Copy)]
pub struct Quaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

pub trait Quat {
    fn quat_from_rotation_vector(&mut self, v: ColRef<f64>);
    fn quat_from_axis_angle(&mut self, angle: f64, axis: ColRef<f64>);
    fn quat_from_rotation_matrix(&mut self, r: MatRef<f64>);
    fn quat_compose(&mut self, q1: ColRef<f64>, q2: ColRef<f64>);
    fn quat_rotate_vector(&self, v: ColMut<f64>);
    fn quat_derivative(&self, m: MatMut<f64>);
    fn quat_from_tangent_twist(&mut self, tangent: ColRef<f64>, twist: f64);
    fn quat_as_matrix(&self, m: MatMut<f64>);
    fn quat_from_identity(&mut self);
}

impl Quat for ColMut<'_, f64> {
    #[inline]
    fn quat_from_identity(&mut self) {
        self[0] = 1.;
        self[1] = 0.;
        self[2] = 0.;
        self[3] = 0.;
    }

    /// Populates matrix with rotation matrix equivalent of quaternion.
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `m.nrows() < 3`.  
    /// Panics if `m.ncols() < 3`.  
    #[inline]
    fn quat_as_matrix(&self, mut m: MatMut<f64>) {
        m[(0, 0)] = self[0] * self[0] + self[1] * self[1] - self[2] * self[2] - self[3] * self[3];
        m[(0, 1)] = 2. * (self[1] * self[2] - self[0] * self[3]);
        m[(0, 2)] = 2. * (self[1] * self[3] + self[0] * self[2]);

        m[(1, 0)] = 2. * (self[1] * self[2] + self[0] * self[3]);
        m[(1, 1)] = self[0] * self[0] - self[1] * self[1] + self[2] * self[2] - self[3] * self[3];
        m[(1, 2)] = 2. * (self[2] * self[3] - self[0] * self[1]);

        m[(2, 0)] = 2. * (self[1] * self[3] - self[0] * self[2]);
        m[(2, 1)] = 2. * (self[2] * self[3] + self[0] * self[1]);
        m[(2, 2)] = self[0] * self[0] - self[1] * self[1] - self[2] * self[2] + self[3] * self[3];
    }

    /// Populates matrix with quaternion derivative
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `m.nrows() < 3`.  
    /// Panics if `m.ncols() < 4`.  
    #[inline]
    fn quat_derivative(&self, mut m: MatMut<f64>) {
        m[(0, 0)] = -self[1];
        m[(0, 1)] = self[0];
        m[(0, 2)] = -self[2];
        m[(0, 3)] = self[3];

        m[(1, 0)] = -self[3];
        m[(1, 1)] = self[2];
        m[(1, 2)] = self[0];
        m[(1, 3)] = -self[1];

        m[(1, 0)] = -self[2];
        m[(1, 1)] = -self[3];
        m[(1, 2)] = self[1];
        m[(1, 3)] = self[0];
    }

    /// Populates Quaternion from rotation vector
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `v.ncols() < 3`.  
    #[inline]
    fn quat_from_rotation_vector(&mut self, v: ColRef<f64>) {
        let angle = v.norm_l2();
        if angle < 1e-12 {
            self[0] = 1.;
            self[1] = 0.;
            self[2] = 0.;
            self[3] = 0.;
        } else {
            let (sin, cos) = (angle / 2.).sin_cos();
            let factor = sin / angle;
            self[0] = cos;
            self[1] = v[0] * factor;
            self[2] = v[1] * factor;
            self[3] = v[2] * factor;
        }
    }

    /// Populates Quaternion from axis-angle representation
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `axis.ncols() < 3`.  
    #[inline]
    fn quat_from_axis_angle(&mut self, angle: f64, axis: ColRef<f64>) {
        if angle < 1e-12 {
            self[0] = 1.;
            self[1] = 0.;
            self[2] = 0.;
            self[3] = 0.;
        } else {
            let (sin, cos) = (angle / 2.).sin_cos();
            let factor = sin / angle;
            self[0] = cos;
            self[1] = angle * axis[0] * factor;
            self[2] = angle * axis[1] * factor;
            self[3] = angle * axis[2] * factor;
        }
    }

    /// Populates Quaternion from rotation matrix
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `m.nrows() < 3`.
    /// Panics if `m.ncols() < 3`.
    #[inline]
    fn quat_from_rotation_matrix(&mut self, m: MatRef<f64>) {
        let m22_p_m33 = m[(1, 1)] + m[(2, 2)];
        let m22_m_m33 = m[(1, 1)] - m[(2, 2)];
        let vals = vec![
            m[(0, 0)] + m22_p_m33,
            m[(0, 0)] - m22_p_m33,
            -m[(0, 0)] + m22_m_m33,
            -m[(0, 0)] - m22_m_m33,
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
            0 => {
                self[0] = half * tmp;
                self[1] = (m[(2, 1)] - m[(1, 2)]) * coef;
                self[2] = (m[(0, 2)] - m[(2, 0)]) * coef;
                self[3] = (m[(1, 0)] - m[(0, 1)]) * coef;
            }
            1 => {
                self[0] = (m[(2, 1)] - m[(1, 2)]) * coef;
                self[1] = half * tmp;
                self[2] = (m[(0, 1)] + m[(1, 0)]) * coef;
                self[3] = (m[(0, 2)] + m[(2, 0)]) * coef;
            }
            2 => {
                self[0] = (m[(0, 2)] - m[(2, 0)]) * coef;
                self[1] = (m[(0, 1)] + m[(1, 0)]) * coef;
                self[2] = half * tmp;
                self[3] = (m[(1, 2)] + m[(2, 1)]) * coef;
            }
            3 => {
                self[0] = (m[(1, 0)] - m[(0, 1)]) * coef;
                self[1] = (m[(0, 2)] + m[(2, 0)]) * coef;
                self[2] = (m[(1, 2)] + m[(2, 1)]) * coef;
                self[3] = half * tmp;
            }
            _ => unreachable!(),
        }
    }

    /// Populates Quaternion from composition of q1 and q2.
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `q1.ncols() < 4`.  
    /// Panics if `q2.ncols() < 4`.  
    #[inline]
    fn quat_compose(&mut self, q1: ColRef<f64>, q2: ColRef<f64>) {
        self[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        self[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        self[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        self[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }

    #[inline]
    fn quat_rotate_vector(&self, mut v: ColMut<f64>) {
        v[0] = (self[0] * self[0] + self[1] * self[1] - self[2] * self[2] - self[3] * self[3])
            * v[0]
            + 2. * (self[1] * self[2] - self[0] * self[3]) * v[1]
            + 2. * (self[1] * self[3] + self[0] * self[2]) * v[2];
        v[1] = 2. * (self[1] * self[2] + self[0] * self[3]) * v[0]
            + (self[0] * self[0] - self[1] * self[1] + self[2] * self[2] - self[3] * self[3])
                * v[1]
            + 2. * (self[2] * self[3] - self[0] * self[1]) * v[2];
        v[2] = 2. * (self[1] * self[3] - self[0] * self[2]) * v[0]
            + 2. * (self[2] * self[3] + self[0] * self[1]) * v[1]
            + (self[0] * self[0] - self[1] * self[1] - self[2] * self[2] + self[3] * self[3])
                * v[2];
    }

    /// Populates Quaternion from tangent vector and twist angle.
    ///
    /// # Panics
    /// Panics if `self.ncols() < 4`.  
    /// Panics if `tangent.ncols() < 4`.  
    fn quat_from_tangent_twist(&mut self, tangent: ColRef<f64>, twist: f64) {
        let e1 = Col::from_fn(3, |i| tangent[i]);
        let a = if e1[0] > 0. { 1. } else { -1. };
        let e2 = col![
            -a * e1[1] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
            a * e1[0] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
            0.,
        ];

        let mut e3 = Col::<f64>::zeros(3);
        cross(e1.as_ref(), e2.as_ref(), e3.as_mut());

        let mut q0 = Col::<f64>::zeros(4);
        q0.as_mut().quat_from_rotation_matrix(
            mat![
                [e1[0], e2[0], e3[0]],
                [e1[1], e2[1], e3[1]],
                [e1[2], e2[2], e3[2]],
            ]
            .as_ref(),
        );

        //  Matrix3::from_columns(&[e1, e2, e3]);
        let mut q_twist = Col::<f64>::zeros(4);
        q_twist
            .as_mut()
            .quat_from_axis_angle(twist * PI / 180., e1.as_ref());
        self.quat_compose(q_twist.as_ref(), q0.as_ref());
    }
}

// Returns the cross product of two vectors
pub fn cross(a: ColRef<f64>, b: ColRef<f64>, mut c: ColMut<f64>) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

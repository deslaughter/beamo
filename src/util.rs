use std::f64::consts::PI;

use faer::{col, mat, unzipped, zipped, Col, ColMut, ColRef, Entity, MatMut, MatRef};

pub trait Quat {
    fn quat_from_rotation_vector(&mut self, v: ColRef<f64>);
    fn quat_from_axis_angle(&mut self, angle: f64, axis: ColRef<f64>);
    fn quat_from_rotation_matrix(&mut self, r: MatRef<f64>);
    fn quat_compose(&mut self, q1: ColRef<f64>, q2: ColRef<f64>);
    fn quat_from_tangent_twist(&mut self, tangent: ColRef<f64>, twist: f64);
    fn quat_from_identity(&mut self);
}

#[inline]
pub fn axial_vector_of_matrix(m: MatRef<f64>, mut v: ColMut<f64>) {
    v[0] = (m[(2, 1)] - m[(1, 2)]) / 2.;
    v[1] = (m[(0, 2)] - m[(2, 0)]) / 2.;
    v[2] = (m[(1, 0)] - m[(0, 1)]) / 2.;
}

#[inline]
/// AX(A) = tr(A)/2 * I - A/2, where I is the identity matrix
pub fn matrix_ax(m: MatRef<f64>, mut ax: MatMut<f64>) {
    zipped!(&mut ax, &m).for_each(|unzipped!(mut ax, m)| *ax = -*m / 2.);
    let trace_2 = m.diagonal().column_vector().sum() / 2.;
    ax[(0, 0)] += trace_2;
    ax[(1, 1)] += trace_2;
    ax[(2, 2)] += trace_2;
}

#[inline]
pub fn quat_inverse(q_in: ColRef<f64>, mut q_out: ColMut<f64>) {
    let length = q_in.norm_l2();
    q_out[0] = q_in[0] / length;
    q_out[1] = -q_in[1] / length;
    q_out[2] = -q_in[2] / length;
    q_out[3] = -q_in[3] / length;
}

#[inline]
pub fn quat_as_rotation_vector(q: ColRef<f64>, mut v: ColMut<f64>) {
    let norm = q.norm_l2();
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    // Calculate the angle (in radians) and the rotation axis
    let angle = 2.0 * w.acos();
    let angle = if angle > std::f64::consts::PI {
        angle - 2.0 * std::f64::consts::PI
    } else {
        angle
    };
    let norm = (1.0 - w * w).sqrt();

    // To avoid division by zero, check if norm is very small
    if norm < 1e-10 {
        // If norm is close to zero, the rotation is very small; return a zero vector
        v.fill_zero();
    } else {
        // Compute the rotation vector (axis * angle)
        let scale = angle / norm;
        v[0] = x * scale;
        v[1] = y * scale;
        v[2] = z * scale;
    }
}

#[inline]
pub fn quat_as_euler_angles(q: ColRef<f64>, mut v: ColMut<f64>) {
    let norm = q.norm_l2();
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    v[0] = (2. * (w * x + y * z)).atan2(1. - 2. * (x * x + y * y));
    let a = (1. + 2. * (w * y - x * z)).sqrt();
    let b = (1. - 2. * (w * y - x * z)).sqrt();
    v[1] = -PI / 2. + 2. * a.atan2(b);
    v[2] = (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z));
}

/// Populates matrix with rotation matrix equivalent of quaternion.
///
/// # Panics
/// Panics if `self.ncols() < 4`.  
/// Panics if `m.nrows() < 3`.  
/// Panics if `m.ncols() < 3`.  
#[inline]
pub fn quat_as_matrix(v: ColRef<f64>, mut m: MatMut<f64>) {
    let (w, i, j, k) = (v[0], v[1], v[2], v[3]);
    let ww = w * w;
    let ii = i * i;
    let jj = j * j;
    let kk = k * k;
    let ij = i * j * 2.;
    let wk = w * k * 2.;
    let wj = w * j * 2.;
    let ik = i * k * 2.;
    let jk = j * k * 2.;
    let wi = w * i * 2.;

    m[(0, 0)] = ww + ii - jj - kk;
    m[(0, 1)] = ij - wk;
    m[(0, 2)] = ik + wj;

    m[(1, 0)] = ij + wk;
    m[(1, 1)] = ww - ii + jj - kk;
    m[(1, 2)] = jk - wi;

    m[(2, 0)] = ik - wj;
    m[(2, 1)] = jk + wi;
    m[(2, 2)] = ww - ii - jj + kk;
}

#[inline]
pub fn quat_rotate_vector(q: ColRef<f64>, v_in: ColRef<f64>, mut v_out: ColMut<f64>) {
    v_out[0] = (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]) * v_in[0]
        + 2. * (q[1] * q[2] - q[0] * q[3]) * v_in[1]
        + 2. * (q[1] * q[3] + q[0] * q[2]) * v_in[2];
    v_out[1] = 2. * (q[1] * q[2] + q[0] * q[3]) * v_in[0]
        + (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]) * v_in[1]
        + 2. * (q[2] * q[3] - q[0] * q[1]) * v_in[2];
    v_out[2] = 2. * (q[1] * q[3] - q[0] * q[2]) * v_in[0]
        + 2. * (q[2] * q[3] + q[0] * q[1]) * v_in[1]
        + (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]) * v_in[2];
}

/// Populates matrix with quaternion derivative
///
/// # Panics
/// Panics if `self.ncols() < 4`.  
/// Panics if `m.nrows() < 3`.  
/// Panics if `m.ncols() < 4`.  
#[inline]
pub fn quat_derivative(q: ColRef<f64>, mut m: MatMut<f64>) {
    m[(0, 0)] = -q[1];
    m[(0, 1)] = q[0];
    m[(0, 2)] = -q[3];
    m[(0, 3)] = q[2];
    m[(1, 0)] = -q[2];
    m[(1, 1)] = q[3];
    m[(1, 2)] = q[0];
    m[(1, 3)] = -q[1];
    m[(2, 0)] = -q[3];
    m[(2, 1)] = -q[2];
    m[(2, 2)] = q[1];
    m[(2, 3)] = q[0];
}

impl Quat for ColMut<'_, f64> {
    #[inline]
    fn quat_from_identity(&mut self) {
        self[0] = 1.;
        self[1] = 0.;
        self[2] = 0.;
        self[3] = 0.;
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
        if angle.abs() < 1e-12 {
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
        let c = half / tmp;

        match max_idx {
            0 => {
                self[0] = half * tmp;
                self[1] = (m[(2, 1)] - m[(1, 2)]) * c;
                self[2] = (m[(0, 2)] - m[(2, 0)]) * c;
                self[3] = (m[(1, 0)] - m[(0, 1)]) * c;
            }
            1 => {
                self[0] = (m[(2, 1)] - m[(1, 2)]) * c;
                self[1] = half * tmp;
                self[2] = (m[(0, 1)] + m[(1, 0)]) * c;
                self[3] = (m[(0, 2)] + m[(2, 0)]) * c;
            }
            2 => {
                self[0] = (m[(0, 2)] - m[(2, 0)]) * c;
                self[1] = (m[(0, 1)] + m[(1, 0)]) * c;
                self[2] = half * tmp;
                self[3] = (m[(1, 2)] + m[(2, 1)]) * c;
            }
            3 => {
                self[0] = (m[(1, 0)] - m[(0, 1)]) * c;
                self[1] = (m[(0, 2)] + m[(2, 0)]) * c;
                self[2] = (m[(1, 2)] + m[(2, 1)]) * c;
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
        let m = self.norm_l2();
        self[0] /= m;
        self[1] /= m;
        self[2] /= m;
        self[3] /= m;
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

pub fn vec_tilde(v: ColRef<f64>, mut m: MatMut<f64>) {
    // [0., -v[2], v[1]]
    // [v[2], 0., -v[0]]
    // [-v[1], v[0], 0.]
    m[(0, 0)] = 0.;
    m[(1, 0)] = v[2];
    m[(2, 0)] = -v[1];
    m[(0, 1)] = -v[2];
    m[(1, 1)] = 0.;
    m[(2, 1)] = v[0];
    m[(0, 2)] = v[1];
    m[(1, 2)] = -v[0];
    m[(2, 2)] = 0.;
}

pub trait ColAsMatMut<'a, T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T>;
    fn as_mat_mut(self, nrows: usize, ncols: usize) -> MatMut<'a, T>;
}

impl<'a, T> ColAsMatMut<'a, T> for Col<T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
    fn as_mat_mut(mut self, nrows: usize, ncols: usize) -> MatMut<'a, T> {
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, ncols, 1, nrows as isize) }
    }
}

impl<'a, T> ColAsMatMut<'a, T> for ColMut<'a, T>
where
    T: Entity,
{
    fn as_shape(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
    fn as_mat_mut(self, nrows: usize, ncols: usize) -> MatMut<'a, T> {
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, ncols, 1, nrows as isize) }
    }
}

pub trait ColAsMatRef<'a, T>
where
    T: Entity,
{
    fn as_mat_ref(self, nrows: usize, ncols: usize) -> MatRef<'a, T>;
}

impl<'a, T> ColAsMatRef<'a, T> for ColRef<'a, T>
where
    T: Entity,
{
    fn as_mat_ref(self, nrows: usize, ncols: usize) -> MatRef<'a, T> {
        unsafe { mat::from_raw_parts(self.as_ptr(), nrows, ncols, 1, nrows as isize) }
    }
}

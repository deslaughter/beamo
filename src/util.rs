use faer::{prelude::*, sparse::*};
use itertools::Itertools;
use std::f64::consts::PI;

// Calculate the mass and stiffness matrices for an annular section
#[allow(non_snake_case)]
pub fn annular_section(
    outer_diameter: f64,
    wall_thickness: f64,
    elastic_modulus: f64,
    shear_modulus: f64,
    poisson_ratio: f64,
    density: f64,
) -> (Mat<f64>, Mat<f64>) {
    // Geometry
    let outer_radius = outer_diameter / 2.0;
    let inner_radius = outer_radius - wall_thickness;
    let area = PI * (outer_radius.powi(2) - inner_radius.powi(2));
    let I = PI * (outer_radius.powi(4) - inner_radius.powi(4)) / 4.;
    let J = PI * (outer_radius.powi(4) - inner_radius.powi(4)) / 2.;

    // Shear correction factor for circular sections
    let k = (6. * (1. + poisson_ratio)) / (7. + 6. * poisson_ratio);

    // Calculate mass properties
    let m = density * area; // Mass per unit length
    let rhoI = density * I; // Mass moment of inertia

    // Calculate stiffness properties
    let EA = elastic_modulus * area;
    let EI = elastic_modulus * I;
    let GKt = shear_modulus * J;
    let GAk = shear_modulus * area * k;

    let m = mat![
        [m, 0., 0., 0., 0., 0.],
        [0., m, 0., 0., 0., 0.],
        [0., 0., m, 0., 0., 0.],
        [0., 0., 0., rhoI, 0., 0.],
        [0., 0., 0., 0., rhoI, 0.],
        [0., 0., 0., 0., 0., 2. * rhoI],
    ];

    let k = mat![
        [GAk, 0., 0., 0., 0., 0.],
        [0., GAk, 0., 0., 0., 0.],
        [0., 0., EA, 0., 0., 0.],
        [0., 0., 0., EI, 0., 0.],
        [0., 0., 0., 0., EI, 0.],
        [0., 0., 0., 0., 0., GKt],
    ];
    (m, k)
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
    zip!(&mut ax, &m).for_each(|unzip!(ax, m)| *ax = -*m / 2.);
    let trace_2 = m.diagonal().column_vector().sum() / 2.;
    ax[(0, 0)] += trace_2;
    ax[(1, 1)] += trace_2;
    ax[(2, 2)] += trace_2;
}

#[inline]
/// AX2(a,b)
pub fn matrix_ax2(a: MatRef<f64>, b: ColRef<f64>, mut out: MatMut<f64>) {
    out[(0, 0)] = ((a[(1, 2)] - a[(2, 1)]) * b[0]) / 2.;
    out[(0, 1)] = (-a[(0, 2)] * b[0] - a[(2, 1)] * b[1] - a[(2, 2)] * b[2]) / 2.;
    out[(0, 2)] = (a[(0, 1)] * b[0] + a[(1, 1)] * b[1] + a[(1, 2)] * b[2]) / 2.;

    out[(1, 0)] = (a[(1, 2)] * b[1] + a[(2, 0)] * b[0] + a[(2, 2)] * b[2]) / 2.;
    out[(1, 1)] = ((-a[(0, 2)] + a[(2, 0)]) * b[1]) / 2.;
    out[(1, 2)] = (-a[(0, 0)] * b[0] - a[(0, 2)] * b[2] - a[(1, 0)] * b[1]) / 2.;

    out[(2, 0)] = (-a[(1, 0)] * b[0] - a[(1, 1)] * b[1] - a[(2, 1)] * b[2]) / 2.;
    out[(2, 1)] = (a[(0, 0)] * b[0] + a[(0, 1)] * b[1] + a[(2, 0)] * b[2]) / 2.;
    out[(2, 2)] = ((a[(0, 1)] - a[(1, 0)]) * b[2]) / 2.;
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
pub fn quat_inverse_alloc(q_in: ColRef<f64>) -> Col<f64> {
    let mut q_out = Col::<f64>::zeros(4);
    quat_inverse(q_in, q_out.as_mut());
    q_out
}

pub fn quat_as_rotation_vector(q: ColRef<f64>, mut v: ColMut<f64>) {
    let norm = q.norm_l2();
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    // Calculate the angle (in radians) and the rotation axis
    let angle = 2. * w.acos();
    let angle = if angle > std::f64::consts::PI {
        angle - 2. * std::f64::consts::PI
    } else {
        angle
    };
    let norm = (1. - w * w).sqrt();

    // To avoid division by zero, check if norm is very small
    if norm < f64::EPSILON {
        // If norm is close to zero, the rotation is very small; return a zero vector
        v.fill(0.);
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
pub fn quat_as_matrix(q: ColRef<f64>, mut m: MatMut<f64>) {
    let (w, i, j, k) = (q[0], q[1], q[2], q[3]);
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
pub fn quat_as_matrix_alloc(q: ColRef<f64>) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(3, 3);
    quat_as_matrix(q, m.as_mut());
    m
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

#[inline]
pub fn quat_rotate_vector_alloc(q: ColRef<f64>, v_in: ColRef<f64>) -> Col<f64> {
    let mut v_out = Col::<f64>::zeros(3);
    quat_rotate_vector(q, v_in, v_out.as_mut());
    v_out
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

#[inline]
pub fn quat_from_rotation_vector(rv: ColRef<f64>, mut q: ColMut<f64>) {
    let angle = rv.norm_l2();
    if angle < f64::EPSILON {
        q[0] = 1.;
        q[1] = 0.;
        q[2] = 0.;
        q[3] = 0.;
    } else {
        let (sin, cos) = (angle / 2.).sin_cos();
        let factor = sin / angle;
        q[0] = cos;
        q[1] = rv[0] * factor;
        q[2] = rv[1] * factor;
        q[3] = rv[2] * factor;
    }
}

#[inline]
pub fn quat_from_rotation_vector_alloc(rv: ColRef<f64>) -> Col<f64> {
    let mut q = Col::<f64>::zeros(4);
    quat_from_rotation_vector(rv, q.as_mut());
    q
}

/// Populates Quaternion from rotation vector
///
/// # Panics
/// Panics if `self.ncols() < 4`.
/// Panics if `v.ncols() < 3`.

/// Populates Quaternion from axis-angle representation
///
/// # Panics
/// Panics if `self.ncols() < 4`.
/// Panics if `axis.ncols() < 3`.
#[inline]
pub fn quat_from_axis_angle(angle: f64, axis: ColRef<f64>, mut q: ColMut<f64>) {
    if angle.abs() < 1e-12 {
        q[0] = 1.;
        q[1] = 0.;
        q[2] = 0.;
        q[3] = 0.;
    } else {
        let (sin, cos) = (angle / 2.).sin_cos();
        let factor = sin / angle;
        q[0] = cos;
        q[1] = angle * axis[0] * factor;
        q[2] = angle * axis[1] * factor;
        q[3] = angle * axis[2] * factor;
    }
}

/// Populates Quaternion from rotation matrix
///
/// # Panics
/// Panics if `self.ncols() < 4`.
/// Panics if `m.nrows() < 3`.
/// Panics if `m.ncols() < 3`.
#[inline]
pub fn quat_from_rotation_matrix(m: MatRef<f64>, mut q: ColMut<f64>) {
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
            q[0] = half * tmp;
            q[1] = (m[(2, 1)] - m[(1, 2)]) * c;
            q[2] = (m[(0, 2)] - m[(2, 0)]) * c;
            q[3] = (m[(1, 0)] - m[(0, 1)]) * c;
        }
        1 => {
            q[0] = (m[(2, 1)] - m[(1, 2)]) * c;
            q[1] = half * tmp;
            q[2] = (m[(0, 1)] + m[(1, 0)]) * c;
            q[3] = (m[(0, 2)] + m[(2, 0)]) * c;
        }
        2 => {
            q[0] = (m[(0, 2)] - m[(2, 0)]) * c;
            q[1] = (m[(0, 1)] + m[(1, 0)]) * c;
            q[2] = half * tmp;
            q[3] = (m[(1, 2)] + m[(2, 1)]) * c;
        }
        3 => {
            q[0] = (m[(1, 0)] - m[(0, 1)]) * c;
            q[1] = (m[(0, 2)] + m[(2, 0)]) * c;
            q[2] = (m[(1, 2)] + m[(2, 1)]) * c;
            q[3] = half * tmp;
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
pub fn quat_compose(q1: ColRef<f64>, q2: ColRef<f64>, mut q_out: ColMut<f64>) {
    q_out[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    q_out[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    q_out[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    q_out[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
}

#[inline]
pub fn quat_compose_alloc(q1: ColRef<f64>, q2: ColRef<f64>) -> Col<f64> {
    let mut q_out = Col::<f64>::zeros(4);
    quat_compose(q1, q2, q_out.as_mut());
    q_out
}

/// Populates Quaternion from tangent vector and twist angle.
///
/// # Panics
/// Panics if `self.ncols() < 4`.
/// Panics if `tangent.ncols() < 4`.
pub fn quat_from_tangent_twist(tangent: ColRef<f64>, twist: f64, q: ColMut<f64>) {
    let e1 = Col::from_fn(3, |i| tangent[i]);
    let a = if e1[0] > 0. { 1. } else { -1. };
    let e2 = col![
        -a * e1[1] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
        a * e1[0] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
        0.,
    ];

    let mut e3 = Col::<f64>::zeros(3);
    cross_product(e1.as_ref(), e2.as_ref(), e3.as_mut());

    let mut q0 = Col::<f64>::zeros(4);
    quat_from_rotation_matrix(
        mat![
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]],
        ]
        .as_ref(),
        q0.as_mut(),
    );

    //  Matrix3::from_columns(&[e1, e2, e3]);
    let mut q_twist = Col::<f64>::zeros(4);
    quat_from_axis_angle(twist * PI / 180., e1.as_ref(), q_twist.as_mut());
    quat_compose(q_twist.as_ref(), q0.as_ref(), q);
}

// Returns the cross product of two vectors
pub fn cross_product(a: ColRef<f64>, b: ColRef<f64>, mut c: ColMut<f64>) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

pub fn dot_product(a: ColRef<f64>, b: ColRef<f64>) -> f64 {
    let mut sum = 0.;
    zip!(&a, &b).for_each(|unzip!(a, b)| sum += *a * *b);
    sum
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

pub fn vec_tilde_alloc(v: ColRef<f64>) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(3, 3);
    vec_tilde(v, m.as_mut());
    m
}

pub fn rotate_section_matrix(matrix: &Mat<f64>, rotation_vector: &Col<f64>) -> Mat<f64> {
    let m =
        quat_as_matrix_alloc(quat_from_rotation_vector_alloc(rotation_vector.as_ref()).as_ref());
    let mut r = Mat::<f64>::zeros(6, 6);
    r.submatrix_mut(0, 0, 3, 3).copy_from(&m);
    r.submatrix_mut(3, 3, 3, 3).copy_from(&m);
    &r * matrix * &r.transpose()
}

pub fn write_matrix(m: MatRef<f64>, file_path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(file_path)?;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if j > 0 {
                write!(file, ",")?;
            }
            write!(file, "{}", m[(i, j)])?;
        }
        writeln!(file)?;
    }
    Ok(())
}

pub trait ColRefReshape<'a, T> {
    fn reshape(self, rows: usize, cols: usize) -> MatRef<'a, T>;
}

impl<'a, T> ColRefReshape<'a, T> for ColRef<'a, T> {
    #[inline]
    fn reshape(self, rows: usize, cols: usize) -> MatRef<'a, T> {
        unsafe { MatRef::from_raw_parts(self.as_ptr() as *const T, rows, cols, 1, rows as isize) }
    }
}

pub trait ColMutReshape<'a, T> {
    fn reshape(self, rows: usize, cols: usize) -> MatRef<'a, T>;
    fn reshape_mut(self, rows: usize, cols: usize) -> MatMut<'a, T>;
}

impl<'a, T> ColMutReshape<'a, T> for ColMut<'a, T> {
    #[inline]
    fn reshape(self, rows: usize, cols: usize) -> MatRef<'a, T> {
        unsafe { MatRef::from_raw_parts(self.as_ptr() as *const T, rows, cols, 1, rows as isize) }
    }
    fn reshape_mut(self, rows: usize, cols: usize) -> MatMut<'a, T> {
        unsafe { MatMut::from_raw_parts_mut(self.as_ptr() as *mut T, rows, cols, 1, rows as isize) }
    }
}

#[inline]
pub fn sparse_matrix_from_triplets(
    nrows: usize,
    ncols: usize,
    triplets: &[Triplet<usize, usize, f64>],
) -> (SparseColMat<usize, f64>, Vec<usize>) {
    // Get argsort order for column major sparse matrix
    let mut sort: Vec<usize> = (0..triplets.len()).collect();
    sort.sort_by_key(|&i| (triplets[i].col, triplets[i].row));

    // Get sort order for updating the sparse matrix values in place
    let mut order = sort.clone();
    sort.iter().enumerate().for_each(|(i, &j)| {
        order[j] = i;
    });

    // Reorder triplets according to sort
    let triplets = sort.iter().map(|&i| triplets[i]).collect_vec();

    // Create sparse matrix from triplets
    let m = SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .expect("Failed to create sparse matrix from triplets");

    (m, order)
}

#[cfg(test)]
mod tests {

    use super::*;
    use equator::assert;
    use faer::utils::approx::*;

    #[test]
    fn test_quat_to_rotation_matrix_1() {
        let approx_eq = CwiseMat(ApproxEq::eps());

        let q = col![(0.5 as f64).sqrt(), (0.5 as f64).sqrt(), 0., 0.];
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        quat_as_matrix(q.as_ref(), m.as_mut());
        assert!(m ~ mat![[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]);

        let q = col![(0.5 as f64).sqrt(), 0., (0.5 as f64).sqrt(), 0.];
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        quat_as_matrix(q.as_ref(), m.as_mut());
        assert!(m ~ mat![[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]);

        let q = col![(0.5 as f64).sqrt(), 0., 0., (0.5 as f64).sqrt()];
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        quat_as_matrix(q.as_ref(), m.as_mut());
        assert!(m ~ mat![[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    }

    #[test]
    fn test_ax2() {
        let approx_eq = CwiseMat(ApproxEq::eps());
        let a = mat![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let b = col![1., 2., 3.];
        let mut out = Mat::<f64>::zeros(3, 3);
        matrix_ax2(a.as_ref(), b.as_ref(), out.as_mut());
        assert!(out ~ mat![[-1., -20., 12.], [20., 4., -6.], [-16., 10., -3.]]);
    }
}

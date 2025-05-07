use faer::prelude::*;
use itertools::{izip, Itertools};

// Returns the dot product of two vectors
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    izip!(a.iter(), b.iter()).map(|(&x, &y)| x * y).sum()
}

// Returns the L2-norm of a vector
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&f| f * f).sum::<f64>().sqrt()
}

// Returns unit vector of given vector or None
pub fn unit_vector(v: &[f64; 3]) -> Option<[f64; 3]> {
    let m = norm(v);
    if m == 0. {
        None
    } else {
        Some([v[0] / m, v[1] / m, v[2] / m])
    }
}

//------------------------------------------------------------------------------
// Lagrange Polynomials
//------------------------------------------------------------------------------

use std::f64::consts::PI;

pub fn lagrange_polynomial(x: f64, xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(j, &xj)| {
            xs.iter()
                .enumerate()
                .filter(|(m, _)| *m != j)
                .map(|(_, &xm)| (x - xm) / (xj - xm))
                .product()
        })
        .collect()
}

pub fn lagrange_polynomial_derivative(x: f64, xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(j, &sj)| {
            xs.iter()
                .enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(i, &si)| {
                    1.0 / (sj - si)
                        * xs.iter()
                            .enumerate()
                            .filter(|(m, _)| *m != i && *m != j)
                            .map(|(_, &sm)| (x - sm) / (sj - sm))
                            .product::<f64>()
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod test_lagrange {

    use super::*;

    #[test]
    fn test_lagrange_polynomial() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 4.0, 9.0];

        let w1 = lagrange_polynomial(1.0, &xs);
        let w2 = lagrange_polynomial(2.0, &xs);
        let w3 = lagrange_polynomial(3.0, &xs);

        assert_eq!(w1, vec![1.0, 0.0, 0.0]);
        assert_eq!(w2, vec![0.0, 1.0, 0.0]);
        assert_eq!(w3, vec![0.0, 0.0, 1.0]);

        assert_eq!(dot(&w1, &ys), 1.0);
        assert_eq!(dot(&w2, &ys), 4.0);
        assert_eq!(dot(&w3, &ys), 9.0);

        let w4 = lagrange_polynomial(1.5, &xs.as_slice());
        assert_eq!(dot(&w4, &ys), 1.5 * 1.5);
    }

    #[test]
    fn test_lagrange_polynomial_derivative() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 4.0, 9.0];

        let w1 = lagrange_polynomial_derivative(1.0, &xs);
        let w2 = lagrange_polynomial_derivative(2.0, &xs);
        let w3 = lagrange_polynomial_derivative(3.0, &xs);

        assert_eq!(w1, vec![-1.5, 2.0, -0.5]);
        assert_eq!(w2, vec![-0.5, 0.0, 0.5]);
        assert_eq!(w3, vec![0.5, -2.0, 1.5]);

        assert_eq!(dot(&w1, &ys), 2.0);
        assert_eq!(dot(&w2, &ys), 4.0);
        assert_eq!(dot(&w3, &ys), 6.0);

        let w4 = lagrange_polynomial_derivative(1.5, &xs);
        assert_eq!(dot(&w4, &ys), 2.0 * 1.5);
    }
}

//------------------------------------------------------------------------------
// Legendre Polynomials
//------------------------------------------------------------------------------

pub fn legendre_polynomial(n: usize, xi: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => xi,
        _ => {
            let n_f = n as f64;
            ((2. * n_f - 1.) * xi * legendre_polynomial(n - 1, xi)
                - (n_f - 1.) * legendre_polynomial(n - 2, xi))
                / n_f
        }
    }
}

pub fn legendre_polynomial_derivative_1(n: usize, xi: f64) -> f64 {
    match n {
        0 => 0.,
        1 => 1.,
        2 => 3. * xi,
        _ => {
            (2. * (n as f64) - 1.) * legendre_polynomial(n - 1, xi)
                + legendre_polynomial_derivative_1(n - 2, xi)
        }
    }
}

pub fn legendre_polynomial_derivative_2(n: usize, xi: f64) -> f64 {
    (2. * xi * legendre_polynomial_derivative_1(n, xi)
        - ((n * (n + 1)) as f64) * legendre_polynomial(n, xi))
        / (1. - (xi * xi))
}

pub fn legendre_polynomial_derivative_3(n: usize, xi: f64) -> f64 {
    (4. * xi * legendre_polynomial_derivative_2(n, xi)
        - ((n * (n + 1) - 2) as f64) * legendre_polynomial_derivative_1(n, xi))
        / (1. - (xi * xi))
}

#[cfg(test)]
mod test_legendre {

    use super::*;

    #[test]
    fn test_legendre_polynomial() {
        assert_eq!(legendre_polynomial(0, -1.), 1.);
        assert_eq!(legendre_polynomial(0, 0.), 1.);
        assert_eq!(legendre_polynomial(0, 1.), 1.);

        assert_eq!(legendre_polynomial(1, -1.), -1.);
        assert_eq!(legendre_polynomial(1, 0.), 0.);
        assert_eq!(legendre_polynomial(1, 1.), 1.);

        assert_eq!(legendre_polynomial(2, -1.), 1.);
        assert_eq!(legendre_polynomial(2, 0.), -0.5);
        assert_eq!(legendre_polynomial(2, 1.), 1.);

        assert_eq!(legendre_polynomial(3, -1.), -1.);
        assert_eq!(legendre_polynomial(3, 0.), 0.);
        assert_eq!(legendre_polynomial(3, 1.), 1.);

        assert_eq!(legendre_polynomial(4, -1.), 1.);
        assert_eq!(
            legendre_polynomial(4, -0.6546536707079771),
            -0.4285714285714286
        );
        assert_eq!(legendre_polynomial(4, 0.), 0.375);
        assert_eq!(
            legendre_polynomial(4, 0.6546536707079771),
            -0.4285714285714286
        );
        assert_eq!(legendre_polynomial(4, 1.), 1.);
    }

    #[test]
    fn test_legendre_polynomial_derivative() {
        assert_eq!(legendre_polynomial_derivative_1(0, -1.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(0, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(0, 1.), 0.);

        assert_eq!(legendre_polynomial_derivative_1(1, -1.), 1.);
        assert_eq!(legendre_polynomial_derivative_1(1, 0.), 1.);
        assert_eq!(legendre_polynomial_derivative_1(1, 1.), 1.);

        assert_eq!(legendre_polynomial_derivative_1(2, -1.), -3.);
        assert_eq!(legendre_polynomial_derivative_1(2, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(2, 1.), 3.);

        assert_eq!(legendre_polynomial_derivative_1(3, -1.), 6.);
        assert_eq!(legendre_polynomial_derivative_1(3, 0.), -1.5);
        assert_eq!(legendre_polynomial_derivative_1(3, 1.), 6.);

        assert_eq!(legendre_polynomial_derivative_1(6, -1.), -21.);
        assert_eq!(legendre_polynomial_derivative_1(6, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(6, 1.), 21.);
    }
}

//------------------------------------------------------------------------------
// Gauss Legendre Lobotto Points and Weights
//------------------------------------------------------------------------------

pub fn gauss_legendre_lobotto_points(order: usize) -> Vec<f64> {
    gauss_legendre_lobotto_quadrature(order).0
}

pub fn gauss_legendre_lobotto_quadrature(order: usize) -> (Vec<f64>, Vec<f64>) {
    let n = order + 1;
    let n_2 = n / 2;
    let nf = n as f64;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    x[0] = -1.;
    x[n - 1] = 1.;
    for i in 1..n_2 {
        let mut xi = (1. - (3. * (nf - 2.)) / (8. * (nf - 1.).powi(3)))
            * ((4. * (i as f64) + 1.) * PI / (4. * (nf - 1.) + 1.)).cos();
        let mut error = 1.0;
        while error > 1e-16 {
            let y = legendre_polynomial_derivative_1(n - 1, xi);
            let y1 = legendre_polynomial_derivative_2(n - 1, xi);
            let y2 = legendre_polynomial_derivative_3(n - 1, xi);
            let dx = 2. * y * y1 / (2. * y1 * y1 - y * y2);
            xi -= dx;
            error = dx.abs();
        }
        x[i] = -xi;
        x[n - i - 1] = xi;
        w[i] = 2. / (nf * (nf - 1.) * legendre_polynomial(n - 1, x[i]).powi(2));
        w[n - i - 1] = w[i];
    }
    if n % 2 != 0 {
        x[n_2] = 0.;
        w[n_2] = 2.0 / ((nf * (nf - 1.)) * legendre_polynomial(n - 1, x[n_2]).powi(2));
    }
    (x, w)
}

#[cfg(test)]
mod test_gll {

    use super::*;

    #[test]
    fn test_gauss_legendre_lobotto_points() {
        assert_eq!(gauss_legendre_lobotto_points(1), vec![-1., 1.]);
        assert_eq!(gauss_legendre_lobotto_points(2), vec![-1., 0., 1.]);
        let p = vec![
            -1.,
            -0.6546536707079771437983,
            0.,
            0.654653670707977143798,
            1.,
        ];
        assert_eq!(gauss_legendre_lobotto_points(4), p);
        let p = vec![
            -1.,
            -0.830223896278567,
            -0.46884879347071423,
            0.,
            0.46884879347071423,
            0.830223896278567,
            1.,
        ];
        assert_eq!(gauss_legendre_lobotto_points(6), p);
    }
}

//------------------------------------------------------------------------------
// Polynomial fitting
//------------------------------------------------------------------------------

pub fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let a = Mat::from_fn(x.len(), degree + 1, |i, j| x[i].powi(j as i32));
    let qr = a.qr();
    let c = qr.solve_lstsq(ColRef::from_slice(y));
    c.try_as_col_major().unwrap().as_slice().to_vec()
}

/// Fit curve that passes through origin
pub fn polyfit_origin(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let a = Mat::from_fn(x.len(), degree, |i, j| x[i].powi(1 + j as i32));
    let qr = a.qr();
    let mut c = Col::zeros(degree + 1);
    c.subrows_mut(1, degree)
        .copy_from(qr.solve_lstsq(ColRef::from_slice(y)));
    c.try_as_col_major().unwrap().as_slice().to_vec()
}

/// Calculate first derivative of coefficients
pub fn polyder(c: &[f64]) -> Vec<f64> {
    c.iter()
        .enumerate()
        .skip(1)
        .map(|(i, v)| v * (i as f64))
        .collect_vec()
}

pub fn polyval(c: &[f64], x: f64) -> f64 {
    let mut y = 0.;
    for (i, &v) in c.iter().enumerate() {
        y += v * x.powi(i as i32);
    }
    y
}

#[cfg(test)]
mod test_poly {

    use super::*;
    use equator::assert;
    use faer::utils::approx::*;
    use itertools::Itertools;

    #[test]
    fn test_poly_funcs() {
        let approx_eq = ApproxEq::eps();

        let x = vec![0., 1., 2., 3., 4.];
        let y = vec![0.5, 2.3, -1., 0.7, 1.2];
        let c = polyfit(&x, &y, 2);
        assert_eq!(c.len(), 3);
        assert!(c[0] ~ 1.1228571428571412);
        assert!(c[1] ~ -0.7057142857142826);
        assert!(c[2] ~ 0.17142857142857063);

        let x = vec![0., 1., 2., 3., 4.];
        let y = x.iter().map(|x| 2. * x + 3. * x * x).collect_vec();
        let c = polyfit_origin(&x, &y, 2);
        assert_eq!(c.len(), 3);
        assert!(c[0] ~ 0.);
        assert!(c[1] ~ 2.);
        assert!(c[2] ~ 3.);
        assert!(polyval(&c, 0.) ~ 0.);
        assert!(polyval(&c, 0.5) ~ 1.75);
        assert!(polyval(&c, 1.) ~ 5.);

        let c = vec![1.1228571428571412, -0.7057142857142826, 0.17142857142857063];
        assert!(polyval(&c, 0.) ~ 1.1228571428571412);
        assert!(polyval(&c, 0.5) ~ 0.8128571428571431);

        let c = vec![1., 3., 2.];
        let dc = polyder(&c);
        assert!(dc[0] ~ 3.);
        assert!(dc[1] ~ 4.);
        assert!(polyval(&dc, 0.) ~ 3.);
        assert!(polyval(&dc, 0.5) ~ 5.);
    }
}

//------------------------------------------------------------------------------
// Interpolation Matrix
//------------------------------------------------------------------------------

pub fn shape_interp_matrix(rows: &[f64], cols: &[f64], mut m: MatMut<f64>) {
    for (i, &si) in rows.iter().enumerate() {
        for (j, &v) in lagrange_polynomial(si, cols).iter().enumerate() {
            m[(i, j)] = v;
        }
    }
}

pub fn shape_deriv_matrix(rows: &[f64], cols: &[f64], mut m: MatMut<f64>) {
    for (i, &si) in rows.iter().enumerate() {
        for (j, &v) in lagrange_polynomial_derivative(si, cols).iter().enumerate() {
            m[(i, j)] = v;
        }
    }
}

//------------------------------------------------------------------------------
// Integration test
//------------------------------------------------------------------------------

#[cfg(test)]
mod test_integration {

    use super::*;
    use crate::util::{quat_as_matrix, quat_from_tangent_twist};
    use equator::assert;
    use faer::utils::approx::*;
    use faer::{linalg::matmul::matmul, Accum};
    use itertools::Itertools;

    #[test]
    fn test_shape_functions() {
        let approx_eq = CwiseMat(ApproxEq::eps());

        // Shape Functions, Derivative of Shape Functions, GLL points for an q-order element
        let p = gauss_legendre_lobotto_points(4);
        assert_eq!(
            p,
            vec![-1., -0.6546536707079772, 0., 0.6546536707079772, 1.]
        );

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let s = p.iter().map(|v| (v + 1.) / 2.).collect_vec();

        // Node x, y, z, twist along reference line
        let fz = |t: f64| -> f64 { t - 2. * t * t };
        let fy = |t: f64| -> f64 { -2. * t + 3. * t * t };
        let fx = |t: f64| -> f64 { 5. * t };
        let ft = |t: f64| -> f64 { 0. * t * t };
        let ref_line = Mat::<f64>::from_fn(4, s.len(), |i, j| match i {
            0 => fx(s[j]),
            1 => fy(s[j]),
            2 => fz(s[j]),
            3 => ft(s[j]),
            _ => unreachable!(),
        });

        // Shape function derivatives at each node along reference line
        let mut shape_deriv = Mat::<f64>::zeros(s.len(), s.len());
        shape_deriv_matrix(&s, &s, shape_deriv.as_mut());

        // Tangent vectors at each node along reference line
        let mut ref_tan = Mat::<f64>::zeros(3, s.len());
        matmul(
            ref_tan.as_mut().transpose_mut(),
            Accum::Replace,
            shape_deriv,
            ref_line.subrows(0, 3).transpose(),
            1.,
            Par::Seq,
        );
        ref_tan.col_iter_mut().for_each(|mut c| {
            let m = c.norm_l2();
            if m != 0. {
                c /= m;
            }
        });

        assert!(
            ref_tan.transpose() ~ mat![
                [0.9128709291752768, -0.365148371670111, 0.1825741858350555],
                [0.9801116185947563, -0.1889578775710052, 0.06063114380768645],
                [0.9622504486493763, 0.19245008972987512, -0.1924500897298752],
                [0.7994326396775596, 0.4738974351647219, -0.3692271327549852],
                [0.7071067811865474, 0.5656854249492382, -0.4242640687119285],
            ]
        );

        // Quaternion at each node along reference line
        let mut ref_q = Mat::<f64>::zeros(4, ref_tan.ncols());
        izip!(
            ref_q.col_iter_mut(),
            ref_tan.col_iter(),
            ref_line.row(3).iter()
        )
        .for_each(|(q, tan, &twist)| quat_from_tangent_twist(tan, twist, q));

        let mut m = Mat::<f64>::zeros(3, 3);
        quat_as_matrix(ref_q.col(3), m.as_mut());
        assert!(
            m ~ mat![
                [0.7994326396775595, -0.509929465806723, 0.3176151673334238],
                [0.47389743516472144, 0.8602162169490122, 0.18827979456709748],
                [-0.3692271327549849, 0.0000000000000000, 0.9293391869697156]
            ]
        );

        assert!(
            ref_q.subcols(3, 1) ~ mat![
                [0.9472312341234699],
                [-0.049692141629315074],
                [0.18127630174800594],
                [0.25965858850765167],
            ]
        );
    }
}

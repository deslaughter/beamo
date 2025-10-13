use itertools::Itertools;

#[derive(Clone)]
pub struct Quadrature {
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Quadrature {
    pub fn gauss(order: usize) -> Self {
        let gl_rule = gauss_quad::GaussLegendre::new(order).unwrap();
        let (points, weights) = gl_rule
            .into_node_weight_pairs()
            .iter()
            .rev()
            .map(|&p| p)
            .unzip();
        Quadrature {
            points: points,
            weights: weights,
        }
    }
    pub fn gauss_legendre_lobotto(order: usize) -> Self {
        use super::interp::gauss_legendre_lobotto_quadrature;
        let (points, weights) = gauss_legendre_lobotto_quadrature(order);
        Quadrature {
            points: points,
            weights: weights,
        }
    }
    pub fn trapezoidal(s: &[f64]) -> Self {
        let n = s.len();
        if n < 2 {
            panic!("insufficient points")
        }

        // Calculate points
        let min = s.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let points = s
            .iter()
            .map(|&x| -1.0 + 2.0 * (x - min) / (max - min))
            .collect_vec();

        // Calculate weights
        let mut weights = vec![0.; n];
        weights[0] = (points[1] - points[0]) / 2.;
        for i in 1..n - 1 {
            weights[i] = (points[i + 1] - points[i - 1]) / 2.;
        }
        weights[n - 1] = (points[n - 1] - points[n - 2]) / 2.;
        Self { points, weights }
    }
    /// Compute composite Simpson's rule weights for arbitrary nodes in [-1,1].
    /// This works for uneven spacing and any number of nodes.
    /// The weights are derived from integrating the Lagrange basis functions.
    pub fn simpsons_rule(x: &[f64]) -> Self {
        let n = x.len();
        assert!(n >= 2, "Need at least 2 nodes");

        let mut w = vec![0.0; n];

        // Calculate points
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let points = x
            .iter()
            .map(|&x| -1.0 + 2.0 * (x - min) / (max - min))
            .collect_vec();

        let mut i = 0;
        while i + 2 < n {
            let x0 = points[i];
            let x1 = points[i + 1];
            let x2 = points[i + 2];

            // Quadratic Lagrange basis integration over [x0,x2]
            let mut local_w = [0.0; 3];
            for (j, &xj) in [x0, x1, x2].iter().enumerate() {
                // Build Lagrange basis L_j
                let mut coeffs = vec![1.0];
                let mut denom = 1.0;
                for (k, &xk) in [x0, x1, x2].iter().enumerate() {
                    if j == k {
                        continue;
                    }
                    denom *= xj - xk;

                    let mut new_coeffs = vec![0.0; coeffs.len() + 1];
                    for (m, &c) in coeffs.iter().enumerate() {
                        new_coeffs[m] -= c * xk;
                        new_coeffs[m + 1] += c;
                    }
                    coeffs = new_coeffs;
                }
                for c in &mut coeffs {
                    *c /= denom;
                }

                // Integrate polynomial on [x0,x2]
                let mut wi = 0.0;
                for (k, &c) in coeffs.iter().enumerate() {
                    wi += c * ((x2.powi(k as i32 + 1) - x0.powi(k as i32 + 1)) / (k as f64 + 1.0));
                }
                local_w[j] = wi;
            }

            // Accumulate into global weights
            w[i] += local_w[0];
            w[i + 1] += local_w[1];
            w[i + 2] += local_w[2];

            i += 2;
        }

        // If an odd interval remains → trapezoidal rule
        if i + 1 < n {
            let h = x[i + 1] - x[i];
            w[i] += h / 2.0;
            w[i + 1] += h / 2.0;
        }

        Self { points, weights: w }
    }
}

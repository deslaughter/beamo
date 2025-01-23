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
}

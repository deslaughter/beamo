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
}

#[derive(Debug, Clone)]
pub struct Inflow {
    typ: InflowType,
    uniform_flow: UniformFlow,
}

#[derive(Debug, Clone)]
pub enum InflowType {
    Uniform = 1,
}

impl Inflow {
    pub fn steady_wind(
        velocity_horizontal: f64,
        height_reference: f64,
        shear_vertical: f64,
        flow_angle_horizontal: f64,
    ) -> Self {
        Inflow {
            typ: InflowType::Uniform,
            uniform_flow: UniformFlow {
                time: vec![0.],
                data: vec![UniformFlowParameters {
                    velocity_horizontal,
                    height_reference,
                    shear_vertical,
                    flow_angle_horizontal,
                }],
            },
        }
    }
    pub fn velocity(&self, t: f64, position: [f64; 3]) -> [f64; 3] {
        match self.typ {
            InflowType::Uniform => self.uniform_flow.velocity(t, position),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniformFlow {
    pub time: Vec<f64>, // Time vector for uniform flow parameters
    pub data: Vec<UniformFlowParameters>,
}

impl UniformFlow {
    pub fn velocity(&self, _t: f64, position: [f64; 3]) -> [f64; 3] {
        match self.time.len() {
            1 => self.data[0].velocity(position),
            _ => unreachable!("Time-dependent uniform flow not implemented"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniformFlowParameters {
    pub velocity_horizontal: f64,   // Horizontal inflow velocity (m/s)
    pub height_reference: f64,      // Reference height (m)
    pub shear_vertical: f64,        // Vertical shear exponent
    pub flow_angle_horizontal: f64, // Flow angle relative to x axis (radians)
}

impl UniformFlowParameters {
    pub fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        // Calculate horizontal velocity
        let vh = self.velocity_horizontal
            * (position[2] / self.height_reference).powf(self.shear_vertical);

        // Get sin and cos of flow angle
        let (sin_flow_angle, cos_flow_angle) = self.flow_angle_horizontal.sin_cos();

        // Apply horizontal direction
        [vh * cos_flow_angle, -vh * sin_flow_angle, 0.]
    }
}

#[cfg(test)]
mod tests {

    use std::vec;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_stead_wind_without_shear() {
        // Define steady 10.0 m/s wind along X-axis for any time and position
        let vel_h = 10.0;
        let ref_height = 100.0;
        let power_law_exp = 0.0;
        let flow_angle_horizontal = 0.0_f64.to_radians();
        let inflow = Inflow::steady_wind(vel_h, ref_height, power_law_exp, flow_angle_horizontal);

        struct Case {
            time: f64,
            position: [f64; 3],
            vel_exp: [f64; 3],
        }

        let test_cases = vec![
            Case {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                vel_exp: [10.0, 0.0, 0.0],
            },
            Case {
                time: 1.0,
                position: [0.0, 0.0, ref_height],
                vel_exp: [10.0, 0.0, 0.0],
            },
            Case {
                time: 1000.0,
                position: [100.0, 100.0, 100.0],
                vel_exp: [10.0, 0.0, 0.0],
            },
        ];

        for case in test_cases {
            let velocity = inflow.velocity(case.time, case.position);
            assert_relative_eq!(velocity[0], case.vel_exp[0], epsilon = 1e-12);
            assert_relative_eq!(velocity[1], case.vel_exp[1], epsilon = 1e-12);
            assert_relative_eq!(velocity[2], case.vel_exp[2], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_stead_wind_with_shear_nonzero_flow_angle() {
        // Define steady 10.0 m/s wind along X-axis at ref height with 0.1 power law shear exponent
        let vel_h = 10.0;
        let ref_height = 100.0;
        let power_law_exp = 0.1;
        let flow_angle_horizontal = 45.0_f64.to_radians();
        let inflow = Inflow::steady_wind(vel_h, ref_height, power_law_exp, flow_angle_horizontal);

        struct Case {
            time: f64,
            position: [f64; 3],
            vel_exp: [f64; 3],
        }

        let test_cases = vec![
            // Test case at reference height
            Case {
                time: 0.0,
                position: [0.0, 0.0, ref_height],
                vel_exp: [7.0710678118654755, -7.0710678118654755, 0.0],
            },
            // Test case at ground level
            Case {
                time: 1.0,
                position: [0.0, 0.0, 0.0],
                vel_exp: [0.0, 0.0, 0.0],
            },
            // Test at half ref height [10*sqrt(2)/2*0.5**0.1, 0, 0]
            Case {
                time: 100.0,
                position: [100.0, 100.0, ref_height / 2.0],
                vel_exp: [6.597539553864471, -6.597539553864471, 0.0],
            },
        ];

        for case in test_cases {
            let velocity = inflow.velocity(case.time, case.position);
            assert_relative_eq!(velocity[0], case.vel_exp[0], epsilon = 1e-12);
            assert_relative_eq!(velocity[1], case.vel_exp[1], epsilon = 1e-12);
            assert_relative_eq!(velocity[2], case.vel_exp[2], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_stead_wind_with_shear() {
        // Define steady 10.0 m/s wind along X-axis at ref height with 0.1 power law shear exponent
        let vel_h = 10.0;
        let ref_height = 100.0;
        let power_law_exp = 0.1;
        let flow_angle_horizontal = 0.0_f64.to_radians();
        let inflow = Inflow::steady_wind(vel_h, ref_height, power_law_exp, flow_angle_horizontal);

        struct Case {
            time: f64,
            position: [f64; 3],
            vel_exp: [f64; 3],
        }

        let test_cases = vec![
            // Test case at reference height
            Case {
                time: 0.0,
                position: [0.0, 0.0, ref_height],
                vel_exp: [10.0, 0.0, 0.0],
            },
            // Test case at ground level
            Case {
                time: 1.0,
                position: [0.0, 0.0, 0.0],
                vel_exp: [0.0, 0.0, 0.0],
            },
            // Test at half ref height [10*0.5**0.1, 0, 0]
            Case {
                time: 100.0,
                position: [100.0, 100.0, ref_height / 2.0],
                vel_exp: [9.330329915368074, 0.0, 0.0],
            },
        ];

        for case in test_cases {
            let velocity = inflow.velocity(case.time, case.position);
            assert_relative_eq!(velocity[0], case.vel_exp[0], epsilon = 1e-12);
            assert_relative_eq!(velocity[1], case.vel_exp[1], epsilon = 1e-12);
            assert_relative_eq!(velocity[2], case.vel_exp[2], epsilon = 1e-12);
        }
    }
}

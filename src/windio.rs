use faer::prelude::*;
use serde::Deserialize;

pub fn read_windio_from_file(file_path: &str) -> WindIO {
    let yaml_file = std::fs::read_to_string(file_path).expect("Unable to read file");
    serde_yaml::from_str(&yaml_file).expect("Unable to parse YAML")
}

#[derive(Debug, Deserialize)]
pub struct WindIO {
    pub components: Components,
    pub materials: Vec<Material>,
}

#[derive(Debug, Deserialize)]
pub struct Components {
    pub blade: Blade,
    pub hub: Hub,
    pub nacelle: Nacelle,
    pub tower: Tower,
}

#[derive(Debug, Deserialize)]
pub struct Blade {
    pub elastic_properties_mb: ElasticPropertiesMB,
}

#[derive(Debug, Deserialize)]
pub struct Hub {
    pub diameter: f64,
    pub cone_angle: f64,
    pub elastic_properties_mb: HubElasticPropertiesMB,
}

#[derive(Debug, Deserialize)]
pub struct Nacelle {
    pub drivetrain: Drivetrain,
    pub elastic_properties_mb: NacelleElasticPropertiesMB,
}

#[derive(Debug, Deserialize)]
pub struct Drivetrain {
    pub uptilt: f64,
    pub distance_tt_hub: f64,
    pub overhang: f64,
}

#[derive(Debug, Deserialize)]
pub struct NacelleElasticPropertiesMB {
    pub system_mass: f64,
    pub yaw_mass: f64,
    pub system_inertia_tt: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct HubElasticPropertiesMB {
    pub system_mass: f64,
    pub system_inertia: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Tower {
    pub outer_shape_bem: TowerOuterShapeBEM,
    pub internal_structure_2d_fem: TowerInternalStructure2DFEM,
}

#[derive(Debug, Deserialize)]
pub struct TowerOuterShapeBEM {
    pub reference_axis: ReferenceAxis,
    pub outer_diameter: GridAndValues,
}

#[derive(Debug, Deserialize)]
pub struct TowerInternalStructure2DFEM {
    pub layers: Vec<TowerLayer>,
}

#[derive(Debug, Deserialize)]
pub struct TowerLayer {
    pub name: String,
    pub material: String,
    pub thickness: GridAndValues,
}

#[derive(Debug, Deserialize)]
pub struct ReferenceAxis {
    pub x: GridAndValues,
    pub y: GridAndValues,
    pub z: GridAndValues,
}

#[derive(Debug, Deserialize)]
pub struct GridAndValues {
    pub grid: Vec<f64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ElasticPropertiesMB {
    pub six_x_six: SixXSix,
}

#[derive(Debug, Deserialize)]
pub struct SixXSix {
    pub reference_axis: ReferenceAxis,
    pub twist: GridAndValues,
    pub stiff_matrix: ElasticMatrix,
    pub inertia_matrix: ElasticMatrix,
}

#[derive(Debug, Deserialize)]
pub struct ElasticMatrix {
    pub grid: Vec<f64>,
    pub values: Vec<FlatMatrix>,
}

#[derive(Debug, Deserialize)]
pub struct FlatMatrix(Vec<f64>);
impl FlatMatrix {
    pub fn as_matrix(&self) -> Mat<f64> {
        mat![
            [self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]],
            [self.0[1], self.0[6], self.0[7], self.0[8], self.0[9], self.0[10]],
            [self.0[2], self.0[7], self.0[11], self.0[12], self.0[13], self.0[14]],
            [self.0[3], self.0[8], self.0[12], self.0[15], self.0[16], self.0[17]],
            [self.0[4], self.0[9], self.0[13], self.0[16], self.0[18], self.0[19]],
            [self.0[5], self.0[10], self.0[14], self.0[17], self.0[19], self.0[20]],
        ]
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ScalarOrVector {
    Scalar(f64),
    Vector(Vec<f64>),
}

#[derive(Debug, Deserialize)]
pub struct Material {
    pub name: String,
    #[serde(alias = "rho")]
    pub density: f64,
    #[serde(alias = "E")]
    pub elastic_modulus: ScalarOrVector,
    #[serde(alias = "G")]
    pub shear_modulus: ScalarOrVector,
    #[serde(alias = "nu")]
    pub poisson_ratio: ScalarOrVector,
}

use std::fs;

use itertools::Itertools;
use ottr::model::Model;

fn main() {}

// fn build_vawt(model: &mut Model) {
fn build_vawt() {
    let n_blades = 2;
    let blade_radius: f64 = 177.2022 * 0.3048; // m
    let blade_height: f64 = 1.02 * blade_radius * 2.; // m
    let tower_height = 15.0; // m
    let chord = 5.0; // m

    let n_blade_nodes = 11;

    let angle = ((2. * blade_radius.powi(2) - blade_height.powi(2)) / (2. * blade_radius.powi(2)));

    println!("angle: {}", angle.to_degrees());

    // // Build blade input
    // let blade_input = BeamInputBuilder::new()
    //     .set_element_order(n_blade_nodes - 1)
    //     .set_section_refinement(0)
    //     .set_reference_axis_z(
    //         &ref_axis["x"]["grid"]
    //             .as_sequence()
    //             .unwrap()
    //             .iter()
    //             .map(|v| v.as_f64().unwrap())
    //             .collect_vec(),
    //         &itertools::izip!(
    //             ref_axis["x"]["values"]
    //                 .as_sequence()
    //                 .unwrap()
    //                 .iter()
    //                 .map(|v| v.as_f64().unwrap())
    //                 .collect_vec(),
    //             ref_axis["y"]["values"]
    //                 .as_sequence()
    //                 .unwrap()
    //                 .iter()
    //                 .map(|v| v.as_f64().unwrap())
    //                 .collect_vec(),
    //             ref_axis["z"]["values"]
    //                 .as_sequence()
    //                 .unwrap()
    //                 .iter()
    //                 .map(|v| v.as_f64().unwrap())
    //                 .collect_vec()
    //         )
    //         .map(|(x, y, z)| [x, y, z])
    //         .collect_vec(),
    //         &blade_twist["grid"]
    //             .as_sequence()
    //             .unwrap()
    //             .iter()
    //             .map(|v| v.as_f64().unwrap())
    //             .collect_vec(),
    //         &blade_twist["values"]
    //             .as_sequence()
    //             .unwrap()
    //             .iter()
    //             .map(|v| v.as_f64().unwrap())
    //             .collect_vec(),
    //     )
    //     .set_sections_z(&sections)
    //     .set_prescribe_root(true)
    //     .set_root_position([
    //         0.,
    //         0.,
    //         0.,
    //         root_orientation[0],
    //         root_orientation[1],
    //         root_orientation[2],
    //         root_orientation[3],
    //     ])
    //     .build();

    // // Build and return the beam component
    // let blade = BeamComponent::new(&blade_input, model);
}

#[derive(Debug)]
pub struct Material {
    id: usize,
    name: String,
    ply_thickness: f64,
    e1: f64,
    e2: f64,
    g12: f64,
    anu: f64,
    rho: f64,
    xt: f64,
    xc: f64,
    yt: f64,
    yc: f64,
}

fn read_numad_material_file(path: &str) -> Vec<Material> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_path(path)
        .unwrap();

    rdr.records()
        .skip(3)
        .map(|result| {
            let row = result.unwrap();
            Material {
                id: row.get(0).unwrap().parse::<usize>().unwrap(),
                name: row.get(1).unwrap().into(),
                ply_thickness: row.get(3).unwrap().parse::<f64>().unwrap() * 1.0e-3, // meters
                e1: row.get(4).unwrap().parse::<f64>().unwrap() * 1.0e6,
                e2: row.get(5).unwrap().parse::<f64>().unwrap() * 1.0e6,
                g12: row.get(7).unwrap().parse::<f64>().unwrap() * 1.0e6,
                anu: row.get(10).unwrap().parse().unwrap(), // ratio
                rho: row.get(13).unwrap().parse().unwrap(), // g/cc * 1000 #kg/m3
                xt: row.get(14).unwrap().parse().unwrap(),  // Pa
                xc: row.get(15).unwrap().parse().unwrap(),  // Pa
                yt: row
                    .get(16)
                    .unwrap_or("100")
                    .replace("", "100")
                    .parse::<f64>()
                    .unwrap()
                    .abs()
                    * 1.0e6,
                yc: row
                    .get(17)
                    .unwrap_or("100")
                    .replace("", "100")
                    .parse::<f64>()
                    .unwrap()
                    .abs()
                    * 1.0e6,
            }
        })
        .collect_vec()
}

#[derive(Debug)]
pub struct Section {
    id: usize,
    name: String,
}

fn read_numad_geometry_file(path: &str) -> Vec<Section> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_path(path)
        .unwrap();

    let mut iter = rdr.records();

    // Get first row
    let row = iter.next().unwrap().unwrap();

    let n_web = row.get(5).unwrap().parse::<usize>().unwrap();
    let n_stack = row.get(7).unwrap().parse::<usize>().unwrap();

    // Get second row
    let row = iter.next().unwrap().unwrap();
    let n_segments = row.get(7).unwrap().parse::<usize>().unwrap();

    println!(
        "n_web: {}, n_stack: {}, n_segments: {}",
        n_web, n_stack, n_segments
    );

    // Get all the section data
    iter.skip(1)
        .map(|result| {
            let row = result.unwrap();
            Section {
                id: row.get(0).unwrap().parse::<usize>().unwrap(),
                name: row.get(1).unwrap().into(),
            }
        })
        .collect_vec()
}

mod test {
    use super::*;

    #[test]
    fn test_build_vawt() {
        build_vawt();
    }

    #[test]
    fn test_read_numad_material_file() {
        let result = read_numad_material_file("examples/vawt/TowerMaterials.csv");
        println!("{:?}", result)
    }

    #[test]
    fn test_read_numad_geometry_file() {
        let result = read_numad_geometry_file("examples/vawt/TowerGeom.csv");
        println!("{:?}", result)
    }
}

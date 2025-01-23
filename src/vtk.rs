use faer::{Col, Mat};
use itertools::{izip, Itertools};
use vtkio::model::*;

use crate::{
    elements::{beams::Beams, springs::Springs},
    state::State,
    util::{quat_as_matrix, Quat},
};

// Currently assumes there is one beam element

pub fn beams_nodes_as_vtk(beams: &Beams) -> Vtk {
    let rotations = izip!(
        beams.node_u.subrows(3, 4).col_iter(),
        beams.node_x0.subrows(3, 4).col_iter(),
    )
    .map(|(r, r0)| {
        let mut q = Col::<f64>::zeros(4);
        q.as_mut().quat_compose(r, r0);
        let mut m = Mat::<f64>::zeros(3, 3);
        quat_as_matrix(q.as_ref(), m.as_mut());
        m
    })
    .collect_vec();
    let orientations = vec!["OrientationX", "OrientationY", "OrientationZ"];
    let n_nodes = beams.node_ids.len();

    Vtk {
        version: Version { major: 4, minor: 2 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(
                izip!(beams.node_u.col_iter(), beams.node_x0.col_iter())
                    .flat_map(|(u, x0)| [u[0] + x0[0], u[1] + x0[1], u[2] + x0[2]])
                    .collect_vec(),
            ),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity: {
                        let mut a = vec![0, n_nodes - 1];
                        let b = (1..n_nodes - 1).collect_vec();
                        a.extend(b);
                        a.iter().map(|&i| i as u64).collect_vec()
                    },
                    offsets: vec![n_nodes as u64],
                },
                types: vec![CellType::LagrangeCurve],
            },
            data: Attributes {
                point: orientations
                    .iter()
                    .enumerate()
                    .map(|(i, &orientation)| {
                        Attribute::DataArray(DataArrayBase {
                            name: orientation.to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| r.col(i).iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        })
                    })
                    .chain(vec![
                        Attribute::DataArray(DataArrayBase {
                            name: "AngularVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                beams
                                    .node_v
                                    .subrows(3, 3) // omega
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "TranslationalVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                beams
                                    .node_v
                                    .subrows(0, 3) // vel
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                    ])
                    .collect_vec(),
                ..Default::default()
            },
        }),
    }
}

pub fn beams_qps_as_vtk(beams: &Beams) -> Vtk {
    let rotations = izip!(
        beams.qp.u.subrows(3, 4).col_iter(),
        beams.qp.x0.subrows(3, 4).col_iter(),
    )
    .map(|(r, r0)| {
        let mut q = Col::<f64>::zeros(4);
        q.as_mut().quat_compose(r, r0);
        let mut m = Mat::<f64>::zeros(3, 3);
        quat_as_matrix(q.as_ref(), m.as_mut());
        m
    })
    .collect_vec();
    let orientations = vec!["OrientationX", "OrientationY", "OrientationZ"];
    let n_qps = beams.qp.m.nrows();

    Vtk {
        version: Version { major: 4, minor: 2 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(
                izip!(beams.qp.u.col_iter(), beams.qp.x0.col_iter())
                    .flat_map(|(u, x0)| [u[0] + x0[0], u[1] + x0[1], u[2] + x0[2]])
                    .collect_vec(),
            ),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity: {
                        let mut a = vec![0, n_qps - 1];
                        let b = (1..n_qps - 1).collect_vec();
                        a.extend(b);
                        a.iter().map(|&i| i as u64).collect_vec()
                    },
                    offsets: vec![n_qps as u64],
                },
                types: vec![CellType::LagrangeCurve],
            },
            data: Attributes {
                point: orientations
                    .iter()
                    .enumerate()
                    .map(|(i, &orientation)| {
                        Attribute::DataArray(DataArrayBase {
                            name: orientation.to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| r.col(i).iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        })
                    })
                    .chain(vec![
                        Attribute::DataArray(DataArrayBase {
                            name: "AngularVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                beams
                                    .qp
                                    .v
                                    .subrows(3, 3) // omega
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                        Attribute::DataArray(DataArrayBase {
                            name: "TranslationalVelocity".to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                beams
                                    .qp
                                    .v
                                    .subrows(0, 3) // vel
                                    .col_iter()
                                    .flat_map(|c| c.iter().map(|&v| v as f32).collect_vec())
                                    .collect_vec(),
                            ),
                        }),
                    ])
                    .collect_vec(),
                ..Default::default()
            },
        }),
    }
}

pub fn lines_as_vtk(lines: &[[usize; 2]], state: &State) -> Vtk {
    let n_lines = lines.len();
    Vtk {
        version: Version { major: 4, minor: 2 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(
                lines
                    .iter()
                    .flat_map(|node_ids| {
                        let x_1 = state.x.col(node_ids[0]);
                        let x_2 = state.x.col(node_ids[1]);
                        [x_1[0], x_1[1], x_1[2], x_2[0], x_2[1], x_2[2]]
                    })
                    .collect_vec(),
            ),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity: (0..2 * n_lines).map(|i| i as u64).collect_vec(),
                    offsets: (1..n_lines + 1).map(|i| 2 * i as u64).collect_vec(),
                },
                types: vec![CellType::Line; n_lines],
            },
            data: Attributes {
                ..Default::default()
            },
        }),
    }
}

pub fn springs_as_vtk(springs: &Springs, state: &State) -> Vtk {
    Vtk {
        version: Version { major: 4, minor: 2 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(
                springs
                    .elem_node_ids
                    .iter()
                    .flat_map(|node_ids| {
                        let x0_1 = state.x0.col(node_ids[0]);
                        let u_1 = state.u.col(node_ids[0]);
                        let x0_2 = state.x0.col(node_ids[1]);
                        let u_2 = state.u.col(node_ids[1]);
                        [
                            u_1[0] + x0_1[0],
                            u_1[1] + x0_1[1],
                            u_1[2] + x0_1[2],
                            u_2[0] + x0_2[0],
                            u_2[1] + x0_2[1],
                            u_2[2] + x0_2[2],
                        ]
                    })
                    .collect_vec(),
            ),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity: (0..2 * springs.n_elem).map(|i| i as u64).collect_vec(),
                    offsets: (1..springs.n_elem + 1).map(|i| 2 * i as u64).collect_vec(),
                },
                types: vec![CellType::Line; springs.n_elem],
            },
            data: Attributes {
                ..Default::default()
            },
        }),
    }
}

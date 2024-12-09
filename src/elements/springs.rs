use faer::{linalg::matmul::matmul, unzipped, zipped, Col, ColMut, Mat, MatMut, Parallelism};

use itertools::{izip, Itertools};

use crate::{
    node::{Node, NodeFreedomMap},
    state::State,
    util::{vec_tilde, ColAsMatMut, ColAsMatRef},
};

pub struct SpringElement {
    pub id: usize,
    pub undeformed_length: Option<f64>,
    pub stiffness: f64,
    pub node_ids: [usize; 2],
}

pub struct Springs {
    /// Number of spring elements
    pub n_elem: usize,
    /// Node ID for each element
    pub elem_node_ids: Vec<[usize; 2]>,
    /// Initial difference in node locations `[3][n_nodes]`
    pub x0: Mat<f64>,
    /// State: node 1 translational displacement `[3][n_nodes]`
    pub u1: Mat<f64>,
    /// State: node 2 translational displacement `[3][n_nodes]`
    pub u2: Mat<f64>,
    /// Current difference in node locations `[3][n_nodes]`
    pub r: Mat<f64>,
    /// Current distance between nodes `[n_nodes]`
    pub l: Col<f64>,
    /// Undeformed length
    pub l_ref: Col<f64>,
    /// Spring stiffness
    pub k: Col<f64>,
    pub c1: Col<f64>,
    pub c2: Col<f64>,
    /// stiff matrices `[3][3][n_nodes]`
    pub a: Mat<f64>,
    /// force components `[3][n_nodes]`
    pub f: Mat<f64>,
}

impl Springs {
    pub fn new(elements: &[SpringElement], nodes: &[Node]) -> Self {
        // Total number of elements
        let n_elem = elements.len();

        // Initial distance between nodes
        let x0 = Mat::from_fn(3, n_elem, |i, j| {
            nodes[elements[j].node_ids[1]].x[i] - nodes[elements[j].node_ids[0]].x[i]
        });

        // Initial spring length
        let l_ref = Col::from_fn(n_elem, |i| match elements[i].undeformed_length {
            Some(l_ref) => l_ref,
            None => x0.as_ref().col(i).norm_l2(),
        });

        Self {
            n_elem,
            elem_node_ids: elements.iter().map(|e| e.node_ids.clone()).collect_vec(),
            x0,
            u1: Mat::zeros(3, n_elem),
            u2: Mat::zeros(3, n_elem),
            r: Mat::zeros(3, n_elem),
            l: Col::zeros(n_elem),
            l_ref,
            k: Col::from_fn(n_elem, |i| elements[i].stiffness),
            c1: Col::zeros(n_elem),
            c2: Col::zeros(n_elem),
            f: Mat::zeros(3, n_elem),
            a: Mat::zeros(3 * 3, n_elem),
        }
    }

    pub fn calculate(&mut self, state: &State) {
        // Copy displacement from state nodes to element nodes
        izip!(
            self.elem_node_ids.iter(),
            self.u1.col_iter_mut(),
            self.u2.col_iter_mut(),
        )
        .for_each(|(&ids, mut u1, mut u2)| {
            u1.copy_from(state.u.col(ids[0]).subrows(0, 3));
            u2.copy_from(state.u.col(ids[1]).subrows(0, 3));
        });

        //----------------------------------------------------------------------
        // Calculate values
        //----------------------------------------------------------------------

        // Calculate components of difference between
        zipped!(&mut self.r, &self.x0, &self.u1, &self.u2)
            .for_each(|unzipped!(mut r, x0, u1, u2)| *r = *x0 + *u2 - *u1);

        // Calculate distance between nodes
        izip!(self.l.iter_mut(), self.r.col_iter()).for_each(|(l, r)| *l = r.norm_l2());

        let mut r_tilde = Mat::<f64>::zeros(3, 3);

        // Calculate coefficients
        zipped!(&mut self.c1, &mut self.c2, &self.k, &self.l_ref, &self.l).for_each(
            |unzipped!(mut c1, mut c2, k, l_ref, l)| {
                *c1 = *k * (*l_ref / *l - 1.);
                *c2 = *k * *l_ref / (*l).powi(3);
            },
        );

        // Calculate force for each element
        izip!(self.f.col_iter_mut(), self.c1.iter(), self.r.col_iter()).for_each(
            |(mut f, &c1, r)| zipped!(&mut f, r).for_each(|unzipped!(mut f, r)| *f = c1 * *r),
        );

        // Calculate stiffness matrix for each element
        izip!(
            self.a.col_iter_mut(),
            self.c1.iter(),
            self.c2.iter(),
            self.r.col_iter(),
            self.l.iter(),
        )
        .for_each(|(a_col, &c1, &c2, r, l)| {
            let mut a = a_col.as_mat_mut(3, 3);
            vec_tilde(r, r_tilde.as_mut());
            a.as_mut()
                .diagonal_mut()
                .column_vector_mut()
                .fill(c1 - c2 * l.powi(2));
            matmul(
                a,
                r_tilde.as_ref(),
                r_tilde.as_ref(),
                Some(1.),
                -c2,
                Parallelism::None,
            );
        });
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &mut self,
        nfm: &NodeFreedomMap,
        state: &State,
        mut k: MatMut<f64>, // Stiffness
        mut r: ColMut<f64>, // Residual
    ) {
        // If no elements, return
        if self.n_elem == 0 {
            return;
        }

        self.calculate(state);

        //----------------------------------------------------------------------
        // Assemble elements into global matrices and vectors
        //----------------------------------------------------------------------

        izip!(
            self.elem_node_ids.iter(),
            self.a.col_iter(),
            self.f.col_iter()
        )
        .for_each(|(node_ids, a_col, f)| {
            // Get index of first dof in each element node
            let elem_dofs_start = [
                nfm.node_dofs[node_ids[0]].first_dof_index,
                nfm.node_dofs[node_ids[1]].first_dof_index,
            ];

            // Residual vector
            elem_dofs_start
                .iter()
                .zip([1., -1.])
                .for_each(|(&i, sign)| {
                    zipped!(&mut r.as_mut().subrows_mut(i, 3), f)
                        .for_each(|unzipped!(mut r, f)| *r += sign * *f);
                });

            // Get start DOFs for each node
            let elem_dof_start_pairs = elem_dofs_start
                .iter()
                .cartesian_product(elem_dofs_start.iter())
                .collect_vec();

            // Stiffness matrix
            let a = a_col.as_mat_ref(3, 3);
            elem_dof_start_pairs.iter().for_each(|(&i, &j)| {
                let sign = if i == j { 1. } else { -1. };
                zipped!(&mut k.as_mut().submatrix_mut(i, j, 3, 3), a)
                    .for_each(|unzipped!(mut k, a)| *k += sign * *a);
            });
        });
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use faer::{assert_matrix_eq, col, mat};

    use crate::model::Model;

    use super::*;

    #[test]
    fn test_force() {
        let mut model = Model::new();

        let n1 = model.add_node().position_xyz(0., 0., 0.).build();
        let n2 = model.add_node().position_xyz(1., 0., 0.).build();

        model.add_spring_element(n1, n2, 10., None);

        let mut state = model.create_state();
        let mut springs = model.create_springs();

        // Zero displacement
        springs.calculate(&state);
        assert_eq!(springs.l_ref[0], 1.);
        assert_eq!(springs.l[0], 1.);
        assert_matrix_eq!(springs.f.col(0).as_2d(), col![0., 0., 0.].as_2d());
        assert_matrix_eq!(
            springs.a.col(0).as_mat_ref(3, 3),
            mat![[-10., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        );

        // Unit displacement
        state.u[(0, 1)] = 1.;
        springs.calculate(&state);
        assert_eq!(springs.l_ref[0], 1.);
        assert_eq!(springs.l[0], 2.);
        assert_eq!(springs.c1[0], -5.);
        assert_eq!(springs.c2[0], 10. * 1. / (2. as f64).powi(3));
        assert_matrix_eq!(springs.f.col(0).as_2d(), col![-10., 0., 0.].as_2d());
        assert_matrix_eq!(
            springs.a.col(0).as_mat_ref(3, 3),
            mat![[-10., -5., -5.], [0., 0., 0.], [0., 0., 0.]]
        );
    }
}

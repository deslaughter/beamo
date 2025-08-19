use faer::prelude::*;
use faer::sparse::*;
use faer::{linalg::matmul::matmul, Accum};

use itertools::{izip, Itertools};

use crate::util::sparse_matrix_from_triplets;
use crate::{
    node::{Node, NodeFreedomMap},
    state::State,
    util::{vec_tilde, ColMutReshape},
};

/// Spring element definition
pub struct SpringElement {
    pub id: usize,
    pub undeformed_length: Option<f64>,
    pub stiffness: f64,
    pub node_ids: [usize; 2],
}

pub struct Springs {
    /// Number of spring elements
    pub n_elem: usize,
    /// Node IDs for each element
    pub elem_node_ids: Vec<[usize; 2]>,
    /// First DOF index for each element node
    pub elem_dofs_start: Vec<[usize; 2]>,
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
    r_tilde: Mat<f64>,

    /// Sparse stiffness matrix
    pub k_sp: SparseColMat<usize, f64>,
    /// Order of data in sparse matrices
    order_sp: Vec<usize>,
}

impl Springs {
    pub fn new(elements: &[SpringElement], nodes: &[Node], nfm: &NodeFreedomMap) -> Self {
        // Total number of elements
        let n_elem = elements.len();

        // Initial distance between nodes
        let x0 = Mat::from_fn(3, n_elem, |i, j| {
            nodes[elements[j].node_ids[1]].xr[i] - nodes[elements[j].node_ids[0]].xr[i]
        });

        // Initial spring length
        let l_ref = Col::from_fn(n_elem, |i| match elements[i].undeformed_length {
            Some(l_ref) => l_ref,
            None => x0.as_ref().col(i).norm_l2(),
        });

        // Element node IDs
        let elem_node_ids = elements.iter().map(|e| e.node_ids.clone()).collect_vec();

        // Element DOF start indices
        let elem_dofs_start = elem_node_ids
            .iter()
            .map(|ids| {
                [
                    nfm.node_dofs[ids[0]].first_dof_index,
                    nfm.node_dofs[ids[1]].first_dof_index,
                ]
            })
            .collect_vec();

        // Sparse matrix triplets
        let triplets = elem_dofs_start
            .iter()
            .flat_map(|dofs_start| {
                dofs_start
                    .iter()
                    .cartesian_product(dofs_start.iter())
                    .flat_map(|(&i, &j)| {
                        vec![
                            Triplet::new(i + 0, j + 0, 0.),
                            Triplet::new(i + 1, j + 0, 0.),
                            Triplet::new(i + 2, j + 0, 0.),
                            Triplet::new(i + 0, j + 1, 0.),
                            Triplet::new(i + 1, j + 1, 0.),
                            Triplet::new(i + 2, j + 1, 0.),
                            Triplet::new(i + 0, j + 2, 0.),
                            Triplet::new(i + 1, j + 2, 0.),
                            Triplet::new(i + 2, j + 2, 0.),
                        ]
                    })
            })
            .collect_vec();

        // Create sparse stiffness matrix from triplets and get data order
        let (k_sp, order_sp) = sparse_matrix_from_triplets(nfm.n_dofs(), nfm.n_dofs(), &triplets);

        // Return initialized struct
        Self {
            n_elem,
            elem_node_ids,
            elem_dofs_start: elem_dofs_start,
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
            r_tilde: Mat::zeros(3, 3),
            k_sp,
            order_sp,
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

        // Calculate components of current difference between node positions
        calc_r(
            self.r.as_mut(),
            self.x0.as_ref(),
            self.u1.as_ref(),
            self.u2.as_ref(),
        );

        // Calculate current distance between nodes
        calc_l(self.l.as_mut(), self.r.as_ref());

        // Calculate coefficients
        calc_c(
            self.c1.as_mut(),
            self.c2.as_mut(),
            self.k.as_ref(),
            self.l_ref.as_ref(),
            self.l.as_ref(),
        );

        // Calculate force for each element
        calc_f(self.f.as_mut(), self.c1.as_ref(), self.r.as_ref());

        // Calculate 3x3 stiffness matrix for each element
        calc_a(
            self.a.as_mut(),
            self.c1.as_ref(),
            self.c2.as_ref(),
            self.r.as_ref(),
            self.l.as_ref(),
            self.r_tilde.as_mut(),
        );
    }

    /// Adds beam elements to mass, damping, and stiffness matrices; and residual vector
    pub fn assemble_system(
        &mut self,
        state: &State,
        mut r: ColMut<f64>, // Residual
    ) {
        // If no elements, return
        if self.n_elem == 0 {
            return;
        }

        // Calculate element force and stiffness from current state
        self.calculate(state);

        //----------------------------------------------------------------------
        // Assemble elements into global matrices and vectors
        //----------------------------------------------------------------------

        // Add element force vector to residual vector
        self.elem_dofs_start
            .iter()
            .zip(self.f.col_iter())
            .for_each(|(dofs_start, f)| {
                dofs_start.iter().zip([1., -1.]).for_each(|(&i_dof, sign)| {
                    zip!(&mut r.as_mut().subrows_mut(i_dof, 3), f)
                        .for_each(|unzip!(r, f)| *r += sign * *f);
                });
            });

        let k_values = self.k_sp.val_mut();
        let mut i = 0;
        self.a.col_iter().for_each(|a_col| {
            a_col.iter().for_each(|&v| {
                k_values[self.order_sp[i]] += v;
                i += 1;
            });
            a_col.iter().for_each(|&v| {
                k_values[self.order_sp[i]] -= v;
                i += 1;
            });
            a_col.iter().for_each(|&v| {
                k_values[self.order_sp[i]] -= v;
                i += 1;
            });
            a_col.iter().for_each(|&v| {
                k_values[self.order_sp[i]] += v;
                i += 1;
            });
        });
    }
}

//------------------------------------------------------------------------------
// Kernels
//------------------------------------------------------------------------------

/// Calculate xyz distance between element nodes
#[inline]
fn calc_r(mut r: MatMut<f64>, x0: MatRef<f64>, u1: MatRef<f64>, u2: MatRef<f64>) {
    zip!(&mut r, &x0, &u1, &u2).for_each(|unzip!(r, x0, u1, u2)| *r = *x0 + *u2 - *u1);
}

/// Calculate distance between element nodes, length of spring
#[inline]
fn calc_l(l: ColMut<f64>, r: MatRef<f64>) {
    izip!(l.iter_mut(), r.col_iter()).for_each(|(l, r)| *l = r.norm_l2());
}

/// Calculate coefficients
#[inline]
fn calc_c(
    mut c1: ColMut<f64>,
    mut c2: ColMut<f64>,
    k: ColRef<f64>,
    l_ref: ColRef<f64>,
    l: ColRef<f64>,
) {
    zip!(&mut c1, &mut c2, &k, &l_ref, &l).for_each(|unzip!(c1, c2, k, l_ref, l)| {
        *c1 = *k * (*l_ref / *l - 1.);
        *c2 = *k * *l_ref / (*l).powi(3);
    });
}

/// Calculate element force vectors
#[inline]
fn calc_f(f: MatMut<f64>, c1: ColRef<f64>, r: MatRef<f64>) {
    izip!(f.col_iter_mut(), c1.iter(), r.col_iter())
        .for_each(|(mut f, &c1, r)| zip!(&mut f, r).for_each(|unzip!(f, r)| *f = c1 * *r));
}

/// Calculate element stiffness matrix
#[inline]
fn calc_a(
    a: MatMut<f64>,
    c1: ColRef<f64>,
    c2: ColRef<f64>,
    r: MatRef<f64>,
    l: ColRef<f64>,
    mut r_tilde: MatMut<f64>,
) {
    izip!(
        a.col_iter_mut(),
        c1.iter(),
        c2.iter(),
        r.col_iter(),
        l.iter(),
    )
    .for_each(|(a_col, &c1, &c2, r, l)| {
        let mut a = a_col.reshape_mut(3, 3);
        vec_tilde(r, r_tilde.as_mut());
        a.as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(c1 - c2 * l.powi(2));
        matmul(
            a,
            Accum::Add,
            r_tilde.as_ref(),
            r_tilde.as_ref(),
            -c2,
            Par::Seq,
        );
    });
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::Model;
    use crate::util::ColRefReshape;
    use equator::assert;
    use faer::utils::approx::*;

    #[test]
    fn test_force() {
        let approx_eq = CwiseMat(ApproxEq::eps());

        let mut model = Model::new();

        let n1 = model.add_node().position_xyz(0., 0., 0.).build();
        let n2 = model.add_node().position_xyz(1., 0., 0.).build();

        model.add_spring_element(n1, n2, 10., None);

        let mut state = model.create_state();
        let nfm = model.create_node_freedom_map();
        let elements = model.create_elements(&nfm);
        let mut springs = elements.springs;

        // Zero displacement
        springs.calculate(&state);
        assert_eq!(springs.l_ref[0], 1.);
        assert_eq!(springs.l[0], 1.);
        assert!(springs.f.col(0) ~ col![0., 0., 0.]);
        assert!(
            springs.a.col(0).reshape(3, 3) ~
            mat![[-10., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        );

        // Unit displacement
        state.u[(0, 1)] = 1.;
        springs.calculate(&state);
        assert_eq!(springs.l_ref[0], 1.);
        assert_eq!(springs.l[0], 2.);
        assert_eq!(springs.c1[0], -5.);
        assert_eq!(springs.c2[0], 10. * 1. / (2. as f64).powi(3));
        assert!(springs.f.col(0) ~ col![-10., 0., 0.]);
        assert!(
            springs.a.col(0).reshape(3, 3) ~
            mat![[-10., 0., 0.], [0., -5., 0.], [0., 0., -5.]]
        );
    }
}

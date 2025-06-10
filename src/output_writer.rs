use crate::state::State;

pub struct OutputWriter<'a> {
    netcdf_file: &'a mut netcdf::FileMut,
    data: [Vec<f64>; 7], // x, y, z, w, i, j, k
}
impl<'a> OutputWriter<'a> {
    pub fn new(netcdf_file: &'a mut netcdf::FileMut, n_nodes: usize) -> Self {
        netcdf_file.add_unlimited_dimension("time").unwrap();
        netcdf_file.add_dimension("nodes", n_nodes).unwrap();
        let dims = &["time", "nodes"];
        let comps = ["x", "y", "z", "w", "i", "j", "k"];
        ["x", "u", "v", "a", "f"].iter().for_each(|&var| {
            comps.iter().for_each(|&comp| {
                if comp == "w" && var != "x" && var != "u" {
                    return; // Skip w component for non-position variables
                }
                let v = format!("{}_{}", var, comp);
                netcdf_file.add_variable::<f64>(&v, dims).unwrap();
            });
        });

        Self {
            netcdf_file,
            data: [
                vec![0.0; n_nodes], // x
                vec![0.0; n_nodes], // y
                vec![0.0; n_nodes], // z
                vec![0.0; n_nodes], // w
                vec![0.0; n_nodes], // i
                vec![0.0; n_nodes], // j
                vec![0.0; n_nodes], // k
            ],
        }
    }

    pub fn write(&mut self, state: &State, time_step: usize) {
        // Write position data
        state.x.col_iter().enumerate().for_each(|(i, c)| {
            c.iter().enumerate().for_each(|(j, &val)| {
                self.data[j][i] = val;
            });
        });
        ["x_x", "x_y", "x_z", "x_w", "x_i", "x_j", "x_k"]
            .iter()
            .enumerate()
            .for_each(|(j, &var)| {
                self.netcdf_file
                    .variable_mut(var)
                    .unwrap()
                    .put_values(&self.data[j], (time_step, ..))
                    .unwrap();
            });

        // Write displacement data
        state.u.col_iter().enumerate().for_each(|(i, c)| {
            c.iter().enumerate().for_each(|(j, &val)| {
                self.data[j][i] = val;
            });
        });
        ["u_x", "u_y", "u_z", "u_w", "u_i", "u_j", "u_k"]
            .iter()
            .enumerate()
            .for_each(|(j, &var)| {
                self.netcdf_file
                    .variable_mut(var)
                    .unwrap()
                    .put_values(&self.data[j], (time_step, ..))
                    .unwrap();
            });

        // Write velocity data
        state.v.col_iter().enumerate().for_each(|(i, c)| {
            c.iter().enumerate().for_each(|(j, &val)| {
                self.data[j][i] = val;
            });
        });
        ["v_x", "v_y", "v_z", "v_i", "v_j", "v_k"]
            .iter()
            .enumerate()
            .for_each(|(j, &var)| {
                self.netcdf_file
                    .variable_mut(var)
                    .unwrap()
                    .put_values(&self.data[j], (time_step, ..))
                    .unwrap();
            });

        // Write acceleration data
        state.a.col_iter().enumerate().for_each(|(i, c)| {
            c.iter().enumerate().for_each(|(j, &val)| {
                self.data[j][i] = val;
            });
        });
        ["a_x", "a_y", "a_z", "a_i", "a_j", "a_k"]
            .iter()
            .enumerate()
            .for_each(|(j, &var)| {
                self.netcdf_file
                    .variable_mut(var)
                    .unwrap()
                    .put_values(&self.data[j], (time_step, ..))
                    .unwrap();
            });

        // Write force data
        state.fx.col_iter().enumerate().for_each(|(i, c)| {
            c.iter().enumerate().for_each(|(j, &val)| {
                self.data[j][i] = val;
            });
        });
        ["f_x", "f_y", "f_z", "f_i", "f_j", "f_k"]
            .iter()
            .enumerate()
            .for_each(|(j, &var)| {
                self.netcdf_file
                    .variable_mut(var)
                    .unwrap()
                    .put_values(&self.data[j], (time_step, ..))
                    .unwrap();
            });
    }
}

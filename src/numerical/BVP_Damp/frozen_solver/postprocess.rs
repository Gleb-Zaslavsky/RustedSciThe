impl NRBVP {
    pub fn dont_save_log(&mut self, dont_save_log: bool) {
        self.no_reports = dont_save_log;
    }
    pub fn save_to_file(&self) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let result_DMatrix = self
            .get_result()
            .expect("Frozen BVP save_to_file requires a computed solution matrix");
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            "result.txt",
            &self.x_mesh,
            &self.arg,
        );
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        let number_of_Ys = self.values.len();
        let reduced = self.result.as_ref()?;
        if number_of_Ys == 0 || reduced.len() != number_of_Ys * self.n_steps {
            return None;
        }

        let mut boundary_values = Vec::new();
        for (state_index, state_name) in self.values.iter().enumerate() {
            for &(side, value) in self.BorderConditions.get(state_name)? {
                let node_index = match side {
                    0 => 0,
                    1 => self.n_steps,
                    _ => return None,
                };
                boundary_values.push((node_index * number_of_Ys + state_index, side, value));
            }
        }
        let full_len = number_of_Ys * (self.n_steps + 1);
        let mut seen_positions = vec![false; full_len];
        if boundary_values.len() != number_of_Ys
            || boundary_values.iter().any(|(position, _, value)| {
                if *position >= full_len || !value.is_finite() || seen_positions[*position] {
                    return true;
                }
                seen_positions[*position] = true;
                false
            })
        {
            return None;
        }

        let full = construct_full_solution(reduced.clone(), boundary_values);
        if full.len() != full_len {
            return None;
        }
        Some(
            DMatrix::from_column_slice(number_of_Ys, self.n_steps + 1, full.as_slice()).transpose(),
        )
    }

    /// Converts the computed solution into the unified postprocessing dataset.
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        if self.result.is_none() {
            return Err(PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector and a valid endpoint constraint set".to_string(),
            ));
        }
        let values = self.get_result().ok_or_else(|| {
            PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector and a valid endpoint constraint set".to_string(),
            )
        })?;
        PostprocessDataset::new(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            values,
        )
    }

    /// Executes a declarative postprocessing plan using the modern facade.
    pub fn execute_postprocessing(
        &self,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, PostprocessError> {
        let dataset = self.postprocess_dataset()?;
        plan.execute(&dataset)
    }

    pub fn plot_result(&self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps + 1;
        let matrix_of_results: DMatrix<f64> = self
            .get_result()
            .expect("Frozen BVP plot_result requires a valid converged full profile");
        for _col in matrix_of_results.column_iter() {
            //   println!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }
        info!(
            "matrix of results has shape {:?}",
            matrix_of_results.shape()
        );
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            matrix_of_results,
        );
        info!("result plotted");
    }
}

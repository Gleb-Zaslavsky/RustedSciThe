impl NRBVP {
    pub fn dont_save_log(&mut self, dont_save_log: bool) {
        self.no_reports = dont_save_log;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     functions to return and save result in different formats
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Saves solution to text file with formatted output
    ///
    /// # Arguments
    /// * `filename` - Optional filename (defaults to "result.txt")
    pub fn save_to_file(&self, filename: Option<String>) {
        let name = if let Some(name) = filename {
            format!("{}.txt", name)
        } else {
            "result.txt".to_string()
        };
        let result_DMatrix = self
            .get_result()
            .expect("Damped BVP save_to_file requires a computed full solution matrix");
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            &name,
            &self.x_mesh,
            &self.arg,
        );
    }

    /// Saves solution to CSV file for data analysis
    ///
    /// # Arguments
    /// * `filename` - Optional filename (defaults to "result_table")
    pub fn save_to_csv(&self, filename: Option<String>) {
        let name = if let Some(name) = filename {
            name
        } else {
            "result_table".to_string()
        };
        let result_DMatrix = self
            .get_result()
            .expect("Damped BVP save_to_csv requires a computed full solution matrix");
        let _ = save_matrix_to_csv(
            &result_DMatrix,
            &self.values,
            &name,
            &self.x_mesh,
            &self.arg,
        );
    }

    /// Returns the complete solution matrix including boundary conditions
    ///
    /// # Returns
    /// Matrix where rows are grid points and columns are variables
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        self.full_result.clone()
    }

    /// Converts the computed solution into the unified postprocessing dataset.
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        let values = self.get_result().ok_or_else(|| {
            PostprocessError::InvalidDataset(
                "Damped BVP postprocess_dataset requires a computed full solution matrix"
                    .to_string(),
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

    /// Processes raw solution vector into full solution matrix
    ///
    /// Reconstructs complete solution by adding boundary conditions
    /// and reshaping into proper matrix format
    pub fn handle_result(&mut self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Damped BVP handle_result requires a converged solution vector")
            .clone();

        let BC_position_and_value = self.BC_position_and_value.clone();
        let full_results_vector = construct_full_solution(vector_of_results, BC_position_and_value);
        let full_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps + 1, full_results_vector.as_slice());
        let full_results = full_results.transpose();
        info!("matrix of results has shape {:?}", full_results.shape());
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        self.full_result = Some(full_results.clone());
    }
    /// Creates plots using gnuplot backend
    ///
    /// Generates publication-quality plots of the solution
    /// Requires gnuplot to be installed and in PATH
    pub fn gnuplot_result(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP gnuplot_result requires a computed full solution matrix");
        plots_gnulot(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }

    /// Creates plots using plotters crate
    ///
    /// Generates solution plots with embedded Rust plotting
    pub fn plot_result(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP plot_result requires a computed full solution matrix");
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }
    pub fn plot_result_in_terminal(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP plot_result_in_terminal requires a computed full solution matrix");
        plots_terminal(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }
    /// Computes and displays solver performance statistics
    ///
    /// Shows memory usage, iteration counts, and timing information
    fn calc_statistics(&self) {
        let mut stats = self.telemetry_counters.snapshot().to_legacy_map();
        if let Some(jac) = self.factor_owner.jacobian() {
            let jac_shape = jac.shape();
            let matrix_weight = checkmem(&**jac);
            stats.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
            stats.insert(
                "number of jacobian elements".to_string(),
                jac_shape.0 * jac_shape.1,
            );
        }
        stats.insert("length of y vector".to_string(), self.y.len() as usize);
        stats.insert(
            "number of grid points".to_string(),
            self.x_mesh.len() as usize,
        );
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());

        // What nodes were added per refinement

        let nodes_added_table: Vec<Vec<String>> = self
            .nodes_added
            .iter()
            .enumerate()
            .map(|(idx, val)| vec![idx.to_string(), val.to_string()])
            .collect();

        info!("\n \n NODES ADDED PER REFINEMENT \n \n");
        let mut table = Builder::from(nodes_added_table).build();
        table.with(Style::modern_rounded());
        info!("\n {} \n", table.to_string());
    }

    ////////////////////////////////////////////////////////////
    // Utility methods
    ////////////////////////////////////////////////////////////

    /// Updates internal timing statistics
    ///
    /// Used internally to accumulate timing data for performance analysis
    pub fn step_with_timer(&mut self, pair_of_times: (std::time::Duration, std::time::Duration)) {
        let (fun_time, linear_sys_time) = pair_of_times;
        self.custom_timer.append_to_fun_time(fun_time);
        self.custom_timer.append_to_linear_sys_time(linear_sys_time);
    }
}

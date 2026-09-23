impl NRBVP {
    /// Computes undamped Newton step and solves linear system.
    ///
    /// This evaluates the residual and solves the current linearized system.
    pub fn step(
        &self,
        p: f64,
        y: &dyn VectorType,
    ) -> (
        Box<dyn VectorType>,
        (std::time::Duration, std::time::Duration),
    ) {
        let (step, timings, _, _) = self.step_with_linear_telemetry(p, y);
        (step, timings)
    }

    /// Computes a Newton step and also returns native factor/RHS timings.
    ///
    /// The public [`Self::step`] method keeps its historical return shape.
    /// Solver internals use this richer form to expose typed factorization
    /// reuse telemetry without changing downstream callers.
    fn try_step_with_linear_telemetry(
        &self,
        p: f64,
        y: &dyn VectorType,
    ) -> Result<
        (
            Box<dyn VectorType>,
            (std::time::Duration, std::time::Duration),
            std::time::Duration,
            std::time::Duration,
        ),
        BvpBackendIntegrationError,
    > {
        let mut factor_owner = self.factor_owner.borrow_mut();
        self.solve_step_with_linear_telemetry(p, y, factor_owner.as_mut())
    }

    fn solve_step_with_linear_telemetry(
        &self,
        p: f64,
        y: &dyn VectorType,
        mut factor_owner: Option<&mut OwnedLinearFactorRuntime>,
    ) -> Result<
        (
            Box<dyn VectorType>,
            (std::time::Duration, std::time::Duration),
            std::time::Duration,
            std::time::Duration,
        ),
        BvpBackendIntegrationError,
    > {
        let fun_time_start = Instant::now();
        let fun = &self.fun;
        self.telemetry_counters.record_residual_call();
        let F_k = fun.try_call(p, y).map_err(|error| {
            BvpBackendIntegrationError::CallbackExecutionFailed {
                stage: "residual".to_string(),
                message: error.to_string(),
            }
        })?;
        let fun_time_end = fun_time_start.elapsed();
        let J_k = self.factor_owner.jacobian().ok_or_else(|| {
            BvpBackendIntegrationError::PipelinePanicked(
                "Damped BVP Newton step requires a cached Jacobian matrix".to_string(),
            )
        })?;
        let (matrix_rows, matrix_columns) = J_k.shape();
        if F_k.len() != matrix_rows || matrix_rows != matrix_columns {
            return Err(BvpBackendIntegrationError::LinearSolveFailed {
                backend: "legacy-matrix".to_string(),
                matrix_rows,
                matrix_columns,
                rhs_len: F_k.len(),
                message: format!(
                    "residual/Jacobian shape mismatch: residual_len={}, Jacobian_shape=({matrix_rows}, {matrix_columns})",
                    F_k.len()
                ),
            });
        }
        let residual_norm = F_k.norm();
        info!("\n \n residual norm = {:?} ", residual_norm);
        //    println!(" \n \n F_k = {:?} \n \n", F_k.to_DVectorType());
        for el in F_k.iterate() {
            if !el.is_finite() {
                error!("\n \n NaN in undamped step residual function \n \n");
                return Err(BvpBackendIntegrationError::LinearSolveFailed {
                    backend: "legacy-matrix".to_string(),
                    matrix_rows,
                    matrix_columns,
                    rhs_len: F_k.len(),
                    message: "residual vector contains a non-finite value before linear solve"
                        .to_string(),
                });
            }
        }
        // solving equation J_k*dy_k=-F_k for undamped dy_k, but Lambda*dy_k - is dumped step
        let owned_factor_path = factor_owner.is_some();
        let linear_sys_time_start = Instant::now();
        let (undamped_step_k, linear_timing) = if let Some(owner) = factor_owner.as_mut() {
            let (step, factorization, rhs_solve) = owner.try_solve(&*F_k).map_err(|error| {
                BvpBackendIntegrationError::LinearSolveFailed {
                    backend: "owned-factor".to_string(),
                    matrix_rows,
                    matrix_columns,
                    rhs_len: F_k.len(),
                    message: format!("{error:?}"),
                }
            })?;
            (
                step,
                crate::numerical::BVP_Damp::BVP_traits::LinearSolveTiming {
                    factorization,
                    rhs_solve,
                },
            )
        } else {
            J_k.try_solve_sys_with_timing(
                &*F_k,
                self.linear_sys_method.clone(),
                self.abs_tolerance,
                self.max_iterations,
                self.factor_owner.bandwidth(),
                y,
            )
            .map_err(|error| BvpBackendIntegrationError::LinearSolveFailed {
                backend: "matrix-try-api".to_string(),
                matrix_rows,
                matrix_columns,
                rhs_len: F_k.len(),
                message: error.to_string(),
            })?
        };
        //  info!("linear system solution {},\n {} \n {}", undamped_step_k.to_DVectorType(), F_k.to_DVectorType(), J_k.to_DMatrixType());
        let linear_sys_time_end = linear_sys_time_start.elapsed();
        let reported_linear_time = if owned_factor_path {
            linear_sys_time_end + linear_timing.factorization
        } else {
            linear_sys_time_end
        };
        for el in undamped_step_k.iterate() {
            if !el.is_finite() {
                log::error!("\n \n NaN in damped step deltaY \n \n");
                return Err(BvpBackendIntegrationError::LinearSolveFailed {
                    backend: "legacy-matrix".to_string(),
                    matrix_rows,
                    matrix_columns,
                    rhs_len: F_k.len(),
                    message: "Newton update contains a non-finite value after linear solve"
                        .to_string(),
                });
            }
        }
        let pair_of_times = (fun_time_end, reported_linear_time);
        Ok((
            undamped_step_k,
            pair_of_times,
            linear_timing.factorization,
            linear_timing.rhs_solve,
        ))
    }

    fn step_with_linear_telemetry(
        &self,
        p: f64,
        y: &dyn VectorType,
    ) -> (
        Box<dyn VectorType>,
        (std::time::Duration, std::time::Duration),
        std::time::Duration,
        std::time::Duration,
    ) {
        self.solve_step_with_linear_telemetry(p, y, None)
            .unwrap_or_else(|error| panic!("Damped BVP step failed: {error:?}"))
    }

    /// Performs damped Newton step with line search
    ///
    /// Implements the core damping algorithm:
    /// 1. Computes undamped Newton step
    /// 2. Applies boundary constraints to determine maximum step size
    /// 3. Uses line search with damping to ensure residual decreases
    /// 4. Returns status code and accepted step
    ///
    /// # Returns
    /// * `(1, Some(step))` - Converged solution found
    /// * `(0, Some(step))` - Step accepted, continue iterations
    /// * `(-2, None)` - No acceptable damping coefficient found
    /// * `(-3, None)` - Step violates bounds
    pub fn try_damped_step(
        &mut self,
    ) -> Result<(i32, Option<Box<dyn VectorType>>), BvpBackendIntegrationError> {
        // macro for saving times
        macro_rules! save_operation_times {
            ($self:expr, $pair_of_times:expr) => {
                let (fun_time, linear_sys_time, factorization_time, rhs_solve_time) =
                    $pair_of_times;
                $self.custom_timer.append_to_fun_time(fun_time);
                $self
                    .custom_timer
                    .append_to_linear_sys_time(linear_sys_time);
                $self
                    .custom_timer
                    .append_to_factorization_time(factorization_time);
                $self.custom_timer.append_to_rhs_solve_time(rhs_solve_time);
            };
        }
        //_________________________________________________________________
        let p = self.p;
        let now = Instant::now();
        // compute the undamped Newton step
        let y_k_minus_1 = &*self.y;
        let factorization_ready = self
            .factor_owner
            .borrow()
            .as_ref()
            .map(|owner| owner.has_solved_rhs())
            .unwrap_or(false)
            || self
                .factor_owner
                .old_jac
                .as_ref()
                .map(|jacobian| jacobian.factorization_ready())
                .unwrap_or(false);
        let (undamped_step_k_minus_1, pair_of_times, factorization_time, rhs_solve_time) =
            self.try_step_with_linear_telemetry(p, y_k_minus_1)?;
        // saving times of corresponding operations
        save_operation_times!(
            self,
            (
                pair_of_times.0,
                pair_of_times.1,
                factorization_time,
                rhs_solve_time,
            )
        );
        self.telemetry_counters.record_linear_solve();
        self.telemetry_counters.record_rhs_solve();
        if factorization_ready {
            self.telemetry_counters.record_factorization_cache_hit();
        } else {
            self.telemetry_counters.record_factorization();
        }

        let fbound = bound_step_Cantera2(y_k_minus_1, &*undamped_step_k_minus_1, &self.bounds_vec);
        if fbound.is_nan() {
            self.telemetry_counters.record_non_finite_value(0);
            error!("\n \n fbound is NaN \n \n");
            panic!("Damped BVP damping failed: boundary step factor is NaN")
        }
        if fbound.is_infinite() {
            self.telemetry_counters.record_non_finite_value(0);
            error!("\n \n fbound is infinite \n \n");
            panic!("Damped BVP damping failed: boundary step factor is infinite")
        }
        if fbound < 1.0 {
            self.telemetry_counters.record_bound_limited_step(fbound);
        }
        // let fbound =1.0;
        info!("\n \n fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound < 1e-10 {
            log::warn!(
                "\n  No damped step can be taken without violating solution component bounds."
            );
            return Ok((-3, None));
        }

        let maxDampIter = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.max_damp_iter)
            .unwrap_or(5);
        let DampFacor = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.damp_factor)
            .unwrap_or(0.5);

        let mut S_k: Option<f64> = None;
        let mut damped_step_result: Option<Box<dyn VectorType>> = None;
        let mut conv: f64 = 0.0;

        // compute the weighted norm of the undamped step size (s0 in C++ code) - calculated OUTSIDE the loop
        let s0 = undamped_step_k_minus_1.norm();

        let mut k = 0;
        while k < maxDampIter {
            self.telemetry_counters.record_damping_trial_at(
                self.telemetry_counters.iterations(),
                lambda,
                f64::NAN,
            );
            if k > 1 {
                info!("\n \n damped_step number {} ", k);
            }
            info!("\n \n Damping coefficient = {}", lambda);

            let damping_trial_started = self.telemetry_counters.start_damping_trial_scope();

            // step the solution by the damped step size: x_{k+1} = x_k + alpha_k*step_k
            let damped_step_k = undamped_step_k_minus_1.mul_float(lambda);
            let y_k: Box<dyn VectorType> = y_k_minus_1 - &*damped_step_k;

            // compute the next undamped step that would result if x1 is accepted
            // J(x_k)^-1 F(x_k+1)
            let factorization_ready = self
                .factor_owner
                .borrow()
                .as_ref()
                .map(|owner| owner.has_solved_rhs())
                .unwrap_or(false)
                || self
                    .factor_owner
                    .old_jac
                    .as_ref()
                    .map(|jacobian| jacobian.factorization_ready())
                    .unwrap_or(false);
            let step_result = self.try_step_with_linear_telemetry(p, &*y_k);
            if step_result.is_err() {
                self.telemetry_counters
                    .finish_damping_trial_scope(damping_trial_started);
            }
            let (undamped_step_k, pair_of_times, factorization_time, rhs_solve_time) = step_result?;
            // saving times of corresponding operations
            save_operation_times!(
                self,
                (
                    pair_of_times.0,
                    pair_of_times.1,
                    factorization_time,
                    rhs_solve_time,
                )
            );
            self.telemetry_counters.record_linear_solve();
            self.telemetry_counters.record_rhs_solve();
            if factorization_ready {
                self.telemetry_counters.record_factorization_cache_hit();
            } else {
                self.telemetry_counters.record_factorization();
            }

            // compute the weighted norm of step1 (s1 in C++ code)
            let s1 = undamped_step_k.norm();
            self.error_old = s1;
            info!("\n \n L2 norm of undamped step = {}", s1);
            let convergence_cond_for_step =
                convergence_condition(&*y_k, &self.abs_tolerance, &self.rel_tolerance_vec);

            // If the norm of s1 is less than the norm of s0, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);

            // C++ acceptance criteria: if (s1 < 1.0 || s1 < s0)
            let accepted = (s1 < 1.0) || (s1 < s0);
            self.telemetry_counters
                .finish_damping_trial_scope(damping_trial_started);
            if accepted {
                // The criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies
                S_k = Some(s1);
                damped_step_result = Some(y_k.clone_box());
                conv = convergence_cond_for_step;
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced damping parameter
            lambda = lambda / (2.0f64.powf(k as f64 + DampFacor));
            self.telemetry_counters.record_damping_rejection_at(
                self.telemetry_counters.iterations(),
                lambda,
                s1,
            );
            info!("damping coefficient decreased to {}", lambda);
            S_k = Some(s1);

            k += 1;
        }

        if k < maxDampIter {
            let step_norm = S_k.expect(
                "Damped BVP damping invariant violated: accepted damping step did not record a step norm",
            );
            // if there is a damping coefficient found (so max damp steps not exceeded)
            if step_norm > conv {
                //found damping coefficient but not converged yet
                info!("\n \n  Damping coefficient found (solution has not converged yet)");
                info!(
                    "\n \n  step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old, step_norm, conv
                );
                Ok((0, damped_step_result))
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old, step_norm, conv
                );
                Ok((1, damped_step_result))
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("\n \n  No damping coefficient found (max damping iterations reached)");
            Ok((-2, None))
        }
    } // end of damped step

    pub fn damped_step(&mut self) -> (i32, Option<Box<dyn VectorType>>) {
        self.try_damped_step()
            .unwrap_or_else(|error| panic!("Damped BVP damped step failed: {error:?}"))
    }
    /// Compatibility wrapper around [`Self::try_calc_residual`].
    pub fn calc_residual(&self, y: Box<dyn VectorType>) -> f64 {
        self.try_calc_residual(y)
            .unwrap_or_else(|error| panic!("Damped BVP residual evaluation failed: {error:?}"))
    }

    /// Evaluates and validates one residual callback at the solver boundary.
    ///
    /// The validation is limited to vector shape and finiteness. It avoids any
    /// matrix conversion and therefore does not add a hidden dense allocation
    /// to the Lambdify hot path.
    pub fn try_calc_residual(
        &self,
        y: Box<dyn VectorType>,
    ) -> Result<f64, BvpBackendIntegrationError> {
        let fun = &self.fun;
        self.telemetry_counters.record_residual_call();
        let expected_len = y.len();
        let residual = fun.try_call(self.p, &*y).map_err(|error| {
            BvpBackendIntegrationError::CallbackExecutionFailed {
                stage: "residual".to_string(),
                message: error.to_string(),
            }
        })?;
        if residual.len() != expected_len {
            return Err(BvpBackendIntegrationError::CallbackShapeMismatch {
                stage: "residual".to_string(),
                expected_rows: expected_len,
                expected_columns: 1,
                actual_rows: residual.len(),
                actual_columns: 1,
            });
        }
        for (index, value) in residual.iterate().enumerate() {
            if !value.is_finite() {
                self.telemetry_counters.record_non_finite_value(index);
                return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                    stage: "residual".to_string(),
                    index,
                });
            }
        }
        Ok(residual.norm())
    }
    /// Main iteration loop for damped Newton-Raphson method
    ///
    /// Orchestrates the complete solution process:
    /// 1. Newton iterations with damping
    /// 2. Jacobian reuse strategy
    /// 3. Adaptive grid refinement when needed
    /// 4. Convergence checking
    ///
    /// # Returns
    /// Solution vector if converged, None if failed
    pub fn main_loop_damped(&mut self) -> Option<DVector<f64>> {
        self.try_main_loop_damped().unwrap_or_else(|err| {
            panic!("Damped BVP main loop failed during fallible runtime path: {err:?}")
        })
    }

    /// Fallible internal Newton loop used by [`NRBVP::try_solver`].
    pub fn try_main_loop_damped(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        ////////////////////////////////////////////////////////////////////////
        info!("\n \n solving system of equations with Newton-Raphson method! \n \n");
        info!("{:?}", self.initial_guess.shape());
        if self.grid_refinemens == 0 {
            let y: DMatrix<f64> = self.initial_guess.clone();
            //  println!("new y = {} \n \n", &y);
            let y: Vec<f64> = y.iter().cloned().collect();
            let y: DVector<f64> = DVector::from_vec(y);
            self.result = Some(y.clone()); // save into result in case the very first iteration
            // with the current n_steps will go wrong and we shall need grid refinement
            self.y = Vectors_type_casting(&y.clone(), self.effective_runtime_method());
        } else {
        }
        let initial_res_nornal = self.try_calc_residual(self.y.clone_box())?;
        info!("norm of the initial residual = {}", initial_res_nornal);
        // println!("y = {:?}", &y);
        let mut nJacReeval = 0;
        let mut i = 0;
        while i < self.max_iterations {
            let iteration_started = self.telemetry_counters.start_iteration_scope();
            self.jac_recalc = jac_recalc(
                &self.strategy_params,
                self.m,
                self.factor_owner.jacobian_slot(),
                &mut self.jac_recalc,
            );
            let jacobian_result = self.try_recalc_jacobian();
            if jacobian_result.is_err() {
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
            jacobian_result?;
            self.m += 1;
            i += 1; // increment the number of iterations
            self.telemetry_counters.record_iteration();
            let step_result = self.try_damped_step();
            if step_result.is_err() {
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
            let (status, damped_step_result) = step_result?;

            if status == 0 {
                // status == 0 means convergence is not reached yet we're going to another iteration
                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        error!("\n \n y_k_plus_1 is None");
                        panic!(
                            "Damped BVP main loop invariant violated: accepted step returned no updated state"
                        )
                    }
                };
                self.y = y_k_plus_1;
                self.jac_recalc = false;
            }
            // status == 0
            else if status == 1 {
                // status == 1 means convergence is reached, save the result
                info!("\n \n Solution has converged, breaking the loop!");

                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        panic!(
                            "Damped BVP main loop invariant violated: converged step returned no updated state"
                        )
                    }
                };
                let residual_result = self.try_calc_residual(y_k_plus_1.clone_box());
                if residual_result.is_err() {
                    self.telemetry_counters
                        .finish_iteration_scope(iteration_started);
                }
                let resid_norm = residual_result?;
                info!("residual norm of the solution = {}", resid_norm);
                let result = Some(y_k_plus_1.to_DVectorType()); // save the successful result of the iteration
                // before refining in case it will go wrong
                self.result = result.clone();
                info!(
                    "\n \n solution found for the current grid {}",
                    &self.result.clone().unwrap().len()
                );

                // if flag for new grid is up we must call adaptive grid refinement
                if self.new_grid_enabled
                    && self
                        .strategy_params
                        .as_ref()
                        .map_or(false, |p| p.adaptive.is_some())
                {
                    self.telemetry_counters
                        .finish_iteration_scope(iteration_started);
                    info!("solving with new grid!");
                    return self.try_solve_with_new_grid();
                } else {
                    // if adapive is None then we just return the result
                    info!("returning the result");

                    self.telemetry_counters
                        .finish_iteration_scope(iteration_started);
                    return Ok(result);
                };
            //  self.max_error = error; // ???
            }
            // status == 1
            else if status < 0 {
                //negative means convergence is not reached yet, damped step is not accepted
                if self.m > 1 {
                    // if we have already tried 2 times with same Jacobian we must recalculate Jacobian
                    self.jac_recalc = true;
                    info!(
                        "\n \n status <0, recalculating Jacobian flag up! Jacobian age = {} \n \n",
                        self.m
                    );
                    if nJacReeval > 3 {
                        break;
                    }
                    nJacReeval += 1;
                } else {
                    info!("\n \n Jacobian age {} =<1 \n \n", self.m);
                    //  self.new_grid_enabled = true;
                    break;
                }
            } // status <0

            self.telemetry_counters
                .finish_iteration_scope(iteration_started);
            info!("\n \n end of iteration {} with jac age {} \n \n", i, self.m);
        }

        // all iterations, recalculations of Jacobian were unsuccessful
        // only that can help - grid refinement

        if self.new_grid_enabled
            && self
                .strategy_params
                .as_ref()
                .map_or(false, |p| p.adaptive.is_some())
        {
            info!("\n \n iterations unsuccessful, calling solve_with_new_grid \n \n");
            return self.try_solve_with_new_grid();
        }

        Ok(None)
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                      functions to create a new grid and recalculate with new grid
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Determines if grid refinement is needed based on solution analysis
    ///
    /// Checks refinement criteria and maximum refinement limits
    pub fn we_need_refinement(&mut self) {
        if let Some(ref params) = self.strategy_params {
            if let Some(ref adaptive_config) = params.adaptive {
                let mut res = match adaptive_config.version {
                    1 => {
                        if self.number_of_refined_intervals == 0 {
                            log::info!(
                                "\n \n number of marked intervals is 0, no new grid is needed \n \n"
                            );
                            false
                        } else {
                            log::info!(
                                "\n \n number of marked intervals is {}, new grid is needed \n \n",
                                self.number_of_refined_intervals
                            );
                            true
                        }
                    }
                    _ => panic!(
                        "Damped BVP adaptive grid refinement failed: unsupported adaptive version"
                    ),
                };

                if adaptive_config.max_refinements <= self.grid_refinemens {
                    info!(
                        "maximum number of grid refinements {} reached {}",
                        adaptive_config.max_refinements, self.grid_refinemens
                    );
                    res = false;
                }
                self.new_grid_enabled = res;
            }
        }
    }

    /// Creates refined mesh based on solution gradients and error estimates
    ///
    /// # Returns
    /// Tuple of (new mesh points, interpolated initial guess, number of refined intervals)
    fn create_new_grid(
        &mut self,
    ) -> Result<(Vec<f64>, DVector<f64>, usize), BvpBackendIntegrationError> {
        info!("================GRID REFINEMENT===================");
        let y = self.result.clone().unwrap().clone_box();
        let y_DVector = y.to_DVectorType();
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;

        // dbg!(&y_DMatrix.);
        let method = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.adaptive.as_ref())
            .map(|a| a.grid_method.clone())
            .ok_or_else(|| BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy_params.adaptive.grid_method".to_string(),
                value: "missing".to_string(),
                message: "grid method must be specified when adaptive refinement is enabled"
                    .to_string(),
            })?;

        self.custom_timer.grid_refinement_tic();

        let BC_position_and_value = self.BC_position_and_value.clone();
        let full_result_vector = construct_full_solution(y_DVector.clone(), BC_position_and_value);

        let y_DMatrix =
            DMatrix::from_column_slice(number_of_Ys, n_steps + 1, full_result_vector.as_slice());
        for (value, row) in self.values.iter().zip(y_DMatrix.clone().row_iter()) {
            let row: Vec<f64> = row.iter().cloned().collect();
            log::debug!(
                "Initial guess for {}: {:?} of len {}",
                value,
                row,
                row.len()
            );
        }

        let residuals: Option<DVector<f64>> = match method {
            GridRefinementMethod::Sci() => {
                // compute residuals on the current grid
                let fun = &self.fun;
                let p = self.p;
                let y_dvector = self.result.clone().unwrap();
                let y = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
                    &y_dvector,
                    self.effective_runtime_method(),
                );
                self.telemetry_counters.record_residual_call();
                let residuals = fun.try_call(p, &*y).map_err(|error| {
                    BvpBackendIntegrationError::CallbackExecutionFailed {
                        stage: "grid refinement residual".to_string(),
                        message: error.to_string(),
                    }
                })?;
                let residuals = residuals.to_DVectorType();
                Some(residuals)
            }
            _ => None,
        };

        let (new_mesh, initial_guess, number_of_nonzero_keys) = new_grid(
            method,
            &y_DMatrix,
            &self.x_mesh,
            self.abs_tolerance,
            residuals,
        );

        let initial_guess = extract_unknown_variables(
            initial_guess,
            &self.BC_position_and_value,
            number_of_nonzero_keys,
        );
        assert_eq!(
            initial_guess.len(),
            (new_mesh.len() - 1) * self.values.len(),
            "Initial guess size mismatch after grid refinement"
        );

        //  dbg!(&initial_guess);

        info!("================GRID REFINEMENT ENDED===================");
        self.custom_timer.grid_refinement_tac();
        Ok((new_mesh, initial_guess, number_of_nonzero_keys))
    }

    /// Continues solving on refined grid
    ///
    /// Updates solver state with new mesh and restarts Newton iterations
    fn try_solve_with_new_grid(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let (new_mesh, initial_guess, number_of_nonzero_keys) = self.create_new_grid()?;
        self.custom_timer.grid_refinement_tac();
        self.number_of_refined_intervals = number_of_nonzero_keys;
        self.nodes_added.push(number_of_nonzero_keys);
        let initial_guess_matrix = DMatrix::from_column_slice(
            self.values.len(),
            new_mesh.len() - 1,
            initial_guess.as_slice(),
        );

        self.initial_guess = initial_guess_matrix;
        self.y = Vectors_type_casting(&initial_guess, self.effective_runtime_method());
        self.x_mesh = DVector::from_vec(new_mesh.clone());
        self.grid_refinemens += 1;
        info!(
            "\n \n grid refinement counter = {} \n \n",
            self.grid_refinemens
        );
        self.telemetry_counters.record_grid_refinement();
        self.we_need_refinement();

        if number_of_nonzero_keys > 0 {
            // Preserve the structural bandwidth before clearing the prepared
            // layout. `clear_layout` intentionally resets it to (0, 0), but
            // the regenerated mesh must reuse the previous structural hint;
            // otherwise the second Banded preparation receives a diagonal-only
            // contract and rejects valid off-diagonal entries.
            let prepared_bandwidth = self.factor_owner.bandwidth();

            // Clear old Jacobian completely to avoid dimension mismatch
            let had_owned_factor = self.factor_owner.borrow().is_some();
            if had_owned_factor
                || self
                    .factor_owner
                    .old_jac
                    .as_ref()
                    .map(|jacobian| jacobian.factorization_ready())
                    .unwrap_or(false)
            {
                self.telemetry_counters.record_factorization_invalidation();
            }
            self.factor_owner.clear_numeric_jacobian();
            self.jac = None; // Clear Jacobian function as well
            self.jac_recalc = true; // Force Jacobian recalculation for new grid
            self.m = 0; // Reset Jacobian age counter

            // Clear other cached state that might be invalid for new grid
            self.bounds_vec.clear();
            self.rel_tolerance_vec.clear();
            self.factor_owner.clear_layout();
            self.variable_string.clear();

            // Update grid parameters
            self.n_steps = new_mesh.len() - 1;

            info!(
                "new guess of shape {} {}",
                self.initial_guess.nrows(),
                self.initial_guess.ncols()
            );
            info!("new mesh length {}", new_mesh.len());

            self.custom_timer.symbolic_operations_tic();
            // Regenerate system with the preserved structural bandwidth.
            self.try_eq_generate(Some(new_mesh), Some(prepared_bandwidth))?;
            self.custom_timer.symbolic_operations_tac();
        } else {
            info!("no new grid needed - returning to main loop");
            return Ok(None);
        }
        self.jac_recalc = true;
        self.try_main_loop_damped()
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       main functions to start the solver
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Internal solver method with timing and statistics
    ///
    /// Coordinates the complete solution process:
    /// 1. Symbolic system generation
    /// 2. Newton iteration loop
    /// 3. Result processing and statistics
    ///
    /// # Returns
    /// Solution vector if successful, None if failed
    /// Fallible solve path without logging setup.
    ///
    /// This is the preferred entrypoint for internal/runtime callers that want
    /// typed backend and execution errors but do not need the higher-level
    /// logging wrapper provided by [`NRBVP::try_solve`].
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        self.telemetry_counters.begin_solve();
        self.custom_timer.start();
        let begin = Instant::now();
        let res = (|| {
            self.custom_timer.symbolic_operations_tic();
            self.try_eq_generate(None, None)?;
            self.custom_timer.symbolic_operations_tac();
            self.try_main_loop_damped()
        })();
        self.custom_timer.finish();
        self.telemetry_counters.record_termination(
            matches!(&res, Ok(Some(_))),
            if matches!(&res, Ok(Some(_))) {
                1.0
            } else {
                0.0
            },
        );
        let res = res?;
        let end = begin.elapsed();
        elapsed_time(end);
        self.handle_result();
        self.calc_statistics();
        self.custom_timer.get_all();
        Ok(res)
    }

    /// Solves a system whose symbolic/runtime callbacks have already been
    /// prepared by [`NRBVP::try_eq_generate`].
    ///
    /// This is intentionally separate from [`NRBVP::try_solver`]: the normal
    /// entry point owns the complete cold lifecycle and regenerates callbacks,
    /// while prepared callers (benchmarks, parameter sweeps, and integrations
    /// that explicitly cache a backend) must not pay that preparation cost a
    /// second time. The result handling and solver statistics remain identical
    /// to the regular path.
    pub fn try_solver_prepared(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let prepared_fingerprint = self.prepared_plan_fingerprint();
        if !self
            .prepared_runtime_revision
            .is_current_with_fingerprint(prepared_fingerprint)
            || !self
                .factor_owner
                .prepared_binding_matches(prepared_fingerprint)
        {
            return Err(BvpBackendIntegrationError::PreparedRuntimeInvalidated {
                reason: "prepared plan is stale: a tracked revision, public compatibility input, or prepared resource binding changed; call try_eq_generate before try_solver_prepared".to_string(),
            });
        }
        self.telemetry_counters.begin_solve();
        self.custom_timer.start();
        let begin = Instant::now();
        let res = self.try_main_loop_damped();
        self.custom_timer.finish();
        self.telemetry_counters.record_termination(
            matches!(&res, Ok(Some(_))),
            if matches!(&res, Ok(Some(_))) {
                1.0
            } else {
                0.0
            },
        );
        let res = res?;
        let end = begin.elapsed();
        elapsed_time(end);
        self.handle_result();
        self.calc_statistics();
        self.custom_timer.get_all();
        Ok(res)
    }

    /// Compatibility-only wrapper over [`NRBVP::try_solver`].
    ///
    /// New production-facing code should call [`NRBVP::try_solver`] or
    /// [`NRBVP::try_solve`] instead.
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.try_solver()
            .unwrap_or_else(|err| panic!("BVP solver failed before Newton loop: {err:?}"))
    }

    /// Main public interface for solving BVP
    ///
    /// Wrapper that handles logging configuration and calls internal solver.
    /// Supports configurable logging levels and automatic log file generation.
    ///
    /// # Returns
    /// Solution vector if successful, None if failed
    pub fn try_solve(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            self.try_solver()
        } else {
            let log_option = self.parse_log_level()?;
            let logger_instance = if self.no_reports {
                // don't want to save txt report
                let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                    log_option,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                )]);
                logger_instance
            } else {
                // want to save txt report
                let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
                let name = format!("log_{}.txt", date_and_time);
                let file = File::create(&name).map_err(|err| {
                    BvpBackendIntegrationError::LogFileCreationFailed {
                        path: name.clone(),
                        message: err.to_string(),
                    }
                })?;
                let logger_instance = CombinedLogger::init(vec![
                    TermLogger::new(
                        log_option,
                        Config::default(),
                        TerminalMode::Mixed,
                        ColorChoice::Auto,
                    ),
                    WriteLogger::new(log_option, Config::default(), file),
                ]);
                logger_instance
            };
            match logger_instance {
                Ok(()) => {
                    let res = self.try_solver()?;
                    info!(" \n \n Program ended");
                    Ok(res)
                }
                Err(_) => self.try_solver(), //end Error
            } // end mat
        }
    }

    /// Compatibility-only wrapper over [`NRBVP::try_solve`].
    ///
    /// New code should prefer [`NRBVP::try_solve`] so AOT/logging/runtime failures
    /// remain typed instead of turning into a panic.
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve()
            .unwrap_or_else(|err| panic!("BVP solve failed before convergence loop: {err:?}"))
    }
}

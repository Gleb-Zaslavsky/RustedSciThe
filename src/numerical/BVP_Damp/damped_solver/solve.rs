impl NRBVP {
    fn validate_active_bounds(&self) -> Result<(), BvpBackendIntegrationError> {
        if let Some((index, value, _lower, _upper)) =
            first_bound_violation(&*self.y, &self.bounds_vec)
        {
            if !value.is_finite() {
                return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                    stage: "Newton iterate".to_string(),
                    index,
                });
            }
            return Err(BvpBackendIntegrationError::BoundStateOutsideBounds {
                iteration: self.telemetry_counters.iterations() as usize,
                index,
            });
        }
        Ok(())
    }

    /// Select the CorrectionNewton acceptance rules for the internal matched A/B gate.
    ///
    /// This remains test-only until fixed-mesh and adaptive-grid correctness gates
    /// justify exposing or promoting the candidate policy.
    #[cfg(test)]
    pub(crate) fn enable_correction_newton_criteria_for_test(&mut self) {
        self.newton_criteria = DampedNewtonCriteria::CorrectionNewton;
    }

    fn ensure_newton_norm_layout(&mut self) -> Result<(), BvpBackendIntegrationError> {
        let components = self.values.len();
        let full_len = self.y.len() + self.BC_position_and_value.len();
        if components == 0 || full_len != components.saturating_mul(self.n_steps + 1) {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "CorrectionNewton weighted-RMS layout".to_string(),
                value: format!(
                    "packed={}, boundaries={}, components={}, intervals={}",
                    self.y.len(),
                    self.BC_position_and_value.len(),
                    components,
                    self.n_steps
                ),
                message:
                    "packed unknowns and boundary entries do not cover the full node-major grid"
                        .to_string(),
            });
        }
        if self.newton_full_state_positions.len() == self.y.len()
            && self.newton_component_sum_scratch.len() == components
        {
            return Ok(());
        }

        let mut boundary_mask = vec![false; full_len];
        for &(position, _boundary_side, _) in &self.BC_position_and_value {
            let component = position % components;
            if position >= full_len || std::mem::replace(&mut boundary_mask[position], true) {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "CorrectionNewton weighted-RMS boundary layout".to_string(),
                    value: format!("position={position}, component={component}"),
                    message:
                        "boundary coordinates must be unique and match node-major component indices"
                            .to_string(),
                });
            }
        }
        let positions = boundary_mask
            .iter()
            .enumerate()
            .filter_map(|(position, is_boundary)| (!is_boundary).then_some(position))
            .collect::<Vec<_>>();
        if positions.len() != self.y.len() {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "CorrectionNewton weighted-RMS packed layout".to_string(),
                value: format!("mapped={}, packed={}", positions.len(), self.y.len()),
                message: "free full-grid positions do not match packed Newton state".to_string(),
            });
        }
        self.newton_full_state_positions = positions;
        self.newton_component_sum_scratch = vec![0.0; components];
        Ok(())
    }

    /// CorrectionNewton's full-grid component-scaled RMS, with eliminated BC
    /// corrections treated as zero and BC values included in component scales.
    fn correction_newton_weighted_rms(
        abs_tolerance: f64,
        rel_tolerances: &[f64],
        components: usize,
        boundary_values: &[(usize, usize, f64)],
        full_state_positions: &[usize],
        component_sums: &mut [f64],
        state: &dyn VectorType,
        correction: &dyn VectorType,
    ) -> Result<f64, BvpBackendIntegrationError> {
        let full_len = full_state_positions.len() + boundary_values.len();
        if state.len() != full_state_positions.len()
            || correction.len() != state.len()
            || rel_tolerances.len() != state.len()
            || component_sums.len() != components
        {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "CorrectionNewton weighted-RMS input".to_string(),
                value: format!(
                    "state={}, correction={}, relative_tolerances={}",
                    state.len(),
                    correction.len(),
                    rel_tolerances.len()
                ),
                message: "state, correction, and per-unknown tolerance dimensions must agree"
                    .to_string(),
            });
        }

        component_sums.fill(0.0);
        for &(position, _, value) in boundary_values {
            let component = position % components;
            if !value.is_finite() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "CorrectionNewton weighted-RMS boundary value".to_string(),
                    value: value.to_string(),
                    message: "boundary values must be finite".to_string(),
                });
            }
            component_sums[component] += value.abs();
        }
        for (index, &position) in full_state_positions.iter().enumerate() {
            let value = state.get_val(index);
            if !value.is_finite() {
                return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                    stage: "Newton state for weighted correction norm".to_string(),
                    index,
                });
            }
            component_sums[position % components] += value.abs();
        }

        let point_count = full_len / components;
        let mut scaled_rms = 0.0_f64;
        for (index, &position) in full_state_positions.iter().enumerate() {
            let component = position % components;
            let relative_tolerance = rel_tolerances[index];
            let scale =
                abs_tolerance + relative_tolerance * component_sums[component] / point_count as f64;
            let value = correction.get_val(index);
            if !relative_tolerance.is_finite()
                || relative_tolerance < 0.0
                || !scale.is_finite()
                || scale <= 0.0
                || !value.is_finite()
            {
                return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                    stage: "Newton correction weighted RMS".to_string(),
                    index,
                });
            }
            scaled_rms = scaled_rms.hypot(value / scale);
        }
        let result = scaled_rms / (full_len as f64).sqrt();
        if !result.is_finite() {
            return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                stage: "Newton correction weighted RMS".to_string(),
                index: 0,
            });
        }
        Ok(result)
    }

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
        let (step, timings, _, _, _) = self.step_with_linear_telemetry(p, y);
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
            f64,
        ),
        BvpBackendIntegrationError,
    > {
        let mut factor_owner = self.factor_owner.borrow_mut();
        self.solve_step_with_linear_telemetry(p, y, None, factor_owner.as_mut())
    }

    fn try_step_with_residual(
        &self,
        p: f64,
        y: &dyn VectorType,
        residual: Box<dyn VectorType>,
    ) -> Result<
        (
            Box<dyn VectorType>,
            (std::time::Duration, std::time::Duration),
            std::time::Duration,
            std::time::Duration,
            f64,
        ),
        BvpBackendIntegrationError,
    > {
        // The loop already evaluated this state before refreshing its Jacobian;
        // reuse that validated vector rather than repeating the callback.
        let mut factor_owner = self.factor_owner.borrow_mut();
        self.solve_step_with_linear_telemetry(p, y, Some(residual), factor_owner.as_mut())
    }

    fn solve_step_with_linear_telemetry(
        &self,
        p: f64,
        y: &dyn VectorType,
        residual: Option<Box<dyn VectorType>>,
        mut factor_owner: Option<&mut OwnedLinearFactorRuntime>,
    ) -> Result<
        (
            Box<dyn VectorType>,
            (std::time::Duration, std::time::Duration),
            std::time::Duration,
            std::time::Duration,
            f64,
        ),
        BvpBackendIntegrationError,
    > {
        let (F_k, fun_time) = if let Some(residual) = residual {
            (residual, std::time::Duration::ZERO)
        } else {
            let fun_time_start = Instant::now();
            let fun = &self.fun;
            self.telemetry_counters.record_residual_call();
            let residual = fun.try_call(p, y).map_err(|error| {
                BvpBackendIntegrationError::CallbackExecutionFailed {
                    stage: "residual".to_string(),
                    message: error.to_string(),
                }
            })?;
            (residual, fun_time_start.elapsed())
        };
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
                error!("\n \n NaN in damped step deltaY \n \n");
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
        let pair_of_times = (fun_time, reported_linear_time);
        Ok((
            undamped_step_k,
            pair_of_times,
            linear_timing.factorization,
            linear_timing.rhs_solve,
            residual_norm,
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
        f64,
    ) {
        self.solve_step_with_linear_telemetry(p, y, None, None)
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
        let (status, state, _) = self.try_damped_step_with_residual(None)?;
        Ok((status, state))
    }

    fn try_damped_step_with_residual(
        &mut self,
        initial_residual: Option<Box<dyn VectorType>>,
    ) -> Result<(i32, Option<Box<dyn VectorType>>, Option<f64>), BvpBackendIntegrationError> {
        self.validate_active_bounds()?;
        if self.newton_criteria == DampedNewtonCriteria::CorrectionNewton {
            self.ensure_newton_norm_layout()?;
        }
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
        let max_damp_iter = self
            .strategy_params
            .as_ref()
            .and_then(|params| params.max_damp_iter)
            .unwrap_or(7);
        if max_damp_iter == 0 {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy_params.max_damp_iter".to_string(),
                value: "0".to_string(),
                message: "at least one damping trial is required".to_string(),
            });
        }
        let damp_factor = if self.newton_criteria == DampedNewtonCriteria::Legacy {
            let value = self
                .strategy_params
                .as_ref()
                .and_then(|params| params.damp_factor)
                .unwrap_or(0.5);
            if !value.is_finite() || !(0.0 < value && value < 1.0) {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "strategy_params.damp_factor".to_string(),
                    value: value.to_string(),
                    message:
                        "the legacy damping exponent must be finite and strictly between 0 and 1"
                            .to_string(),
                });
            }
            value
        } else {
            // CorrectionNewton uses a fixed sqrt(2) reduction and does not
            // consult the legacy exponent-offset option.
            0.5
        };
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
        let base_step = if let Some(residual) = initial_residual {
            self.try_step_with_residual(p, y_k_minus_1, residual)?
        } else {
            self.try_step_with_linear_telemetry(p, y_k_minus_1)?
        };
        let (undamped_step_k_minus_1, pair_of_times, factorization_time, rhs_solve_time, _) =
            base_step;
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

        // The linear solve returns delta = J^-1 F, while the Newton state
        // update below is y_next = y - lambda * delta. Bound feasibility must
        // be evaluated using that same subtractive direction.
        let (mut fbound, limiting_index) = bound_step_Cantera2_with_update_sign_and_limit(
            y_k_minus_1,
            &*undamped_step_k_minus_1,
            &self.bounds_vec,
            -1.0,
        );
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
        fbound = feasible_update_factor(
            y_k_minus_1,
            &*undamped_step_k_minus_1,
            &self.bounds_vec,
            -1.0,
            fbound,
        );
        if fbound < 1.0 {
            self.telemetry_counters
                .record_bound_limited_step_at(fbound, limiting_index);
            if let Some(index) = limiting_index {
                let value = y_k_minus_1.get_val(index);
                let delta = undamped_step_k_minus_1.get_val(index);
                let bounds = self.bounds_vec[index];
                let full_step_value = value - delta;
                let limited_step_value = value - fbound * delta;
                info!(
                    "Damped bound limit: packed_index={} variable={} value={:.17e} delta={:.17e} full_step={:.17e} limited_step={:.17e} bounds={:?} factor={:.17e}",
                    index,
                    self.variable_string
                        .get(index)
                        .map(String::as_str)
                        .unwrap_or("<unknown>"),
                    value,
                    delta,
                    full_step_value,
                    limited_step_value,
                    bounds,
                    fbound,
                );
            }
        }
        // let fbound =1.0;
        info!("\n \n fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound < 1e-10 {
            warn!("\n  No damped step can be taken without violating solution component bounds.");
            return Ok((-3, None, None));
        }

        let mut S_k: Option<f64> = None;
        let mut damped_step_result: Option<Box<dyn VectorType>> = None;
        let mut accepted_residual_norm: Option<f64> = None;
        let mut conv: f64 = 0.0;
        // Legacy acceptance used a raw Euclidean correction; the candidate
        // policy uses CorrectionNewton's dimensionless component-scaled RMS.
        let s0 = match self.newton_criteria {
            DampedNewtonCriteria::Legacy => undamped_step_k_minus_1.norm(),
            DampedNewtonCriteria::CorrectionNewton => Self::correction_newton_weighted_rms(
                self.abs_tolerance,
                &self.rel_tolerance_vec,
                self.values.len(),
                &self.BC_position_and_value,
                &self.newton_full_state_positions,
                &mut self.newton_component_sum_scratch,
                y_k_minus_1,
                &*undamped_step_k_minus_1,
            )?,
        };

        let mut k = 0;
        while k < max_damp_iter {
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
            let (
                undamped_step_k,
                pair_of_times,
                factorization_time,
                rhs_solve_time,
                trial_residual_norm,
            ) = step_result?;
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

            let s1 = match self.newton_criteria {
                DampedNewtonCriteria::Legacy => undamped_step_k.norm(),
                DampedNewtonCriteria::CorrectionNewton => Self::correction_newton_weighted_rms(
                    self.abs_tolerance,
                    &self.rel_tolerance_vec,
                    self.values.len(),
                    &self.BC_position_and_value,
                    &self.newton_full_state_positions,
                    &mut self.newton_component_sum_scratch,
                    &*y_k,
                    &*undamped_step_k,
                )?,
            };
            self.error_old = s1;
            info!("\n \n Newton correction norm = {}", s1);
            let convergence_cond_for_step = match self.newton_criteria {
                DampedNewtonCriteria::Legacy => {
                    convergence_condition(&*y_k, &self.abs_tolerance, &self.rel_tolerance_vec)
                }
                DampedNewtonCriteria::CorrectionNewton => 1.0,
            };

            // If the norm of s1 is less than the norm of s0, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);

            // Both policies retain the same acceptance shape. Legacy used raw
            // L2 here; CorrectionNewton applies it to normalized weighted RMS.
            let accepted = (s1 < 1.0) || (s1 < s0);
            self.telemetry_counters
                .finish_damping_trial_scope(damping_trial_started);
            if accepted {
                // The criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies
                S_k = Some(s1);
                damped_step_result = Some(y_k.clone_box());
                accepted_residual_norm = Some(trial_residual_norm);
                conv = convergence_cond_for_step;
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced damping parameter
            match self.newton_criteria {
                DampedNewtonCriteria::Legacy => {
                    // Legacy schedule: exponent offset increases after every rejection.
                    lambda /= 2.0f64.powf(k as f64 + damp_factor);
                }
                DampedNewtonCriteria::CorrectionNewton => {
                    // Match CorrectionNewton's fixed geometric backtracking.
                    lambda /= 2.0_f64.sqrt();
                }
            }
            self.telemetry_counters.record_damping_rejection_at(
                self.telemetry_counters.iterations(),
                lambda,
                s1,
            );
            info!("damping coefficient decreased to {}", lambda);
            S_k = Some(s1);

            k += 1;
        }

        if k < max_damp_iter {
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
                Ok((0, damped_step_result, accepted_residual_norm))
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old, step_norm, conv
                );
                Ok((1, damped_step_result, accepted_residual_norm))
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("\n \n  No damping coefficient found (max damping trials reached)");
            Ok((-2, None, None))
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
        Ok(self.try_eval_residual(y)?.norm())
    }

    fn try_eval_residual(
        &self,
        y: Box<dyn VectorType>,
    ) -> Result<Box<dyn VectorType>, BvpBackendIntegrationError> {
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
        Ok(residual)
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
        let use_prepared_iterate = self.prepared_iterate;
        self.prepared_iterate = false;
        if self.grid_refinemens == 0 && !use_prepared_iterate {
            let y: DMatrix<f64> = self.initial_guess.clone();
            //  println!("new y = {} \n \n", &y);
            let y: Vec<f64> = y.iter().cloned().collect();
            let y: DVector<f64> = DVector::from_vec(y);
            self.result = Some(y.clone()); // save into result in case the very first iteration
                                           // with the current n_steps will go wrong and we shall need grid refinement
            self.y = Vectors_type_casting(&y.clone(), self.effective_runtime_method());
        } else {
        }
        self.validate_active_bounds()?;
        let initial_residual_started = Instant::now();
        // Keep the legacy initial-residual error ordering, then consume this
        // validated vector as the first Newton step's base residual.
        let mut initial_residual = Some(self.try_eval_residual(self.y.clone_box())?);
        let initial_residual_norm = initial_residual
            .as_ref()
            .expect("initial residual was just evaluated")
            .norm();
        self.custom_timer
            .append_to_fun_time(initial_residual_started.elapsed());
        info!("norm of the initial residual = {}", initial_residual_norm);
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
            let step_result = self.try_damped_step_with_residual(initial_residual.take());
            if step_result.is_err() {
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
            let (status, damped_step_result, accepted_residual_norm) = step_result?;

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
                let resid_norm = accepted_residual_norm.ok_or_else(|| {
                    BvpBackendIntegrationError::PipelinePanicked(
                        "converged damping step did not retain its accepted residual norm"
                            .to_string(),
                    )
                })?;
                info!("residual norm of the solution = {}", resid_norm);
                let result = Some(y_k_plus_1.to_DVectorType()); // save the successful result of the iteration
                                                                // before refining in case it will go wrong
                self.y = y_k_plus_1.clone_box();
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
                    return self.try_solve_with_new_grid(true);
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
            return self.try_solve_with_new_grid(false);
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
                            info!(
                                "\n \n number of marked intervals is 0, no new grid is needed \n \n"
                            );
                            false
                        } else {
                            info!(
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

        let expected_unknowns = self.n_steps.checked_mul(self.values.len()).ok_or_else(|| {
            BvpBackendIntegrationError::InvalidProblem {
                field: "grid_refinement_state".to_string(),
                message: "current unknown count overflowed while refining the mesh".to_string(),
            }
        })?;
        let y_DVector = (self.y.len() == expected_unknowns)
            .then(|| self.y.to_DVectorType())
            .or_else(|| {
                self.result
                    .as_ref()
                    .filter(|result| result.len() == expected_unknowns)
                    .map(|result| result.to_DVectorType())
            })
            .or_else(|| {
                (self.initial_guess.len() == expected_unknowns)
                    .then(|| DVector::from_column_slice(self.initial_guess.as_slice()))
            })
            .ok_or_else(|| BvpBackendIntegrationError::InvalidProblem {
                field: "grid_refinement_state".to_string(),
                message: format!(
                    "no stored profile matches the active grid: expected {expected_unknowns} unknowns, result={}, iterate={}, initial_guess={}",
                    self.result.as_ref().map_or(0, |result| result.len()),
                    self.y.len(),
                    self.initial_guess.len()
                ),
            })?;
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;

        // dbg!(&y_DMatrix.);
        self.custom_timer.grid_refinement_tic();

        let BC_position_and_value = self.BC_position_and_value.clone();
        let full_result_vector = construct_full_solution(y_DVector.clone(), BC_position_and_value);

        let y_DMatrix =
            DMatrix::from_column_slice(number_of_Ys, n_steps + 1, full_result_vector.as_slice());
        for (value, row) in self.values.iter().zip(y_DMatrix.clone().row_iter()) {
            let row: Vec<f64> = row.iter().cloned().collect();
            debug!(
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
                let y = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
                    &y_DVector,
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
        current_grid_converged: bool,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let (new_mesh, initial_guess, number_of_nonzero_keys) = self.create_new_grid()?;
        self.custom_timer.grid_refinement_tac();

        let mesh_changed = new_mesh.as_slice() != self.x_mesh.as_slice();
        if !mesh_changed {
            self.number_of_refined_intervals = 0;
            self.we_need_refinement();
            if current_grid_converged {
                return Ok(self.result.clone());
            }
            return Ok(None);
        }

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

        // Preserve the structural bandwidth before clearing the prepared
        // layout. `clear_layout` intentionally resets it to (0, 0), but the
        // regenerated mesh must reuse the previous structural hint; otherwise
        // the second Banded preparation receives a diagonal-only contract and
        // rejects valid off-diagonal entries.
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
    /// Fallible solve path with typed backend, execution, and logging setup
    /// errors. Enabled logging must initialize its persistent file sink before
    /// numerical work begins.
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let _logging_scope = SolverLogGuard::try_enter(self.effective_logging_level()?)
            .map_err(|source| BvpBackendIntegrationError::LogSetupFailed { source })?;
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
        self.telemetry_counters.finish_solve(&res);
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
        // A failed Newton attempt may leave an iterate in `self.result` for
        // refinement/restart diagnostics, but it is not a publishable profile.
        // In particular, its mesh may no longer match the active refined mesh.
        if res.is_some() {
            self.handle_result();
        }
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
        let _logging_scope = SolverLogGuard::try_enter(self.effective_logging_level()?)
            .map_err(|source| BvpBackendIntegrationError::LogSetupFailed { source })?;
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
        self.telemetry_counters.finish_solve(&res);
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
        // Keep failure handling consistent with the cold solve path: only a
        // converged result may be expanded with boundary values and published.
        if res.is_some() {
            self.handle_result();
        }
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
        self.try_solver()
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

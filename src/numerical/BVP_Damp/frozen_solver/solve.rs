impl NRBVP {
    /// Fallible Newton iteration used by the typed solve path.
    pub fn try_iteration(&mut self) -> Result<Box<dyn VectorType>, BvpBackendIntegrationError> {
        let p = self.p;
        let y = &*self.y;
        let fun = &self.fun;
        let fun_begin = Instant::now();
        self.telemetry_counters.record_residual_call();
        let new_fun = fun.try_call(p, y).map_err(|error| {
            BvpBackendIntegrationError::CallbackExecutionFailed {
                stage: "residual".to_string(),
                message: error.to_string(),
            }
        })?;
        self.custom_timer.append_to_fun_time(fun_begin.elapsed());
        let now = Instant::now();

        let reused_factorization;
        if self.jac_recalc {
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            self.custom_timer.jac_tic();
            let callback_result = match self.jac.as_mut() {
                Some(jacobian) => jacobian.try_call(p, y),
                None => {
                    self.custom_timer.jac_tac();
                    return Err(BvpBackendIntegrationError::PipelinePanicked(
                        "Frozen BVP iteration requires an installed Jacobian callback".into(),
                    ));
                }
            };
            let new_j = match callback_result {
                Ok(jacobian) => jacobian,
                Err(error) => {
                    self.custom_timer.jac_tac();
                    return Err(BvpBackendIntegrationError::CallbackExecutionFailed {
                        stage: "Jacobian".to_string(),
                        message: error.to_string(),
                    });
                }
            };
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.custom_timer.jac_tac();
            if self
                .factor_owner
                .old_jac
                .as_ref()
                .map(|jacobian| jacobian.factorization_ready())
                .unwrap_or(false)
            {
                self.telemetry_counters.record_factorization_invalidation();
            }
            // The cached Jacobian is read-only during a frozen reuse window.
            // Store the callback result directly instead of cloning the full
            // matrix on every iteration.
            self.factor_owner.replace_numeric_jacobian(new_j);
            let fresh_jacobian = self.factor_owner.jacobian().ok_or_else(|| {
                BvpBackendIntegrationError::PipelinePanicked(
                    "Frozen BVP Jacobian callback returned no cached matrix".into(),
                )
            })?;
            let prepared_factor = prepare_factor_owner_runtime(
                fresh_jacobian.as_ref(),
                self.factor_owner.bandwidth(),
                self.linear_sys_method.as_deref(),
            );
            self.factor_owner.replace_factor(prepared_factor);
            self.prepared_runtime_revision
                .mark_numeric_jacobian_current();
            if self.factor_owner.borrow().is_some() {
                self.prepared_runtime_revision.mark_factor_current();
            }
            reused_factorization = false;
            self.m = 0;
            self.telemetry_counters.record_jacobian_recalculation();
        } else {
            self.m = self.m + 1;
            reused_factorization = self
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
        }

        let new_j = self
            .factor_owner
            .old_jac
            .as_ref()
            .ok_or_else(|| {
                BvpBackendIntegrationError::PipelinePanicked(
                    "Frozen BVP iteration requires a cached Jacobian matrix when reuse is enabled"
                        .into(),
                )
            })?
            .as_ref();

        //   println!("new fun = {:?}", &new_fun);
        let linear_begin = Instant::now();
        let (delta, linear_timing) = if let Some(owner) = self.factor_owner.borrow_mut().as_mut() {
            let (delta, factorization, rhs_solve) =
                owner.try_solve(&*new_fun).map_err(|error| {
                    BvpBackendIntegrationError::LinearSolveFailed {
                        backend: "owned-factor".to_string(),
                        matrix_rows: new_j.shape().0,
                        matrix_columns: new_j.shape().1,
                        rhs_len: new_fun.len(),
                        message: format!("{error:?}"),
                    }
                })?;
            (
                delta,
                LinearSolveTiming {
                    factorization,
                    rhs_solve,
                },
            )
        } else {
            new_j.solve_sys_with_timing(
                &*new_fun,
                self.linear_sys_method.clone(),
                self.tolerance,
                self.max_iterations,
                self.factor_owner.bandwidth(),
                y,
            )
        };
        self.custom_timer
            .append_to_linear_sys_time(linear_begin.elapsed() + linear_timing.factorization);
        self.custom_timer
            .append_to_factorization_time(linear_timing.factorization);
        self.custom_timer
            .append_to_rhs_solve_time(linear_timing.rhs_solve);
        self.telemetry_counters.record_linear_solve();
        self.telemetry_counters.record_rhs_solve();
        if reused_factorization {
            self.telemetry_counters.record_factorization_cache_hit();
        } else {
            self.telemetry_counters.record_factorization();
        }
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        //  println!(" \n \n dy= {:?}", &delta);
        // element wise subtraction
        let new_y = y - &*delta;

        Ok(new_y)
    }

    /// Legacy panic-wrapper retained for callers using the historical API.
    pub fn iteration(&mut self) -> Box<dyn VectorType> {
        self.try_iteration().unwrap_or_else(|error| {
            panic!("Frozen BVP iteration failed during fallible runtime path: {error:?}")
        })
    }
    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        self.try_main_loop().unwrap_or_else(|err| {
            panic!("Frozen BVP main loop failed during fallible runtime path: {err:?}")
        })
    }

    /// Fallible internal Newton loop used by [`NRBVP::try_solver`].
    pub fn try_main_loop(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        info!("solving system of equations with Newton-Raphson method! \n \n");
        let y: DMatrix<f64> = self.initial_guess.clone();
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);
        self.y = Vectors_type_casting(&y.clone(), self.method.clone());
        let mut i = 0;

        while i < self.max_iterations {
            self.telemetry_counters.record_iteration();
            let iteration_started = self.telemetry_counters.start_iteration_scope();
            let iteration_result = self.try_iteration();
            if iteration_result.is_err() {
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
            let new_y = iteration_result?;
            let y1 = new_y.subtract(&*self.y);
            let dy: Box<dyn VectorType> = y1.clone_box();

            let error = dy.norm();
            self.jac_recalc = frozen_jac_recalc(
                &self.strategy,
                &self.strategy_params,
                self.factor_owner.jacobian_slot(),
                self.m,
                error,
                self.error_old,
            );
            self.error_old = error;
            info!(" \n \n error = {:?} \n \n", &error);
            if error < self.tolerance {
                log::info!("converged in {} iterations, error = {}", i, error);
                self.result = Some(new_y.to_DVectorType());
                self.max_error = error;
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
                return Ok(Some(new_y.to_DVectorType()));
            } else {
                let new_y: Box<dyn VectorType> = new_y.clone_box();
                self.y = new_y;
                i += 1;
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
        }
        Ok(None)
    }
    /// Fallible solve path without logging setup.
    ///
    /// This is the preferred entrypoint for internal/runtime callers that want
    /// typed backend and execution errors but do not need the higher-level
    /// logging wrapper provided by [`NRBVP::try_solve`].
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        // TODO! СЃСЂР°РІРЅРёС‚СЊ СЏРІРЅС‹Р№ РјСЌС€ СЃ РЅРµСЏРІРЅС‹Рј
        // let test_mesh = Some((0..100).map(|x| 0.01 * x as f64).collect::<Vec<f64>>());
        self.telemetry_counters.begin_solve();
        self.custom_timer.start();
        let begin = Instant::now();
        let res = (|| {
            self.custom_timer.symbolic_operations_tic();
            self.try_eq_generate()?;
            self.custom_timer.symbolic_operations_tac();
            self.try_main_loop()
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

        Ok(res)
    }

    /// Solves using callbacks prepared by an earlier `try_eq_generate` call.
    ///
    /// Numeric parameter values may be rebound between calls; structural or
    /// configuration changes are rejected until preparation is repeated.
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
                reason:
                    "prepared Frozen plan is stale: a tracked revision, public compatibility input, or prepared resource binding changed; call try_eq_generate before try_solver_prepared".to_string(),
            });
        }
        self.try_main_loop()
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solver`].
    ///
    /// New production-facing code should call [`NRBVP::try_solver`] or
    /// [`NRBVP::try_solve`] instead.
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.try_solver()
            .unwrap_or_else(|err| panic!("Frozen BVP solver failed before Newton loop: {err:?}"))
    }

    /// Main public fallible solve entrypoint with logging support.
    ///
    /// This is the preferred production-facing solve path.
    pub fn try_solve(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let logger_instance = if self.no_reports {
            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                LevelFilter::Info,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);
            logger_instance
        } else {
            let date_and_time = Local::now().format("%Y-%m-%d_%H-%M");
            let name = format!("log_{}.txt", date_and_time);
            let file = File::create(&name).map_err(|err| {
                BvpBackendIntegrationError::LogFileCreationFailed {
                    path: name.clone(),
                    message: err.to_string(),
                }
            })?;
            let logger_instance = CombinedLogger::init(vec![
                TermLogger::new(
                    LevelFilter::Info,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                ),
                WriteLogger::new(LevelFilter::Info, Config::default(), file),
            ]);
            logger_instance
        };
        match logger_instance {
            Ok(()) => {
                let res = self.try_solver()?;
                log::info!("Program ended");
                Ok(res)
            }
            Err(_) => self.try_solver(),
        }
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solve`].
    ///
    /// New code should prefer [`NRBVP::try_solve`] so AOT/logging/runtime failures
    /// remain typed instead of turning into a panic.
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve().unwrap_or_else(|err| {
            panic!("Frozen BVP solve failed before convergence loop: {err:?}")
        })
    }
}

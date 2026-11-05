//! Fixed-mesh ExprLegacy preparation for the independent Cantera Newton core.
//!
//! This module owns only trapezoid discretization and callback preparation. It
//! reuses the established pointwise ExprLegacy Lambdify frontend, but does not
//! construct or invoke a Damped/BVP_sci nonlinear solver.

use super::{
    pack_bvp_initial_state, CanteraCallbackFailure, CanteraExprBvpOptions, CanteraExprBvpProblem,
    CanteraJacobianPattern, CanteraNewtonConfig, CanteraNewtonSolver, CanteraSolverError,
};
use crate::numerical::BVP_sci::new::{BvpSciAssembly, BvpSciLambdifyPlan, BvpSciTelemetry};
use nalgebra::DVector;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
struct Contribution {
    output_slot: usize,
    node: Option<usize>,
    local_jacobian_slot: usize,
    scale: f64,
}

/// Prepare fixed-grid trapezoid callbacks and their global structural pattern.
pub(super) fn prepare(
    problem: CanteraExprBvpProblem,
    options: CanteraExprBvpOptions,
) -> Result<CanteraNewtonSolver, CanteraSolverError> {
    super::validate_expr_problem(&problem, &options)?;
    let packed = pack_bvp_initial_state(&problem)?;
    let component_count = problem.state_names.len();
    let steps = problem.steps;
    let node_count = steps + 1;
    let full_size = component_count * node_count;
    let h = (problem.interval.1 - problem.interval.0) / steps as f64;
    let state_names = problem
        .state_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let external_parameter_names = problem
        .equations
        .iter()
        .flat_map(|equation| equation.all_arguments_are_variables())
        .filter(|name| {
            name != &problem.independent_variable && !state_names.contains(name.as_str())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let assembly = match options.symbolic_frontend {
        super::CanteraSymbolicFrontend::ExprLegacy => BvpSciAssembly::ExprLegacy,
        super::CanteraSymbolicFrontend::AtomView => BvpSciAssembly::AtomViewNative,
    };
    let pointwise = match options.symbolic_execution {
        super::CanteraSymbolicExecution::Lambdify => {
            BvpSciLambdifyPlan::prepare_with_external_parameters(
                assembly,
                &problem.equations,
                &problem.state_names,
                &[],
                &external_parameter_names,
                problem.independent_variable.clone(),
                BvpSciTelemetry::disabled(),
            )
        }
        super::CanteraSymbolicExecution::Aot => {
            let config = options.aot_config.clone().ok_or_else(|| {
                CanteraSolverError::InvalidConfiguration {
                    message: "AOT execution requires an explicit generated-backend configuration"
                        .into(),
                }
            })?;
            BvpSciLambdifyPlan::prepare_aot_with_external_parameters(
                assembly,
                match options.matrix_layout {
                    super::CanteraMatrixLayout::Dense => {
                        crate::numerical::BVP_sci::new::BvpSciMatrixLayout::Dense
                    }
                    super::CanteraMatrixLayout::Sparse => {
                        crate::numerical::BVP_sci::new::BvpSciMatrixLayout::Sparse
                    }
                    super::CanteraMatrixLayout::BandedAuto => {
                        crate::numerical::BVP_sci::new::BvpSciMatrixLayout::BandedAuto
                    }
                },
                problem.equations.clone(),
                problem.state_names.clone(),
                Vec::new(),
                &external_parameter_names,
                problem.independent_variable.clone(),
                config,
                BvpSciTelemetry::disabled(),
            )
        }
    }
    .map_err(|source| CanteraSolverError::SymbolicPreparation { source })?;
    let mut local_pattern = pointwise.jacobian_pattern();
    if pointwise.is_aot()
        && options.matrix_layout == super::CanteraMatrixLayout::Dense
        && local_pattern.is_empty()
    {
        local_pattern.extend(
            (0..component_count)
                .flat_map(|row| (0..component_count).map(move |column| (row, column))),
        );
    }
    let local_nnz = local_pattern.len();

    // Index local entries by equation once. Scanning the whole pointwise
    // pattern for every equation makes preparation O(component_count * nnz).
    let mut local_row_offsets = vec![0; component_count + 1];
    for &(row, _) in &local_pattern {
        local_row_offsets[row + 1] += 1;
    }
    for row in 0..component_count {
        local_row_offsets[row + 1] += local_row_offsets[row];
    }
    let mut local_row_slots = vec![0; local_nnz];
    let mut next_local_row_slot = local_row_offsets[..component_count].to_vec();
    for (slot, &(row, _)) in local_pattern.iter().enumerate() {
        local_row_slots[next_local_row_slot[row]] = slot;
        next_local_row_slot[row] += 1;
    }

    let mut full_to_unknown = vec![None; full_size];
    for (unknown, &full_position) in packed.full_state_positions.iter().enumerate() {
        full_to_unknown[full_position] = Some(unknown);
    }

    // Per interval there are at most two endpoint terms per equation and two
    // trapezoid contributions per local Jacobian entry.
    let pending_capacity = component_count
        .checked_mul(2)
        .and_then(|endpoint_terms| {
            local_nnz
                .checked_mul(2)
                .and_then(|jacobian_terms| endpoint_terms.checked_add(jacobian_terms))
        })
        .and_then(|entries_per_interval| steps.checked_mul(entries_per_interval))
        .ok_or_else(|| CanteraSolverError::InvalidConfiguration {
            message: "trapezoid Jacobian contribution count overflows usize".into(),
        })?;
    let mut global_coordinates = BTreeSet::new();
    let mut pending = Vec::with_capacity(pending_capacity);
    for interval in 0..steps {
        for equation in 0..component_count {
            let global_row = interval * component_count + equation;
            for (node, scale) in [(interval, -1.0), (interval + 1, 1.0)] {
                let full_column = node * component_count + equation;
                if let Some(global_column) = full_to_unknown[full_column] {
                    global_coordinates.insert((global_row, global_column));
                    pending.push((global_row, global_column, None, 0, scale));
                }
            }
            let row_slots =
                &local_row_slots[local_row_offsets[equation]..local_row_offsets[equation + 1]];
            for &local_slot in row_slots {
                let (_, local_column) = local_pattern[local_slot];
                for node in [interval, interval + 1] {
                    let full_column = node * component_count + local_column;
                    if let Some(global_column) = full_to_unknown[full_column] {
                        global_coordinates.insert((global_row, global_column));
                        pending.push((global_row, global_column, Some(node), local_slot, -0.5 * h));
                    }
                }
            }
        }
    }

    let coordinates = global_coordinates.into_iter().collect::<Vec<_>>();
    let pattern = CanteraJacobianPattern::try_new(packed.values.len(), coordinates.clone())
        .map_err(|source| CanteraSolverError::InvalidConfiguration {
            message: format!("failed to construct trapezoid Jacobian pattern: {source}"),
        })?;
    let pattern_nnz = pattern.nnz();
    let coordinate_slots = coordinates
        .iter()
        .enumerate()
        .map(|(slot, &coordinate)| (coordinate, slot))
        .collect::<BTreeMap<_, _>>();
    let contributions = pending
        .into_iter()
        .map(
            |(row, column, node, local_jacobian_slot, scale)| Contribution {
                output_slot: coordinate_slots[&(row, column)],
                node,
                local_jacobian_slot,
                scale,
            },
        )
        .collect::<Vec<_>>();

    let initial_state = packed.values;
    let state_bounds = problem
        .state_names
        .iter()
        .map(|name| options.bounds.get(name).copied().unwrap_or((-1e300, 1e300)))
        .collect::<Vec<_>>();
    let relative_tolerances = problem
        .state_names
        .iter()
        .map(|name| {
            options
                .relative_tolerances
                .get(name)
                .copied()
                .unwrap_or(options.default_relative_tolerance)
        })
        .collect::<Vec<_>>();
    let bounds = packed
        .full_state_positions
        .iter()
        .map(|&position| state_bounds[position % component_count])
        .collect::<Vec<_>>();

    let residual_plan = pointwise.clone();
    let left_endpoint = problem.interval.0;
    let residual_positions = packed.full_state_positions.clone();
    let residual_fixed = packed.fixed_state_values.clone();
    let mut residual_full_state = vec![0.0; full_size];
    let mut residual_rhs = vec![0.0; full_size];
    let mut residual_arguments = vec![0.0; 1 + component_count + external_parameter_names.len()];
    let residual =
        move |unknowns: &DVector<f64>, external_parameters: &[f64], output: &mut DVector<f64>| {
            residual_full_state.fill(0.0);
            for (unknown, &full_position) in residual_positions.iter().enumerate() {
                residual_full_state[full_position] = unknowns[unknown];
            }
            for &(position, value) in &residual_fixed {
                residual_full_state[position] = value;
            }
            residual_rhs.fill(0.0);
            for node in 0..node_count {
                let state_start = node * component_count;
                let output_start = state_start;
                residual_plan
                    .evaluate_rhs(
                        left_endpoint + node as f64 * h,
                        &residual_full_state[state_start..state_start + component_count],
                        external_parameters,
                        &mut residual_arguments,
                        &mut residual_rhs[output_start..output_start + component_count],
                    )
                    .map_err(CanteraCallbackFailure::Symbolic)?;
            }

            if output.len() != component_count * steps {
                return Err(CanteraCallbackFailure::External(format!(
                    "global residual output has {} entries, expected {}",
                    output.len(),
                    component_count * steps
                )));
            }
            for interval in 0..steps {
                for equation in 0..component_count {
                    let row = interval * component_count + equation;
                    let left = interval * component_count + equation;
                    let right = (interval + 1) * component_count + equation;
                    output[row] = residual_full_state[right]
                        - residual_full_state[left]
                        - 0.5 * h * (residual_rhs[left] + residual_rhs[right]);
                }
            }
            Ok(())
        };

    let jacobian_plan = pointwise;
    let left_endpoint = problem.interval.0;
    let jacobian_positions = packed.full_state_positions.clone();
    let jacobian_fixed = packed.fixed_state_values.clone();
    let mut jacobian_full_state = vec![0.0; full_size];
    let mut jacobian_pointwise_values = vec![0.0; node_count * local_nnz];
    let mut jacobian_arguments = vec![0.0; 1 + component_count + external_parameter_names.len()];
    let mut jacobian_dense_scratch = if jacobian_plan.is_aot() {
        vec![0.0; component_count * component_count]
    } else {
        Vec::new()
    };
    let mut jacobian_aot_scratch = if jacobian_plan.is_aot() {
        vec![
            0.0;
            jacobian_plan
                .jacobian_callback_scratch_len()
                .map_err(|source| CanteraSolverError::SymbolicPreparation { source })?
        ]
    } else {
        Vec::new()
    };
    let jacobian = move |unknowns: &DVector<f64>,
                         external_parameters: &[f64],
                         output: &mut [f64]| {
        if output.len() != pattern_nnz {
            return Err(CanteraCallbackFailure::External(format!(
                "global Jacobian output has {} values, expected {}",
                output.len(),
                pattern_nnz
            )));
        }
        jacobian_full_state.fill(0.0);
        for (unknown, &full_position) in jacobian_positions.iter().enumerate() {
            jacobian_full_state[full_position] = unknowns[unknown];
        }
        for &(position, value) in &jacobian_fixed {
            jacobian_full_state[position] = value;
        }
        jacobian_pointwise_values.fill(0.0);
        for node in 0..node_count {
            let state_start = node * component_count;
            let value_start = node * local_nnz;
            let state = &jacobian_full_state[state_start..state_start + component_count];
            let values = &mut jacobian_pointwise_values[value_start..value_start + local_nnz];
            if jacobian_plan.is_aot() {
                jacobian_dense_scratch.fill(0.0);
                jacobian_plan
                    .evaluate_jacobian_dense_with_scratch(
                        left_endpoint + node as f64 * h,
                        state,
                        external_parameters,
                        &mut jacobian_arguments,
                        &mut jacobian_dense_scratch,
                        &mut jacobian_aot_scratch,
                    )
                    .map_err(CanteraCallbackFailure::Symbolic)?;
                for (slot, &(row, column)) in local_pattern.iter().enumerate() {
                    values[slot] = jacobian_dense_scratch[row * component_count + column];
                }
            } else {
                jacobian_plan
                    .evaluate_jacobian_values(
                        left_endpoint + node as f64 * h,
                        state,
                        external_parameters,
                        &mut jacobian_arguments,
                        values,
                    )
                    .map_err(CanteraCallbackFailure::Symbolic)?;
            }
        }
        output.fill(0.0);
        for contribution in &contributions {
            let value = match contribution.node {
                Some(node) => {
                    jacobian_pointwise_values[node * local_nnz + contribution.local_jacobian_slot]
                }
                None => 1.0,
            };
            output[contribution.output_slot] += contribution.scale * value;
        }
        Ok(())
    };

    let config = CanteraNewtonConfig {
        matrix_layout: options.matrix_layout,
        component_count,
        absolute_tolerances: DVector::from_element(component_count, options.absolute_tolerance),
        relative_tolerances: DVector::from_vec(relative_tolerances),
        bounds,
        full_state_positions: packed.full_state_positions,
        fixed_state_values: packed.fixed_state_values,
        max_iterations: options.max_iterations,
        max_jacobian_age: options.max_jacobian_age,
        max_damping_trials: options.max_damping_trials,
        logging_level: options.logging_level,
        telemetry_mode: options.telemetry_mode,
    };
    let mut solver = CanteraNewtonSolver::new_with_parameterized_buffered_callbacks(
        initial_state,
        residual,
        pattern,
        jacobian,
        external_parameter_names,
        config,
    )?;
    solver.symbolic_mesh = Some(
        (0..=steps)
            .map(|node| {
                if node == steps {
                    problem.interval.1
                } else {
                    problem.interval.0 + node as f64 * h
                }
            })
            .collect(),
    );
    Ok(solver)
}

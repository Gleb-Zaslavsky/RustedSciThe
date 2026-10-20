pub mod banded;
pub mod bdf;
pub mod be;
pub mod bvp_damp;
pub mod bvp_sci;
pub mod lsode2;
pub mod nonlinear;
pub mod nonlinear_matrix;
pub mod optimization;
pub mod radau;

#[cfg(test)]
mod tests {
    use crate::registry::SCENARIOS;
    use std::collections::BTreeSet;

    #[test]
    fn registered_scenarios_pass_their_debug_correctness_gates() {
        for scenario in SCENARIOS.iter() {
            if scenario.execution.starts_with("AOT") {
                // AOT requires a compiler and is exercised by explicit route
                // invocations, not by every debug unit-test run.
                continue;
            }
            let output = (scenario.run)()
                .unwrap_or_else(|error| panic!("scenario {} failed: {error}", scenario.id));
            assert!(
                !output.metrics.is_empty(),
                "scenario {} returned no metrics",
                scenario.id
            );
            for gate in scenario.gate_metrics {
                assert!(
                    output.metrics.iter().any(|(id, _, _)| id == gate),
                    "scenario {} did not emit declared gate metric {gate}",
                    scenario.id
                );
            }
        }
    }

    #[test]
    fn representative_solver_families_complete_separate_gate_and_diagnostic_phases() {
        let mut checked = BTreeSet::new();
        for scenario in SCENARIOS.iter().filter(|scenario| {
            scenario.diagnose.is_some() && !scenario.execution.starts_with("AOT")
        }) {
            if !checked.insert(scenario.solver) {
                continue;
            }
            let series = crate::measurement::measure(scenario, 0, 3).unwrap_or_else(|error| {
                panic!("scenario {} measurement failed: {error}", scenario.id)
            });
            assert!(
                series
                    .metrics
                    .iter()
                    .any(|metric| metric.id == "scenario_wall_ms" && metric.phase == "gate_off"),
                "scenario {} has no Off-mode wall gate",
                scenario.id
            );
            assert!(
                series.metrics.iter().all(|metric| {
                    metric.phase == "gate_off" || metric.phase == "instrumented_diagnostic"
                }),
                "scenario {} has an unknown measurement phase",
                scenario.id
            );
            if scenario.solver == "radau" {
                let factor = series
                    .metrics
                    .iter()
                    .find(|metric| metric.id == "stage_factorization_ms")
                    .expect("Radau detailed route should report factorization timing");
                assert_eq!(factor.scope, "native_child_factorization_ms");
                assert_eq!(factor.parent_scope.as_deref(), Some("linear_ms"));
            }
            if scenario.solver == "bvp_sci" {
                let factor = series
                    .metrics
                    .iter()
                    .find(|metric| metric.id == "factorization_ms")
                    .expect("BVP_sci detailed route should report factorization timing");
                assert!(factor.scope.starts_with("native_"));
                assert!(factor.scope.ends_with("_Factorization"));
            }
            if scenario.solver == "bvp_damp" {
                let solve_scope = series
                    .metrics
                    .iter()
                    .find(|metric| metric.id == "native_scope_solve_ms")
                    .expect("BVP_Damp should expose its typed solve scope");
                assert_eq!(solve_scope.scope, "native_typed_aggregate_Solve");
                assert_eq!(solve_scope.parent_scope, None);
            }
        }
        assert!(
            checked.len() >= 8,
            "expected representative coverage for all solver families"
        );
    }
}

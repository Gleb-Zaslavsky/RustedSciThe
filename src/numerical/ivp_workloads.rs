//! Solver-agnostic symbolic IVP workloads shared by solver tests and benches.
//!
//! This module contains problem definitions and deterministic continuation
//! targets only. Solver policy, telemetry, filesystem access, and reporting
//! belong in solver-specific adapters. BDF, BE, LSODE2, and Radau can therefore
//! use the same equations and initial conditions without fixture drift.

use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

/// Workload families used by solver correctness and performance corpora.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorkloadKind {
    DiffusionChain,
    CombustionLike,
    StiffScalar,
    Robertson,
    ThreeBody,
}

impl WorkloadKind {
    pub const ALL: [Self; 5] = [
        Self::DiffusionChain,
        Self::CombustionLike,
        Self::StiffScalar,
        Self::Robertson,
        Self::ThreeBody,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::DiffusionChain => "diffusion-chain",
            Self::CombustionLike => "combustion-like",
            Self::StiffScalar => "stiff-scalar",
            Self::Robertson => "robertson",
            Self::ThreeBody => "three-body",
        }
    }
}

/// Owned symbolic problem data without solver-specific configuration.
#[derive(Clone, Debug)]
pub struct SymbolicWorkload {
    pub kind: WorkloadKind,
    pub equations: Vec<Expr>,
    pub variables: Vec<String>,
    pub time_variable: String,
    pub initial_state: DVector<f64>,
    pub parameter_names: Vec<String>,
    pub parameter_values: DVector<f64>,
}

/// Fully coupled dense symbolic workload used for scaling studies.
///
/// Kept separate from `WorkloadKind` because it is a size-controlled stress
/// fixture rather than a canonical physical benchmark family.
#[derive(Clone, Debug)]
pub struct DenseCoupledWorkload {
    pub equations: Vec<Expr>,
    pub variables: Vec<String>,
    pub time_variable: String,
    pub initial_state: DVector<f64>,
    pub parameter_names: Vec<String>,
    pub parameter_values: DVector<f64>,
}

/// Returns the deterministic numeric target used by continuation stories and benches.
/// The first parameter is swept upward; remaining nonzero parameters move down.
pub fn parameter_continuation_target(base: &DVector<f64>, step: usize) -> DVector<f64> {
    let step = step as f64;
    DVector::from_iterator(
        base.len(),
        base.iter().enumerate().map(|(index, value)| {
            if *value == 0.0 {
                0.0
            } else if index == 0 {
                *value * (1.0 + 0.00025 * step)
            } else {
                *value * (1.0 - 0.00010 * step)
            }
        }),
    )
}

impl SymbolicWorkload {
    pub fn is_parameterized(&self) -> bool {
        !self.parameter_names.is_empty()
    }
}

/// Builds one workload. `dimension` is meaningful for the diffusion chain;
/// fixed-size workloads ignore it and retain their canonical state dimension.
pub fn build_workload(kind: WorkloadKind, dimension: usize) -> SymbolicWorkload {
    match kind {
        WorkloadKind::DiffusionChain => diffusion_chain(dimension),
        WorkloadKind::CombustionLike => combustion_like(),
        WorkloadKind::StiffScalar => stiff_scalar(),
        WorkloadKind::Robertson => robertson(),
        WorkloadKind::ThreeBody => three_body(),
    }
}

pub fn diffusion_chain(dimension: usize) -> SymbolicWorkload {
    assert!(dimension > 0, "diffusion chain dimension must be positive");
    let equations = (0..dimension)
        .map(|index| {
            let left = (index > 0)
                .then(|| format!("y{}", index - 1))
                .unwrap_or_else(|| "0".to_string());
            let right = (index + 1 < dimension)
                .then(|| format!("y{}", index + 1))
                .unwrap_or_else(|| "0".to_string());
            Expr::parse_expression(&format!(
                "-k*y{index} + d*({left} - 2*y{index} + {right}) + q*exp(-t) - nl*y{index}*y{index}"
            ))
        })
        .collect();

    SymbolicWorkload {
        kind: WorkloadKind::DiffusionChain,
        equations,
        variables: (0..dimension).map(|index| format!("y{index}")).collect(),
        time_variable: "t".to_string(),
        initial_state: DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| 0.2 + 0.01 * (index % 11) as f64),
        ),
        parameter_names: vec!["k".into(), "d".into(), "q".into(), "nl".into()],
        parameter_values: DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]),
    }
}

/// Builds a stable, fully coupled dense system with one nonzero Jacobian entry
/// for every state-equation pair. Its purpose is frontend/linear-algebra
/// scaling, not physical modelling.
pub fn dense_coupled(dimension: usize) -> DenseCoupledWorkload {
    assert!(dimension > 0, "dense coupled dimension must be positive");
    let variables: Vec<_> = (0..dimension).map(|index| format!("y{index}")).collect();
    let coupled_sum = variables.join(" + ");
    let equations = variables
        .iter()
        .map(|variable| {
            Expr::parse_expression(&format!("-k*{variable} + c*({coupled_sum}) + q*exp(-t)"))
        })
        .collect();

    DenseCoupledWorkload {
        equations,
        variables,
        time_variable: "t".to_string(),
        initial_state: DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| 0.2 + 0.01 * (index % 11) as f64),
        ),
        parameter_names: vec!["k".into(), "c".into(), "q".into()],
        parameter_values: DVector::from_vec(vec![20.0, 0.25 / dimension as f64, 0.20]),
    }
}

pub fn combustion_like() -> SymbolicWorkload {
    SymbolicWorkload {
        kind: WorkloadKind::CombustionLike,
        equations: vec![
            Expr::parse_expression("-k*exp(-E/(R*T))*A*A"),
            Expr::parse_expression("0.5*k*exp(-E/(R*T))*A*A - kloss*B"),
            Expr::parse_expression("Qcrho*k*exp(-E/(R*T))*A*A - cooling*(T - T0)"),
        ],
        variables: vec!["A".into(), "B".into(), "T".into()],
        time_variable: "t".into(),
        initial_state: DVector::from_vec(vec![1.0, 0.0, 300.0]),
        parameter_names: vec![
            "k".into(),
            "E".into(),
            "R".into(),
            "T0".into(),
            "Qcrho".into(),
            "kloss".into(),
            "cooling".into(),
        ],
        parameter_values: DVector::from_vec(vec![1.0e7, 5.0e4, 8.314, 300.0, 5.0e2, 0.0, 0.5]),
    }
}

pub fn stiff_scalar() -> SymbolicWorkload {
    SymbolicWorkload {
        kind: WorkloadKind::StiffScalar,
        equations: vec![Expr::parse_expression("-1000*(y-cos(t))-sin(t)")],
        variables: vec!["y".into()],
        time_variable: "t".into(),
        initial_state: DVector::from_vec(vec![1.0]),
        parameter_names: Vec::new(),
        parameter_values: DVector::zeros(0),
    }
}

pub fn robertson() -> SymbolicWorkload {
    SymbolicWorkload {
        kind: WorkloadKind::Robertson,
        equations: vec![
            Expr::parse_expression("-0.04*y0 + 10000*y1*y2"),
            Expr::parse_expression("0.04*y0 - 10000*y1*y2 - 30000000*y1*y1"),
            Expr::parse_expression("30000000*y1*y1"),
        ],
        variables: vec!["y0".into(), "y1".into(), "y2".into()],
        time_variable: "t".into(),
        initial_state: DVector::from_vec(vec![1.0, 0.0, 0.0]),
        parameter_names: Vec::new(),
        parameter_values: DVector::zeros(0),
    }
}

pub fn three_body() -> SymbolicWorkload {
    let r01 = Expr::parse_expression("((x0 - x1)^2 + (y0 - y1)^2)^0.5");
    let r02 = Expr::parse_expression("((x0 - x2)^2 + (y0 - y2)^2)^0.5");
    let r12 = Expr::parse_expression("((x1 - x2)^2 + (y1 - y2)^2)^0.5");

    let eq_vx0 = Expr::parse_expression("-k * (m1*(x0 - x1)/R01^3 + m2*(x0 - x2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02);
    let eq_vy0 = Expr::parse_expression("-k * (m1*(y0 - y1)/R01^3 + m2*(y0 - y2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02);
    let eq_vx1 = Expr::parse_expression("-k * (m0*(x1 - x0)/R01^3 + m2*(x1 - x2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12);
    let eq_vy1 = Expr::parse_expression("-k * (m0*(y1 - y0)/R01^3 + m2*(y1 - y2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12);
    let eq_vx2 = Expr::parse_expression("-k * (m0*(x2 - x0)/R02^3 + m1*(x2 - x1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12);
    let eq_vy2 = Expr::parse_expression("-k * (m0*(y2 - y0)/R02^3 + m1*(y2 - y1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12);

    SymbolicWorkload {
        kind: WorkloadKind::ThreeBody,
        equations: vec![
            Expr::parse_expression("vx0"),
            eq_vx0,
            Expr::parse_expression("vy0"),
            eq_vy0,
            Expr::parse_expression("vx1"),
            eq_vx1,
            Expr::parse_expression("vy1"),
            eq_vy1,
            Expr::parse_expression("vx2"),
            eq_vx2,
            Expr::parse_expression("vy2"),
            eq_vy2,
        ],
        variables: vec![
            "x0".into(),
            "vx0".into(),
            "y0".into(),
            "vy0".into(),
            "x1".into(),
            "vx1".into(),
            "y1".into(),
            "vy1".into(),
            "x2".into(),
            "vx2".into(),
            "y2".into(),
            "vy2".into(),
        ],
        time_variable: "t".into(),
        initial_state: DVector::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, -0.4, 33.0, 0.0,
        ]),
        parameter_names: vec!["k".into(), "m0".into(), "m1".into(), "m2".into()],
        parameter_values: DVector::from_vec(vec![39.47841760435743, 1.0, 0.5, 0.75]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_has_expected_shapes_and_parameter_metadata() {
        let diffusion = diffusion_chain(7);
        assert_eq!(diffusion.equations.len(), 7);
        assert_eq!(diffusion.initial_state.len(), 7);
        assert!(diffusion.is_parameterized());

        let compatibility = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(7);
        assert_eq!(compatibility.equations.len(), diffusion.equations.len());
        assert_eq!(compatibility.parameter_names, diffusion.parameter_names);

        for workload in [combustion_like(), stiff_scalar(), robertson(), three_body()] {
            assert_eq!(workload.equations.len(), workload.variables.len());
            assert_eq!(workload.initial_state.len(), workload.variables.len());
            assert_eq!(
                workload.parameter_names.len(),
                workload.parameter_values.len()
            );
        }

        let dense = dense_coupled(32);
        assert_eq!(dense.equations.len(), 32);
        assert_eq!(dense.variables.len(), 32);
        assert_eq!(dense.initial_state.len(), 32);
        assert_eq!(dense.parameter_names.len(), dense.parameter_values.len());
    }

    #[test]
    fn every_corpus_workload_has_a_stable_label() {
        let labels: Vec<_> = WorkloadKind::ALL.iter().map(|kind| kind.label()).collect();
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().all(|label| !label.is_empty()));
    }

    #[test]
    fn continuation_targets_use_one_shared_parameter_path() {
        let base = DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010, 0.0]);
        let target = parameter_continuation_target(&base, 64);
        assert_eq!(target[0], 20.32);
        assert_eq!(target[1], 3.9744);
        assert_eq!(target[2], 0.19872);
        assert_eq!(target[3], 0.009936);
        assert_eq!(target[4], 0.0);
    }
}

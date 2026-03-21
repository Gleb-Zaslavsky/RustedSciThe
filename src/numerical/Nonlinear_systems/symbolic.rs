use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector};

/// Adapter that exposes symbolic equations through the generic problem traits.
pub struct SymbolicNonlinearProblem {
    /// Prepared symbolic Jacobian backend.
    jacobian: Jacobian,
    /// Original symbolic equations.
    equations: Vec<Expr>,
    /// Ordered list of variables.
    variables: Vec<String>,
    /// Optional symbolic parameter names.
    equation_parameters: Option<Vec<String>>,
    /// Optional parameter values used during evaluation.
    equation_parameter_values: Option<DVector<f64>>,
}

impl SymbolicNonlinearProblem {
    /// Builds a symbolic problem from parsed expressions.
    pub fn from_expressions(
        equations: Vec<Expr>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
    ) -> Result<Self, SolveError> {
        if equations.is_empty() {
            return Err(SolveError::InvalidConfig(
                "equation system must not be empty".to_string(),
            ));
        }

        let variables = match variables {
            Some(variables) => variables,
            None => {
                let mut args = equations
                    .iter()
                    .flat_map(|expr| expr.all_arguments_are_variables())
                    .collect::<Vec<_>>();
                args.sort();
                args.dedup();
                args
            }
        };

        if variables.is_empty() {
            return Err(SolveError::InvalidConfig(
                "failed to infer variables from symbolic equations".to_string(),
            ));
        }
        if variables.len() != equations.len() {
            return Err(SolveError::DimensionMismatch {
                expected: equations.len(),
                actual: variables.len(),
                context: "symbolic variables vs equations",
            });
        }

        let mut jacobian = Jacobian::new();
        let variable_refs = variables
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        jacobian.set_vector_of_functions(equations.clone());
        jacobian.set_variables(variable_refs);

        if let Some(parameters) = &equation_parameters {
            jacobian.set_params(parameters.clone());
            jacobian.calc_jacobian();
            jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();
            jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();
        } else {
            jacobian.calc_jacobian();
            jacobian.lambdify_jacobian_DMatrix_parallel();
            jacobian.lambdify_vector_funvector_DVector();
        }

        Ok(Self {
            jacobian,
            equations,
            variables,
            equation_parameters,
            equation_parameter_values,
        })
    }

    /// Builds a symbolic problem from equation strings.
    pub fn from_strings(
        equations: Vec<String>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
    ) -> Result<Self, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions(
            expressions,
            variables,
            equation_parameters,
            equation_parameter_values,
        )
    }

    /// Returns the variable names used by the symbolic system.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Returns the original symbolic equations.
    pub fn equations(&self) -> &[Expr] {
        &self.equations
    }

    /// Updates parameter values for parameterized equations.
    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), SolveError> {
        if let Some(parameters) = &self.equation_parameters {
            if parameters.len() != values.len() {
                return Err(SolveError::DimensionMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                    context: "symbolic equation parameters",
                });
            }
        }
        self.equation_parameter_values = Some(values);
        Ok(())
    }

    /// Shared implementation of residual evaluation.
    fn residual_impl(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        if x.len() != self.variables.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.variables.len(),
                actual: x.len(),
                context: "symbolic residual input",
            });
        }

        let result = if let Some(parameter_values) = &self.equation_parameter_values {
            if let Some(parameters) = &self.equation_parameters {
                if parameters.len() != parameter_values.len() {
                    return Err(SolveError::DimensionMismatch {
                        expected: parameters.len(),
                        actual: parameter_values.len(),
                        context: "symbolic parameter values",
                    });
                }
            }
            let residual = &self.jacobian.lambdified_function_with_params;
            residual(parameter_values, x)
        } else {
            let residual = &self.jacobian.lambdified_function_DVector;
            residual(x)
        };

        if result.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::ResidualEvaluation(
                "symbolic residual returned NaN or Inf".to_string(),
            ));
        }
        Ok(result)
    }

    /// Shared implementation of Jacobian evaluation.
    fn jacobian_impl(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        if x.len() != self.variables.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.variables.len(),
                actual: x.len(),
                context: "symbolic jacobian input",
            });
        }

        let result = if let Some(parameter_values) = &self.equation_parameter_values {
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix_with_params;
            jacobian(parameter_values, x)
        } else {
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix;
            jacobian(x)
        };

        if result.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::JacobianEvaluation(
                "symbolic jacobian returned NaN or Inf".to_string(),
            ));
        }
        Ok(result)
    }
}

impl NonlinearProblem for SymbolicNonlinearProblem {
    /// Returns the number of unknowns in the symbolic system.
    fn dimension(&self) -> usize {
        self.variables.len()
    }

    /// Evaluates the symbolic residual vector.
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.residual_impl(x)
    }
}

impl JacobianProvider for SymbolicNonlinearProblem {
    /// Evaluates the symbolic Jacobian matrix.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.jacobian_impl(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::LM_Nielsen::NielsenLevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::LM_vanilla::LevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::problem::Bounds;
    use crate::numerical::Nonlinear_systems::trust_region::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;
    use std::f64;

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("symbolic problem should build")
    }

    fn solve_elementary_with_bounds<M>(method: M) -> DVector<f64>
    where
        M: crate::numerical::Nonlinear_systems::engine::NonlinearMethod,
    {
        let options = SolveOptions {
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            ..SolveOptions::default()
        };
        SolverEngine::new(method, options)
            .solve(&elementary_problem(), DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve should succeed")
            .x
    }

    fn chemistry_problem(dgm: f64) -> (SymbolicNonlinearProblem, Vec<String>) {
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");
        let d_g0 = Expr::Const(-450.0e3);
        let d_g1 = Expr::Const(-150.0e3);
        let d_g2 = Expr::Const(-50e3);
        let d_gm = Expr::Const(dgm);
        let n0 = symbolic[0].clone();
        let n1 = symbolic[1].clone();
        let n2 = symbolic[2].clone();
        let np = symbolic[3].clone();
        let lambda0 = symbolic[4].clone();
        let lambda1 = symbolic[5].clone();
        let rt = Expr::Const(8.314) * Expr::Const(273.15);

        let eq_mu = vec![
            lambda0.clone()
                + Expr::Const(2.0) * lambda1.clone()
                + (d_g0.clone() + rt.clone() * Expr::ln(n0.clone() / np.clone())) / d_gm.clone(),
            lambda0
                + lambda1.clone()
                + (d_g1 + rt.clone() * Expr::ln(n1.clone() / np.clone())) / d_gm.clone(),
            Expr::Const(2.0) * lambda1 + (d_g2 + rt * Expr::ln(n2.clone() / np.clone())) / d_gm,
        ];
        let eq_sum = vec![n0.clone() + n1.clone() + n2.clone() - np.clone()];
        let eq_comp = vec![
            n0.clone() + n1.clone() - Expr::Const(0.999),
            Expr::Const(2.0) * n0 + n1 + Expr::Const(2.0) * n2 - Expr::Const(1.501),
        ];

        let mut system = Vec::new();
        system.extend(eq_mu);
        system.extend(eq_sum);
        system.extend(eq_comp);
        let variables = symbolic.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        (
            SymbolicNonlinearProblem::from_expressions(system, Some(variables.clone()), None, None)
                .expect("chemistry problem should build"),
            variables,
        )
    }

    fn assert_chemistry_constraints(solution: &DVector<f64>, variables: &[String], tol: f64) {
        let map = variables
            .iter()
            .zip(solution.iter())
            .map(|(k, v)| (k.clone(), *v))
            .collect::<HashMap<String, f64>>();
        let n0 = map["N0"];
        let n1 = map["N1"];
        let n2 = map["N2"];
        let np = map["Np"];
        assert!((n0 + n1 - 0.999).abs() < tol);
        assert!((n0 + n1 + n2 - np).abs() < tol);
        assert!((2.0 * n0 + n1 + 2.0 * n2 - 1.501).abs() < tol);
    }

    #[test]
    fn symbolic_problem_solves_from_expressions() {
        let problem = elementary_problem();
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn symbolic_problem_solves_from_strings() {
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&elementary_problem(), DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn various_nonlinear_equations_simple() {
        let problem = SymbolicNonlinearProblem::from_strings(
            vec!["x+y-100".to_string(), "1/x - 1/y - 1/200".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("problem should build");
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        let x = -50.0 * (f64::sqrt(17.0) - 5.0);
        let y = 50.0 * (f64::sqrt(17.0) - 3.0);
        assert_relative_eq!(result.x[0], x, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], y, epsilon = 1e-3);
    }

    #[test]
    fn symbolic_damped_method_respects_bounds() {
        let solution = solve_elementary_with_bounds(DampedNewtonMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn symbolic_lm_method_solves_elementary_problem() {
        let solution = solve_elementary_with_bounds(LevenbergMarquardtMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn symbolic_trust_region_method_solves_elementary_problem() {
        let solution = solve_elementary_with_bounds(TrustRegionMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn chemistry_problem_solves_with_lm() {
        let (problem, variables) = chemistry_problem(8.314 * 450e4);
        let bounds = Bounds::new(vec![
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 10.0),
            (-1e-1, 1e-2),
            (-1e-1, 1e-2),
        ])
        .expect("bounds");
        let options = SolveOptions {
            tolerance: 2.0e-3,
            max_iterations: 130,
            bounds: Some(bounds),
            ..SolveOptions::default()
        };
        let method = LevenbergMarquardtMethod {
            increase_factor: 11.0,
            decrease_factor: 9.0,
            ..LevenbergMarquardtMethod::default()
        };
        let result = SolverEngine::new(method, options)
            .solve(
                &problem,
                DVector::from_vec(vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0]),
            )
            .expect("solve should succeed");
        assert_chemistry_constraints(&result.x, &variables, 8e-3);
    }

    #[test]
    fn chemistry_problem_solves_with_nielsen_lm() {
        let (problem, variables) = chemistry_problem(8.314 * 60e5);
        let bounds = Bounds::new(vec![
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 10.0),
            (-10000.0, 1e6),
            (-100000.0, 1e6),
        ])
        .expect("bounds");
        let options = SolveOptions {
            tolerance: 2.0e-3,
            max_iterations: 130,
            bounds: Some(bounds),
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NielsenLevenbergMarquardtMethod::default(), options)
            .solve(
                &problem,
                DVector::from_vec(vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0]),
            )
            .expect("solve should succeed");
        assert_chemistry_constraints(&result.x, &variables, 8e-3);
    }
}

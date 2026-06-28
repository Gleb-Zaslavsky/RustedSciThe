use crate::numerical::optimization::universal_fitting::{
    Method, UniversalFitting, UniversalFittingResult,
};
use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashMap;
use thiserror::Error;

/// A small container for a named kinetic fitting model.
#[derive(Clone, Debug, PartialEq)]
pub struct FittingModel {
    /// Pre-exponential or rate constant.
    pub k: f64,
    /// Additional model parameters for multi-parameter expressions.
    pub params: HashMap<String, f64>,
}

/// Named kinetic models supported by the convenience API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FittingModelName {
    /// Autocatalytic model.
    Autocatalytic,
    /// First-order kinetic model.
    FirstOrder,
    /// Second-order kinetic model.
    SecondOrder,
    /// Third-order kinetic model.
    ThirdOrder,
    /// Sestak-Berggren model.
    SestakBerggren,
    /// Johnson-Mehl-Avrami model.
    JohnsonMehlAvrami,
    /// Acceleratory model.
    Acceleratory,
    /// Deceleratory model.
    Deceleratory,
    /// Truncated Sestak-Berggren model.
    SestacBergrenTrunc,
    /// Parameterized Sestak-Berggren model.
    SestacBergrenParam,
    /// Single exponential decay.
    DecExp,
    /// Sum of two exponentials.
    TwoExp,
    /// Sum of three exponentials.
    ThreeExp,
    /// Linear model.
    Linear,
    /// Polynomial model of degree `n`.
    Polynom { n: usize },
}

impl FittingModelName {
    /// Return the symbolic equation for this kinetic model.
    pub fn equation(&self) -> Expr {
        match self {
            Self::Autocatalytic => Expr::parse_expression("k*x*(1-x)"),
            Self::FirstOrder => Expr::parse_expression("k*(1-x)"),
            Self::SecondOrder => Expr::parse_expression("k*(1-x)^2"),
            Self::ThirdOrder => Expr::parse_expression("k*(1-x)^3"),
            Self::SestakBerggren => Expr::parse_expression("k*x^n*(1-x)^m*(-ln(1-x))^p"),
            Self::JohnsonMehlAvrami => Expr::parse_expression("m*(1-x)*(-ln(1-x))^(1-1/m)"),
            Self::Acceleratory => Expr::parse_expression("m*(1-x)^(1-1/m)"),
            Self::Deceleratory => Expr::parse_expression("(1-x)^m"),
            Self::SestacBergrenTrunc => Expr::parse_expression("x^m*(1-x)^n"),
            Self::SestacBergrenParam => Expr::parse_expression("c*x^m*(1-x)^n"),
            Self::DecExp => Expr::parse_expression("c*exp(-k*x)"),
            Self::TwoExp => Expr::parse_expression("a*exp(-p*x) + b*exp(-k*x)"),
            Self::ThreeExp => Expr::parse_expression("a*exp(-p*x) + b*exp(-k*x) + c*exp(-r*x)"),
            Self::Linear => Expr::parse_expression("k*x + b"),
            Self::Polynom { n } => Expr::polyval(*n, "x").0,
        }
    }

    /// Return the ordered parameter names for this model.
    pub fn vec_of_params(&self) -> Vec<String> {
        match self {
            Self::Autocatalytic => vec!["k".to_string()],
            Self::FirstOrder => vec!["k".to_string()],
            Self::SecondOrder => vec!["k".to_string()],
            Self::ThirdOrder => vec!["k".to_string()],
            Self::SestakBerggren => vec![
                "k".to_string(),
                "n".to_string(),
                "m".to_string(),
                "p".to_string(),
            ],
            Self::JohnsonMehlAvrami => vec!["m".to_string()],
            Self::Acceleratory => vec!["m".to_string()],
            Self::Deceleratory => vec!["m".to_string()],
            Self::SestacBergrenTrunc => vec!["m".to_string(), "n".to_string()],
            Self::SestacBergrenParam => {
                vec!["m".to_string(), "n".to_string(), "c".to_string()]
            }
            Self::DecExp => vec!["c".to_string(), "k".to_string()],
            Self::TwoExp => vec![
                "a".to_string(),
                "p".to_string(),
                "b".to_string(),
                "k".to_string(),
            ],
            Self::ThreeExp => vec![
                "a".to_string(),
                "p".to_string(),
                "b".to_string(),
                "k".to_string(),
                "c".to_string(),
                "r".to_string(),
            ],
            Self::Linear => vec!["k".to_string(), "b".to_string()],
            Self::Polynom { n } => Expr::polyval(*n, "x").1,
        }
    }

    /// Return a simple initial guess vector.
    pub fn vec_of_initial_guess(&self) -> Vec<f64> {
        match self {
            Self::Autocatalytic => vec![0.5],
            Self::FirstOrder => vec![0.5],
            Self::SecondOrder => vec![0.5],
            Self::ThirdOrder => vec![0.5],
            Self::SestakBerggren => vec![1.0, 1.0, 1.0, 1.0],
            Self::JohnsonMehlAvrami => vec![1.0],
            Self::Acceleratory => vec![1.0],
            Self::Deceleratory => vec![1.0],
            Self::SestacBergrenTrunc => vec![1.0, 1.0],
            Self::SestacBergrenParam => vec![1.0, 1.0, 1.0],
            Self::DecExp => vec![1.0, 0.5],
            Self::TwoExp => vec![1.0, 0.5, 1.0, 0.25],
            Self::ThreeExp => vec![1.0, 0.5, 1.0, 0.25, 1.0, 0.1],
            Self::Linear => vec![0.5, 0.5],
            Self::Polynom { n } => vec![1.0; Expr::polyval(*n, "x").1.len()],
        }
    }

    /// Return the nonlinear parameter names used by the VarPro backend.
    ///
    /// Only separable exponential models are handled by VarPro here, because
    /// the amplitudes are linear coefficients and the decay rates are the true
    /// nonlinear unknowns.
    pub fn varpro_unknowns(&self) -> Option<Vec<String>> {
        match self {
            Self::DecExp => Some(vec!["k".to_string()]),
            Self::TwoExp => Some(vec!["p".to_string(), "k".to_string()]),
            Self::ThreeExp => Some(vec!["p".to_string(), "k".to_string(), "r".to_string()]),
            _ => None,
        }
    }

    /// Return the separable basis functions used by the VarPro backend.
    ///
    /// The basis order matches the order of the linear coefficients in the
    /// final solution map.
    pub fn varpro_basis_strs(&self) -> Option<Vec<(&'static str, &'static str)>> {
        match self {
            Self::DecExp => Some(vec![("k", "exp(-k*x)")]),
            Self::TwoExp => Some(vec![("p", "exp(-p*x)"), ("k", "exp(-k*x)")]),
            Self::ThreeExp => Some(vec![
                ("p", "exp(-p*x)"),
                ("k", "exp(-k*x)"),
                ("r", "exp(-r*x)"),
            ]),
            _ => None,
        }
    }

    /// Return the linear coefficient names used by the VarPro backend.
    ///
    /// These are the amplitudes that get solved internally by projection and
    /// later renamed back to the public kinetic-model names.
    pub fn varpro_linear_names(&self) -> Option<Vec<String>> {
        match self {
            Self::DecExp => Some(vec!["c".to_string()]),
            Self::TwoExp => Some(vec!["a".to_string(), "b".to_string()]),
            Self::ThreeExp => Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            _ => None,
        }
    }

    /// Return the default nonlinear initial guess for the VarPro backend.
    pub fn varpro_initial_guess(&self) -> Option<Vec<f64>> {
        match self {
            Self::DecExp => Some(vec![0.5]),
            Self::TwoExp => Some(vec![0.5, 0.25]),
            Self::ThreeExp => Some(vec![0.5, 0.25, 0.1]),
            _ => None,
        }
    }

    /// Return the symbolic independent variable name used by these models.
    pub fn arg_name(&self) -> &'static str {
        "x"
    }

    /// Return the preferred numerical backend for this model.
    ///
    /// The separable exponential models benefit from VarPro, while the rest of
    /// the kinetic library stays on the classic symbolic LM path.
    pub fn preferred_method(&self) -> Method {
        match self {
            Self::DecExp | Self::TwoExp | Self::ThreeExp => Method::VARPRO,
            _ => Method::LM,
        }
    }

    /// Return `true` if the model should be treated as a simple LM fit.
    pub fn uses_classic_lm(&self) -> bool {
        matches!(self.preferred_method(), Method::LM)
    }
}

/// Result of a kinetic fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fit {
    /// Selected model.
    pub kinetic_model: FittingModelName,
    /// Independent variable samples.
    pub x_data: Vec<f64>,
    /// Observed data samples.
    pub y_data: Vec<f64>,
    /// Optional initial guess.
    pub initial_guess: Option<Vec<f64>>,
    /// Target tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Fitted parameter map.
    pub map_of_solutions: Option<HashMap<String, f64>>,
    /// Coefficient of determination.
    pub r2: f64,
}

/// Errors returned by the kinetic fitting convenience layer.
#[derive(Debug, Error)]
pub enum KineticFittingError {
    /// The x data is empty.
    #[error("x data cannot be empty")]
    XDataEmpty,
    /// The y data is empty.
    #[error("y data cannot be empty")]
    YDataEmpty,
    /// The initial guess is missing.
    #[error("initial guess cannot be empty")]
    InitialGuessMissing,
    /// The solver backend failed.
    #[error("fit failed: {0}")]
    FitFailed(String),
    /// The solver result was missing a map of solutions.
    #[error("fit completed but did not produce a solution map")]
    MissingSolutionMap,
}

impl Fit {
    /// Create a new kinetic fitting builder for a selected model.
    pub fn new(kinetic_model: FittingModelName) -> Self {
        Self {
            kinetic_model,
            x_data: Vec::new(),
            y_data: Vec::new(),
            initial_guess: None,
            tolerance: 1e-6,
            max_iter: 300,
            map_of_solutions: None,
            r2: 0.0,
        }
    }

    /// Set the x and y data together.
    pub fn with_data(mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        self.x_data = x_data;
        self.y_data = y_data;
        self
    }

    /// Set the initial guess for the parameters.
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.initial_guess = Some(initial_guess);
        self
    }

    /// Set the numerical tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Run the selected kinetic fit and store the result in this builder.
    pub fn fit(mut self) -> Result<Self, KineticFittingError> {
        self.validate()?;

        let initial_guess = self.effective_initial_guess();
        let method = self.kinetic_model.preferred_method();

        let universal = match method {
            Method::LM => UniversalFitting::new()
                .with_method(Method::LM)
                .with_data(self.x_data.clone(), self.y_data.clone())
                .with_equation(self.kinetic_model.equation())
                .with_arg(self.kinetic_model.arg_name().to_string())
                .with_unknowns(self.kinetic_model.vec_of_params())
                .with_initial_guess(initial_guess)
                .with_tolerance(self.tolerance)
                .with_max_iterations(self.max_iter)
                .build(),
            Method::VARPRO => {
                let mut universal = UniversalFitting::new()
                    .with_method(Method::VARPRO)
                    .with_data(self.x_data.clone(), self.y_data.clone())
                    .with_arg(self.kinetic_model.arg_name().to_string())
                    .with_parameters(
                        self.kinetic_model
                            .varpro_unknowns()
                            .expect("VarPro model should provide nonlinear unknowns"),
                    )
                    .with_initial_guess(initial_guess);

                for (parameter_name, basis) in self
                    .kinetic_model
                    .varpro_basis_strs()
                    .expect("VarPro model should provide basis functions")
                {
                    universal = universal.with_basis_str(parameter_name, basis);
                }

                universal.build()
            }
        }
        .map_err(|err| KineticFittingError::FitFailed(err.to_string()))?;

        match universal {
            UniversalFittingResult::LM(fit) => {
                self.map_of_solutions = fit.solution_map();
                self.r2 = fit.r_squared().unwrap_or(0.0);
                if self.map_of_solutions.is_none() {
                    return Err(KineticFittingError::MissingSolutionMap);
                }
                Ok(self)
            }
            UniversalFittingResult::VARPRO(fit) => {
                let map = fit
                    .solution_map()
                    .ok_or(KineticFittingError::MissingSolutionMap)?;
                self.map_of_solutions = Some(self.remap_varpro_solution(map));
                self.r2 = fit.r_squared().unwrap_or(0.0);
                Ok(self)
            }
        }
    }

    /// Store the fitted coefficients as a name-to-value map.
    pub fn with_solution_map(mut self, map: HashMap<String, f64>) -> Self {
        self.map_of_solutions = Some(map);
        self
    }

    /// Return the fitted coefficients as a name-to-value map.
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        self.map_of_solutions.clone()
    }

    /// Short alias for [`solution_map`](Self::solution_map).
    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.solution_map()
    }

    /// Store the coefficient of determination of the last fit.
    pub fn with_r_squared(mut self, r2: f64) -> Self {
        self.r2 = r2;
        self
    }

    /// Return the coefficient of determination of the last fit.
    pub fn r_squared(&self) -> f64 {
        self.r2
    }

    /// Short alias for [`r_squared`](Self::r_squared).
    pub fn r2(&self) -> f64 {
        self.r_squared()
    }

    /// Evaluate the fitted model at the stored x values.
    pub fn evaluate_solution(&self) -> Result<Vec<f64>, KineticFittingError> {
        let params = self
            .solution_map()
            .ok_or(KineticFittingError::MissingSolutionMap)?;
        let formula = self.kinetic_model.equation();
        let formula_with_params = formula.set_variable_from_map(&params);
        let closure = formula_with_params.lambdify1D();

        Ok(self.x_data.iter().map(|x_i| closure(*x_i)).collect())
    }

    fn validate(&self) -> Result<(), KineticFittingError> {
        if self.x_data.is_empty() {
            return Err(KineticFittingError::XDataEmpty);
        }
        if self.y_data.is_empty() {
            return Err(KineticFittingError::YDataEmpty);
        }
        let expected_guess_len = match self.kinetic_model.preferred_method() {
            Method::LM => self.kinetic_model.vec_of_params().len(),
            Method::VARPRO => self
                .kinetic_model
                .varpro_unknowns()
                .map(|unknowns| unknowns.len())
                .unwrap_or_else(|| self.kinetic_model.vec_of_params().len()),
        };
        let guess_len = self.effective_initial_guess().len();
        if guess_len != expected_guess_len {
            return Err(KineticFittingError::FitFailed(
                "initial guess length does not match parameter count".to_string(),
            ));
        }
        Ok(())
    }

    /// Return the actual initial guess used by the solver.
    ///
    /// If the caller does not provide an explicit guess, the model-specific
    /// default vector is used instead.
    fn effective_initial_guess(&self) -> Vec<f64> {
        match self.kinetic_model.preferred_method() {
            Method::LM => self
                .initial_guess
                .clone()
                .unwrap_or_else(|| self.kinetic_model.vec_of_initial_guess()),
            Method::VARPRO => self
                .initial_guess
                .clone()
                .map(|guess| self.varpro_initial_guess_from_any_guess(&guess))
                .or_else(|| self.kinetic_model.varpro_initial_guess())
                .unwrap_or_else(|| self.kinetic_model.vec_of_initial_guess()),
        }
    }

    /// Extract the nonlinear seed values from any accepted VarPro guess shape.
    fn varpro_initial_guess_from_any_guess(&self, guess: &[f64]) -> Vec<f64> {
        if let Some(varpro_unknowns) = self.kinetic_model.varpro_unknowns() {
            if guess.len() == varpro_unknowns.len() {
                return guess.to_vec();
            }
        }

        match self.kinetic_model {
            FittingModelName::DecExp if guess.len() >= 2 => vec![guess[1]],
            FittingModelName::TwoExp if guess.len() >= 4 => vec![guess[1], guess[3]],
            FittingModelName::ThreeExp if guess.len() >= 6 => vec![guess[1], guess[3], guess[5]],
            _ => guess.to_vec(),
        }
    }

    /// Rename VarPro coefficient keys so the public API stays model-centric.
    fn remap_varpro_solution(&self, mut map: HashMap<String, f64>) -> HashMap<String, f64> {
        if let Some(linear_names) = self.kinetic_model.varpro_linear_names() {
            for (idx, name) in linear_names.into_iter().enumerate() {
                if let Some(value) = map.remove(&format!("c{idx}")) {
                    map.insert(name, value);
                }
            }
        }
        map
    }
}

impl Default for FittingModelName {
    fn default() -> Self {
        Self::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn linspace(start: f64, end: f64, len: usize) -> Vec<f64> {
        let step = if len > 1 {
            (end - start) / (len - 1) as f64
        } else {
            0.0
        };
        (0..len).map(|idx| start + idx as f64 * step).collect()
    }

    fn fit_case(
        model: FittingModelName,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        initial_guess: Vec<f64>,
    ) -> Fit {
        Fit::new(model)
            .with_data(x_data, y_data)
            .with_initial_guess(initial_guess)
            .fit()
            .expect("kinetic model should fit")
    }

    fn assert_map_value(map: &HashMap<String, f64>, key: &str, expected: f64, eps: f64) {
        assert!(
            map.contains_key(key),
            "solution map should contain key {key}"
        );
        assert_relative_eq!(map[key], expected, epsilon = eps);
    }

    fn assert_expression_values(
        model: FittingModelName,
        x_data: &[f64],
        params: &[(&str, f64)],
        expected: &[f64],
    ) {
        let param_map: HashMap<String, f64> = params
            .iter()
            .map(|(name, value)| ((*name).to_string(), *value))
            .collect();
        let expr = model.equation().set_variable_from_map(&param_map);
        let eval = expr.lambdify1D();
        let actual: Vec<f64> = x_data.iter().map(|&x| eval(x)).collect();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }
    }

    macro_rules! kinetic_test {
        ($name:ident, $model:expr, $x:expr, $y:expr, $guess:expr, $( $key:expr => $expected:expr ),+ $(,)?) => {
            #[test]
            fn $name() {
                let fit = fit_case($model, $x, $y, $guess);
                let map = fit.solution_map().expect("fit should expose a solution map");
                $(assert_map_value(&map, $key, $expected, 1e-6);)+
                assert!(fit.r_squared() > 0.999);
                let y_pred = fit.evaluate_solution().expect("fit should evaluate");
                assert_eq!(y_pred.len(), fit.x_data.len());
            }
        };
    }

    kinetic_test!(
        autocatalytic_fits,
        FittingModelName::Autocatalytic,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| 2.0 * v * (1.0 - v)).collect::<Vec<_>>()
        },
        vec![1.5],
        "k" => 2.0
    );

    kinetic_test!(
        first_order_fits,
        FittingModelName::FirstOrder,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| 1.8 * (1.0 - v)).collect::<Vec<_>>()
        },
        vec![1.0],
        "k" => 1.8
    );

    kinetic_test!(
        second_order_fits,
        FittingModelName::SecondOrder,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| 1.4 * (1.0 - v).powi(2)).collect::<Vec<_>>()
        },
        vec![1.0],
        "k" => 1.4
    );

    kinetic_test!(
        third_order_fits,
        FittingModelName::ThirdOrder,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| 1.2 * (1.0 - v).powi(3)).collect::<Vec<_>>()
        },
        vec![1.0],
        "k" => 1.2
    );

    #[test]
    fn sestak_berggren_smoke_test() {
        let model = FittingModelName::SestakBerggren;
        let expr = model.equation();
        assert!(!expr.to_string().is_empty());
        assert_eq!(model.vec_of_params(), vec!["k", "n", "m", "p"]);
        assert_eq!(model.vec_of_initial_guess().len(), 4);
    }

    #[test]
    fn jma_smoke_test() {
        let model = FittingModelName::JohnsonMehlAvrami;
        let expr = model.equation();
        assert!(!expr.to_string().is_empty());
        assert_eq!(model.vec_of_params(), vec!["m"]);
        assert_eq!(model.vec_of_initial_guess().len(), 1);
    }

    #[test]
    fn exponential_models_prefer_varpro_backend() {
        assert_eq!(FittingModelName::DecExp.preferred_method(), Method::VARPRO);
        assert_eq!(FittingModelName::TwoExp.preferred_method(), Method::VARPRO);
        assert_eq!(
            FittingModelName::ThreeExp.preferred_method(),
            Method::VARPRO
        );
        assert!(FittingModelName::Linear.uses_classic_lm());
        assert!(FittingModelName::FirstOrder.uses_classic_lm());
    }

    #[test]
    fn dec_exp_uses_default_initial_guess_when_not_provided() {
        let x = linspace(0.0, 5.0, 50);
        let y = x
            .iter()
            .map(|&v| 2.0 * (-0.6 * v).exp())
            .collect::<Vec<_>>();

        let fit = Fit::new(FittingModelName::DecExp)
            .with_data(x, y)
            .fit()
            .expect("default initial guess should be enough for the VarPro path");

        let map = fit
            .solution_map()
            .expect("fit should expose a solution map");
        assert_relative_eq!(map["c"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["k"], 0.6, epsilon = 1e-6);
        assert!(fit.r_squared() > 0.999_999);
    }

    #[test]
    fn sestak_berggren_expression_parameters_are_ordered() {
        assert_eq!(
            FittingModelName::SestakBerggren.vec_of_params(),
            vec![
                "k".to_string(),
                "n".to_string(),
                "m".to_string(),
                "p".to_string()
            ]
        );
    }

    #[test]
    fn jma_expression_parameters_are_ordered() {
        assert_eq!(
            FittingModelName::JohnsonMehlAvrami.vec_of_params(),
            vec!["m".to_string()]
        );
    }

    kinetic_test!(
        acceleratory_fits,
        FittingModelName::Acceleratory,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| 1.6 * (1.0 - v).powf(1.0 - 1.0 / 1.6)).collect::<Vec<_>>()
        },
        vec![1.0],
        "m" => 1.6
    );

    kinetic_test!(
        deceleratory_fits,
        FittingModelName::Deceleratory,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| (1.0 - v).powf(1.4)).collect::<Vec<_>>()
        },
        vec![1.0],
        "m" => 1.4
    );

    kinetic_test!(
        sestac_bergren_trunc_fits,
        FittingModelName::SestacBergrenTrunc,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter().map(|&v| v.powf(1.1) * (1.0 - v).powf(0.8)).collect::<Vec<_>>()
        },
        vec![1.0, 1.0],
        "m" => 1.1,
        "n" => 0.8
    );

    kinetic_test!(
        sestac_bergren_param_fits,
        FittingModelName::SestacBergrenParam,
        linspace(0.05, 0.85, 50),
        {
            let x = linspace(0.05, 0.85, 50);
            x.iter()
                .map(|&v| 1.5 * v.powf(0.9) * (1.0 - v).powf(0.7))
                .collect::<Vec<_>>()
        },
        vec![1.0, 1.0, 1.0],
        "m" => 0.9,
        "n" => 0.7,
        "c" => 1.5
    );

    kinetic_test!(
        dec_exp_fits,
        FittingModelName::DecExp,
        linspace(0.0, 5.0, 50),
        {
            let x = linspace(0.0, 5.0, 50);
            x.iter().map(|&v| 2.0 * (-0.6 * v).exp()).collect::<Vec<_>>()
        },
        vec![1.5, 0.3],
        "c" => 2.0,
        "k" => 0.6
    );

    kinetic_test!(
        two_exp_fits,
        FittingModelName::TwoExp,
        linspace(0.0, 5.0, 50),
        {
            let x = linspace(0.0, 5.0, 50);
            x.iter()
                .map(|&v| 1.1 * (-0.8 * v).exp() + 0.7 * (-0.25 * v).exp())
                .collect::<Vec<_>>()
        },
        vec![1.0, 0.5, 1.0, 0.2],
        "a" => 1.1,
        "p" => 0.8,
        "b" => 0.7,
        "k" => 0.25
    );

    kinetic_test!(
        three_exp_fits,
        FittingModelName::ThreeExp,
        linspace(0.0, 5.0, 60),
        {
            let x = linspace(0.0, 5.0, 60);
            x.iter()
                .map(|&v| 1.0 * (-0.9 * v).exp() + 0.8 * (-0.3 * v).exp() + 0.5 * (-0.1 * v).exp())
                .collect::<Vec<_>>()
        },
        vec![1.0, 0.5, 1.0, 0.2, 0.4, 0.1],
        "a" => 1.0,
        "p" => 0.9,
        "b" => 0.8,
        "k" => 0.3,
        "c" => 0.5,
        "r" => 0.1
    );

    kinetic_test!(
        linear_fits,
        FittingModelName::Linear,
        linspace(0.0, 10.0, 50),
        {
            let x = linspace(0.0, 10.0, 50);
            x.iter().map(|&v| 2.5 * v + 1.25).collect::<Vec<_>>()
        },
        vec![1.0, 1.0],
        "k" => 2.5,
        "b" => 1.25
    );

    kinetic_test!(
        polynomial_fits,
        FittingModelName::Polynom { n: 3 },
        linspace(0.0, 3.0, 50),
        {
            let x = linspace(0.0, 3.0, 50);
            x.iter()
                .map(|&v| 5.0 * v.powi(3) + 2.0 * v.powi(2) + 3.0 * v + 1.0)
                .collect::<Vec<_>>()
        },
        vec![1.0; 4],
        "c3" => 5.0,
        "c2" => 2.0,
        "c1" => 3.0,
        "c0" => 1.0
    );
}

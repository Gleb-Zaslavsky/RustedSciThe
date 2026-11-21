//! Shared preparation-time substitution helpers for symbolic solver inputs.

use std::error::Error;
use std::fmt;

use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symexpr_graphs::{Definition, SymbolicSystem, SymbolicSystemError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicSubstitutionError {
    EmptyName,
    ReservedName(String),
    NonSymbolicCallbackRoute,
    System(SymbolicSystemError),
}

impl fmt::Display for SymbolicSubstitutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => f.write_str("substitution names must not be empty"),
            Self::ReservedName(name) => write!(
                f,
                "substitution name `{name}` conflicts with a declared variable, time, or parameter"
            ),
            Self::NonSymbolicCallbackRoute => f.write_str(
                "symbolic substitutions cannot rewrite user-provided residual/Jacobian callbacks",
            ),
            Self::System(error) => write!(f, "invalid substitution definitions: {error}"),
        }
    }
}

impl Error for SymbolicSubstitutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::System(error) => Some(error),
            Self::EmptyName | Self::ReservedName(_) | Self::NonSymbolicCallbackRoute => None,
        }
    }
}

/// Expands named aliases in-place in symbolic equations before solver setup.
/// Unknown names in alias expressions remain ordinary runtime symbols.
pub fn apply_symbolic_substitutions<I, N>(
    expressions: &mut [Expr],
    definitions: I,
    reserved_names: &[String],
) -> Result<(), SymbolicSubstitutionError>
where
    I: IntoIterator<Item = (N, Expr)>,
    N: Into<String>,
{
    let definitions = definitions
        .into_iter()
        .map(|(name, expression)| Definition::new(name, expression))
        .collect::<Vec<_>>();
    if definitions.is_empty() {
        return Ok(());
    }

    for definition in &definitions {
        if definition.name.is_empty() {
            return Err(SymbolicSubstitutionError::EmptyName);
        }
        if reserved_names.iter().any(|name| name == &definition.name) {
            return Err(SymbolicSubstitutionError::ReservedName(
                definition.name.clone(),
            ));
        }
    }

    let system = SymbolicSystem::new(definitions).map_err(SymbolicSubstitutionError::System)?;
    let expanded = system
        .expand_all()
        .map_err(SymbolicSubstitutionError::System)?;
    for (name, replacement) in expanded {
        for expression in expressions.iter_mut() {
            *expression = expression.substitute_variable(&name, &replacement);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_alias_chains_but_leaves_runtime_symbols_free() {
        let mut equations = vec![Expr::parse_expression("outer*y")];
        apply_symbolic_substitutions(
            &mut equations,
            [
                ("outer", Expr::parse_expression("inner + rate")),
                ("inner", Expr::parse_expression("2*x")),
            ],
            &["y".into(), "t".into()],
        )
        .unwrap();

        let variables = equations[0].all_arguments_are_variables();
        assert!(variables.iter().any(|name| name == "x"));
        assert!(variables.iter().any(|name| name == "rate"));
        assert!(
            !variables
                .iter()
                .any(|name| name == "outer" || name == "inner")
        );
    }

    #[test]
    fn reports_duplicate_cycle_and_reserved_alias_names() {
        let mut equations = vec![Expr::parse_expression("alias*y")];
        let duplicate = apply_symbolic_substitutions(
            &mut equations,
            [
                ("alias", Expr::parse_expression("x")),
                ("alias", Expr::parse_expression("z")),
            ],
            &[],
        );
        assert!(matches!(
            duplicate,
            Err(SymbolicSubstitutionError::System(
                SymbolicSystemError::DuplicateDefinition(_)
            ))
        ));

        let cycle = apply_symbolic_substitutions(
            &mut equations,
            [
                ("alias", Expr::parse_expression("other")),
                ("other", Expr::parse_expression("alias")),
            ],
            &[],
        );
        assert!(matches!(
            cycle,
            Err(SymbolicSubstitutionError::System(
                SymbolicSystemError::CyclicDependency(_)
            ))
        ));

        assert!(matches!(
            apply_symbolic_substitutions(
                &mut equations,
                [("y", Expr::parse_expression("x"))],
                &["y".into()],
            ),
            Err(SymbolicSubstitutionError::ReservedName(name)) if name == "y"
        ));
    }
}

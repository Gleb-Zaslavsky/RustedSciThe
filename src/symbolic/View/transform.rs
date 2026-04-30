//! Structural variable renaming and constant substitution for packed atoms.
//!
//! This is the basic building block needed to move BVP discretization to the
//! View pipeline earlier than Jacobian generation.

use std::collections::HashMap;

use super::{
    Atom, AtomView, atom::FunctionBuilder, conversions::approximate_f64_atom, state::Symbol,
};

/// Rename variable symbols in a packed atom.
pub fn rename_symbols(atom: &Atom, rename_map: &HashMap<Symbol, Symbol>) -> Atom {
    rename_symbols_view(atom.as_view(), rename_map)
}

/// Substitute selected variable symbols by numeric constants.
pub fn substitute_symbol_values(atom: &Atom, value_map: &HashMap<Symbol, f64>) -> Atom {
    substitute_symbol_values_view(atom.as_view(), value_map)
}

/// Rename variable symbols in a borrowed atom view.
pub fn rename_symbols_view(view: AtomView<'_>, rename_map: &HashMap<Symbol, Symbol>) -> Atom {
    match view {
        AtomView::Num(_) => view.to_owned(),
        AtomView::Var(v) => {
            let symbol = v.get_symbol();
            Atom::new_var(rename_map.get(&symbol).copied().unwrap_or(symbol))
        }
        AtomView::Fun(f) => {
            let mut builder = FunctionBuilder::new(f.get_symbol());
            for arg in f.iter() {
                builder = builder.add_arg(rename_symbols_view(arg, rename_map));
            }
            builder.finish()
        }
        AtomView::Pow(p) => {
            let (base, exp) = p.get_base_exp();
            let mut result = Atom::default();
            let renamed_base = rename_symbols_view(base, rename_map);
            let renamed_exp = rename_symbols_view(exp, rename_map);
            result.to_pow(renamed_base.as_view(), renamed_exp.as_view());
            result
        }
        AtomView::Mul(m) => {
            let mut result = Atom::default();
            let mul = result.to_mul();
            for arg in m.iter() {
                let renamed = rename_symbols_view(arg, rename_map);
                mul.extend(renamed.as_view());
            }
            result
        }
        AtomView::Add(a) => {
            let mut result = Atom::default();
            let add = result.to_add();
            for arg in a.iter() {
                let renamed = rename_symbols_view(arg, rename_map);
                add.extend(renamed.as_view());
            }
            result
        }
    }
}

/// Substitute selected variable symbols by numeric constants in a borrowed view.
pub fn substitute_symbol_values_view(view: AtomView<'_>, value_map: &HashMap<Symbol, f64>) -> Atom {
    match view {
        AtomView::Num(_) => view.to_owned(),
        AtomView::Var(v) => {
            let symbol = v.get_symbol();
            if let Some(value) = value_map.get(&symbol) {
                approximate_f64_atom(*value)
            } else {
                Atom::new_var(symbol)
            }
        }
        AtomView::Fun(f) => {
            let mut builder = FunctionBuilder::new(f.get_symbol());
            for arg in f.iter() {
                builder = builder.add_arg(substitute_symbol_values_view(arg, value_map));
            }
            builder.finish()
        }
        AtomView::Pow(p) => {
            let (base, exp) = p.get_base_exp();
            let mut result = Atom::default();
            let substituted_base = substitute_symbol_values_view(base, value_map);
            let substituted_exp = substitute_symbol_values_view(exp, value_map);
            result.to_pow(substituted_base.as_view(), substituted_exp.as_view());
            result
        }
        AtomView::Mul(m) => {
            let mut result = Atom::default();
            let mul = result.to_mul();
            for arg in m.iter() {
                let substituted = substitute_symbol_values_view(arg, value_map);
                mul.extend(substituted.as_view());
            }
            result
        }
        AtomView::Add(a) => {
            let mut result = Atom::default();
            let add = result.to_add();
            for arg in a.iter() {
                let substituted = substitute_symbol_values_view(arg, value_map);
                add.extend(substituted.as_view());
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::symbolic::View::conversions::{atom_to_expr, expr_to_atom};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn rename_symbols_rewrites_variables_without_leaving_expr_path() {
        let expr = Expr::parse_expression("x + y^2");
        let atom = expr_to_atom(&expr);
        let mut rename = HashMap::default();
        rename.insert(
            Symbol::new(crate::wrap_symbol!("x")),
            Symbol::new(crate::wrap_symbol!("x_1")),
        );
        rename.insert(
            Symbol::new(crate::wrap_symbol!("y")),
            Symbol::new(crate::wrap_symbol!("y_1")),
        );

        let renamed = rename_symbols(&atom, &rename);
        let rendered = atom_to_expr(&renamed).to_string();
        assert!(rendered.contains("x_1"));
        assert!(rendered.contains("y_1"));
    }

    #[test]
    fn substitute_symbol_values_replaces_selected_symbols_with_constants() {
        let expr = Expr::parse_expression("x + t");
        let atom = expr_to_atom(&expr);
        let mut values = HashMap::default();
        values.insert(Symbol::new(crate::wrap_symbol!("t")), 2.0);

        let substituted = substitute_symbol_values(&atom, &values);
        let rendered = atom_to_expr(&substituted).to_string();
        assert!(rendered.contains("2"));
        assert!(!rendered.contains("t"));
    }
}

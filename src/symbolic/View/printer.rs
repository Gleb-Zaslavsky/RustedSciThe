//! Pretty-printer for packed expressions.
//!
//! Printing walks [`AtomView`] directly, so it does not need to allocate temporary
//! trees or convert into an owned AST. The implementation is deliberately simple:
//! precedence is encoded numerically and the printer emits the minimum parentheses
//! needed to preserve the normalized expression structure.

use super::{
    atom::{Atom, AtomView},
    coefficient::CoefficientView,
};

/// Formatting switches for the simplified printer.
#[derive(Clone, Debug)]
pub struct PrintOptions {
    /// Reserved compact-printing toggle.
    pub compact: bool,
    /// Hide all namespaces when printing variable and function names.
    pub hide_all_namespaces: bool,
    /// Hide one specific namespace when printing names.
    pub hide_namespace: Option<&'static str>,
    /// Print a stable fully-qualified representation intended for debugging.
    pub stable_debug: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            compact: false,
            hide_all_namespaces: true,
            hide_namespace: None,
            stable_debug: false,
        }
    }
}

/// Print an owned atom using default formatting options.
pub fn print(atom: &Atom) -> String {
    print_with_options(atom, &PrintOptions::default())
}

/// Print an owned atom with explicit formatting options.
pub fn print_with_options(atom: &Atom, options: &PrintOptions) -> String {
    let mut out = String::new();
    print_view_to_string(&atom.as_view(), options, &mut out);
    out
}

/// Print an owned atom in a stable fully-qualified debug format.
pub fn print_debug(atom: &Atom) -> String {
    print_with_options(
        atom,
        &PrintOptions {
            hide_all_namespaces: false,
            hide_namespace: None,
            stable_debug: true,
            ..PrintOptions::default()
        },
    )
}

/// Append a textual representation of a borrowed atom view to an existing buffer.
pub fn print_view_to_string(view: &AtomView<'_>, options: &PrintOptions, out: &mut String) {
    print_view_prec(view, options, out, 0);
}

/// Print a view with the precedence of the surrounding context.
fn print_view_prec(view: &AtomView<'_>, options: &PrintOptions, out: &mut String, parent_prec: u8) {
    match view {
        AtomView::Num(n) => match n.get_coeff_view() {
            CoefficientView::Natural(num, den) => {
                if den == 1 {
                    out.push_str(&num.to_string());
                } else {
                    out.push_str(&format!("{}/{}", num, den));
                }
            }
            CoefficientView::Large(_) => out.push_str("<large>"),
        },
        AtomView::Var(v) => out.push_str(&format_symbol(v.get_symbol(), options)),
        AtomView::Fun(f) => {
            out.push_str(&format_symbol(f.get_symbol(), options));
            out.push('(');
            for (i, arg) in f.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                print_view_prec(&arg, options, out, 0);
            }
            out.push(')');
        }
        AtomView::Pow(p) => {
            let need = parent_prec > 3;
            if need {
                out.push('(');
            }
            let (base, exp) = p.get_base_exp();
            print_view_prec(&base, options, out, 3);
            out.push('^');
            print_view_prec(&exp, options, out, 3);
            if need {
                out.push(')');
            }
        }
        AtomView::Mul(m) => {
            let need = parent_prec > 2;
            if need {
                out.push('(');
            }
            for (i, factor) in m.iter().enumerate() {
                if i > 0 {
                    out.push('*');
                }
                print_view_prec(&factor, options, out, 2);
            }
            if need {
                out.push(')');
            }
        }
        AtomView::Add(a) => {
            let need = parent_prec > 1;
            if need {
                out.push('(');
            }
            for (i, term) in a.iter().enumerate() {
                if i > 0 {
                    out.push('+');
                }
                print_view_prec(&term, options, out, 1);
            }
            if need {
                out.push(')');
            }
        }
    }
    let _ = options.compact;
}

fn format_symbol(symbol: super::state::Symbol, options: &PrintOptions) -> String {
    if options.stable_debug {
        return symbol.get_name().to_string();
    }

    if options.hide_all_namespaces {
        return symbol.get_stripped_name().to_string();
    }

    if let Some(namespace) = options.hide_namespace {
        if symbol.get_namespace() == namespace {
            return symbol.get_stripped_name().to_string();
        }
    }

    symbol.get_name().to_string()
}

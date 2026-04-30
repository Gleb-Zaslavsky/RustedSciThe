#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::symbolic::View::{
        Coefficient, ExactSymbolMap, FloatSymbolMap,
        atom::Atom,
        evaluate::{
            EvaluationSymbolFn, FunctionMap, evaluate_exact_with_symbols, evaluate_with_symbols,
            prepare_evaluator,
        },
        printer::{PrintOptions, print, print_debug, print_with_options},
    };
    use crate::{parse, symbol};
    use ahash::HashMap;
    #[test]
    fn print_parse_roundtrip_preserves_normalized_form() {
        let expr = parse!("(x+y)^2/(2*z)").unwrap();
        let printed = expr.to_string();
        let reparsed = parse!(&printed).unwrap();
        assert_eq!(reparsed, expr);
    }

    #[test]
    fn derivative_and_evaluation_match_simple_polynomial() {
        let x = symbol!("x");
        let expr = parse!("x^3+2*x").unwrap();
        let deriv = expr.derivative(x);
        assert_eq!(deriv, parse!("3*x^2+2").unwrap());

        let mut const_map = HashMap::default();
        const_map.insert(Atom::new_var(x), 5.0);
        let value = deriv.evaluate(&const_map, &FunctionMap::new()).unwrap();
        assert!((value - 77.0).abs() < 1e-10);
    }

    #[test]
    fn nested_custom_function_evaluation_works() {
        let x = symbol!("x");
        let f = symbol!("f");
        let g = symbol!("g");
        let expr = parse!("g(f(x)) + f(3)").unwrap();

        let mut const_map = HashMap::default();
        const_map.insert(Atom::new_var(x), 4.0);

        let mut fn_map = FunctionMap::new();
        fn_map.insert(
            f,
            Arc::new(|args: &[f64], _, _, _| Ok(args[0] * args[0] + 1.0)),
        );
        fn_map.insert(
            g,
            Arc::new(move |args: &[f64], const_map, fn_map, cache| {
                let f_eval = fn_map
                    .get(&f)
                    .ok_or_else(|| "Missing function f".to_string())?;
                let inner = f_eval(&[args[0] + 2.0], const_map, fn_map, cache)?;
                Ok(inner - 3.0)
            }),
        );

        let value = expr.evaluate(&const_map, &fn_map).unwrap();
        assert!((value - 369.0).abs() < 1e-10);
    }

    #[test]
    fn builtin_constants_can_be_overridden_by_exact_atom_bindings() {
        let pi_atom = parse!("pi").unwrap();
        let expr = parse!("pi+1").unwrap();
        let mut const_map = HashMap::default();
        const_map.insert(pi_atom, 3.0);
        let value = expr.evaluate(&const_map, &FunctionMap::new()).unwrap();
        assert!((value - 4.0).abs() < 1e-10);
    }

    #[test]
    fn symbol_macro_supports_multiple_symbols() {
        let (x, y, f) = symbol!("x", "y", "f");
        let expr = parse!("f(x,y)").unwrap();
        assert_eq!(x.get_stripped_name(), "x");
        assert_eq!(y.get_stripped_name(), "y");
        assert_eq!(f.get_stripped_name(), "f");
        assert_eq!(expr.to_string(), "f(x,y)");
    }

    #[test]
    fn printer_preserves_precedence_for_add_mul_pow() {
        let expr = parse!("(x+y)*(z+1)^2").unwrap();
        assert_eq!(print(&expr), "(x+y)*(z+1)^2");
    }

    #[test]
    fn compact_print_option_matches_default_for_current_printer() {
        let expr = parse!("sin(x)+1/2*y").unwrap();
        let default = print(&expr);
        let compact = print_with_options(
            &expr,
            &PrintOptions {
                compact: true,
                ..PrintOptions::default()
            },
        );
        assert_eq!(default, compact);
        assert_eq!(compact, "y*1/2+sin(x)");
    }

    #[test]
    fn printer_can_show_namespaces() {
        let expr = parse!("symbolica::sin(mycrate::x)").unwrap();
        let printed = print_with_options(
            &expr,
            &PrintOptions {
                hide_all_namespaces: false,
                ..PrintOptions::default()
            },
        );
        assert_eq!(printed, "symbolica::sin(mycrate::x)");
    }

    #[test]
    fn printer_can_hide_one_namespace_selectively() {
        let expr = parse!("mycrate::f(mycrate::x,symbolica::sin(mycrate::y))").unwrap();
        let printed = print_with_options(
            &expr,
            &PrintOptions {
                hide_all_namespaces: false,
                hide_namespace: Some("mycrate"),
                ..PrintOptions::default()
            },
        );
        assert_eq!(printed, "f(x,symbolica::sin(y))");
    }

    #[test]
    fn stable_debug_print_is_fully_qualified() {
        let expr = parse!("symbolica::sin(mycrate::x)+mycrate::y").unwrap();
        assert_eq!(print_debug(&expr), "mycrate::y+symbolica::sin(mycrate::x)");
    }

    #[test]
    fn exact_symbol_map_evaluation_handles_normalized_rational_pipeline() {
        let (x, y) = symbol!("x", "y");
        let expr = parse!("(x+y)^2/2").unwrap();

        let mut const_map = ExactSymbolMap::default();
        const_map.insert(x, Coefficient::from((1, 2)));
        const_map.insert(y, Coefficient::from((3, 2)));

        let exact = evaluate_exact_with_symbols(&expr, &const_map).unwrap();
        assert_eq!(exact, Coefficient::from(2));
    }

    #[test]
    fn float_symbol_map_evaluation_handles_numeric_pipeline() {
        let (x, y) = symbol!("x", "y");
        let expr = parse!("(x+y)^2/2").unwrap();

        let mut const_map = FloatSymbolMap::default();
        const_map.insert(x, 0.5);
        const_map.insert(y, 1.5);

        let value = evaluate_with_symbols(&expr, &const_map, &FunctionMap::new()).unwrap();
        assert!((value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn symbol_keyed_custom_function_can_read_symbol_bindings() {
        let (x, y, shift, delta) = symbol!("x", "y", "shift", "delta");
        let expr = parse!("shift(x)+shift(y)").unwrap();

        let mut const_map = FloatSymbolMap::default();
        const_map.insert(x, 1.5);
        const_map.insert(y, -0.5);
        const_map.insert(delta, 2.0);
        let mut fn_map = FunctionMap::new();
        fn_map.insert_symbol_fn(shift, move |args, sym_map, _, _| {
            let offset = sym_map
                .get(&delta)
                .copied()
                .ok_or_else(|| "Missing delta binding".to_string())?;
            Ok(args[0] + offset)
        });

        let value = evaluate_with_symbols(&expr, &const_map, &fn_map).unwrap();
        assert!((value - 5.0).abs() < 1e-10);
    }

    #[test]
    fn nested_symbol_keyed_custom_functions_can_chain() {
        let (x, f, g, bias) = symbol!("x", "f", "g", "bias");
        let expr = parse!("g(f(x))").unwrap();

        let mut const_map = FloatSymbolMap::default();
        const_map.insert(x, 3.0);
        const_map.insert(bias, 4.0);

        let mut fn_map = FunctionMap::new();
        fn_map.insert_symbol_fn(f, |args, _, _, _| Ok(args[0] * 2.0));

        let g_eval: EvaluationSymbolFn = Arc::new(move |args, sym_map, fn_map, cache| {
            let f_eval = fn_map
                .get_symbol(&f)
                .ok_or_else(|| "Missing function f".to_string())?;
            let base = f_eval(&[args[0] + 1.0], sym_map, fn_map, cache)?;
            let extra = sym_map
                .get(&bias)
                .copied()
                .ok_or_else(|| "Missing bias binding".to_string())?;
            Ok(base + extra)
        });
        fn_map.insert_symbol(g, g_eval);

        let value = evaluate_with_symbols(&expr, &const_map, &fn_map).unwrap();
        assert!((value - 18.0).abs() < 1e-10);
    }

    #[test]
    fn prepared_evaluator_reuses_compiled_numeric_plan() {
        let (x, y) = symbol!("x", "y");
        let expr = parse!("(x+y)^2+sin(x)").unwrap();
        let prepared = prepare_evaluator(&expr, &[x, y], &FunctionMap::new()).unwrap();

        let v1 = prepared.evaluate(&[1.0, 2.0]).unwrap();
        let v2 = prepared.evaluate(&[2.0, 3.0]).unwrap();

        assert!((v1 - (9.0 + 1.0_f64.sin())).abs() < 1e-10);
        assert!((v2 - (25.0 + 2.0_f64.sin())).abs() < 1e-10);
    }

    #[test]
    fn prepared_evaluator_supports_symbol_keyed_custom_functions() {
        let (x, shift, delta) = symbol!("x", "shift", "delta");
        let expr = parse!("shift(x)^2").unwrap();

        let mut fn_map = FunctionMap::new();
        fn_map.insert_symbol_fn(shift, move |args, sym_map, _, _| {
            let delta_value = sym_map
                .get(&delta)
                .copied()
                .ok_or_else(|| "Missing delta binding".to_string())?;
            Ok(args[0] + delta_value)
        });

        let prepared = prepare_evaluator(&expr, &[x, delta], &fn_map).unwrap();
        let value = prepared.evaluate(&[3.0, 2.0]).unwrap();
        assert!((value - 25.0).abs() < 1e-10);
    }

    #[test]
    fn prepared_evaluator_supports_legacy_atom_keyed_custom_functions() {
        let (x, f) = symbol!("x", "f");
        let expr = parse!("f(x)+1").unwrap();

        let mut fn_map = FunctionMap::new();
        fn_map.insert_fn(f, move |args, atom_map, _, _| {
            let x_value = atom_map
                .get(&Atom::new_var(x))
                .copied()
                .ok_or_else(|| "Missing x binding".to_string())?;
            Ok(args[0] + x_value)
        });

        let prepared = prepare_evaluator(&expr, &[x], &fn_map).unwrap();
        let value = prepared.evaluate(&[4.0]).unwrap();
        assert!((value - 9.0).abs() < 1e-10);
    }
}

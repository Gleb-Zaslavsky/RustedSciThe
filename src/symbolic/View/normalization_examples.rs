mod normalization_examples {
    use crate::symbolic::View::atom::Atom;
    use crate::{function, parse, symbol};
    #[test]
    fn builder_example_normalizes_like_original() {
        let x = symbol!("x");
        let y = symbol!("y");
        let f = symbol!("f");
        let fun = function!(f, Atom::new_var(x), Atom::new_var(y), 2);
        let expr = (-(Atom::new_var(y) + Atom::new_var(x) + 2) * Atom::new_var(y) * 6).npow(5)
            / Atom::new_var(y)
            * fun
            / 4;
        let expected = parse!("1/4*y^-1*(-6*(x+y+2)*y)^5*f(x,y,2)").unwrap();
        assert_eq!(expr, expected);
    }

    #[test]
    fn parser_and_builder_match_on_normalized_expression() {
        let parsed = parse!("1/4*y^-1*(-6*(x+y+2)*y)^5*f(x,y,2)").unwrap();
        let rebuilt = parse!("(-(y+x+2)*y*6)^5/y*f(x,y,2)/4").unwrap();
        assert_eq!(rebuilt, parsed);
    }
}

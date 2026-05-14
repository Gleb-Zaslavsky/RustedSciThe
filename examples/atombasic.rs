use RustedSciThe::symbolic::View::atom::Atom;
use RustedSciThe::{function, parse, symbol};
fn main() {
    let x = symbol!("x");
    let y = symbol!("y");
    let f = symbol!("f");
    let fun = function!(f, Atom::new_var(x), Atom::new_var(y), 2);
    let expr = (-(Atom::new_var(y) + Atom::new_var(x) + 2) * Atom::new_var(y) * 6).npow(2)
        / Atom::new_var(y)
        * fun
        / 4;
    let parsed = parse!("1/4*y^-1*(-6*(x+y+2)*y)^2*f(x,y,2)").unwrap();

    println!("builder: {}", expr);
    println!("parsed : {}", parsed);
    println!("equal  : {}", expr == parsed);
}

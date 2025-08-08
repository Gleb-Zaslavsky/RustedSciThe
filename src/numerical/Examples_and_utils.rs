/// a collection of test examples of exect solutions of BVPs for testing purposes
use crate::Utils::plots::plots;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use strum_macros::EnumIter;
/// Plot a 1D graph from an expression using a given range and number of values.
pub fn plot_from_expr(
    input: String,
    x_name: &str,
    y_name: &str,
    start: f64,
    end: f64,
    num_values: usize,
) {
    let expr = Expr::parse_expression(&input);
    let y = Expr::lambdify1D_from_linspace(&expr, start, end, num_values);
    let y_DMatrix = DMatrix::from_row_slice(num_values, 1, &y);
    let step = (end - start) / (num_values - 1) as f64;
    let x = (0..num_values)
        .map(|i| start + i as f64 * step)
        .collect::<Vec<_>>();
    // println!("result = {:?} \n", y);
    plots(
        x_name.to_string(),
        vec![y_name.to_string()],
        DVector::from_vec(x),
        y_DMatrix,
    );
}

//EXAMPLES OF EXACT SOLUTION OF BVP OF NONLINEAR DIFFERENTIAL EQUATION
// https://math.stackexchange.com/questions/397461/examples-of-nonlinear-ordinary-differential-equations-with-elementary-solutions
/*
 the Lane-Emden equation of index 5:
y′′+2y′+y**5=0,y(0)=1,y′(0)=0
y'=z
z'=- 2*z/x - y**5
With initial conditions y(0)=1,y′(0)=0
exact solution:
y = (1+(x^2)/3)^(-0.5)


A very simple non-linear system to analyze is what I like to call the "Parachute Equation" which is essentially

y''+ ky'^2−g=0(1)
With initial conditions y(0)=0
 and y˙(0)=0
exact solution:
y= k*(log( (e2√g√kt+1)/2)−√g√kt)


Clairaut's equation.
y′′′=(x−1^2)2+y2+y′−2
y(1)=1, y′(1)=0, y′′(1)=2
exact solution:
yp(x)=1+(x−1)**2−1/6(x−1)**3+1/12(x−1)**4+...

two-point boundary value problem:
y''= -(2.0/a)*(1+2.0*ln(y))*y
y(-1) = exp(-1/a)
y(1) = exp(-1/a)
exact solution:
y(x)=exp(-x^2/a)
.
*/
#[derive(Debug, PartialEq, Eq, EnumIter)]
pub enum NonlinEquation {
    LaneEmden5,
    ParachuteEquation,
    Clairaut,
    TwoPointBVP,
}

const A: f64 = 1.0;
impl NonlinEquation {
    pub fn setup(&self) -> Vec<Expr> {
        match self {
            NonlinEquation::LaneEmden5 => {
                let _values = vec!["y".to_string(), "z".to_string()];
                let eqs = vec!["z", "-2*x*z - y^5"];
                let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
                vec_eqs
            }
            NonlinEquation::ParachuteEquation => {
                let _values = vec!["y".to_string(), "z".to_string()];
                let eqs = vec!["z", "-x^2 +1"];//??? y not x
                let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
                vec_eqs
            }
            NonlinEquation::Clairaut => {
                let _values = vec!["y".to_string(), "z".to_string(), "zz".to_string()];
                let eqs = vec!["z", "zz", "(x-1)^2 + y^2 + z-2"]; // (x−1)2+y2+y′−2
                let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
                vec_eqs
            }
            NonlinEquation::TwoPointBVP => {
                let _values = vec!["y".to_string(), "z".to_string()];
                let eqs = vec!["z", "-(2.0/1.0)*(1+ln( (y)^2.0 ))*y"];
                let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
                vec_eqs
            }
        } // end match
    } // end setup
    pub fn values(&self) -> Vec<String> {
        match self {
            NonlinEquation::LaneEmden5 => vec!["y".to_string(), "z".to_string()],
            NonlinEquation::ParachuteEquation => vec!["y".to_string(), "z".to_string()],
            NonlinEquation::Clairaut => vec!["y".to_string(), "z".to_string(), "zz".to_string()],
            NonlinEquation::TwoPointBVP => vec!["y".to_string(), "z".to_string()],
        }
    }
    pub fn boundary_conditions(&self) -> HashMap<String, (usize, f64)> {
        match self {
            NonlinEquation::LaneEmden5 => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("z".to_string(), (0usize, 0.0f64));
                BorderConditions.insert("y".to_string(), (0usize, 1.0f64));
                BorderConditions
            }
            NonlinEquation::ParachuteEquation => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("z".to_string(), (0usize, 0.0f64));
                BorderConditions.insert("y".to_string(), (0usize, 0.0f64));
                BorderConditions
            }
            NonlinEquation::Clairaut => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("zz".to_string(), (1usize, 2.0f64));
                BorderConditions.insert("z".to_string(), (1usize, 0.0f64));
                BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
                BorderConditions
            }
            NonlinEquation::TwoPointBVP => {
                let mut BorderConditions = HashMap::new();
                let z_at_1 = -2.0 * (-1.0 / A).exp();
                let y_at_min_1 = (-1.0 / A).exp();
                BorderConditions.insert("y".to_string(), (0usize, y_at_min_1));
                BorderConditions.insert("z".to_string(), (1usize, z_at_1));
                BorderConditions
            }
        }
    }
    pub fn boundary_conditions2(&self) -> HashMap<String, Vec<(usize, f64)>> {
        match self {
            NonlinEquation::LaneEmden5 => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("z".to_string(), vec![(0usize, 0.0f64)]);
                BorderConditions.insert("y".to_string(), vec![(0usize, 1.0f64)]);
                BorderConditions
            }
            NonlinEquation::ParachuteEquation => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("z".to_string(), vec![(0usize, 0.0f64)]);
                BorderConditions.insert("y".to_string(), vec![(0usize, 0.0f64)]);
                BorderConditions
            }
            NonlinEquation::Clairaut => {
                let mut BorderConditions = HashMap::new();
                BorderConditions.insert("zz".to_string(), vec![(1usize, 2.0f64)]);
                BorderConditions.insert("z".to_string(), vec![(1usize, 0.0f64)]);
                BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
                BorderConditions
            }
            NonlinEquation::TwoPointBVP => {
                let mut BorderConditions = HashMap::new();
                let z_at_1 = -2.0 * (-1.0 / A).exp();
                let y_at_min_1 = (-1.0 / A).exp();
                BorderConditions.insert("y".to_string(), vec![(0usize, y_at_min_1)]);
                BorderConditions.insert("z".to_string(), vec![(1usize, z_at_1)]);
                BorderConditions
            }
        }
    }
    pub fn exact_solution(
        &self,
        start: Option<f64>,
        end: Option<f64>,
        num_values: Option<usize>,
    ) -> Vec<f64> {
        match self {
            NonlinEquation::LaneEmden5 => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                let num_values = if let Some(num_values) = num_values {
                    num_values
                } else {
                    100
                };
                let ressult = "(1+(x^2)/3)^(-0.5)".to_string();
                let expr = Expr::parse_expression(&ressult);
                let y = Expr::lambdify1D_from_linspace(&expr, start, end, num_values);
                plot_from_expr(ressult, "x", "y_exact", start, end, num_values);
                y
            }
            NonlinEquation::ParachuteEquation => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                let num_values = if let Some(num_values) = num_values {
                    num_values
                } else {
                    100
                };
                let ressult = "( ln( (exp(2.0*x) +1 )/2  )  -x)".to_string();
                let expr = Expr::parse_expression(&ressult);
                let y = Expr::lambdify1D_from_linspace(&expr, start, end, num_values);
                plot_from_expr(ressult, "x", "y_exact", start, end, num_values);
                y
            }
            NonlinEquation::Clairaut => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                let num_values = if let Some(num_values) = num_values {
                    num_values
                } else {
                    100
                };
                let ressult = " 1+ (x- 1)^2 - (1/6)*(x-1)^3 + (1/12)*(x-1)^4 ".to_string(); //
                let expr = Expr::parse_expression(&ressult);
                let y = Expr::lambdify1D_from_linspace(&expr, start, end, num_values);
                plot_from_expr(ressult, "x", "y_exact", start, end, num_values);
                y
            }
            NonlinEquation::TwoPointBVP => {
                let start = if let Some(start) = start { start } else { -1.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                let num_values = if let Some(num_values) = num_values {
                    num_values
                } else {
                    100
                };
                let ressult = " exp(-x^2/1.0) ".to_string(); //
                let expr = Expr::parse_expression(&ressult);
                let y = Expr::lambdify1D_from_linspace(&expr, start, end, num_values);
                plot_from_expr(ressult, "x", "y_exact", start, end, num_values);
                y
            }
        }
    }

    pub fn span(&self, start: Option<f64>, end: Option<f64>) -> (f64, f64) {
        match self {
            NonlinEquation::LaneEmden5 => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                (start, end)
            }
            NonlinEquation::ParachuteEquation => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                (start, end)
            }
            NonlinEquation::Clairaut => {
                let start = if let Some(start) = start { start } else { 0.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                (start, end)
            }
            NonlinEquation::TwoPointBVP => {
                let start = if let Some(start) = start { start } else { -1.0 };
                let end = if let Some(end) = end { end } else { 1.0 };
                (start, end)
            }
        }
    }

    pub fn Bounds(&self) -> HashMap<String, (f64, f64)> {
        match self {
            NonlinEquation::LaneEmden5 => {
                let Bounds = HashMap::from([
                    ("z".to_string(), (-10.0, 10.0)),
                    ("y".to_string(), (-7.0, 7.0)),
                ]);
                Bounds
            }
            NonlinEquation::ParachuteEquation => {
                let Bounds = HashMap::from([
                    ("z".to_string(), (-10.0, 10.0)),
                    ("y".to_string(), (-7.0, 7.0)),
                ]);

                Bounds
            }
            NonlinEquation::Clairaut => {
                let Bounds = HashMap::from([
                    ("z".to_string(), (-10.0, 10.0)),
                    ("zz".to_string(), (-10.0, 10.0)),
                    ("y".to_string(), (-7.0, 7.0)),
                ]);

                Bounds
            }
            NonlinEquation::TwoPointBVP => HashMap::from([
                ("z".to_string(), (-1.0, 1.0)),
                ("y".to_string(), (1e-26, 1.0)),
            ]),
        }
    } //Bounds
    pub fn rel_tolerance(&self) -> HashMap<String, f64> {
        match self {
            NonlinEquation::LaneEmden5 => {
                HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)])
            }
            NonlinEquation::ParachuteEquation => {
                HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)])
            }
            NonlinEquation::Clairaut => HashMap::from([
                ("zz".to_string(), 1e-4),
                ("z".to_string(), 1e-4),
                ("y".to_string(), 1e-4),
            ]),
            NonlinEquation::TwoPointBVP => {
                HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)])
            }
        }
    }
}

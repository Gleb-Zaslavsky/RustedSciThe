use crate::numerical::optimization::sym_fitting::Fitting;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::collections::HashMap;
/// some special cases of fitting
/// sew two functions:
/// There are 2 symbolic functions that need to be sewn together:
/// 1. f1 is valid for x < x_central
/// 2. f2 is valid for x > x_central
/// 3 we need to find function f3 that is valid for x_left < x < x_right

pub struct SewTwoFunctions {
    pub f1: Expr,
    pub f2: Expr,
    pub x_left: f64,
    pub x_central: f64,
    pub x_right: f64,
    pub n_points: usize,

    pub fitting_data: (Vec<f64>, Vec<f64>),
    pub fitting: Fitting,
}
impl SewTwoFunctions {
    pub fn new(
        f1: Expr,
        f2: Expr,
        x_left: f64,
        x_central: f64,
        x_right: f64,
        n_points: usize,
    ) -> Self {
        SewTwoFunctions {
            f1,
            f2,
            x_left,
            x_central,
            x_right,
            n_points,

            fitting_data: (Vec::new(), Vec::new()),
            fitting: Fitting::new(),
        }
    }
    pub fn create_fitting_data(&mut self) {
        let f1 = self.f1.clone();
        let f2 = self.f2.clone();
        let x_left = self.x_left;
        let x_central = self.x_central;
        let x_right = self.x_right;
        let n_points = self.n_points;
        let (mut x_data, mut y_data) = create_fitting_data_partial(f1, x_left, x_central, n_points);
        let (x_data2, y_data2) = create_fitting_data_partial(f2, x_central, x_right, n_points);
        x_data.extend(x_data2);
        y_data.extend(y_data2);
        self.fitting_data = (x_data, y_data);
    }
    pub fn set_x_y(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.fitting_data = (x, y);
    }
    pub fn fit(
        &mut self,
        eq: Expr,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.fitting.set_fitting(
            self.fitting_data.0.clone(),
            self.fitting_data.1.clone(),
            eq,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
        self.fitting.eq_generate();
        self.fitting.solve();
    }
    pub fn fit_easy(
        &mut self,
        eq: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
    ) {
        self.fitting.easy_fitting(
            self.fitting_data.0.clone(),
            self.fitting_data.1.clone(),
            eq.clone(),
            unknowns,
            arg,
            initial_guess,
        );
    }
    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.fitting.result.clone()
    }
    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.fitting.map_of_solutions.clone()
    }
    pub fn get_r_ssquared(&self) -> Option<f64> {
        self.fitting.r_ssquared.clone()
    }
}

fn create_fitting_data_partial(
    f: Expr,
    x0: f64,
    x_end: f64,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let eq_fun = f.lambdify1D();
    // evenly spaced points between x0 and x_end
    let step = (x_end - x0) / (n_points - 1) as f64;
    for i in 0..n_points {
        let x = x0 + i as f64 * step;
        // evaluate the function at each point
        let y = eq_fun(x);
        x_data.push(x);
        y_data.push(y);
    }
    (x_data, y_data)
}

/// Sew multiple functions across multiple ranges
pub struct SewMultipleFunctions {
    functions: Vec<Expr>,
    ranges: Vec<(f64, f64)>, // (start, end) for each function
    n_points: usize,
    fitting_data: (Vec<f64>, Vec<f64>),
    fitting: Fitting,
}

impl SewMultipleFunctions {
    pub fn new(functions: Vec<Expr>, ranges: Vec<(f64, f64)>, n_points: usize) -> Self {
        assert_eq!(
            functions.len(),
            ranges.len(),
            "Number of functions must match number of ranges"
        );
        SewMultipleFunctions {
            functions,
            ranges,
            n_points,
            fitting_data: (Vec::new(), Vec::new()),
            fitting: Fitting::new(),
        }
    }

    pub fn create_fitting_data(&mut self) {
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for (func, &(x_start, x_end)) in self.functions.iter().zip(self.ranges.iter()) {
            let (x_partial, y_partial) =
                create_fitting_data_partial(func.clone(), x_start, x_end, self.n_points);
            x_data.extend(x_partial);
            y_data.extend(y_partial);
        }

        self.fitting_data = (x_data, y_data);
    }

    pub fn fit(
        &mut self,
        eq: Expr,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.fitting.set_fitting(
            self.fitting_data.0.clone(),
            self.fitting_data.1.clone(),
            eq,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
        self.fitting.eq_generate();
        self.fitting.solve();
    }

    pub fn fit_easy(
        &mut self,
        eq: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
    ) {
        self.fitting.easy_fitting(
            self.fitting_data.0.clone(),
            self.fitting_data.1.clone(),
            eq,
            unknowns,
            arg,
            initial_guess,
        );
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.fitting.result.clone()
    }

    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.fitting.map_of_solutions.clone()
    }

    pub fn get_r_ssquared(&self) -> Option<f64> {
        self.fitting.r_ssquared.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    #[test]
    fn test_sew_two_functions() {
        /*
        https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Units=SI&Mask=1#Thermo-Gas

        CO2 heat capacity (two ranges)
        T, K 298. to 1200.	1200. to 6000
        A    24.99735	58.16639
        B	55.18696	2.720074
        C	-33.69137	-0.492289
        D	7.948387	0.038844
        E	-0.136638	-6.447293
        F	-403.6075	-425.9186
        G	228.2431	263.6125
        H	-393.5224
        Cp = A + B*t + C*t2 + D*t3 + E/t2
        where t = T(K)/1000
        let's sew two functions together to get the heat capacity of CO2 from 1000. to 1500. K
        */
        let f = Expr::parse_expression("A + B*t + C*t^2 + D*t^3 + E/t^2");
        let var_map1: HashMap<String, f64> = HashMap::from([
            ("A".to_string(), 24.99735),
            ("B".to_string(), 55.18696),
            ("C".to_string(), -33.69137),
            ("D".to_string(), 7.948387),
            ("E".to_string(), -0.136638),
        ]);
        let var_map2: HashMap<String, f64> = HashMap::from([
            ("A".to_string(), 58.16639),
            ("B".to_string(), 2.720074),
            ("C".to_string(), -0.492289),
            ("D".to_string(), 0.038844),
            ("E".to_string(), -6.447293),
        ]);

        let f1 = f.clone().set_variable_from_map(&var_map1);
        println!("Function 1: {}", f1);
        let f2 = f.clone().set_variable_from_map(&var_map2);
        println!("Function 2: {}", f2);

        let x_left = 1000.0 / 1000.0;
        let x_central = 1200.0 / 1000.0;
        let x_right = 1500.0 / 1000.0;
        let n_points = 100;
        let mut sew_two_functions =
            SewTwoFunctions::new(f1, f2, x_left, x_central, x_right, n_points);
        sew_two_functions.create_fitting_data();
        let (ref x_data, ref y_data) = sew_two_functions.fitting_data;
        for i in 0..x_data.len() {
            println!("x: {}, y: {}", x_data[i], y_data[i]);
        }
        sew_two_functions.fit(
            f.clone(),
            Some(vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
            ]),
            "t".to_string(),
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            None,
            None,
            None,
            None,
            None,
        );
        let map_of_solutions = sew_two_functions.get_map_of_solutions();
        println!("{:?}", map_of_solutions);
        let r_ssquared = sew_two_functions.get_r_ssquared();
        println!("r_ssquared: {}", r_ssquared.unwrap());
        assert!(1.0 - r_ssquared.unwrap() < 1e-2);
    }

    #[test]
    fn test_sew_multiple_functions() {
        // Example: sewing 3 polynomial functions across different ranges
        let f1 = Expr::parse_expression("a*x^2 + b*x + c");
        let f2 = Expr::parse_expression("d*x + e");
        let f3 = Expr::parse_expression("f*x^3 + g");

        // Set coefficients for each function
        let var_map1: HashMap<String, f64> = HashMap::from([
            ("a".to_string(), 1.0),
            ("b".to_string(), 2.0),
            ("c".to_string(), 3.0),
        ]);
        let var_map2: HashMap<String, f64> =
            HashMap::from([("d".to_string(), 5.0), ("e".to_string(), 10.0)]);
        let var_map3: HashMap<String, f64> =
            HashMap::from([("f".to_string(), 0.5), ("g".to_string(), 20.0)]);

        let func1 = f1.set_variable_from_map(&var_map1);
        let func2 = f2.set_variable_from_map(&var_map2);
        let func3 = f3.set_variable_from_map(&var_map3);

        let functions = vec![func1, func2, func3];
        let ranges = vec![(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)];
        let n_points = 50;

        let mut sew_multiple = SewMultipleFunctions::new(functions, ranges, n_points);
        sew_multiple.create_fitting_data();

        // Fit with a general polynomial
        let target_eq = "A*x^3 + B*x^2 + C*x + D";
        sew_multiple.fit_easy(
            target_eq.to_string(),
            Some(vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ]),
            "x".to_string(),
            vec![1.0, 1.0, 1.0, 1.0],
        );

        let map_of_solutions = sew_multiple.get_map_of_solutions();
        println!("Multiple functions fit result: {:?}", map_of_solutions);

        let r_ssquared = sew_multiple.get_r_ssquared();
        println!("R-squared: {}", r_ssquared.unwrap());
        assert!(r_ssquared.unwrap() > 0.8); // Should have reasonable fit
    }
}

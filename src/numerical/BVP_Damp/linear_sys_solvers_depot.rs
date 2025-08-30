use crate::somelinalg::RustedLINPACK::lu_band_nalg::LU_nalgebra;
use nalgebra::{DMatrix, DVector};

pub fn nalgebra_solvers_depot(
    A: &DMatrix<f64>,
    b: &DVector<f64>,
    linear_sys_method: Option<String>,
    bandwidth: (usize, usize),
) -> DVector<f64> {
    let condition_for_banded_matrix =
        A.nrows() > 10 * (bandwidth.0 as usize + bandwidth.1 as usize);
    let default_method = if condition_for_banded_matrix {
        "band".to_string()
    } else {
        "full".to_string()
    };
    let method = if let Some(method_) = linear_sys_method {
        method_
    } else {
        default_method
    };
    match method.as_str() {
        "band" => {
            let mut lu = LU_nalgebra::new(A.to_owned(), Some(bandwidth));
            lu.LU();

            let res = lu.solve_linear_system_easy(b);

            res
        }

        "full" => {
            assert_eq!(
                A.nrows(),
                b.len(),
                "dimensons of matrix and vector must match"
            );
            //    println!("{:?}, \n  {:?}", &A, A.shape());
            // println!("{:?}, \n  {:?}", &b, b.len());
            // panic!();
            //  dbg!(&b, b.len());
            let lu = A.to_owned().lu();
            let res = lu.solve(b).unwrap();
            res
        }
        _ => {
            panic!("Unknown method: choose 'full' or 'band' ");
        }
    }
}

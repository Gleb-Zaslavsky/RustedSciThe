#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::numerical::BVP_Damp::BVP_utils::round_to_n_digits;
use log::{error, info};
use nalgebra::{DMatrix, DVector};

pub fn twpnt_refinement(
    y_DMatrix: DMatrix<f64>,
    x_mesh: DVector<f64>,
    d: f64,
    g: f64,
    C: f64,
    abs_tol: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    info!("TWOPOINT algorithm \n \n");
    assert_eq!(y_DMatrix.shape().1, x_mesh.len());
    let mut grid = Grid::new(y_DMatrix, x_mesh, d, g, C, abs_tol);
    grid.calculate();
    (grid.new_grid, grid.y_DMatrix, grid.more)
}
struct Grid {
    y_DMatrix: DMatrix<f64>,
    n: usize,
    x_mesh: DVector<f64>,
    abs_tol: f64,
    d: f64,
    g: f64,
    #[allow(dead_code)]
    C: f64,
    relative_dy: DVector<f64>,
    vary_y: DVector<usize>, //vector of counter of intervals where grid condition for difference is violated
    relative_dy_dx: DVector<f64>,
    vary_dy_dx: DVector<usize>, //vector of counter of intervals where grid condition for derivative is violated
    weights: Vec<usize>,
    mark: Vec<bool>,
    more: usize,
    new_grid: Vec<f64>,
}
impl Grid {
    fn new(
        y_DMatrix: DMatrix<f64>,
        x_mesh: DVector<f64>,
        d: f64,
        g: f64,
        C: f64,
        abs_tol: f64,
    ) -> Self {
        let n = x_mesh.len();
        Self {
            y_DMatrix,
            n,
            x_mesh,
            abs_tol,
            d,
            g,
            C,
            relative_dy: DVector::zeros(n),
            vary_y: DVector::zeros(n),
            relative_dy_dx: DVector::zeros(n),
            vary_dy_dx: DVector::zeros(n),
            weights: vec![],
            mark: vec![],
            more: 0,
            new_grid: vec![],
        }
    }
    fn calculate(&mut self) {
        self.criteria_check();
        self.select_intervals();
        self.add_points();
    }
    fn count_non_zero_elements(vector: &DVector<usize>) -> usize {
        vector.iter().filter(|&x| *x != 0).count()
    }
    fn count_non_false_elements(vector: &Vec<bool>) -> usize {
        vector.iter().filter(|&x| *x != false).count()
    }
    fn assert_monotonically_growing(vector: &[f64]) {
        for window in vector.windows(2) {
            if !(window[0] <= window[1]) {
                println!(
                    "  vector of length {} is non-monotonically growing{:?}, ",
                    vector.len(),
                    vector
                )
            }
            assert!(
                window[0] <= window[1],
                "Vector elements are not monotonously growing"
            );
        }
    }
    #[allow(dead_code)]
    fn grad(&self, y_j: DVector<f64>, i: usize) -> f64 {
        let x_vector = &self.x_mesh;
        let dy = y_j[i + 1] - y_j[i];
        let dx = x_vector[i + 1] - x_vector[i];
        let grad = dy / dx;
        grad
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //          Check if the intermediate solution of the Newton system meets the grid conditions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    fn criteria_check(&mut self) {
        //matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
        let y_DMatrix: &DMatrix<f64> = &self.y_DMatrix;
        //  println!("y_DMatrix {}", y_DMatrix.clone());
        info!("y shape \n {:?}", y_DMatrix.shape());
        let x_mesh: &DVector<f64> = &self.x_mesh;
        // let x_vector:&DVector<f64> = &self.x_vector;
        let abs_tol = self.abs_tol;
        let d = self.d;
        let g = self.g;
        let mut signif: usize = 0;
        let mut relative_dy: DVector<f64> = DVector::zeros(y_DMatrix.ncols()); // relative variation in consecutive points
        let mut vary_y: DVector<usize> = DVector::zeros(y_DMatrix.ncols());
        let mut relative_dy_dx: DVector<f64> = DVector::zeros(y_DMatrix.ncols());
        let mut vary_dy_dx: DVector<usize> = DVector::zeros(y_DMatrix.ncols());
        for (j, y_j) in y_DMatrix.row_iter().enumerate() {
            // for each component j find the maximum and minimum values and its range
            let lower = y_j.min();
            let upper = y_j.max();
            let range_y_j = upper - lower;
            let n = y_j.len();
            self.n = n;
            let delta = d * range_y_j;
            info!("delta {} for component j: {}", delta, j);
            let maxmag_j = f64::max(lower.abs(), upper.abs());
            // decide whether the component is significant. Check if the absolute range of the component is greater than
            //a tolerance (abs_tol) times the maximum of 1 and maxmag. If not, skip the rest of the current iteration.
            if !(range_y_j.abs() > abs_tol * f64::max(1.0, maxmag_j)) {
                continue;
            }
            //if not, this is a significant component.
            signif = signif + 1;

            // Calculate the difference (differ) between consecutive points.
            let mut list_dy_dx_i = Vec::new();
            let mut list_dy_i = Vec::new();
            for i in 0..n - 1 {
                let dy_i = y_j[i + 1] - y_j[i];
                if dy_i.abs() > delta {
                    println!("dy_i/dalta: {}", round_to_n_digits(dy_i.abs() / delta, 3))
                };
                list_dy_i.push(dy_i);
                let dx_i = x_mesh[i + 1] - x_mesh[i];
                let dy_dx_i = dy_i / dx_i;
                list_dy_dx_i.push(dy_dx_i);
                //Update ratio1(k) with the maximum of its current value and the ratio of differ to range.
                if 0.0 < range_y_j {
                    let relative_dy_i = f64::max(relative_dy[i], dy_i.abs() / range_y_j);
                    relative_dy[i] = relative_dy_i;
                }
                if dy_i.abs() > delta {
                    info!("violation of difference grid condition at interval: {}", i);
                    vary_y[i] += 1; // tick the counter - it means that the difference condition is violated
                }
            } // end for k in 0..y_i.len()-1
            let list_dy_dx_i_min = list_dy_dx_i
                .iter()
                .cloned()
                .min_by(|a, b| a.total_cmp(b))
                .unwrap();
            let list_dy_dx_i_max = list_dy_dx_i
                .iter()
                .cloned()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();
            let derivative_range_i = list_dy_dx_i_max - list_dy_dx_i_min;
            let gamma = derivative_range_i * g;
            info!("gamma {} for component j: {}", gamma, j);
            //  Calculate the gradient at the first point and store it in right.
            //  Loop through the interior points, updating left and right with consecutive gradient values.
            //  Calculate the difference (differ) between consecutive gradients.
            // Update relative_dy_dx with the maximum of its current value and the ratio of differ to range.
            // Increment vary_dy_dx if differ exceeds toler2 times range.
            let temp = list_dy_i[0];
            let right = temp;
            for i in 1..n - 1 {
                let left = right;
                let right = list_dy_i[i];
                let differ_dy = (right - left).abs();
                if differ_dy > gamma {
                    println!(
                        "differ_dy/gamma = {}",
                        round_to_n_digits(differ_dy / gamma, 3)
                    )
                };
                if 0.0 < derivative_range_i {
                    let relative_dy_dx_i =
                        f64::max(relative_dy_dx[i], differ_dy / derivative_range_i);
                    relative_dy_dx[i] = relative_dy_dx_i
                }
                if differ_dy > gamma {
                    info!("violation of gradient grid condition at interval: {}", i);
                    vary_dy_dx[i] += 1; // tick the counter - it means that the difference condition is violated
                };
            }
        } //  for (j, y_j)
        info!(
            "\n \n difference grid condition violated in {} intervals",
            Self::count_non_zero_elements(&vary_y)
        );
        info!(
            "gradient grid condition violated in {} intervals",
            Self::count_non_zero_elements(&vary_dy_dx)
        );
        self.relative_dy = relative_dy;
        self.vary_y = vary_y;
        self.relative_dy_dx = relative_dy_dx;
        self.vary_dy_dx = vary_dy_dx;
    } // end fn mark

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                     select the intervals to halve
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    fn select_intervals(&mut self) {
        //  weight the intervals in which variations that are too large.
        // now we have 1) vector of counter of intervals where grid condition for difference is violated
        // 2) vector of counter of intervals where grid condition for derivative is violated
        // now we should combine them together,
        // the weight assigned to each interval, which is determined by the number of times the solution variables' variations exceed certain thresholds.
        let mut most: usize = 0; // number of non zero weights
        let x_mesh: &DVector<f64> = &self.x_mesh;
        let vary_y: &DVector<usize> = &self.vary_y;
        let vary_dy_dx: &DVector<usize> = &self.vary_dy_dx;
        let mut weights = vec![0; x_mesh.len()];
        for i in 0..self.n - 1 {
            weights[i] = vary_y[i];
            if i > 0 {
                weights[i] += vary_dy_dx[i]
            }
            if i < self.n - 1 {
                // add
                weights[i] += vary_dy_dx[i + 1]
            }
            if weights[i] > 0 {
                most += 1
            }
        }
        info!(
            "indexes of non-zero weights {:?} ",
            weights
                .iter()
                .enumerate()
                .filter(|&(_, &x)| x != 0)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        );
        info!(
            "number of non-zero weights {} ",
            Self::count_non_zero_elements(&DVector::from_vec(weights.clone()))
        );
        info!("most parameter {}", most);
        // sorts the weights in ascending order using an interchange sort algorithm. This sorting is done to prioritize the intervals
        //that need to be refined.
        let mut sorted_weights = weights.clone();
        sorted_weights.sort();

        let more = usize::max(0, most);

        let least = if more > 0 {
            sorted_weights[more]
        }
        //  If the number of intervals to halve is greater than zero, the least weight is set to the weight of the interval with the corresponding index
        else {
            1 + sorted_weights[0]
        }; // Otherwise, the least weight is set to one plus the weight of the first interval.
        info!("least element of weights {}", least);
        let mut counted: usize = 0;
        let mut mark = vec![false; x_mesh.len()];
        for i in 0..x_mesh.len() - 1 {
            if weights[i] >= least && more > counted {
                mark[i] = true;
                counted += 1;
            }
        }
        info!(
            "number of marked intervals {} ",
            Self::count_non_false_elements(&mark)
        );
        self.mark = mark;
        self.more = more;
        self.weights = weights;
    } // fn select_intervals

    fn add_points(&mut self) {
        info!("adding points");
        let n_total = self.x_mesh.len() + self.more;
        let x = self.x_mesh.clone();
        let n_former = self.x_mesh.len();
        let more = self.more;
        let mark = self.mark.clone();
        let y = self.y_DMatrix.clone();
        let n_rows = y.nrows();
        // Check this interval is not degenerate
        let mut counted = 0; //  counter counted to keep track of the number of degenerate intervals.
        if more > 0 {
            // let length: f64 = (x[n_former] - x[0]).abs();
            for k in 0..n_former - 1 {
                if mark[k] {
                    let mean = 0.5 * (x[k] + x[k + 1]); // For each interval marked for refinement (mark[k] == true), it calculates the mean
                    //of the interval and checks if the mean is within the interval. If not, it increments the counted counter.
                    if !((x[k] < mean && mean < x[k + 1]) || (x[k + 1] < mean && mean < x[k])) {
                        counted += 1;
                    } // degenerance condition
                } //if mark[k]
            } // or k in 0..
        } // if nore>0
        let error = counted > 0;
        if error {
            error!("error");
        } else {
            info!("no degenerated intervals");
        }

        let mut new_initial_guess: DMatrix<f64> = DMatrix::zeros(n_rows, n_total);
        let mut new_grid: Vec<f64> = vec![0.0; n_total];

        let mut n_new = n_total;
        // construct the new grid vector and new initial guess for the calculation
        //  For each interval, it adds the corresponding point to the new grid and copies the corresponding column from the current
        //solution to the new initial guess. If the interval is marked for refinement (mark[i_old - 1] == true), it calculates the midpoint of the interval
        //and adds it to the new grid. It then calculates the average of the solution at the neighboring points and adds it to the new initial guess.
        new_grid[0] = x[0];
        new_initial_guess.column_mut(0).copy_from(&y.column(0));
        for i_old in (1..n_former).rev() {
            new_grid[n_new - 1] = x[i_old];
            let y_i_old = &y.column(i_old);
            new_initial_guess.column_mut(n_new - 1).copy_from(y_i_old);
            n_new -= 1;

            if mark[i_old - 1] {
                new_grid[n_new - 1] = 0.5 * (x[i_old] + x[i_old - 1]);
                info!(
                    "inserted new grid point {} between points {} and {} at index {} ",
                    round_to_n_digits(new_grid[n_new - 1], 3),
                    round_to_n_digits(new_grid[n_new], 3),
                    round_to_n_digits(new_grid[n_new - 2], 3),
                    i_old - 1
                );
                let y_min_1 = y.column(i_old - 1);
                let y_i = y.column(i_old);
                let dy_i = y_min_1 + 0.5 * (y_i - y_min_1);
                new_initial_guess.column_mut(n_new - 1).copy_from(&dy_i);
                info!("\n \n column added: {} at index {} ", &dy_i, i_old - 1);
                n_new -= 1;
            }
        }

        log::info!(
            "created new grid {:?} of length {}",
            new_grid,
            new_grid.len()
        );
        log::info!(
            "\n \n new_initial_guess: {} of shape{:?}",
            new_initial_guess,
            new_initial_guess.shape()
        );
        assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
        // if no points added code returns the same arrays
        if self.more == 0 {
            assert_eq!(new_initial_guess, self.y_DMatrix);
            assert_eq!(x, DVector::from_vec(new_grid.clone()));
        };

        /*
                  let mut n_new = n_total;
                  for i_old in (1..n_former).rev() {
                      mark[n_new - 1] = false;
                      n_new -= 1;
                      if mark[i_old - 1] {
                          mark[n_new - 1] = true;
                          n_new -= 1;
                      }
                  }
                  mark[n_new - 1] = false;
        */
        Self::assert_monotonically_growing(&new_grid);
        self.new_grid = new_grid;
        self.y_DMatrix = new_initial_guess;
    }
}

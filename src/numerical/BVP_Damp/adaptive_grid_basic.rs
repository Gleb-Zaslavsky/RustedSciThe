//! # Basic Adaptive Grid Refinement Algorithms
//!
//! ## Module Purpose
//! This module implements fundamental adaptive mesh refinement algorithms for boundary value problems,
//! providing the mathematical foundation for intelligent grid adaptation. Each algorithm analyzes
//! solution characteristics to determine where additional mesh points are needed for accuracy.
//!
//! ## Core Algorithms
//! - [`refine_all_grid`]: Uniform refinement - doubles mesh density everywhere
//! - [`easy_grid_refinement`]: Simple gradient-based refinement with tolerance control
//! - [`pearson_grid_refinement`]: Classical boundary layer algorithm with mesh smoothing
//! - [`grcar_smooke_grid_refinement`]: Advanced combustion-oriented method with dual criteria
//! - [`scipy_grid_refinement`]: Residual-based refinement inspired by SciPy's solve_bvp
//!
//! ## Mathematical Foundation
//! All algorithms are based on truncation error analysis and finite difference theory.
//! The module includes detailed mathematical derivations for:
//! - Forward/backward/central difference truncation errors
//! - Taylor series expansions for error estimation
//! - Mesh adaptation criteria based on solution gradients
//!
//! ## Interesting Code Features
//! - **HashMap-based marking system**: Efficiently tracks which intervals need refinement
//! - **Biased indexing**: Clever index management during mesh construction to handle variable insertion
//! - **Multi-row analysis**: Each algorithm analyzes all solution components simultaneously
//! - **Buffering mechanisms**: Prevents abrupt mesh size changes that could hurt accuracy
//! - **Linear interpolation**: Provides smooth initial guess on refined mesh
//! - **Threshold-based filtering**: Avoids refining in regions with negligible solution values
//!
//! ## Performance Optimizations
//! - Pre-allocates matrices with known final size to avoid reallocations
//! - Uses iterator-based row processing for cache efficiency
//! - Employs early termination when no refinement is needed
//! - Leverages nalgebra's optimized linear algebra operations
//!
//! ## Algorithm Selection Strategy
//! Different algorithms excel in different scenarios:
//! - **Uniform**: Guaranteed improvement but expensive
//! - **Easy**: Fast and simple, good for smooth solutions
//! - **Pearson**: Optimal for boundary layer problems
//! - **Grcar-Smooke**: Best for combustion/reaction systems
//! - **SciPy**: Excellent when residual information is available

use log::info;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
/*
Some math considerations...
1. truncation error analysis
Consider a forkward finite difference approximation of the first-order derivative u
[Dy]_n = (y_(n+1)-y_n)/dt (1)
Here, y_n means the value of some function u(t) at a point tn, and [Dy]_n is the discrete derivative
of y(t) at t = t_n. The discrete derivative computed by a finite difference is not exactly equal to
the derivative y'(t_n) at t = t_n. The error in the approximation is
R_n = [Dy]_n -  y'(t_n) (2)
The common way of calculating Rn is to
1. expand y(t) in a Taylor series around the point where the derivative is evaluated, here t_n,
2. insert this Taylor series in (2), and
3. collect terms that cancel and simplify the expression.
 - The Taylor series of y_n at t_n is simply y_n=y(t_n),
 - The Taylor series of y_(n+1) at t_n is y_(n+1) =y(t_n + dt)= y(t_n) + y'(t_n)*dt + y''(t_n)*dt^2/2 + ...
 Inserting the Taylor series above in the left-hand side of (2) gives
 R_n = [Dy]_n -  y'(t_n) = (y_(n+1)-y_n)/dt - y'(t_n) = (y_n + y'(t_n)*dt + y''(t_n)*dt^2/2 -y_n )/dt - y'(t_n) =   y''(t_n)*dt/2
so for  forkward finite difference truncation error is R_n =   y''(t_n)*dt/2
For backwarw finite difference [Dy]_n = (y_n - y_(n-1))/d truncation error is R_n =  - y''(t_n)*dt/2 (because y_(n-1) =y(t_n - dt)= y(t_n) - y'(t_n)*dt + y''(t_n)*dt^2/2 + ...  )
For the central difference approximation,

 [Dy]_n = (y_(n+0.5)-y_(n-0.5))/dt (3)
we write
R_n = [Dy]_n -  y'(t_n)
 The Taylor series of y_(n+0.5)  at t_n is y_(n+0.5) =y(t_n + 0.5dt)= y(t_n) +0.5 y'(t_n)*dt + y''(t_n)*(0.5*dt)^2/2 + y'''(t_n)*(0.5*dt)^3/6
  The Taylor series of y_(n+0.5)  at t_n is y_(n-0.5) =y(t_n - 0.5dt)= y(t_n) - 0.5 y'(t_n)*dt + y''(t_n)*(0.5*dt)^2/2 - y'''(t_n)*(0.5*dt)^3/6
  y_(n+0.5) - y_(n+0.5)  =  y'(t_n)*dt + 2* y'''(t_n)*(0.5*dt)^3/6 =  y'(t_n)*dt +  y'''(t_n)*(dt)^3/24 => [Dy]_n = y'(t_n) +  y'''(t_n)*(dt)^2/24 =>
  R_n =  y'''(t_n)*(dt)^2/24
 based on book Truncation error analysis
Hans Petter Langtangen
dimension of R_n is always just the same as the dimension of derivative
2. On the choise of mesh
"We have found that starting the itration on a coarse mesh has several important advntages. One is that the Newton iteration is more likely to
converge on a coarse mesh than on a fine mesh. Moreover, the number of variables is small on a coarse mesh and thus the cost per iteration is
relatively small. Since the iteration begins from a user-specfied “guess” at the solution, it is likly that many iterations will be required.
Ultimately, of course, to be accurate, the solution must be obtained on a fine mesh. However, as the solution is computed on each successively finer
mesh, the starting estimates are better, since they come from the converged solution on the previous coarse mesh. In general, the solution on one
mesh lies within the domain of convergence of Newton’s method on the next finer mesh.Thus, even though the  cost per iteration is increasing, the
number of required iterations is decreasing. The adaptve placement of the mesh points to form the finer meshes is done in such a
way that the total number of mesh points needed to represent the solution accurately is minimized
 " Chemkin Theory Manual p.263
*/
// sometimes we just need to double the  number of meshes in the grid (actially we get 2*meshes-1) inserting new points int the center of
//intervals
/// refine all intervals in the grid by adding a point in the center of each interval - naive and straightforward but
/// expensive approach. This method is used as a reference to compare with other more sophisticated methods
///  Also if other methods have conditiones if it is needed to add points to the grid and if mothing was added it means
/// that the solution is converged and we can stop the iteration. This method always adds points to the grid so newton
/// iteration will stop only when the maximum number of grid refinements is reached
///
pub fn refine_all_grid(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let mut h: Vec<f64> = Vec::new();
    for i in 0..x_mesh.len() - 1 {
        let h_i = x_mesh[i + 1] - x_mesh[i];
        h.push(h_i);
    }
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    let mut biased_i = 0;
    let number_of_nonzero_keys = x_mesh.len() - 1;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + number_of_nonzero_keys);
    // println!("mark {:?}, x_mesh {} ", mark, x_mesh);
    for i in 0..x_mesh.len() - 1 {
        let y = y_DMatrix.column(i);
        // copy points from the old grid to the new one
        new_grid.push(x_mesh[i]);
        // adding points to the grid
        let h_i = x_mesh[i + 1] - x_mesh[i];
        let x_new = x_mesh[i] + h_i * 0.5;
        new_grid.push(x_new);

        // copy points from the previous step solution to the new guess
        new_initial_guess.column_mut(biased_i).copy_from(&y);
        biased_i += 1;
        //
        // making interpolation between neighbor points of the previous step solution to form the new initial guess

        let y_pl_1 = y_DMatrix.column(i + 1);
        let dy_i = y_pl_1 - y;
        // add new point to new_initial_guess
        let column_to_add = y + &dy_i * 0.5;
        new_initial_guess
            .column_mut(biased_i as usize)
            .copy_from(&column_to_add);
        biased_i += 1;

        // add all points from the old grid to the new grid
    }
    // add right border points to the new grid
    let y_last = y_DMatrix.column(x_mesh.len() - 1);
    new_grid.push(x_mesh[x_mesh.len() - 1]);
    new_initial_guess
        .column_mut(biased_i as usize)
        .copy_from(&y_last);
    log::info!("created new grid of length {}", new_grid.len());
    log::info!(
        "\n \n new_initial_guess of shape{:?}",
        //   new_initial_guess,
        new_initial_guess.shape()
    );

    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, number_of_nonzero_keys)
}

/// Parallel version of refine_all_grid - processes intervals in parallel
pub fn refine_all_grid_par(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let (n_rows, _) = y_DMatrix.shape();
    let number_of_nonzero_keys = x_mesh.len() - 1;
    let mut new_grid: Vec<f64> = Vec::with_capacity(x_mesh.len() + number_of_nonzero_keys);
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + number_of_nonzero_keys);

    // Parallel computation of grid points and interpolated values
    let grid_data: Vec<(f64, f64, DVector<f64>)> = (0..x_mesh.len() - 1)
        .into_par_iter()
        .map(|i| {
            let h_i = x_mesh[i + 1] - x_mesh[i];
            let x_new = x_mesh[i] + h_i * 0.5;
            let y = y_DMatrix.column(i);
            let y_pl_1 = y_DMatrix.column(i + 1);
            let dy_i = y_pl_1 - y;
            let column_to_add = y + &dy_i * 0.5;
            (x_mesh[i], x_new, column_to_add)
        })
        .collect();

    // Sequential assembly (required for proper ordering)
    let mut biased_i = 0;
    for (i, (x_orig, x_new, interpolated)) in grid_data.into_iter().enumerate() {
        new_grid.push(x_orig);
        new_grid.push(x_new);

        let y = y_DMatrix.column(i);
        new_initial_guess.column_mut(biased_i).copy_from(&y);
        biased_i += 1;
        new_initial_guess
            .column_mut(biased_i)
            .copy_from(&interpolated);
        biased_i += 1;
    }

    // Add final point
    let y_last = y_DMatrix.column(x_mesh.len() - 1);
    new_grid.push(x_mesh[x_mesh.len() - 1]);
    new_initial_guess.column_mut(biased_i).copy_from(&y_last);

    log::info!("created new grid of length {}", new_grid.len());
    log::info!("new_initial_guess of shape{:?}", new_initial_guess.shape());

    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, number_of_nonzero_keys)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          EASY  GRID  REFINEMENT  PROCEDURE
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pub fn easy_grid_refinement(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    tolerance: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    info!(
        "shape of solution {}, {}",
        y_DMatrix.nrows(),
        y_DMatrix.ncols()
    );
    info!("x_mesh len {:?}", x_mesh.len());
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    let mut mark: DVector<i32> = DVector::zeros(x_mesh.len());

    // Mark columns that need refinement by checking all rows
    for (j, y) in y_DMatrix.row_iter().enumerate() {
        let y_j_max = y.max();
        let y_j_min = y.min();
        let delta = tolerance * (y_j_max - y_j_min);
        info!("for component j: {} delta {}", j, delta);
        for i in 1..x_mesh.len() - 1 {
            let tau_i = (y[i] - y[i - 1]).abs();
            if tau_i > delta {
                info!("tau {} for i {}", tau_i, i);
                mark[i - 1] = 1;
            }
        }
        info!(
            "\n \n for row {} mark: {:?} len {} \n \n",
            j,
            mark,
            mark.len()
        );
    }

    //  total number of new points to add
    let total_new_points: i32 = mark.sum();
    log::info!("total new points to add: {}", total_new_points);

    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);
    // println!("mark {:?}, x_mesh {} ", mark, x_mesh);
    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);

        // Always add the original point first
        new_grid.push(x_mesh[i]);
        new_initial_guess.column_mut(biased_i).copy_from(&y);
        biased_i += 1;

        if mark[i] != 0_i32 && i < x_mesh.len() - 1 {
            // Add new point between current and next
            let h_i = x_mesh[i + 1] - x_mesh[i];
            let x_new = x_mesh[i] + h_i * 0.5;
            new_grid.push(x_new);

            // Interpolate for new initial guess
            let y_pl_1 = y_DMatrix.column(i + 1);
            let dy_i = y_pl_1 - y;
            let column_to_add = y + &dy_i * 0.5;
            new_initial_guess
                .column_mut(biased_i as usize)
                .copy_from(&column_to_add);
            biased_i += 1;
        }
    }

    log::info!("created new grid of length {}", new_grid.len());
    log::info!(
        "\n \n new_initial_guess of shape{:?}",
        new_initial_guess.shape()
    );
    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

/// Parallel version of easy_grid_refinement - parallelizes the marking phase
pub fn easy_grid_refinement_par(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    tolerance: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    info!(
        "shape of solution {}, {}",
        y_DMatrix.nrows(),
        y_DMatrix.ncols()
    );
    info!("x_mesh len {:?}", x_mesh.len());
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    let mark = Mutex::new(DVector::zeros(x_mesh.len()));

    // Parallel marking phase - each row processed in parallel
    y_DMatrix
        .row_iter()
        .enumerate()
        .par_bridge()
        .for_each(|(j, y)| {
            let y_j_max = y.max();
            let y_j_min = y.min();
            let delta = tolerance * (y_j_max - y_j_min);
            info!("for component j: {} delta {}", j, delta);

            let mut local_mark = Vec::new();
            for i in 1..x_mesh.len() - 1 {
                let tau_i = (y[i] - y[i - 1]).abs();
                if tau_i > delta {
                    info!("tau {} for i {}", tau_i, i);
                    local_mark.push(i - 1);
                }
            }

            // Update global mark under lock
            if !local_mark.is_empty() {
                let mut global_mark = mark.lock().unwrap();
                for &idx in &local_mark {
                    global_mark[idx] = 1;
                }
            }

            info!("for row {} found {} intervals to mark", j, local_mark.len());
        });

    let mark = mark.into_inner().unwrap();
    let total_new_points: i32 = mark.sum();
    log::info!("total new points to add: {}", total_new_points);

    // Sequential insertion phase (required for proper ordering)
    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);

    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);

        // Always add the original point first
        new_grid.push(x_mesh[i]);
        new_initial_guess.column_mut(biased_i).copy_from(&y);
        biased_i += 1;

        if mark[i] != 0_i32 && i < x_mesh.len() - 1 {
            // Add new point between current and next
            let h_i = x_mesh[i + 1] - x_mesh[i];
            let x_new = x_mesh[i] + h_i * 0.5;
            new_grid.push(x_new);

            // Interpolate for new initial guess
            let y_pl_1 = y_DMatrix.column(i + 1);
            let dy_i = y_pl_1 - y;
            let column_to_add = y + &dy_i * 0.5;
            new_initial_guess
                .column_mut(biased_i as usize)
                .copy_from(&column_to_add);
            biased_i += 1;
        }
    }

    log::info!("created new grid of length {}", new_grid.len());
    log::info!("new_initial_guess of shape{:?}", new_initial_guess.shape());
    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          PEARSON  GRID  REFINEMENT  PROCEDURE
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Saurces:
1) ON  A  DIFFERENTIAL  EQUATION  OF  BOUNDARY  LAYER  TYPE
By CARL  E.  PEARSON,
p. 138
2)
eng
New mesh points are now inserted between any pair of adjacent mesh points-say x_i and x_i+1  for which (y_i+1 - y_i).abs() exceeds a predetermined
limit delta; the number of such mesh points inserted (uniformly)  between x_i and x_i+1 is approximately equal to (y_i+1 - y_i).abs()/delta.
Discrete Equations  are then solved again, new  mesh points inserted, and so  on; the process continues iteratively until (y_i+1 - y_i).abs()< delta
everywhere. The value of delta is  adjusted during the computation, so  as to always bear a fixed ratio  (typically 1e-3)  to the computed value of
 (max_i {y_i)}  - min_i {y_i} }.   Since the insertion of new  mesh points may result in a locally abrupt change in mesh interval size,  with some
 consequent  loss  in  the accuracy  with which Discrete Eq. approximates continuous  Eq.,  a  smoothing  process  is  carried  out  prior  to each
 new Gaussian elimination sweep.  This smoothing process simply consists in replacing each mesh point  x_i  by a new mesh point  x_i' = 0.5(x_i+ x_i+1)

 Новые точки сетки теперь вставляются между любой парой соседних точек сетки, скажем, x_i и x_i+1, для которых (y_i+1 - y_i).abs() превышает
 определенный предел delta; количество таких точек сетки, вставленных (равномерно) между x_i и x_i+1, приблизительно равно
 (y_i+1 - y_i).abs()/delta. Затем снова решаются дискретные уравнения, вставляются новые точки сетки и так далее; процесс продолжается итеративно
 до тех пор, пока (y_i+1 - y_i).abs()< delta везде. Значение delta корректируется во время вычисления, чтобы всегда иметь фиксированное отношение
  (обычно 1e-3) к вычисленному значению (max_i {y_i)} - min_i {y_i} }. Поскольку вставка новых точек сетки может привести к локальному резкому
  изменению размера интервала сетки с некоторой последующей потерей точности, с которой дискретные уравнения. аппроксимируют непрерывные, перед каждым новым
  гауссовым исключением выполняется процесс сглаживания. Этот процесс сглаживания заключается просто в замене каждой точки сетки x_i новой точкой
  сетки x_i' = 0.5(x_i+ x_i+1)

 */

pub fn pearson_grid_refinement(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    d: f64,
    C: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let mut h: Vec<f64> = Vec::new();
    for i in 0..x_mesh.len() - 1 {
        let h_i = x_mesh[i + 1] - x_mesh[i];
        h.push(h_i);
    }
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    // hashmap key: in what position insert points, value: how many points to insert
    // mark[i] = how many points to insert in i-th position
    let mut mark: HashMap<usize, i32> = HashMap::new();

    // each row is the solution of the ODE at a points in the grid
    for (j, y) in y_DMatrix.clone().row_iter().enumerate() {
        // println!("y, {:}", y);
        let threshold = 1e-4;
        let y_j_max = y.max();
        let y_j_min = y.min();
        let delta = d * (y_j_max - y_j_min);
        info!("delta {} for component j: {}", delta, j);

        for i in 0..x_mesh.len() {
            if !mark.contains_key(&i) {
                mark.insert(i, 0); // default value 0 - no point inserted
            }

            if x_mesh.len() - 1 > i && i > 0 {
                let tau_i = (y[i] - y[i - 1]).abs();

                let both_ys_are_not_too_small =
                    (y[i].abs() > threshold) & (y[i + 1].abs() > threshold);
                // eq 1, eq 2
                //
                if tau_i > delta && both_ys_are_not_too_small {
                    info!(
                        "tau {:3} ({:3}, {:3})> delta {:3} for i {}",
                        tau_i,
                        y[i],
                        y[i - 1],
                        delta,
                        i
                    );

                    // how many new points should be added
                    let N = if (tau_i / delta) as i32 >= 1 {
                        (tau_i / delta) as i32
                    } else {
                        1
                    };
                    // info!(" conditions vaiolation at index {}, => N = {}", i, N);

                    // Only update if new N is larger than existing value
                    let current_N = *mark.get(&i).unwrap_or(&0);
                    if N > current_N {
                        mark.insert(i - 1, N);
                    }
                }
                // if not mark element remains 0
            } //i>0
            //for i==0 and i==-1 mark elements remain 0
        } // for i in 0..x_mesh.len()
        //  info!("mark {:?}", mark);
        // find keys corresponding to non-zero values in the HashMap
        let non_zero_keys: Vec<usize> = mark
            .iter()
            .filter(|&(_, &value)| value != 0)
            .map(|(key, _)| *key)
            .collect();
        info!(
            "\n \n for row {} found intervals to be refined: {:?} of length {} \n \n",
            j,
            non_zero_keys,
            non_zero_keys.len()
        );
    }

    // bufferisation to avoid rapid changes in grid inervals
    //the ratio of adjacent grid intervals must be bounded above and below by constants.
    for i in 1..x_mesh.len() - 1 {
        let buffer_condition_1 = h[i] / h[i - 1] <= C; // if true at i-1 index point should be added (between x_mesh[i] - x_mesh[i-1];)
        let buffer_condition_2 = h[i] / h[i - 1] >= 1.0 / C; // if true at i index point should be added  (between x_mesh[i+1] - x_mesh[i];
        if !buffer_condition_1 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] <= C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );
            if (i - 1) != 0 {
                let current_N = *mark.get(&(i - 1)).unwrap_or(&0);
                if 1 > current_N {
                    mark.insert(i - 1, 1);
                }
            }
        }
        if !buffer_condition_2 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] >= 1.0/C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );

            let current_N = *mark.get(&i).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
    }
    //  total number of new points to add
    let total_new_points: i32 = mark.values().sum();
    log::info!("total new points to add: {}", total_new_points);

    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);
    // println!("mark {:?}, x_mesh {} ", mark, x_mesh);
    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);
        if mark.get(&i).unwrap() != &0 {
            //
            // adding points to the grid
            let N = *mark.get(&i).unwrap();
            if i < x_mesh.len() - 1 {
                let h_i = x_mesh[i + 1] - x_mesh[i];
                for k in 0..N + 1 {
                    // k=0 refers to yhe element existing in the old mesh
                    let x_new = x_mesh[i] + h_i * (k as f64) / (N as f64 + 1.0);
                    if k != 0 {
                        log::info!("\n \n points added: {} at index {} ", x_new, i);
                    }
                    new_grid.push(x_new);
                }
                //
                // making interpolation between neighbor points of the previous step solution to form the new initial guess
                let y_pl_1 = y_DMatrix.column(i + 1);
                let dy_i = y_pl_1 - y;
                for k in 0..N + 1 {
                    // add new point to new_initial_guess
                    let column_to_add = y + &dy_i * (k as f64) / (N as f64 + 1.0);
                    if k != 0 {
                        // log::info!( "\n \n column added: {} at index {} ", column_to_add,biased_i);
                    }
                    new_initial_guess
                        .column_mut(biased_i as usize)
                        .copy_from(&column_to_add);
                    biased_i += 1;
                }
            } else {
                // Last point - just copy
                new_grid.push(x_mesh[i]);
                new_initial_guess.column_mut(biased_i).copy_from(&y);
                biased_i += 1;
            }
        } else {
            // copy points from the old grid to the new one
            new_grid.push(x_mesh[i]);
            //
            // copy points from the previous step solution to the new guess
            new_initial_guess.column_mut(biased_i).copy_from(&y);
            biased_i += 1;
        } // add all points from the old grid to the new grid
    }

    log::info!("created new grid of length {}", new_grid.len());
    log::info!(
        "\n \n new_initial_guess of shape{:?}",
        new_initial_guess.shape()
    );

    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

/// Parallel version of pearson_grid_refinement - parallelizes the marking phase
pub fn pearson_grid_refinement_par(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    d: f64,
    C: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let mut h: Vec<f64> = Vec::new();
    for i in 0..x_mesh.len() - 1 {
        let h_i = x_mesh[i + 1] - x_mesh[i];
        h.push(h_i);
    }
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    let mark = Mutex::new(HashMap::new());

    // Parallel marking phase
    y_DMatrix
        .clone()
        .row_iter()
        .enumerate()
        .par_bridge()
        .for_each(|(j, y)| {
            let threshold = 1e-4;
            let y_j_max = y.max();
            let y_j_min = y.min();
            let delta = d * (y_j_max - y_j_min);
            info!("delta {} for component j: {}", delta, j);

            let mut local_marks = Vec::new();
            for i in 0..x_mesh.len() {
                if x_mesh.len() - 1 > i && i > 0 {
                    let tau_i = (y[i] - y[i - 1]).abs();
                    let both_ys_are_not_too_small =
                        (y[i].abs() > threshold) & (y[i + 1].abs() > threshold);

                    if tau_i > delta && both_ys_are_not_too_small {
                        let N = if (tau_i / delta) as i32 >= 1 {
                            (tau_i / delta) as i32
                        } else {
                            1
                        };
                        local_marks.push((i - 1, N));
                    }
                }
            }

            if !local_marks.is_empty() {
                let mut global_mark = mark.lock().unwrap();
                for (idx, N) in local_marks {
                    let current_N = *global_mark.get(&idx).unwrap_or(&0);
                    if N > current_N {
                        global_mark.insert(idx, N);
                    }
                }
            }
        });

    let mut mark = mark.into_inner().unwrap();

    // Sequential bufferization
    for i in 1..x_mesh.len() - 1 {
        let buffer_condition_1 = h[i] / h[i - 1] <= C;
        let buffer_condition_2 = h[i] / h[i - 1] >= 1.0 / C;
        if !buffer_condition_1 && (i - 1) != 0 {
            let current_N = *mark.get(&(i - 1)).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
        if !buffer_condition_2 {
            let current_N = *mark.get(&i).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
    }

    let total_new_points: i32 = mark.values().sum();
    log::info!("total new points to add: {}", total_new_points);

    // Sequential grid construction
    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);

    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);
        if mark.get(&i).unwrap_or(&0) != &0 {
            let N = *mark.get(&i).unwrap();
            if i < x_mesh.len() - 1 {
                let h_i = x_mesh[i + 1] - x_mesh[i];
                for k in 0..N + 1 {
                    let x_new = x_mesh[i] + h_i * (k as f64) / (N as f64 + 1.0);
                    new_grid.push(x_new);
                }
                let y_pl_1 = y_DMatrix.column(i + 1);
                let dy_i = y_pl_1 - y;
                for k in 0..N + 1 {
                    let column_to_add = y + &dy_i * (k as f64) / (N as f64 + 1.0);
                    new_initial_guess
                        .column_mut(biased_i as usize)
                        .copy_from(&column_to_add);
                    biased_i += 1;
                }
            } else {
                new_grid.push(x_mesh[i]);
                new_initial_guess.column_mut(biased_i).copy_from(&y);
                biased_i += 1;
            }
        } else {
            new_grid.push(x_mesh[i]);
            new_initial_guess.column_mut(biased_i).copy_from(&y);
            biased_i += 1;
        }
    }

    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//             GRCAR  AND  SMOOKE  GRID  REFINEMENT  PROCEDURE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Saurces
A  HYBRID NEWTON/TIME-INTEGRATION  PROCEDURE FOR THE
SOLUTION OF STEADY, LAMINAR,  ONE-DIMENSIONAL,  PREMIXED
FLAMES by JOSEPH F. GRCAR, ROBERT J.  KEE, MITCHELL D. SMOOKE and JAMES A. MILLER

The  starting  estimate  for  the  dependent variable  vector  y  on  a  new,  finer  mesh  is determined by a  linear interpolation of the old
coarse  mesh  solution.  After  obtaining a  converged  solution on the  new mesh,  the adapta-tion  procedure  is  performed  once  again.  A
sequence  of  solutions  on  successively  finer meshes  is  computed  until  the  inequalities in Eqs. (1) and (2) (see code below) are satisfied between all mesh
points.


olution  of  Burner-Stabilized  Premixed  Laminar  Flames
by  Boundary  Value  Methods
by MITCHELL  D.  SMOOKE


*/
pub fn grcar_smooke_grid_refinement(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    d: f64,
    g: f64,
    C: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    info!("old mesh len {:?}", x_mesh.len());
    info!(
        "shape of solution {}, {}",
        y_DMatrix.nrows(),
        y_DMatrix.ncols()
    );
    for (j, y) in y_DMatrix.clone().row_iter().enumerate() {
        log::debug!("row {} : y= {:}", j, y);
    }
    let mut h: Vec<f64> = Vec::new();
    for i in 0..x_mesh.len() - 1 {
        let h_i = x_mesh[i + 1] - x_mesh[i];
        h.push(h_i);
    }
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    // hashmap key: in what position insert points, value: how many points to insert
    // mark[i] = how many points to insert in i-th position
    let mut mark: HashMap<usize, i32> = HashMap::new();

    // each row is the solution of the ODE at a points in the grid
    for (j, y) in y_DMatrix.clone().row_iter().enumerate() {
        info!("y= {:}, {:?}", y[0], y[y.len() - 1]);
        let threshold = 1e-4;
        let y_j_max = y.max();
        let y_j_min = y.min();
        let delta = d * (y_j_max - y_j_min);
        info!(
            "delta {} ({}, {}) for component j: {}",
            delta, y_j_max, y_j_min, j
        );
        let mut list_dy_dx_i = Vec::new();

        for i in 0..x_mesh.len() - 1 {
            let dy_i = y[i + 1] - y[i];
            let h_i = h[i];
            let dy_dx_i = dy_i / h_i;
            list_dy_dx_i.push(dy_dx_i);
        }
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
        let derivative_range = list_dy_dx_i_max - list_dy_dx_i_min;
        let gamma = g * derivative_range;
        info!("gamma {} for component j: {}", gamma, j);
        for i in 0..x_mesh.len() {
            if !mark.contains_key(&i) {
                mark.insert(i, 0); // default value 0 - no point inserted
            }

            if x_mesh.len() - 1 > i && i > 0 {
                let dy_i = y[i + 1] - y[i];
                let h_i = h[i];
                let dy_dx_i = dy_i / h_i;
                // dy_[i-1]/dx_[u-1]
                let dy_i_min_1 = y[i] - y[i - 1];
                let h_i_min_1 = h[i - 1];
                let dy_dx_i_min_1 = dy_i_min_1 / h_i_min_1;

                let eta_i = (dy_dx_i - dy_dx_i_min_1).abs();
                let tau_i = (y[i] - y[i - 1]).abs();

                let both_ys_are_not_too_small =
                    (y[i].abs() > threshold) & (y[i + 1].abs() > threshold);
                // eq 1, eq 2
                //
                if (tau_i > delta && both_ys_are_not_too_small)
                    || (eta_i > gamma && both_ys_are_not_too_small)
                {
                    //  info!("tau_i {}, criterion 1 {},  eta_i {}, criterion 2 {}, ctriterion 3 {}", tau_i, tau_i > delta && both_ys_are_not_too_small ,  eta_i, eta_i > gamma && both_ys_are_not_too_small,  both_ys_are_not_too_small);
                    info!(
                        "tau {:3} ({:3}, {:3})> delta {:3} for i {}",
                        tau_i,
                        y[i],
                        y[i - 1],
                        delta,
                        i
                    );
                    info!(
                        "eta {:3}  ({:3}, {:3})> gamma {:3} for i {}",
                        eta_i, dy_dx_i, dy_dx_i_min_1, gamma, i
                    );
                    // how many new points should be added
                    let N = if (tau_i / delta) as i32 >= 1 {
                        (tau_i / delta) as i32
                    } else {
                        1
                    };
                    info!(" conditions vaiolation at index {}, => N = {}", i, N);

                    // Only update if new N is larger than existing value
                    let current_N = *mark.get(&i).unwrap_or(&0);
                    if N > current_N {
                        mark.insert(i - 1, N);
                    }
                }
                // if not mark element remains 0
            } //i>0
            //for i==0 and i==-1 mark elements remain 0
        } // for i in 0..x_mesh.len()
        //  info!("mark {:?}", mark);
        // find keys corresponding to non-zero values in the HashMap
        let non_zero_keys: Vec<usize> = mark
            .iter()
            .filter(|&(_, &value)| value != 0)
            .map(|(key, _)| *key)
            .collect();
        info!(
            "\n \n for row {} found intervals to be refined: {:?} of length {} \n \n",
            j,
            non_zero_keys,
            non_zero_keys.len()
        );
    }

    // bufferisation to avoid rapid changes in grid inervals
    //the ratio of adjacent grid intervals must be bounded above and below by constants.
    for i in 1..x_mesh.len() - 1 {
        let buffer_condition_1 = h[i] / h[i - 1] <= C; // if true at i-1 index point should be added (between x_mesh[i] - x_mesh[i-1];)
        let buffer_condition_2 = h[i] / h[i - 1] >= 1.0 / C; // if true at i index point should be added  (between x_mesh[i+1] - x_mesh[i];
        if !buffer_condition_1 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] <= C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );
            if (i - 1) != 0 {
                let current_N = *mark.get(&(i - 1)).unwrap_or(&0);
                if 1 > current_N {
                    mark.insert(i - 1, 1);
                }
            }
        }
        if !buffer_condition_2 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] >= 1.0/C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );

            let current_N = *mark.get(&i).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
    }
    //  total number of new points to add
    let total_new_points: i32 = mark.values().sum();
    log::info!("total new points to add: {}", total_new_points);

    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);
    // println!("mark {:?}, x_mesh {} ", mark, x_mesh);
    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);
        if mark.get(&i).unwrap() != &0 {
            //
            // adding points to the grid
            let N = *mark.get(&i).unwrap();
            if i < x_mesh.len() - 1 {
                let h_i = x_mesh[i + 1] - x_mesh[i];
                for k in 0..N + 1 {
                    // k=0 refers to yhe element existing in the old mesh
                    let x_new = x_mesh[i] + h_i * (k as f64) / (N as f64 + 1.0);
                    if k != 0 {
                        log::info!("\n \n points added: {} at index {} ", x_new, i);
                    }
                    new_grid.push(x_new);
                }
                //
                // making interpolation between neighbor points of the previous step solution to form the new initial guess
                let y_pl_1 = y_DMatrix.column(i + 1);
                let dy_i = y_pl_1 - y;
                for k in 0..N + 1 {
                    // add new point to new_initial_guess
                    let column_to_add = y + &dy_i * (k as f64) / (N as f64 + 1.0);
                    if k != 0 {
                        log::info!(
                            "\n \n column added: {} at index {} ",
                            column_to_add,
                            biased_i
                        );
                    }
                    new_initial_guess
                        .column_mut(biased_i as usize)
                        .copy_from(&column_to_add);
                    biased_i += 1;
                }
            } else {
                // Last point - just copy
                new_grid.push(x_mesh[i]);
                new_initial_guess.column_mut(biased_i).copy_from(&y);
                biased_i += 1;
            }
        } else {
            // copy points from the old grid to the new one
            new_grid.push(x_mesh[i]);
            //
            // copy points from the previous step solution to the new guess
            new_initial_guess.column_mut(biased_i).copy_from(&y);
            biased_i += 1;
        } // add all points from the old grid to the new grid
    }

    log::info!("created new grid of length {}", new_grid.len());
    log::info!(
        "\n \n new_initial_guess of shape{:?}",
        new_initial_guess.shape()
    );

    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

/// Parallel version of grcar_smooke_grid_refinement - parallelizes the marking phase
pub fn grcar_smooke_grid_refinement_par(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    d: f64,
    g: f64,
    C: f64,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let mut h: Vec<f64> = Vec::new();
    for i in 0..x_mesh.len() - 1 {
        let h_i = x_mesh[i + 1] - x_mesh[i];
        h.push(h_i);
    }
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid: Vec<f64> = Vec::new();
    let mark = Mutex::new(HashMap::new());

    // Parallel marking phase
    y_DMatrix
        .clone()
        .row_iter()
        .enumerate()
        .par_bridge()
        .for_each(|(j, y)| {
            let threshold = 1e-4;
            let y_j_max = y.max();
            let y_j_min = y.min();
            let delta = d * (y_j_max - y_j_min);
            info!(
                "delta {} ({}, {}) for component j: {}",
                delta, y_j_max, y_j_min, j
            );
            let mut list_dy_dx_i = Vec::new();
            for i in 0..x_mesh.len() - 1 {
                let dy_i = y[i + 1] - y[i];
                let h_i = h[i];
                let dy_dx_i = dy_i / h_i;
                list_dy_dx_i.push(dy_dx_i);
            }
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
            let derivative_range = list_dy_dx_i_max - list_dy_dx_i_min;
            let gamma = g * derivative_range;
            info!("gamma {} for component j: {}", gamma, j);
            let mut local_marks = Vec::new();
            for i in 0..x_mesh.len() {
                if x_mesh.len() - 1 > i && i > 0 {
                    let dy_i = y[i + 1] - y[i];
                    let h_i = h[i];
                    let dy_dx_i = dy_i / h_i;
                    let dy_i_min_1 = y[i] - y[i - 1];
                    let h_i_min_1 = h[i - 1];
                    let dy_dx_i_min_1 = dy_i_min_1 / h_i_min_1;
                    let eta_i = (dy_dx_i - dy_dx_i_min_1).abs();
                    let tau_i = (y[i] - y[i - 1]).abs();
                    let both_ys_are_not_too_small =
                        (y[i].abs() > threshold) & (y[i + 1].abs() > threshold);

                    if (tau_i > delta && both_ys_are_not_too_small)
                        || (eta_i > gamma && both_ys_are_not_too_small)
                    {
                        info!(
                            "tau {:3} ({:3}, {:3})> delta {:3} for i {}",
                            tau_i,
                            y[i],
                            y[i - 1],
                            delta,
                            i
                        );
                        info!(
                            "eta {:3}  ({:3}, {:3})> gamma {:3} for i {}",
                            eta_i, dy_dx_i, dy_dx_i_min_1, gamma, i
                        );
                        let N = if (tau_i / delta) as i32 >= 1 {
                            (tau_i / delta) as i32
                        } else {
                            1
                        };
                        local_marks.push((i - 1, N));
                        info!(" conditions vaiolation at index {}, => N = {}", i, N);
                    }
                }
            }

            if !local_marks.is_empty() {
                let mut global_mark = mark.lock().unwrap();
                for (idx, N) in local_marks {
                    let current_N = *global_mark.get(&idx).unwrap_or(&0);
                    if N > current_N {
                        global_mark.insert(idx, N);
                    }
                }
            }
        });

    let mut mark = mark.into_inner().unwrap();

    // Sequential bufferization
    for i in 1..x_mesh.len() - 1 {
        let buffer_condition_1 = h[i] / h[i - 1] <= C;
        let buffer_condition_2 = h[i] / h[i - 1] >= 1.0 / C;
        if !buffer_condition_1 && (i - 1) != 0 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] <= C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );
            let current_N = *mark.get(&(i - 1)).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
        if !buffer_condition_2 {
            log::info!(
                "bufferization at index {} needed, as h[i]/h[i-1] >= 1.0/C, h[i], h[i-1] = {} , {}",
                i,
                h[i],
                h[i - 1]
            );
            let current_N = *mark.get(&i).unwrap_or(&0);
            if 1 > current_N {
                mark.insert(i - 1, 1);
            }
        }
    }

    let total_new_points: i32 = mark.values().sum();
    log::info!("total new points to add: {}", total_new_points);

    // Sequential grid construction
    let mut biased_i = 0;
    let mut new_initial_guess: DMatrix<f64> =
        DMatrix::zeros(n_rows, x_mesh.len() + total_new_points as usize);

    for i in 0..x_mesh.len() {
        let y = y_DMatrix.column(i);
        if mark.get(&i).unwrap_or(&0) != &0 {
            let N = *mark.get(&i).unwrap();
            if i < x_mesh.len() - 1 {
                let h_i = x_mesh[i + 1] - x_mesh[i];
                for k in 0..N + 1 {
                    let x_new = x_mesh[i] + h_i * (k as f64) / (N as f64 + 1.0);
                    log::info!("\n \n points added: {} at index {} ", x_new, i);
                    new_grid.push(x_new);
                }
                let y_pl_1 = y_DMatrix.column(i + 1);
                let dy_i = y_pl_1 - y;
                for k in 0..N + 1 {
                    let column_to_add = y + &dy_i * (k as f64) / (N as f64 + 1.0);
                    log::info!(
                        "\n \n column added: {} at index {} ",
                        column_to_add,
                        biased_i
                    );
                    new_initial_guess
                        .column_mut(biased_i as usize)
                        .copy_from(&column_to_add);
                    biased_i += 1;
                }
            } else {
                new_grid.push(x_mesh[i]);
                new_initial_guess.column_mut(biased_i).copy_from(&y);
                biased_i += 1;
            }
        } else {
            new_grid.push(x_mesh[i]);
            new_initial_guess.column_mut(biased_i).copy_from(&y);
            biased_i += 1;
        }
    }
    log::info!("created new grid of length {}", new_grid.len());
    log::info!(
        "\n \n new_initial_guess of shape{:?}",
        new_initial_guess.shape()
    );
    assert_eq!(new_initial_guess.len(), new_grid.len() * n_rows);
    (new_grid, new_initial_guess, total_new_points as usize)
}

/////////////////////////////////////////////////////////////////////////////////////////////
//            SCIPY (solve bvp) INSPIRED GRID  REFINEMENT  PROCEDURE
/////////////////////////////////////////////////////////////////////////////////////////////
pub fn scipy_grid_refinement(
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    tolerance: f64,
    residual: Option<DVector<f64>>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let nrows = y_DMatrix.nrows();
    let residuals = residual.expect("residual vector is required for scipy grid refinement");
    // Determine which intervals need refinement
    let mut insert_1 = Vec::new();
    let mut insert_2 = Vec::new();
    let m = residuals.len();
    for j in 0..(m - 1) {
        if residuals[j] > tolerance && residuals[j] < 100.0 * tolerance {
            insert_1.push(j);
        } else if residuals[j] >= 100.0 * tolerance {
            insert_2.push(j);
        }
    }

    let nodes_added = insert_1.len() + 2 * insert_2.len();
    info!("nodes to add {}", nodes_added);
    if nodes_added > 0 {
        let new_grid = modify_mesh(&x_mesh, &insert_1, &insert_2);
        let new_grid: Vec<f64> = new_grid.iter().cloned().collect();
        // Evaluate solution at new mesh points
        let new_initial_guess = interpolate_solution(
            &y_DMatrix,
            insert_1.as_slice(),
            insert_2.as_slice(),
            &new_grid,
        );
        assert_eq!(new_initial_guess.len(), new_grid.len() * nrows);
        (new_grid, new_initial_guess, nodes_added as usize)
    } else {
        info!("no refinement needed");
        let x_mesh: Vec<f64> = x_mesh.iter().cloned().collect();
        (x_mesh.clone(), y_DMatrix.clone(), 0_usize)
    }
}

pub fn modify_mesh(x: &DVector<f64>, insert_1: &[usize], insert_2: &[usize]) -> DVector<f64> {
    let mut new_points = x.iter().cloned().collect::<Vec<f64>>();

    // Insert 1 node in middle of intervals
    for &i in insert_1 {
        let mid = 0.5 * (x[i] + x[i + 1]);
        new_points.push(mid);
    }

    // Insert 2 nodes to divide interval into 3 parts
    for &i in insert_2 {
        let p1 = (2.0 * x[i] + x[i + 1]) / 3.0;
        let p2 = (x[i] + 2.0 * x[i + 1]) / 3.0;
        new_points.push(p1);
        new_points.push(p2);
    }

    new_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    DVector::from_vec(new_points)
}

pub fn interpolate_solution(
    y: &DMatrix<f64>,
    insert_1: &[usize],
    insert_2: &[usize],
    new_x: &[f64],
) -> DMatrix<f64> {
    let (n_rows, old_ncols) = y.shape();
    let mut new_y = DMatrix::zeros(n_rows, new_x.len());
    let mut col_idx = 0;

    // Create marking system like other algorithms
    let mut mark: HashMap<usize, i32> = HashMap::new();
    for &i in insert_1 {
        mark.insert(i, 1);
    }
    for &i in insert_2 {
        mark.insert(i, 2);
    }

    for i in 0..old_ncols {
        // Always copy original column
        new_y.column_mut(col_idx).copy_from(&y.column(i));
        col_idx += 1;

        if let Some(&N) = mark.get(&i) {
            if i < old_ncols - 1 {
                let y_curr = y.column(i);
                let y_next = y.column(i + 1);
                let dy = y_next - y_curr;

                for k in 1..=N {
                    let t = k as f64 / (N as f64 + 1.0);
                    let interpolated = y_curr + &dy * t;
                    info!("interpolated: {:?} for col {}", interpolated, col_idx);
                    new_y.column_mut(col_idx).copy_from(&interpolated);
                    col_idx += 1;
                }
            }
        }
    }

    new_y
}

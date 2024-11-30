#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
/*
For each grid point :
Calculate local error estimate εᵢ:
εᵢ = |h²ᵢ/12 * (y''ᵢ₊₁ - 2y''ᵢ + y''ᵢ₋₁)|
Given: tolerance TOL, safety factor σ (typically 0.8-0.9)
For each interval i:
    IF εᵢ > TOL THEN
        mark[i] = REFINE
    ELSE IF εᵢ < σ*TOL/2 THEN
        mark[i] = POSSIBLE_COARSEN
    ELSE
        mark[i] = KEEP
For each marked point i:
    IF mark[i] = REFINE THEN
        // Prevent too large jumps in mesh size
        mark[i-1] = max(mark[i-1], KEEP)
        mark[i+1] = max(mark[i+1], KEEP)
For each interval i:
    // Check gradient criterion
    IF |yᵢ₊₁ - yᵢ|/hᵢ > gradient_threshold THEN
        mark[i] = REFINE

    // Check curvature criterion
    IF |y''ᵢ| > curvature_threshold THEN
        mark[i] = REFINE
Define maximum refinement level MAX_LEVEL
For each interval i:
    IF mark[i] = REFINE AND level[i] < MAX_LEVEL THEN
        subdivide_interval(i)
    ELSE IF mark[i] = POSSIBLE_COARSEN AND
            level[i] > 1 AND
            can_coarsen_neighbors(i) THEN
        coarsen_interval(i)
Avoid creating too large differences in adjacent cell sizes
Typically limit ratio of adjacent cells to 2:1
// Create buffer zones around refinement regions
For each marked point i:
    IF mark[i] = REFINE THEN
        For j in range(i-buffer_size, i+buffer_size):
            mark[j] = max(mark[j], KEEP)
// Prevent too frequent refinement/coarsening
IF time_since_last_refinement < MIN_TIME THEN
    return
// Additional quality measures
For each interval i:
    IF solution_quality_measure(i) < threshold THEN
        mark[i] = REFINE
*/
//use crate::numerical::BVP_Damp::BVP_traits::{Fun, FunEnum, Jac, JacEnum, MatrixType, VectorType, Vectors_type_casting};
use nalgebra::{DMatrix, DVector, Matrix};

struct Grid {
    y_DMatrix: DMatrix<f64>,
    x_mesh: DVector<f64>,
    toler_0: f64,
    toler_1: f64,
    toler_2: f64,
    ratio1: DVector<f64>,
    vary1: DVector<f64>,
    ratio2: DVector<f64>,
    vary2: DVector<f64>,
    weights: DVector<f64>,
    mark: DVector<usize>,
    level: DVector<usize>,
}
impl Grid {
    fn grad(&self, y_j: DVector<f64>, i: usize) -> f64 {
        let x_vector = &self.x_mesh;
        let dy = y_j[i + 1] - y_j[i];
        let dx = x_vector[i + 1] - x_vector[i];
        let grad = dy / dx;
        grad
    }

    fn mark(&mut self) {
        //matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
        let y_DMatrix: &DMatrix<f64> = &self.y_DMatrix;
        let x_vector: &DVector<f64> = &self.x_mesh;
        // let x_vector:&DVector<f64> = &self.x_vector;
        let toler_0 = self.toler_0;
        let toler_1 = self.toler_1;
        let toler_2 = self.toler_2;
        let mut signif: usize = 0;
        let mut ratio1: DVector<f64> = DVector::zeros(y_DMatrix.ncols());
        let mut vary1: DVector<f64> = DVector::zeros(y_DMatrix.ncols());
        let mut ratio2: DVector<f64> = DVector::zeros(y_DMatrix.ncols());
        let mut vary2: DVector<f64> = DVector::zeros(y_DMatrix.ncols());
        for (j, y_j) in y_DMatrix.row_iter().enumerate() {
            let lower = y_j.min();
            let upper = y_j.max();
            let range_y_j = upper - lower;
            let maxmag_j = f64::max(lower.abs(), upper.abs());
            // decide whether the component is significant. Check if the absolute range of the component is greater than
            //a tolerance (toler0) times the maximum of 1 and maxmag. If not, skip the rest of the current iteration.
            if !(range_y_j.abs() > toler_0 * f64::max(1.0, maxmag_j)) {
                continue;
            }
            // this is a significant component.
            signif = signif + 1;

            // Calculate the difference (differ) between consecutive points.
            let mut list_dy_dx_i = Vec::new();
            for i in 0..y_j.len() - 1 {
                let dy_i = (y_j[i + 1] - y_j[i]).abs();
                let dx_i = x_vector[j + 1] - x_vector[j];
                let dy_dx_i = dy_i / dx_i;
                list_dy_dx_i.push(dy_dx_i);
                //Update ratio1(k) with the maximum of its current value and the ratio of differ to range.
                if 0.0 < range_y_j {
                    let ratio1_i = f64::max(ratio1[i], dy_i / range_y_j);
                    ratio1[i] = ratio1_i;
                }
                if toler_1 * range_y_j < dy_i {
                    vary1[i] = vary1[i] + 1.0;
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
            let derivative_range_i = (list_dy_dx_i_max - list_dy_dx_i_min).abs();

            //  Calculate the gradient at the first point and store it in right.
            //  Loop through the interior points, updating left and right with consecutive gradient values.
            //  Calculate the difference (differ) between consecutive gradients.
            // Update ratio2 with the maximum of its current value and the ratio of differ to range.
            // Increment vary2 if differ exceeds toler2 times range.
            let temp = self.grad(y_j.transpose(), 0);
            let right = temp;
            for i in 1..y_j.len() - 1 {
                let left = right;
                let right = self.grad(y_j.transpose(), i);
                let differ = (right - left).abs();
                if 0.0 < derivative_range_i {
                    let ratio2_i = f64::max(ratio2[i], differ / derivative_range_i);
                    ratio2[i] = ratio2_i
                }
                if toler_2 * derivative_range_i < differ {
                    vary2[i] = vary2[i] + 1.0;
                };
            }
        } //
        self.ratio1 = ratio1;
        self.vary1 = vary1;
        self.ratio2 = ratio2;
        self.vary2 = vary2;
    }
    fn refine_grid(&mut self) {
        let y_DMatrix: &DMatrix<f64> = &self.y_DMatrix;

        for (j, y_j) in y_DMatrix.row_iter().enumerate() {
            let mut most: usize = 0;
            let weights = &mut self.weights;
            for i in 0..y_j.len() - 1 {
                weights[i] = self.vary1[i];
                if 1 < i {
                    weights[i] = weights[i] + self.vary2[i]
                }
                if i < y_j.len() - 1 {
                    weights[i] = weights[i] + self.vary2[i + 1]
                }
                if 0.0 < weights[i] {
                    most = most + 1
                }
            }
            for i in 0..y_j.len() - 1 {
                for k in i + 1..y_j.len() - 1 {
                    if weights[k] > weights[i] {
                        let itemp = weights[k];
                        weights[k] = weights[i];
                        weights[i] = itemp;
                    }
                }
                if weights[i] == 0.0 {
                    break;
                }
            }
        } // for (i, row_i) in y_vector
    }
}

/*
pub fn truncation_error_vector(y_vector:DMatrix<f64>, x_vector:DVector<f64>) ->DVector<f64> {



}


fn adaptive_grid_refinement(problem: &BvpProblem, solution: &[f64], grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut new_solution = solution.to_vec();
    let mut new_grid = grid.to_vec();

    let tolerance = 1e-6; // Set your desired tolerance for refinement.

    loop {
        let mut refine_intervals = Vec::new();

        for i in 1..grid.len() - 1 {
            let residual = calculate_residual(problem, grid[i]);
            let max_residual = residual.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if max_residual > tolerance {
                refine_intervals.push(i);
            }
        }

        if refine_intervals.is_empty() {
            break;
        }

        // Add new points to the grid at the marked intervals, and interpolate the solution at these new points.
        for &interval in refine_intervals.iter().rev() {
            let midpoint = 0.5 * (grid[interval] + grid[interval + 1]);
            new_grid.insert(interval + 1, midpoint);

            // Interpolate the solution at the new point.
            let left_solution = solution[interval];
            let right_solution = solution[interval + 1];
            let new_solution_value = 0.5 * (left_solution + right_solution);
            new_solution.insert(interval + 1, new_solution_value);
        }
    }

    (new_solution, new_grid)
}
*/

/*Here's the detailed algorithm for marking points that need subdivision in adaptive grid refinement:

$\textbf{Algorithm for Marking Points for Subdivision:}$

$\textbf{1. Error Estimation:}$
For each grid point $i$:
```
Calculate local error estimate εᵢ:
εᵢ = |h²ᵢ/12 * (y''ᵢ₊₁ - 2y''ᵢ + y''ᵢ₋₁)|
```

$\textbf{2. Marking Strategy:}$
```
Given: tolerance TOL, safety factor σ (typically 0.8-0.9)
For each interval i:
    IF εᵢ > TOL THEN
        mark[i] = REFINE
    ELSE IF εᵢ < σ*TOL/2 THEN
        mark[i] = POSSIBLE_COARSEN
    ELSE
        mark[i] = KEEP
```

$\textbf{3. Smoothing the Marking:}$
```
For each marked point i:
    IF mark[i] = REFINE THEN
        // Prevent too large jumps in mesh size
        mark[i-1] = max(mark[i-1], KEEP)
        mark[i+1] = max(mark[i+1], KEEP)
```

$\textbf{4. Additional Criteria:}$
```
For each interval i:
    // Check gradient criterion
    IF |yᵢ₊₁ - yᵢ|/hᵢ > gradient_threshold THEN
        mark[i] = REFINE

    // Check curvature criterion
    IF |y''ᵢ| > curvature_threshold THEN
        mark[i] = REFINE
```

$\textbf{5. Practical Implementation:}$
```
Define maximum refinement level MAX_LEVEL
For each interval i:
    IF mark[i] = REFINE AND level[i] < MAX_LEVEL THEN
        subdivide_interval(i)
    ELSE IF mark[i] = POSSIBLE_COARSEN AND
            level[i] > 1 AND
            can_coarsen_neighbors(i) THEN
        coarsen_interval(i)
```

$\textbf{Important Considerations:}$

1. $\textbf{Balancing:}$
- Avoid creating too large differences in adjacent cell sizes
- Typically limit ratio of adjacent cells to 2:1

2. $\textbf{Buffer Zones:}$
```
// Create buffer zones around refinement regions
For each marked point i:
    IF mark[i] = REFINE THEN
        For j in range(i-buffer_size, i+buffer_size):
            mark[j] = max(mark[j], KEEP)
```

3. $\textbf{Efficiency Measures:}$
```
// Prevent too frequent refinement/coarsening
IF time_since_last_refinement < MIN_TIME THEN
    return
```

4. $\textbf{Solution Quality Checks:}$
```
// Additional quality measures
For each interval i:
    IF solution_quality_measure(i) < threshold THEN
        mark[i] = REFINE
```

Would you like me to generate a video explaining how this algorithm works in practice with specific examples? */

/*For boundary points, we need special treatment when computing local error estimates since we can't use the standard centered difference formula. Here are the main approaches:

$\textbf{1. One-Sided Differences:}$

For left boundary point $(i=0)$:
```
ε₀ = |h²/12 * (y''₁ - 2y''₀ + y''_ghost)|
```
where $y''_{ghost}$ is computed using boundary conditions

For right boundary point $(i=N)$:
```
εₙ = |h²/12 * (y''_ghost - 2y''ₙ + y''ₙ₋₁)|
```

$\textbf{2. Using Boundary Conditions:}$

If we have Dirichlet boundary conditions:
```
y(a) = α, y(b) = β
```
The error at boundaries can be estimated using:
```
ε₀ = |y₀ - α|
εₙ = |yₙ - β|
```

For Neumann boundary conditions $y'(a)=α$:
```
ε₀ = |h * (y₁ - y₀)/h - α|
```

$\textbf{3. Modified Truncation Error:}$

For left boundary:
```
ε₀ = |h²/6 * (y''₁ - y''₀)|
```

For right boundary:
```
εₙ = |h²/6 * (y''ₙ - y''ₙ₋₁)|
```

$\textbf{4. Extrapolation Method:}$

```
// For left boundary
y''_ghost = 2y''₀ - y''₁
ε₀ = |h²/12 * (y''₁ - 2y''₀ + y''_ghost)|

// For right boundary
y''_ghost = 2y''ₙ - y''ₙ₋₁
εₙ = |h²/12 * (y''_ghost - 2y''ₙ + y''ₙ₋₁)|
```

$\textbf{5. Conservative Approach:}$

Take maximum of nearby interior points:
```
ε₀ = max(ε₁, ε₂)
εₙ = max(εₙ₋₁, εₙ₋₂)
```

$\textbf{6. Combined Strategy:}$
```
// For boundary points
IF isDirichletBC THEN
    εboundary = |y - BC_value|
ELSE IF isNeumannBC THEN
    εboundary = |dy/dx - BC_value|
ELSE
    // Use one-sided difference
    εboundary = |h²/6 * (y''₁ - y''₀)|
END

// Apply safety factor
εboundary *= safety_factor
```

$\textbf{Important Considerations:}$

1. Consistency with boundary conditions
2. Maintaining accuracy order
3. Stability of the error estimate
4. Conservation properties

$\textbf{Implementation Example:}$
```
function compute_boundary_error(y, h, BC_type, BC_value):
    if BC_type == "Dirichlet":
        return abs(y[0] - BC_value)

    elif BC_type == "Neumann":
        return abs((y[1] - y[0])/h - BC_value)

    else:  // General case
        y_second_deriv = compute_second_derivative(y, h)
        return abs(h²/6 * (y_second_deriv[1] - y_second_deriv[0]))
```

$\textbf{Quality Check:}$
```
// Verify reliability of boundary error estimate
if boundary_error < interior_error_minimum/10:
    // Use conservative estimate
    boundary_error = interior_error_minimum
```

The choice of method depends on:
1. Type of boundary conditions
2. Required accuracy
3. Stability considerations
4. Computational efficiency needs */

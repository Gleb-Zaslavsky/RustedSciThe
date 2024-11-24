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
use nalgebra::{DMatrix, DVector, };

struct Grid {
    y_vector:DMatrix<f64>, 
    x_vector:DVector<f64>,
    toler0:f64,
    toler1:f64,
    toler2:f64,
    ratio1:DVector<f64>,
    vary1:DVector<f64>,
    ratio2:DVector<f64>,
    vary2:DVector<f64>,
    weights:DVector<f64>,
    mark:DVector<usize>,
    level: DVector<usize>,
}
impl Grid {
fn grad(&self, i:usize, j:usize) -> f64 {
    let y_vector:&DMatrix<f64> = &self.y_vector;
    let x_vector:&DVector<f64> = &self.x_vector;
    let row_i = y_vector.row(i);
    let dy = row_i[j+1]-row_i[j];
    let dx = x_vector[j+1]-x_vector[j];
    let grad = dy/dx;
    grad
}

// initialize lower and upper to the first value of the row.
// Loop through all points of the row to find the minimum (lower) and maximum (upper) values.
// Calculate the range (range) as the difference between upper and lower.
// Calculate the maximum magnitude (maxmag) as the maximum of the absolute values of lower and upper.
fn range(&mut self) {
//matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    let y_vector:&DMatrix<f64> = &self.y_vector;
   // let x_vector:&DVector<f64> = &self.x_vector;
    let toler0= self.toler0;
    let toler1= self.toler1;
    let toler2= self.toler2; 
    let mut signif:usize = 0;
    let mut ratio1:DVector<f64> = DVector::zeros(y_vector.ncols()); 
    let mut vary1:DVector<f64>  = DVector::zeros(y_vector.ncols()); 
    let mut ratio2:DVector<f64>  = DVector::zeros(y_vector.ncols()); 
    let mut vary2:DVector<f64>  = DVector::zeros(y_vector.ncols()); 
    for (i, row_i)  in y_vector.row_iter().enumerate() {
        let mut lower = row_i[0];
        let mut upper = row_i[0];
        for  element_j in row_i.iter() {
            lower=f64::min(lower, *element_j);
            upper=f64::max(upper, *element_j);
        }
        let range_i = upper - lower;
        let maxmag_i = f64::max(lower.abs(), upper.abs());
        // decide whether the component is significant. Check if the absolute range of the component is greater than 
        //a tolerance (toler0) times the maximum of 1 and maxmag. If not, skip the rest of the current iteration.
         if !(range_i > toler0*f64::max(1.0, maxmag_i)) {continue;}
        // this is a significant component.
        signif = signif + 1;

        // Calculate the difference (differ) between consecutive points.
        for  k in 0..row_i.len()-1 {
            let element_i = row_i[k];
            let element_i_plus_1 = row_i[k+1];
            let differ = (element_i_plus_1 - element_i).abs();
            //Update ratio1(k) with the maximum of its current value and the ratio of differ to range.
            if 0.0<range_i {
                let ratio1_k = f64::max(ratio1[k], differ/range_i);
                ratio1[k] = ratio1_k;
            }
            if toler1*range_i<differ {
                vary1[k] = vary1[k] + 1.0;
            } 

        }// end for k in 0..row_i.len()-1

        // Calculate the gradient (grad) of the component at the first point and initialize lower and upper with this value.
        // Loop through the interior points to find the minimum and maximum gradient values.
        // Calculate the range of the gradient.
        let temp = self.grad( i, 1);  
        let mut lower = temp;
        let mut upper = temp;
        for  k in 1..row_i.len()-1 {
            let temp = self.grad( i, k);
           let lower_ = f64::min(lower, temp);
           lower = lower_;
           let upper_ = f64::max(upper, temp);
           upper = upper_;
        }
        let range_i = upper - lower;//???

      //  Calculate the gradient at the first point and store it in right.
      //  Loop through the interior points, updating left and right with consecutive gradient values.
      //  Calculate the difference (differ) between consecutive gradients.
      // Update ratio2 with the maximum of its current value and the ratio of differ to range.
      // Increment vary2 if differ exceeds toler2 times range.
        let right = temp;
        for k in 1..row_i.len()-1 {
            let  left = right;
            let right = self.grad( i, k);
            let differ = (right - left).abs();
            if 0.0 < range_i { let ratio2_k = f64::max(ratio2[k], differ/range_i); 
                ratio2[k] = ratio2_k}
            if toler2*range_i < differ { vary2[k] = vary2[k] + 1.0; } ;
        }
        
    }//
    self.ratio1 = ratio1;
    self.vary1 = vary1;
    self.ratio2 = ratio2;
    self.vary2 = vary2;

}
fn refine_grid(&self){
    let y_vector:&DMatrix<f64> = &self.y_vector;
    for (i, row_i)  in y_vector.row_iter().enumerate() {
    let mut most:usize =0;
    let mut weight = DVector::zeros(y_vector.ncols()); 
    let n_points = row_i.len()-1;
    for k in 1..n_points {
        weight[k] = self.vary1[k];
        if 1<k {    weight[k] = weight[k] + self.vary2[k]}
        if k<n_points {  weight[k] = weight[k] + self.vary2[k+1] }
        if 0.0<weight[k] {  most =most+1}
    }


    }// for (i, row_i) in y_vector
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
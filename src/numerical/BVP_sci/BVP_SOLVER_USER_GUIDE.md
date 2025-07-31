# BVP Solver User Guide

This guide provides detailed instructions for using the Boundary Value Problem (BVP) solver in RustedSciThe with the `faer` linear algebra backend.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Defining the ODE System](#defining-the-ode-system)
3. [Setting Boundary Conditions](#setting-boundary-conditions)
4. [Initial Guess](#initial-guess)
5. [Solver Parameters](#solver-parameters)
6. [Complete Examples](#complete-examples)
7. [Advanced Features](#advanced-features)

## Basic Setup

First, import the necessary types and functions:

```rust
use crate::numerical::BVP_sci::BVP_sci_faer::solve_bvp;
use faer::col::Col;
use faer::mat::Mat;

type faer_col = Col<f64>;
type faer_dense_mat = Mat<f64>;
```

## Defining the ODE System

The ODE system is defined as a function that takes three parameters:
- `x`: mesh points (faer_col)
- `y`: solution values at mesh points (faer_dense_mat, shape: n×m)
- `p`: parameters (faer_col)

Returns: derivatives (faer_dense_mat, shape: n×m)

### Example: Second-order ODE y'' = -y

```rust
let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
    let mut f = faer_dense_mat::zeros(2, y.ncols());
    for j in 0..y.ncols() {
        *f.get_mut(0, j) = *y.get(1, j);  // y1' = y2
        *f.get_mut(1, j) = -*y.get(0, j); // y2' = -y1
    }
    f
};
```

### Matrix Layout Convention
- `y.get(i, j)`: component `i` of the solution at mesh point `j`
- `f.get_mut(i, j)`: derivative of component `i` at mesh point `j`

### Common ODE Patterns

#### Linear System with Constant Coefficients
```rust
// y'' + 2y' + y = 0 → [y1' = y2, y2' = -y1 - 2*y2]
let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
    let mut f = faer_dense_mat::zeros(2, y.ncols());
    for j in 0..y.ncols() {
        *f.get_mut(0, j) = *y.get(1, j);
        *f.get_mut(1, j) = -*y.get(0, j) - 2.0 * *y.get(1, j);
    }
    f
};
```

#### Nonlinear System
```rust
// y'' = y^3 - sin(x)
let fun = |x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
    let mut f = faer_dense_mat::zeros(2, y.ncols());
    for j in 0..y.ncols() {
        let y1 = *y.get(0, j);
        *f.get_mut(0, j) = *y.get(1, j);
        *f.get_mut(1, j) = y1 * y1 * y1 - x[j].sin();
    }
    f
};
```

#### Parametric System
```rust
// y'' + p[0]*y' + p[1]*y = 0
let fun = |_x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
    let mut f = faer_dense_mat::zeros(2, y.ncols());
    for j in 0..y.ncols() {
        *f.get_mut(0, j) = *y.get(1, j);
        *f.get_mut(1, j) = -p[1] * *y.get(0, j) - p[0] * *y.get(1, j);
    }
    f
};
```

## Setting Boundary Conditions

Boundary conditions are defined as a function that takes:
- `ya`: solution values at left boundary (faer_col)
- `yb`: solution values at right boundary (faer_col)  
- `p`: parameters (faer_col)

Returns: boundary condition residuals (faer_col, length: n + k)

### Common Boundary Condition Types

#### Dirichlet Conditions
```rust
// y(0) = 1, y(1) = 0
let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
    faer_col::from_fn(2, |i| match i {
        0 => ya[0] - 1.0,  // y(0) = 1
        1 => yb[0] - 0.0,  // y(1) = 0
        _ => 0.0,
    })
};
```

#### Mixed Conditions
```rust
// y(0) = 0, y'(1) = 1
let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
    faer_col::from_fn(2, |i| match i {
        0 => ya[0],        // y(0) = 0
        1 => yb[1] - 1.0,  // y'(1) = 1
        _ => 0.0,
    })
};
```

#### Periodic Conditions
```rust
// y(0) = y(1), y'(0) = y'(1)
let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
    faer_col::from_fn(2, |i| match i {
        0 => ya[0] - yb[0],  // y(0) = y(1)
        1 => ya[1] - yb[1],  // y'(0) = y'(1)
        _ => 0.0,
    })
};
```

#### Parameter Estimation
```rust
// y(0) = 0, y(1) = 1, ∫y dx = target_integral
let bc = |ya: &faer_col, yb: &faer_col, p: &faer_col| {
    faer_col::from_fn(3, |i| match i {
        0 => ya[0],                    // y(0) = 0
        1 => yb[0] - 1.0,             // y(1) = 1
        2 => p[0] - target_integral,   // parameter constraint
        _ => 0.0,
    })
};
```

## Initial Guess

The initial guess is a matrix of shape (n, m) where:
- n = number of equations
- m = number of mesh points

### Creating Initial Guess

#### Constant Values
```rust
let mut y = faer_dense_mat::zeros(2, 5);
for j in 0..5 {
    *y.get_mut(0, j) = 1.0;  // y1 = 1 everywhere
    *y.get_mut(1, j) = 0.0;  // y2 = 0 everywhere
}
```

#### Linear Interpolation
```rust
let x = faer_col::from_fn(5, |i| i as f64 * 0.25); // [0, 0.25, 0.5, 0.75, 1.0]
let mut y = faer_dense_mat::zeros(2, 5);
for j in 0..5 {
    *y.get_mut(0, j) = x[j];      // linear from 0 to 1
    *y.get_mut(1, j) = 1.0;       // constant derivative
}
```

#### Analytical Approximation
```rust
// For y'' = -y with y(0)=0, y(π)=0, use sin(x) as initial guess
let x = faer_col::from_fn(5, |i| i as f64 * std::f64::consts::PI / 4.0);
let mut y = faer_dense_mat::zeros(2, 5);
for j in 0..5 {
    *y.get_mut(0, j) = x[j].sin();
    *y.get_mut(1, j) = x[j].cos();
}
```

## Solver Parameters

### Basic Parameters

```rust
let result = solve_bvp(
    &fun,           // ODE function
    &bc,            // Boundary conditions
    x,              // Initial mesh
    y,              // Initial guess
    None,           // ODE Jacobian (optional)
    None,           // BC Jacobian (optional)
    None,           // Parameters (optional)
    None,           // Parameter bounds (optional)
    1e-6,           // BVP tolerance
    1000,           // Max iterations
    2,              // Max mesh refinements
    None,           // BC tolerance (optional)
);
```

### Parameter Descriptions

- **BVP tolerance** (`1e-6`): Controls accuracy of collocation residuals
- **Max iterations** (`1000`): Maximum Newton iterations per mesh
- **Max refinements** (`2`): Maximum number of adaptive mesh refinements
- **BC tolerance** (`None`): Boundary condition tolerance (defaults to BVP tolerance)

### Tolerance Guidelines

| Problem Type | BVP Tolerance | BC Tolerance |
|--------------|---------------|--------------|
| Simple linear | 1e-8 | 1e-10 |
| Nonlinear smooth | 1e-6 | 1e-8 |
| Stiff/oscillatory | 1e-4 | 1e-6 |
| Parameter estimation | 1e-5 | 1e-7 |

## Complete Examples

### Example 1: Simple Linear BVP

```rust
use crate::numerical::BVP_sci::BVP_sci_faer::solve_bvp;
use faer::col::Col;
use faer::mat::Mat;

type faer_col = Col<f64>;
type faer_dense_mat = Mat<f64>;

fn solve_linear_bvp() {
    // y'' = 0, y(0) = 0, y(1) = 1
    // Exact solution: y = x
    
    let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
        let mut f = faer_dense_mat::zeros(2, y.ncols());
        for j in 0..y.ncols() {
            *f.get_mut(0, j) = *y.get(1, j);  // y' = y2
            *f.get_mut(1, j) = 0.0;           // y2' = 0
        }
        f
    };

    let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
        faer_col::from_fn(2, |i| match i {
            0 => ya[0],        // y(0) = 0
            1 => yb[0] - 1.0,  // y(1) = 1
            _ => 0.0,
        })
    };

    // Initial mesh
    let x = faer_col::from_fn(3, |i| i as f64 * 0.5);
    
    // Initial guess
    let mut y = faer_dense_mat::zeros(2, 3);
    for j in 0..3 {
        *y.get_mut(0, j) = x[j];  // linear guess
        *y.get_mut(1, j) = 1.0;   // constant derivative
    }

    let result = solve_bvp(
        &fun, &bc, x, y, None, None, None, None,
        1e-8, 1000, 0, None
    );

    match result {
        Ok(res) => {
            if res.success {
                println!("Solution found!");
                for i in 0..res.x.nrows() {
                    println!("x = {:.3}, y = {:.6}", res.x[i], res.y.get(0, i));
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
```

### Example 2: Nonlinear BVP with Parameters

```rust
fn solve_parametric_bvp() {
    // y'' + p*y = 0, y(0) = 0, y(π) = 0, max(y) = 1
    
    let fun = |_x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
        let mut f = faer_dense_mat::zeros(2, y.ncols());
        for j in 0..y.ncols() {
            *f.get_mut(0, j) = *y.get(1, j);
            *f.get_mut(1, j) = -p[0] * *y.get(0, j);
        }
        f
    };

    let bc = |ya: &faer_col, yb: &faer_col, p: &faer_col| {
        faer_col::from_fn(3, |i| match i {
            0 => ya[0],                    // y(0) = 0
            1 => yb[0],                    // y(π) = 0
            2 => p[0] - 1.0,              // p = 1 (parameter constraint)
            _ => 0.0,
        })
    };

    let x = faer_col::from_fn(5, |i| i as f64 * std::f64::consts::PI / 4.0);
    
    let mut y = faer_dense_mat::zeros(2, 5);
    for j in 0..5 {
        *y.get_mut(0, j) = x[j].sin();
        *y.get_mut(1, j) = x[j].cos();
    }

    let p = faer_col::from_fn(1, |_| 1.0);  // Initial parameter guess

    let result = solve_bvp(
        &fun, &bc, x, y, None, None, Some(p), None,
        1e-6, 1000, 2, None
    );

    match result {
        Ok(res) => {
            if res.success {
                println!("Parameter found: p = {:.6}", res.p.as_ref().unwrap()[0]);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
```

## Advanced Features

### Custom Jacobians

For better performance and convergence, provide analytical Jacobians:

```rust
let fun_jac = |x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
    let m = x.nrows();
    let mut df_dy = Vec::new();
    let mut df_dp = Vec::new();
    
    for j in 0..m {
        // Jacobian w.r.t. y at each mesh point
        let mut jac_y = faer::sparse::SparseColMat::try_new_from_triplets(
            2, 2,
            &[
                faer::sparse::Triplet::new(0, 1, 1.0),  // ∂f1/∂y2 = 1
                faer::sparse::Triplet::new(1, 0, -p[0]), // ∂f2/∂y1 = -p
            ]
        ).unwrap();
        df_dy.push(jac_y);
        
        // Jacobian w.r.t. parameters
        let mut jac_p = faer::sparse::SparseColMat::try_new_from_triplets(
            2, 1,
            &[faer::sparse::Triplet::new(1, 0, -*y.get(0, j))] // ∂f2/∂p = -y1
        ).unwrap();
        df_dp.push(jac_p);
    }
    
    (df_dy, Some(df_dp))
};

let bc_jac = |ya: &faer_col, yb: &faer_col, p: &faer_col| {
    let dbc_dya = faer::sparse::SparseColMat::try_new_from_triplets(
        3, 2,
        &[faer::sparse::Triplet::new(0, 0, 1.0)] // ∂bc1/∂ya1 = 1
    ).unwrap();
    
    let dbc_dyb = faer::sparse::SparseColMat::try_new_from_triplets(
        3, 2,
        &[faer::sparse::Triplet::new(1, 0, 1.0)] // ∂bc2/∂yb1 = 1
    ).unwrap();
    
    let dbc_dp = faer::sparse::SparseColMat::try_new_from_triplets(
        3, 1,
        &[faer::sparse::Triplet::new(2, 0, 1.0)] // ∂bc3/∂p = 1
    ).unwrap();
    
    (dbc_dya, dbc_dyb, Some(dbc_dp))
};

// Use with solver
let result = solve_bvp(
    &fun, &bc, x, y, 
    Some(&fun_jac),  // ODE Jacobian
    Some(&bc_jac),   // BC Jacobian
    Some(p), None, 1e-6, 1000, 2, None
);
```

### Mesh Refinement Strategy

The solver automatically refines the mesh when:
- Collocation residuals are too large
- Solution changes rapidly between mesh points
- Maximum refinements not exceeded

Control refinement with the `max_refinements` parameter:
- `0`: No refinement (fixed mesh)
- `1-3`: Conservative refinement
- `4-10`: Aggressive refinement (may be slow)

### Error Handling

```rust
match solve_bvp(&fun, &bc, x, y, None, None, None, None, 1e-6, 1000, 2, None) {
    Ok(result) => {
        match result.status {
            0 => println!("Success: Solution converged"),
            1 => println!("Warning: Max refinements reached"),
            2 => println!("Error: Singular Jacobian"),
            3 => println!("Error: BC tolerance not met"),
            _ => println!("Error: Max iterations reached"),
        }
    }
    Err(e) => println!("Solver error: {}", e),
}
```

This guide covers the essential aspects of using the BVP solver. For more complex problems, consider starting with simpler initial guesses and gradually increasing accuracy requirements.
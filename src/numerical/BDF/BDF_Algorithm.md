# Backward Differentiation Formula (BDF) Algorithm

## Mathematical Foundation

### Problem Statement

We solve the initial value problem (IVP):
```
dy/dt = f(t, y),  y(t₀) = y₀
```

where `y ∈ ℝⁿ` and `f: ℝ × ℝⁿ → ℝⁿ`.

### BDF Method Theory

#### General BDF Formula

A k-step BDF method has the form:
```
Σ(i=0 to k) αᵢ yₙ₋ᵢ = h βₖ f(tₙ, yₙ)
```

Where:
- `yₙ` approximates `y(tₙ)`
- `h = tₙ - tₙ₋₁` is the step size
- `αᵢ` are the BDF coefficients
- `βₖ = 1` for all BDF methods (making them implicit)

#### BDF Coefficients for Different Orders

| Order | α₀ | α₁ | α₂ | α₃ | α₄ | α₅ | Local Truncation Error |
|-------|----|----|----|----|----|----|----------------------|
| 1     | 1  | -1 | 0  | 0  | 0  | 0  | O(h²)               |
| 2     | 3/2| -2 | 1/2| 0  | 0  | 0  | O(h³)               |
| 3     | 11/6|-3 | 3/2|-1/3| 0  | 0  | O(h⁴)               |
| 4     | 25/12|-4| 3  |-4/3|1/4| 0  | O(h⁵)               |
| 5     | 137/60|-5|5  |-10/3|5/4|-1/5| O(h⁶)               |

#### Stability Properties

- **Orders 1-2**: A-stable (unconditionally stable)
- **Orders 3-5**: A(α)-stable with decreasing stability regions
- **Order 6+**: Unstable (not recommended)

### Nordsieck Form Implementation

#### Scaled Derivatives Representation

Instead of storing previous solution values, we use the Nordsieck form with scaled derivatives:

```
D = [y, h·y', h²·y''/2!, h³·y'''/3!, ..., hᵏ·y⁽ᵏ⁾/k!]ᵀ
```

#### Advantages of Nordsieck Form

1. **Efficient order changes**: No need to recompute coefficients
2. **Step size changes**: Simple scaling operations
3. **Memory efficient**: Fixed storage regardless of order
4. **Numerical stability**: Better conditioning

### Newton-Raphson Solution

#### Nonlinear System

At each step, solve:
```
G(y) = y - (h/α₀)f(t, y) - ψ = 0
```

Where `ψ` contains contributions from previous values:
```
ψ = (1/α₀) Σ(i=1 to k) αᵢ yₙ₋ᵢ
```

#### Newton Iteration

```
yᵐ⁺¹ = yᵐ - [I - (h/α₀)J]⁻¹ G(yᵐ)
```

Where `J = ∂f/∂y` is the Jacobian matrix.

#### Convergence Criteria

Stop when either:
1. `||Δy|| < tol` (solution converged)
2. `rate/(1-rate) ||Δy|| < tol` (predicted convergence)
3. `rate ≥ 1` or maximum iterations reached (divergence)

## Algorithm Pseudocode

### Main BDF Step

```pseudocode
function BDF_STEP(t, y, h, order, D, alpha, gamma, error_const):
    // 1. Prediction Phase
    y_predict = sum(D[0:order+1])  // Nordsieck prediction
    
    // 2. Correction Phase (Newton-Raphson)
    psi = dot(D[1:order+1], gamma[1:order+1]) / alpha[order]
    c = h / alpha[order]
    
    converged = false
    iteration = 0
    
    while not converged and iteration < MAX_ITER:
        f_val = f(t + h, y_predict)
        
        if LU_factorization is None:
            J = jacobian(t + h, y_predict)
            LU = factorize(I - c * J)
        
        residual = c * f_val - psi - d
        dy = solve(LU, residual)
        
        y_predict += dy
        d += dy
        
        // Check convergence
        dy_norm = norm(dy / scale)
        if dy_norm < tolerance:
            converged = true
        
        iteration += 1
    
    // 3. Error Estimation
    error = error_const[order] * d
    error_norm = norm(error / scale)
    
    // 4. Step Acceptance
    if error_norm <= 1.0:
        accept_step = true
        update_nordsieck_array(D, d, order)
    else:
        accept_step = false
        reduce_step_size(h, error_norm, order)
    
    return accept_step, y_predict, error_norm
```

### Order Selection Algorithm

```pseudocode
function SELECT_ORDER(error_norms, current_order):
    // error_norms = [error_m, error_current, error_p]
    // Corresponding to orders [k-1, k, k+1]
    
    factors = []
    for i in range(3):
        order_candidate = current_order - 1 + i
        if order_candidate >= 1 and order_candidate <= MAX_ORDER:
            factor = error_norms[i]^(-1/(order_candidate + 1))
            factors.append(factor)
        else:
            factors.append(0)  // Invalid order
    
    best_index = argmax(factors)
    delta_order = best_index - 1  // Can be -1, 0, or +1
    new_order = clamp(current_order + delta_order, 1, MAX_ORDER)
    
    return new_order, factors[best_index]
```

### Step Size Control

```pseudocode
function CONTROL_STEP_SIZE(error_norm, order, safety_factor):
    if error_norm > 1.0:
        // Reject step, reduce step size
        factor = max(MIN_FACTOR, safety_factor * error_norm^(-1/(order+1)))
        new_h = h * factor
        return false, new_h
    else:
        // Accept step, possibly increase step size
        factor = min(MAX_FACTOR, safety_factor * error_norm^(-1/(order+1)))
        new_h = h * factor
        return true, new_h
```

### Nordsieck Array Update

```pseudocode
function UPDATE_NORDSIECK(D, d, order):
    // Update differences after successful step
    D[order + 2] = d - D[order + 1]  // New highest difference
    D[order + 1] = d                 // Correction becomes new difference
    
    // Propagate differences upward
    for i in range(order, -1, -1):
        D[i] += D[i + 1]
    
    return D
```

## Implementation Details

### Data Structures

#### BDF Struct Fields

```rust
pub struct BDF {
    // Problem definition
    fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    
    // Current state
    t: f64,                    // Current time
    y: DVector<f64>,          // Current solution
    h_abs: f64,               // Current step size
    order: usize,             // Current order
    
    // Nordsieck array
    D: DMatrix<f64>,          // Scaled derivatives matrix
    
    // Method coefficients
    alpha: DVector<f64>,      // BDF coefficients
    gamma: DVector<f64>,      // Integration coefficients  
    error_const: DVector<f64>, // Error estimation coefficients
    
    // Linear algebra
    LU: Option<LU<f64, Dyn, Dyn>>, // Cached LU factorization
    
    // Control parameters
    rtol: NumberOrVec,        // Relative tolerance
    atol: NumberOrVec,        // Absolute tolerance
    max_step: f64,           // Maximum step size
    
    // Statistics
    nfev: usize,             // Function evaluations
    njev: usize,             // Jacobian evaluations
    nlu: usize,              // LU factorizations
}
```

#### Coefficient Initialization

```rust
// Kappa coefficients for modified Newton iteration
let kappa = vec![0.0, -0.1850, -1/9, -0.0823, -0.0415, 0.0];

// Gamma coefficients: γₖ = Σ(i=1 to k) 1/i
let gamma = {
    let mut g = vec![0.0];
    let mut cumsum = 0.0;
    for i in 1..=MAX_ORDER {
        cumsum += 1.0 / (i as f64);
        g.push(cumsum);
    }
    g
};

// Alpha coefficients: αₖ = (1 - κₖ) * γₖ  
let alpha = (ones - kappa).component_mul(&gamma);

// Error constants for local error estimation
let error_const = kappa.component_mul(&gamma) + 1/(1..=MAX_ORDER+1);
```

### Error Control Strategy

#### Weighted RMS Norm

```rust
fn weighted_norm(error: &DVector<f64>, scale: &DVector<f64>) -> f64 {
    let n = error.len() as f64;
    let sum_squares: f64 = error.iter().zip(scale.iter())
        .map(|(e, s)| (e / s).powi(2))
        .sum();
    (sum_squares / n).sqrt()
}
```

#### Scale Vector Computation

```rust
fn compute_scale(rtol: f64, atol: f64, y: &DVector<f64>) -> DVector<f64> {
    y.map(|yi| atol + rtol * yi.abs())
}
```

### Performance Optimizations

#### Jacobian Reuse Strategy

1. **Initial**: Compute Jacobian at step start
2. **Reuse**: Keep same Jacobian while Newton converges quickly
3. **Update**: Recompute if convergence slows or fails
4. **Caching**: Store LU factorization to avoid repeated decomposition

#### Step Size Prediction

```rust
fn predict_step_size(error_norm: f64, order: usize, safety: f64) -> f64 {
    if error_norm == 0.0 {
        MAX_FACTOR
    } else {
        safety * error_norm.powf(-1.0 / (order as f64 + 1.0))
    }
}
```

#### Order Selection Heuristics

- **Conservative**: Only change order when clearly beneficial
- **Stability**: Prefer lower orders for better stability
- **Efficiency**: Higher orders for smooth problems

## Convergence and Stability Analysis

### Local Truncation Error

For a k-step BDF method, the local truncation error is:
```
LTE = Cₖ₊₁ hᵏ⁺¹ y⁽ᵏ⁺¹⁾(tₙ) + O(hᵏ⁺²)
```

Where `Cₖ₊₁` is the error constant.

### Global Error Bound

Under standard assumptions (Lipschitz continuity, bounded derivatives):
```
||yₙ - y(tₙ)|| ≤ C h^k
```

for some constant `C` independent of `h`.

### Stiffness Ratio

For stiff problems with eigenvalues `λᵢ`, the stiffness ratio is:
```
S = max|Re(λᵢ)| / min|Re(λᵢ)|
```

BDF methods remain stable for large `S`, unlike explicit methods.

## Usage Examples

### Basic Usage

```rust
let mut solver = BDF::new();
solver.set_initial(
    Box::new(|t, y| /* ODE function */),
    0.0,                    // t0
    DVector::from_vec(vec![1.0]), // y0
    10.0,                   // t_bound
    0.01,                   // max_step
    NumberOrVec::Number(1e-6), // rtol
    NumberOrVec::Number(1e-9), // atol
    Some(Box::new(|t, y| /* Jacobian */)), // Optional Jacobian
    None,                   // jac_sparsity
    false,                  // vectorized
    None,                   // first_step
);

while solver.t < solver.t_bound {
    let (success, message) = solver._step_impl();
    if !success {
        panic!("Integration failed: {:?}", message);
    }
}
```

### With Analytical Jacobian

```rust
// For system: y' = Ay where A is a matrix
let A = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 0.0, -2.0]);

let ode_fn = Box::new(move |_t: f64, y: &DVector<f64>| &A * y);
let jac_fn = Box::new(move |_t: f64, _y: &DVector<f64>| A.clone());

solver.set_initial(ode_fn, 0.0, y0, 1.0, 0.1, rtol, atol, 
                  Some(jac_fn), None, false, None);
```

## References and Further Reading

1. **Byrne, G.D., Hindmarsh, A.C.** (1975). "A Polyalgorithm for the Numerical Solution of Ordinary Differential Equations". *ACM Transactions on Mathematical Software*, 1(1), 71-96.

2. **Shampine, L.F., Reichelt, M.W.** (1997). "The MATLAB ODE Suite". *SIAM Journal on Scientific Computing*, 18(1), 1-22.

3. **Hairer, E., Wanner, G.** (2010). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer.

4. **Brenan, K.E., Campbell, S.L., Petzold, L.R.** (1996). *Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations*. SIAM.

5. **Ascher, U.M., Petzold, L.R.** (1998). *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*. SIAM.
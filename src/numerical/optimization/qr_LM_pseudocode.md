// PIVOTED QR FACTORIZATION AND LINEAR LEAST SQUARES SOLVER
// FOR LEVENBERG-MARQUARDT ALGORITHM

// ============================================================================
// STRUCTURE DEFINITIONS
// ============================================================================

STRUCT PivotedQR:
    column_norms: Vector[f64]     // Original column norms of input matrix A
    qr: Matrix[f64]              // Combined storage: upper part of R + Householder vectors
    r_diag: Vector[f64]          // Diagonal entries of R matrix
    permutation: Vector[usize]    // Column permutation indices
    work: Vector[f64]            // Working vector for computations

STRUCT LinearLeastSquaresDiagonalProblem:
    qt_b: Vector[f64]            // First n entries of Q^T * b
    upper_r: Matrix[f64]         // Upper part of R, later used for L storage
    l_diag: Vector[f64]          // Diagonal entries of Cholesky factor L
    permutation: Vector[usize]    // Column permutation indices
    column_norms: Vector[f64]    // Original column norms
    work: Vector[f64]            // Working vector
    m: usize                     // Number of rows in original matrix

STRUCT CholeskyFactor:
    permutation: &Vector[usize]   // Reference to permutation
    l: &Matrix[f64]              // Reference to Cholesky factor matrix
    work: &mut Vector[f64]       // Mutable reference to working vector
    qt_b: &Vector[f64]           // Reference to Q^T * b
    lower: bool                  // Flag: true for lower triangular, false for upper
    l_diag: &Vector[f64]         // Reference to diagonal entries

// ============================================================================
// PIVOTED QR FACTORIZATION ALGORITHM (with Housholder Redlection)
// ============================================================================
 1)   Mathematical considerations.
https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections

 2) Simplified pseudocode
 def qr_householder_pivot(A):
    m, n = A.shape
    Q = eye(m)          # Initialize Q as identity
    R = A.copy()        # Initialize R as a copy of A
    P = eye(n)          # Initialize permutation matrix
    
    # Track column norms (squared for efficiency)
    col_norms = [norm(R[:, j])**2 for j in range(n)]
    
    for k in range(n):
        # --- Column Pivoting ---
        # Before processing column k, find the column pivot_col (from k to n-1) with the largest remaining nor
        pivot_col = argmax(col_norms[k:]) + k
        
        # Swap columns k and pivot_col in R and P
        R[:, [k, pivot_col]] = R[:, [pivot_col, k]]
        P[:, [k, pivot_col]] = P[:, [pivot_col, k]]
        col_norms[[k, pivot_col]] = col_norms[[pivot_col, k]]
        
        # --- Householder Transformation ---
        x = R[k:m, k]                     # Column k (from row k to end)
        norm_x = norm(x)
        sign = 1 if x[0] >= 0 else -1     # Avoid cancellation
        v = x.copy()
        v[0] += sign * norm_x              # Householder vector
        v = v / norm(v)                   # Normalize
        
        # --- Apply H to R ---
        # Apply H = I - 2vv^T to R[k:m, k:n] to triangularize.
        R[k:m, k:n] -= 2 * outer(v, v.T @ R[k:m, k:n])
        
        # --- Accumulate Q ---
        # Update Q by multiplying with H (since Q = H₁ H₂ ... Hₙ).
        # Q = Q * H = Q * (I - 2vv^T)
        Q[:, k:m] -= 2 * (Q[:, k:m] @ outer(v, v))
        
        # --- Update column norms (for pivoting in next steps) ---
        if k < n - 1:
            col_norms[k+1:n] -= R[k, k+1:n]**2  # Subtract squared row k
    
    return Q, R, P   
//=============================================================================
3) Detailed pseudocode
ALGORITHM PivotedQR_new(A: Matrix[f64]) -> PivotedQR:
    INPUT: A ∈ ℝ^(m×n) - matrix to decompose
    OUTPUT: Pivoted QR decomposition where P^T * A * P = Q * R
    
    (m, n) ← shape(A)
    
    // STEP 1: Initialize column norms and permutation
    column_norms ← Vector[n]
    FOR j = 0 TO n-1:
        column_norms[j] ← ||A[:,j]||₂  // Euclidean norm of column j
    
    r_diag ← copy(column_norms)
    work ← copy(column_norms)
    permutation ← [0, 1, 2, ..., n-1]  // Identity permutation initially
    
    // STEP 2: Main QR factorization loop with pivoting
    FOR j = 0 TO min(m,n)-1:
        
        // SUBSTEP 2a: Column pivoting
        // Find column with maximum norm among remaining columns
        kmax ← argmax(r_diag[j:n]) + j
        
        IF kmax ≠ j:
            swap_columns(A, j, kmax)
            swap(permutation[j], permutation[kmax])
            r_diag[kmax] ← r_diag[j]
            work[kmax] ← work[j]
        
        // SUBSTEP 2b: Compute Householder reflection
        // Get submatrix A[j:m, j:n]
        axis ← A[j:m, j]  // j-th column from row j onwards
        aj_norm ← ||axis||₂
        
        IF aj_norm = 0:
            r_diag[j] ← 0
            CONTINUE
        
        // Determine sign for numerical stability
        IF axis[0] < 0:
            aj_norm ← -aj_norm
        
        r_diag[j] ← -aj_norm
        
        // Compute Householder vector w_j
        axis ← axis / aj_norm
        axis[0] ← axis[0] + 1
        // Now axis contains the Householder vector
        
        // SUBSTEP 2c: Apply Householder reflection to remaining columns
        FOR k = j+1 TO n-1:
            col ← A[j:m, k]
            
            // Apply reflection: col ← col - 2 * (w_j^T * col / w_j^T * w_j) * w_j
            // Since w_j[0] = axis[0], we use: col ← col - (w_j^T * col / w_j[0]) * w_j
            dot_product ← dot(col, axis)
            col ← col - (dot_product / axis[0]) * axis
            
            // SUBSTEP 2d: Update partial column norms (Lapack Working Note 176)
            IF r_diag[k] = 0:
                CONTINUE
            
            // Fast update formula
            temp ← (col[0] / r_diag[k])²
            r_diag[k] ← r_diag[k] * √(max(0, 1 - temp))
            
            // Check if fast update is reliable
            z05 ← 0.05
            IF z05 * (r_diag[k] / work[k])² ≤ machine_epsilon:
                // Recompute norm exactly
                r_diag[k] ← ||col[1:end]||₂
                work[k] ← r_diag[k]
    
    RETURN PivotedQR{column_norms, A, permutation, r_diag, work}

// ============================================================================
// CONVERSION TO LEAST SQUARES PROBLEM
// ============================================================================

// ============================================================================
3) Detailed pseudocode
ALGORITHM into_least_squares_diagonal_problem(qr: PivotedQR, b: Vector[f64]) 
         -> LinearLeastSquaresDiagonalProblem:
    INPUT: qr - Pivoted QR decomposition, b - right-hand side vector
    OUTPUT: Parametrized least squares problem structure
    
    (m, n) ← shape(qr.qr)
    qt_b ← Vector[n] initialized to zero
    
    // STEP 1: Compute first n entries of Q^T * b
    FOR j = 0 TO min(m,n)-1:
        axis ← qr.qr[j:m, j]  // Householder vector
        
        IF axis[0] ≠ 0:
            // Apply Householder transformation to b
            temp ← -dot(b[j:m], axis) / axis[0]
            b[j:m] ← b[j:m] + temp * axis
        
        IF j < n:
            qt_b[j] ← b[j]
    
    // STEP 2: Restore diagonal of R matrix
    FOR j = 0 TO min(m,n)-1:
        IF j < length(qr.r_diag):
            qr.qr[j,j] ← qr.r_diag[j]
    
    // STEP 3: Resize matrix if needed
    upper_r ← resize(qr.qr, max(m,n), n)
    
    RETURN LinearLeastSquaresDiagonalProblem{
        qt_b, qr.column_norms, upper_r, qr.r_diag, qr.permutation, qr.work, m
    }

// ============================================================================
// DIAGONAL ELIMINATION ALGORITHM (CORE OF LM TRUST REGION)
// ============================================================================

ALGORITHM eliminate_diag(lls: LinearLeastSquaresDiagonalProblem, 
                        diag: Vector[f64], rhs: Vector[f64]) -> Vector[f64]:
    INPUT: lls - least squares problem, diag - diagonal matrix D, rhs - right-hand side
    OUTPUT: Modified right-hand side after eliminating diagonal
    PURPOSE: Solve min ||[A; D] * x - [b; 0]||² by eliminating D using Givens rotations
    
    n ← number of columns in lls.upper_r
    
    // STEP 1: Prepare R^T in lower triangle (will be overwritten with L)
    r_and_l ← lls.upper_r[0:n, 0:n]
    fill_lower_triangle_with_upper_triangle(r_and_l)
    
    // Save diagonal of R for later restoration
    FOR j = 0 TO n-1:
        lls.work[j] ← r_and_l[j,j]
    
    // STEP 2: Eliminate diagonal entries using Givens rotations
    p5 ← 0.5
    p25 ← 0.25
    
    FOR j = 0 TO n-1:
        // Get diagonal entry (with permutation)
        diag_entry ← diag[lls.permutation[j]] IF j < length(lls.permutation) ELSE 0
        
        IF diag_entry ≠ 0:
            lls.l_diag[j] ← diag_entry
            FOR i = j+1 TO n-1:
                lls.l_diag[i] ← 0
            
            qtbpj ← 0
            
            // STEP 3: Apply sequence of Givens rotations
            FOR k = j TO n-1:
                IF lls.l_diag[k] ≠ 0:
                    r_kk ← r_and_l[k,k]
                    
                    // Compute Givens rotation parameters
                    IF |r_kk| < |lls.l_diag[k]|:
                        cot ← r_kk / lls.l_diag[k]
                        sin ← p5 / √(p25 + p25 * cot²)
                        cos ← sin * cot
                    ELSE:
                        tan ← lls.l_diag[k] / r_kk
                        cos ← p5 / √(p25 + p25 * tan²)
                        sin ← cos * tan
                    
                    // Apply rotation to diagonal element and RHS
                    r_and_l[k,k] ← cos * r_kk + sin * lls.l_diag[k]
                    temp ← cos * rhs[k] + sin * qtbpj
                    qtbpj ← -sin * rhs[k] + cos * qtbpj
                    rhs[k] ← temp
                    
                    // Accumulate transformation in row of L
                    FOR i = k+1 TO n-1:
                        r_ik ← r_and_l[i,k]
                        temp ← cos * r_ik + sin * lls.l_diag[i]
                        lls.l_diag[i] ← -sin * r_ik + cos * lls.l_diag[i]
                        r_and_l[i,k] ← temp
        
        // Store diagonal and restore R diagonal
        lls.l_diag[j] ← r_and_l[j,j]
        r_and_l[j,j] ← lls.work[j]
    
    RETURN rhs

// ============================================================================
// SOLVE AFTER ELIMINATION
// ============================================================================

ALGORITHM solve_after_elimination(lls: LinearLeastSquaresDiagonalProblem, 
                                 x: Vector[f64]) -> (Vector[f64], CholeskyFactor):
    INPUT: lls - problem after diagonal elimination, x - solution vector
    OUTPUT: Solution and Cholesky factor for further operations
    
    rank ← position of first zero in lls.l_diag OR length(lls.l_diag)
    rhs ← lls.work
    
    // STEP 1: Set remaining elements to zero
    FOR i = rank TO length(rhs)-1:
        rhs[i] ← 0
    
    n ← number of columns in lls.upper_r
    l ← lls.upper_r[0:n, 0:n]
    
    // STEP 2: Solve L^T * y = rhs (back substitution)
    FOR j = rank-1 DOWN TO 0:
        dot_product ← 0
        FOR i = j+1 TO rank-1:
            dot_product ← dot_product + l[i,j] * rhs[i]
        
        IF lls.l_diag[j] ≠ 0:
            rhs[j] ← (rhs[j] - dot_product) / lls.l_diag[j]
    
    // STEP 3: Apply inverse permutation
    FOR j = 0 TO n-1:
        x[lls.permutation[j]] ← rhs[j]
    
    cholesky_factor ← CholeskyFactor{
        l: lls.upper_r, work: lls.work, permutation: lls.permutation,
        qt_b: lls.qt_b, lower: true, l_diag: lls.l_diag
    }
    
    RETURN (x, cholesky_factor)

// ============================================================================
// SOLVE WITH DIAGONAL MATRIX
// ============================================================================

ALGORITHM solve_with_diagonal(lls: LinearLeastSquaresDiagonalProblem, 
                             diag: Vector[f64], out: Vector[f64]) 
                             -> (Vector[f64], CholeskyFactor):
    INPUT: lls - least squares problem, diag - diagonal matrix D, out - output vector
    OUTPUT: Solution to min ||[A; D] * x - [b; 0]||²
    PURPOSE: Main solver for LM trust region subproblem
    
    out ← copy(lls.qt_b)
    rhs ← eliminate_diag(lls, diag, out)
    swap(lls.work, rhs)
    RETURN solve_after_elimination(lls, rhs)

// ============================================================================
// SOLVE WITH ZERO DIAGONAL (GAUSS-NEWTON STEP) - COMPLETE
// ============================================================================

ALGORITHM solve_with_zero_diagonal(lls: LinearLeastSquaresDiagonalProblem) 
                                  -> (Vector[f64], CholeskyFactor):
    INPUT: lls - least squares problem
    OUTPUT: Solution to min ||A * x - b||² (standard least squares)
    PURPOSE: Compute Gauss-Newton step when trust region is inactive
    
    n ← number of columns in lls.upper_r
    l ← lls.upper_r[0:n, 0:n]
    lls.work ← copy(lls.qt_b)
    rank ← r_rank(lls)  // Find rank of R matrix
    
    // STEP 1: Set elements beyond rank to zero
    FOR i = rank TO length(lls.work)-1:
        lls.work[i] ← 0
    
    // STEP 2: Solve upper triangular system R * x = Q^T * b (back substitution)
    FOR i = rank-1 DOWN TO 0:
        sum ← 0
        FOR j = i+1 TO rank-1:
            IF i < l.nrows AND j < l.ncols:
                sum ← sum + l[i,j] * lls.work[j]
        
        IF i < l.nrows AND i < l.ncols AND l[i,i] ≠ 0:
            lls.work[i] ← (lls.work[i] - sum) / l[i,i]
    
    // STEP 3: Apply inverse permutation P^T
    x ← Vector[n] initialized to zero
    FOR j = 0 TO n-1:
        IF j < length(lls.permutation) AND lls.permutation[j] < length(x):
            x[lls.permutation[j]] ← lls.work[j]
    
    // STEP 4: Create Cholesky factor for further operations
    chol ← CholeskyFactor{
        permutation: lls.permutation,
        l: lls.upper_r,
        work: lls.work,
        qt_b: lls.qt_b,
        lower: false,  // Upper triangular for this case
        l_diag: lls.l_diag
    }
    
    RETURN (x, chol)

// ============================================================================
// SOLVE AFTER ELIMINATION - COMPLETE
// ============================================================================

ALGORITHM solve_after_elimination(lls: LinearLeastSquaresDiagonalProblem, 
                                 x: Vector[f64]) -> (Vector[f64], CholeskyFactor):
    INPUT: lls - problem after diagonal elimination, x - solution vector
    OUTPUT: Solution and Cholesky factor for further operations
    PURPOSE: Complete the solution after Givens rotations have eliminated diagonal
    
    rank ← rank(lls)  // Find rank of L matrix
    rhs ← lls.work
    
    // STEP 1: Fill remaining elements with zero beyond rank
    FOR i = rank TO length(rhs)-1:
        rhs[i] ← 0
    
    n ← number of columns in lls.upper_r
    l ← lls.upper_r[0:n, 0:n]
    
    // STEP 2: Solve L^T * y = rhs (back substitution for lower triangular transpose)
    FOR j = rank-1 DOWN TO 0:
        dot_product ← 0
        
        // Compute dot product with already solved components
        FOR i = j+1 TO rank-1:
            IF i < l.nrows AND j < l.ncols:
                dot_product ← dot_product + l[i,j] * rhs[i]
        
        // Solve for current component
        IF j < length(lls.l_diag) AND lls.l_diag[j] ≠ 0:
            rhs[j] ← (rhs[j] - dot_product) / lls.l_diag[j]
    
    // STEP 3: Apply inverse permutation P^T to get final solution
    FOR j = 0 TO n-1:
        IF (j < length(lls.permutation) AND 
            lls.permutation[j] < length(x) AND 
            j < length(rhs)):
            x[lls.permutation[j]] ← rhs[j]
    
    // STEP 4: Create Cholesky factor structure for subsequent operations
    cholesky_factor ← CholeskyFactor{
        l: lls.upper_r,
        work: lls.work,
        permutation: lls.permutation,
        qt_b: lls.qt_b,
        lower: true,  // Lower triangular factorization
        l_diag: lls.l_diag
    }
    
    RETURN (x, cholesky_factor)

// ============================================================================
// ELIMINATE DIAGONAL - COMPLETE
// ============================================================================

ALGORITHM eliminate_diag(lls: LinearLeastSquaresDiagonalProblem, 
                        diag: Vector[f64], rhs: Vector[f64]) -> Vector[f64]:
    INPUT: lls - least squares problem, diag - diagonal matrix D, rhs - right-hand side
    OUTPUT: Modified right-hand side after eliminating diagonal
    PURPOSE: Transform min ||[A; D] * x - [b; 0]||² to triangular form using Givens rotations
    
    n ← number of columns in lls.upper_r
    
    // STEP 1: Copy R^T to lower triangle (will be overwritten with L)
    r_and_l ← lls.upper_r[0:n, 0:n]
    fill_lower_triangle_with_upper_triangle(r_and_l)
    
    // STEP 2: Save diagonal of R for later restoration
    FOR j = 0 TO n-1:
        IF j < length(lls.work):
            lls.work[j] ← r_and_l[j,j]
    
    // STEP 3: Process each column with Givens rotations
    p5 ← 0.5
    p25 ← 0.25
    
    FOR j = 0 TO n-1:
        // Get diagonal entry from D with permutation
        diag_entry ← IF (j < length(lls.permutation) AND 
                        lls.permutation[j] < length(diag))
                     THEN diag[lls.permutation[j]]
                     ELSE 0.0
        
        IF diag_entry ≠ 0:
            // Initialize for this column
            lls.l_diag[j] ← diag_entry
            FOR i = j+1 TO n-1:
                IF i < length(lls.l_diag):
                    lls.l_diag[i] ← 0
            
            qtbpj ← 0  // Accumulator for transformed RHS
            
            // STEP 4: Apply sequence of Givens rotations
            FOR k = j TO n-1:
                IF k < length(lls.l_diag) AND lls.l_diag[k] ≠ 0:
                    r_kk ← r_and_l[k,k]
                    
                    // STEP 4a: Compute Givens rotation parameters (c,s)
                    IF |r_kk| < |lls.l_diag[k]|:
                        // Case: |diagonal| > |R element|
                        cot ← r_kk / lls.l_diag[k]
                        sin ← p5 / √(p25 + p25 * cot²)
                        cos ← sin * cot
                    ELSE:
                        // Case: |R element| ≥ |diagonal|
                        tan ← lls.l_diag[k] / r_kk
                        cos ← p5 / √(p25 + p25 * tan²)
                        sin ← cos * tan
                    
                    // STEP 4b: Apply rotation to modify R diagonal and RHS
                    r_and_l[k,k] ← cos * r_kk + sin * lls.l_diag[k]
                    temp ← cos * rhs[k] + sin * qtbpj
                    qtbpj ← -sin * rhs[k] + cos * qtbpj
                    rhs[k] ← temp
                    
                    // STEP 4c: Accumulate transformation in L matrix
                    FOR i = k+1 TO n-1:
                        IF i < length(lls.l_diag):
                            r_ik ← r_and_l[i,k]
                            temp ← cos * r_ik + sin * lls.l_diag[i]
                            lls.l_diag[i] ← -sin * r_ik + cos * lls.l_diag[i]
                            r_and_l[i,k] ← temp
        
        // STEP 5: Store final diagonal and restore R diagonal
        lls.l_diag[j] ← r_and_l[j,j]
        r_and_l[j,j] ← lls.work[j]
    
    RETURN rhs

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

ALGORITHM r_rank(lls: LinearLeastSquaresDiagonalProblem) -> usize:
    // Find rank of R matrix (number of non-zero diagonal elements)
    n ← number of columns in lls.upper_r
    max_rank ← min(lls.m, n)
    
    FOR i = 0 TO max_rank-1:
        IF lls.upper_r[i,i] = 0:
            RETURN i
    
    RETURN max_rank

ALGORITHM rank(lls: LinearLeastSquaresDiagonalProblem) -> usize:
    // Find rank of L matrix (number of non-zero diagonal elements in l_diag)
    FOR i = 0 TO length(lls.l_diag)-1:
        IF lls.l_diag[i] = 0:
            RETURN i
    
    RETURN length(lls.l_diag)

use crate::symbolic::symbolic_engine::Expr;

use nalgebra::{DMatrix, DVector};
use std::ops::{Add, Index, IndexMut, Mul, Sub};
#[derive(Clone, Debug, PartialEq)]
/// Symbolic vector
pub struct ExprVector {
    pub data: Vec<Expr>,
}

impl ExprVector {
    /// Create new symbolic vector
    pub fn new(data: Vec<Expr>) -> Self {
        Self { data }
    }

    /// Create zero vector of given size
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![Expr::Const(0.0); size],
        }
    }
    /// crate indexed variable vector
    pub fn indexed_vars_vector(size: usize, var_name: &str) -> Self {
        Self {
            data: (0..size).map(|i| Expr::IndexedVar(i, var_name)).collect(),
        }
    }
    /// Create from variable names
    pub fn from_variables(vars: &[&str]) -> Self {
        Self {
            data: vars.iter().map(|&v| Expr::Var(v.to_string())).collect(),
        }
    }
    /// turn symbolic vector into vector of strings
    pub fn to_strings(&self) -> Vec<String> {
        self.data.iter().map(|expr| expr.to_string()).collect()
    }
    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<Expr> {
        self.data.iter()
    }
    /// Element access
    pub fn get(&self, index: usize) -> Option<&Expr> {
        self.data.get(index)
    }

    /// Mutable element access
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Expr> {
        self.data.get_mut(index)
    }

    /// Push new element
    pub fn push(&mut self, expr: Expr) {
        self.data.push(expr);
    }

    /// Dot product (symbolic)
    pub fn dot(&self, other: &ExprVector) -> Expr {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let mut result = Expr::Const(0.0);
        for i in 0..self.len() {
            result = result + (self.data[i].clone() * other.data[i].clone());
        }
        result.simplify_()
    }

    /// AXPY operation: self = a * x + y*b
    pub fn axpy(&mut self, a: &Expr, x: &ExprVector, b: &Expr) {
        assert_eq!(self.len(), x.len(), "Vector dimensions must match");

        for i in 0..self.len() {
            self.data[i] =
                (a.clone() * x.data[i].clone() + (self.data[i].clone() * b.clone())).simplify_();
        }
    }

    /// Scalar multiplication
    pub fn scale(&mut self, scalar: &Expr) {
        for expr in &mut self.data {
            *expr = (scalar.clone() * expr.clone()).simplify_();
        }
    }

    /// Evaluate vector numerically
    pub fn evaluate(&self, vars: &[&str], values: &[f64]) -> DVector<f64> {
        let evaluated: Vec<f64> = self
            .data
            .iter()
            .map(|expr| expr.eval_expression(vars.to_vec(), values))
            .collect();
        DVector::from_vec(evaluated)
    }

    /// Substitute variables
    pub fn substitute(&self, var: &str, value: &Expr) -> ExprVector {
        ExprVector {
            data: self
                .data
                .iter()
                .map(|expr| expr.substitute_variable(var, value))
                .collect(),
        }
    }

    /// Differentiate with respect to variable
    pub fn diff(&self, var: &str) -> ExprVector {
        ExprVector {
            data: self
                .data
                .iter()
                .map(|expr| (expr.diff(var)).simplify_())
                .collect(),
        }
    }

    /// Simplify all expressions
    pub fn simplify(&self) -> ExprVector {
        ExprVector {
            data: self.data.iter().map(|expr| expr.simplify_()).collect(),
        }
    }
    pub fn as_vec(&self) -> Vec<Expr> {
        self.data.clone()
    }
    pub fn as_dvec(&self) -> DVector<Expr> {
        DVector::from_vec(self.data.clone())
    }
    /// Convert to lambdified function
    pub fn lambdify(&self, vars: &[&str]) -> Box<dyn Fn(&[f64]) -> DVector<f64>> {
        let vars_owned: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
        let exprs = self.data.clone();

        Box::new(move |values: &[f64]| {
            let var_refs: Vec<&str> = vars_owned.iter().map(|s| s.as_str()).collect();
            let evaluated: Vec<f64> = exprs
                .iter()
                .map(|expr| expr.eval_expression(var_refs.clone(), values))
                .collect();
            DVector::from_vec(evaluated)
        })
    }

    pub fn show(&self) {
        for expr in &self.data {
            println!("{}", expr.clone());
        }
    }
}

// Implement indexing
impl Index<usize> for ExprVector {
    type Output = Expr;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for ExprVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// Vector addition
impl Add for ExprVector {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let result: Vec<Expr> = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| (a + b).simplify_())
            .collect();

        ExprVector { data: result }
    }
}

// Vector subtraction
impl Sub for ExprVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let result: Vec<Expr> = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| (a - b).simplify_())
            .collect();

        ExprVector { data: result }
    }
}

// Scalar multiplication
impl Mul<Expr> for ExprVector {
    type Output = Self;

    fn mul(self, scalar: Expr) -> Self::Output {
        let result: Vec<Expr> = self
            .data
            .into_iter()
            .map(|expr| (scalar.clone() * expr).simplify_())
            .collect();

        ExprVector { data: result }
    }
}

impl Mul<ExprVector> for Expr {
    type Output = ExprVector;

    fn mul(self, vector: ExprVector) -> Self::Output {
        vector * self
    }
}

///////////////////////////////////////////////////////////////////////////
// Matrix
///////////////////////////////////////////////////////////////////////////
#[derive(Clone, Debug, PartialEq)]
pub struct ExprMatrix {
    pub data: Vec<Vec<Expr>>,
    pub nrows: usize,
    pub ncols: usize,
}

impl ExprMatrix {
    /// Create new symbolic matrix
    pub fn new(data: Vec<Vec<Expr>>) -> Self {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };

        // Validate dimensions
        for row in &data {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
        }

        Self { data, nrows, ncols }
    }

    /// Create zero matrix
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let data = vec![vec![Expr::Const(0.0); ncols]; nrows];
        Self { data, nrows, ncols }
    }

    /// Create identity matrix
    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![Expr::Const(0.0); size]; size];
        for i in 0..size {
            data[i][i] = Expr::Const(1.0);
        }
        Self {
            data,
            nrows: size,
            ncols: size,
        }
    }

    /// create matrix of indexed variables
    pub fn indexed_vars_matrix(nrows: usize, ncols: usize, var_name: &str) -> Self {
        let mut data = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut row = Vec::with_capacity(ncols);
            for j in 0..ncols {
                row.push(Expr::IndexedVar2D(i, j, var_name));
            }
            data.push(row);
        }
        Self { data, nrows, ncols }
    }
    /// Create matrix from variable names with indexing
    pub fn from_variables(nrows: usize, ncols: usize, var_name: &str) -> Self {
        let mut data = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut row = Vec::with_capacity(ncols);
            for j in 0..ncols {
                row.push(Expr::IndexedVar2D(i, j, var_name));
            }
            data.push(row);
        }
        Self { data, nrows, ncols }
    }

    /// Create diagonal matrix from vector
    pub fn diagonal(diag: &ExprVector) -> Self {
        let size = diag.len();
        let mut data = vec![vec![Expr::Const(0.0); size]; size];
        for i in 0..size {
            data[i][i] = diag[i].clone();
        }
        Self {
            data,
            nrows: size,
            ncols: size,
        }
    }
    /// turn symbolic matrix into vector of strings
    pub fn to_strings(&self) -> Vec<String> {
        self.flatten().to_strings()
    }

    pub fn to_strings_column_major(&self) -> Vec<String> {
        self.flatten_column_major()
            .data
            .iter()
            .map(|expr| expr.to_string())
            .collect()
    }
    /// Create matrix from rows
    pub fn from_rows(rows: Vec<ExprVector>) -> Self {
        if rows.is_empty() {
            return Self::zeros(0, 0);
        }

        let nrows = rows.len();
        let ncols = rows[0].len();

        // Validate all rows have same length
        for row in &rows {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
        }

        let data: Vec<Vec<Expr>> = rows.into_iter().map(|row| row.data).collect();

        Self { data, nrows, ncols }
    }

    /// Create matrix from columns
    pub fn from_columns(cols: Vec<ExprVector>) -> Self {
        if cols.is_empty() {
            return Self::zeros(0, 0);
        }

        let nrows = cols[0].len();
        let ncols = cols.len();

        // Validate all columns have same length
        for col in &cols {
            assert_eq!(col.len(), nrows, "All columns must have the same length");
        }

        let mut data = vec![vec![Expr::Const(0.0); ncols]; nrows];
        for (j, col) in cols.iter().enumerate() {
            for i in 0..nrows {
                data[i][j] = col[i].clone();
            }
        }

        Self { data, nrows, ncols }
    }

    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Check if matrix is square
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Check if matrix is empty
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<&Expr> {
        self.data.get(i)?.get(j)
    }

    /// Get mutable element at (i, j)
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut Expr> {
        self.data.get_mut(i)?.get_mut(j)
    }

    /// Get row as ExprVector
    pub fn row(&self, i: usize) -> ExprVector {
        assert!(i < self.nrows, "Row index out of bounds");
        ExprVector::new(self.data[i].clone())
    }

    /// Get column as ExprVector
    pub fn column(&self, j: usize) -> ExprVector {
        assert!(j < self.ncols, "Column index out of bounds");
        let col_data: Vec<Expr> = self.data.iter().map(|row| row[j].clone()).collect();
        ExprVector::new(col_data)
    }

    /// Set row
    pub fn set_row(&mut self, i: usize, row: &ExprVector) {
        assert!(i < self.nrows, "Row index out of bounds");
        assert_eq!(
            row.len(),
            self.ncols,
            "Row length must match matrix columns"
        );
        self.data[i] = row.data.clone();
    }

    /// Set column
    pub fn set_column(&mut self, j: usize, col: &ExprVector) {
        assert!(j < self.ncols, "Column index out of bounds");
        assert_eq!(
            col.len(),
            self.nrows,
            "Column length must match matrix rows"
        );
        for i in 0..self.nrows {
            self.data[i][j] = col[i].clone();
        }
    }

    /// Transpose
    pub fn transpose(&self) -> ExprMatrix {
        let mut data = vec![vec![Expr::Const(0.0); self.nrows]; self.ncols];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                data[j][i] = self.data[i][j].clone();
            }
        }
        ExprMatrix {
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Matrix-vector multiplication
    pub fn mul_vector(&self, vec: &ExprVector) -> ExprVector {
        assert_eq!(
            self.ncols,
            vec.len(),
            "Matrix columns must match vector length"
        );

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut sum = Expr::Const(0.0);
            for j in 0..self.ncols {
                sum = sum + (self.data[i][j].clone() * vec[j].clone());
            }
            result_data.push(sum.simplify_());
        }

        ExprVector::new(result_data)
    }

    /// Trace (sum of diagonal elements)
    pub fn trace(&self) -> Expr {
        assert!(self.is_square(), "Matrix must be square to compute trace");

        let mut sum = Expr::Const(0.0);
        for i in 0..self.nrows {
            sum = sum + self.data[i][i].clone();
        }
        sum.simplify_()
    }

    /// Determinant (for 2x2 and 3x3 matrices)
    pub fn determinant(&self) -> Expr {
        assert!(
            self.is_square(),
            "Matrix must be square to compute determinant"
        );

        match self.nrows {
            1 => self.data[0][0].clone(),
            2 => {
                let a = &self.data[0][0];
                let b = &self.data[0][1];
                let c = &self.data[1][0];
                let d = &self.data[1][1];
                (a.clone() * d.clone() - b.clone() * c.clone()).simplify_()
            }
            3 => {
                // Using cofactor expansion along first row
                let mut det = Expr::Const(0.0);
                for j in 0..3 {
                    let minor = self.minor(0, j);
                    let cofactor = if j % 2 == 0 {
                        self.data[0][j].clone() * minor.determinant().simplify_()
                    } else {
                        -(self.data[0][j].clone() * minor.determinant().simplify_())
                    };
                    det = det.simplify_() + cofactor.simplify_();
                }
                det.simplify_()
            }
            _ => panic!("Determinant not implemented for matrices larger than 3x3"),
        }
    }

    /// Get minor matrix (remove row i and column j)
    pub fn minor(&self, row: usize, col: usize) -> ExprMatrix {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");

        let mut data = Vec::new();
        for i in 0..self.nrows {
            if i == row {
                continue;
            }
            let mut new_row = Vec::new();
            for j in 0..self.ncols {
                if j == col {
                    continue;
                }
                new_row.push(self.data[i][j].clone());
            }
            data.push(new_row);
        }

        ExprMatrix::new(data)
    }

    /// Scalar multiplication
    pub fn scale(&mut self, scalar: &Expr) {
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                self.data[i][j] = (scalar.clone() * self.data[i][j].clone()).simplify_();
            }
        }
    }

    /// Element-wise operations
    pub fn map<F>(&self, f: F) -> ExprMatrix
    where
        F: Fn(&Expr) -> Expr,
    {
        let new_data: Vec<Vec<Expr>> = self
            .data
            .iter()
            .map(|row| row.iter().map(&f).collect())
            .collect();
        ExprMatrix::new(new_data)
    }

    /// Simplify all expressions
    pub fn simplify(&self) -> ExprMatrix {
        self.map(|expr| expr.simplify_())
    }

    /// Substitute variables
    pub fn substitute(&self, var: &str, value: &Expr) -> ExprMatrix {
        self.map(|expr| expr.substitute_variable(var, value))
    }

    /// Differentiate with respect to variable
    pub fn diff(&self, var: &str) -> ExprMatrix {
        self.map(|expr| expr.diff(var))
    }

    /// Evaluate matrix numerically
    pub fn evaluate(&self, vars: &[&str], values: &[f64]) -> DMatrix<f64> {
        let mut result = DMatrix::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result[(i, j)] = self.data[i][j].eval_expression(vars.to_vec(), values);
            }
        }
        result
    }

    /// Convert to lambdified function
    pub fn lambdify(&self, vars: &[&str]) -> Box<dyn Fn(&[f64]) -> DMatrix<f64>> {
        let vars_owned: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
        let data = self.data.clone();
        let nrows = self.nrows;
        let ncols = self.ncols;

        Box::new(move |values: &[f64]| {
            let var_refs: Vec<&str> = vars_owned.iter().map(|s| s.as_str()).collect();
            let mut result = DMatrix::zeros(nrows, ncols);
            for i in 0..nrows {
                for j in 0..ncols {
                    result[(i, j)] = data[i][j].eval_expression(var_refs.clone(), values);
                }
            }
            result
        })
    }

    /// Flatten to vector (row-major order)
    pub fn flatten(&self) -> ExprVector {
        let flat_data: Vec<Expr> = self
            .data
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect();
        ExprVector::new(flat_data)
    }
    pub fn flatten_column_major(&self) -> ExprVector {
        let mut flat_data: ExprVector = ExprVector::zeros(self.nrows * self.ncols);
        for j in 0..self.ncols {
            // columns first
            for i in 0..self.nrows {
                // then rows
                flat_data[j * self.nrows + i] = self.data[i][j].clone();
            }
        }
        flat_data
    }
    /// Reshape from vector (row-major order)
    pub fn from_vector(vec: &ExprVector, nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            vec.len(),
            nrows * ncols,
            "Vector length must match matrix dimensions"
        );

        let mut data = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut row = Vec::with_capacity(ncols);
            for j in 0..ncols {
                row.push(vec[i * ncols + j].clone());
            }
            data.push(row);
        }

        Self { data, nrows, ncols }
    }
}

// Implement indexing
impl Index<(usize, usize)> for ExprMatrix {
    type Output = Expr;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i][j] // i - row, j - column
    }
}

impl IndexMut<(usize, usize)> for ExprMatrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i][j]
    }
}

// Matrix addition
impl Add for ExprMatrix {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row = Vec::with_capacity(self.ncols);
            for j in 0..self.ncols {
                row.push((self.data[i][j].clone() + other.data[i][j].clone()).simplify_());
            }
            result_data.push(row);
        }

        ExprMatrix::new(result_data)
    }
}

// Matrix subtraction
impl Sub for ExprMatrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row = Vec::with_capacity(self.ncols);
            for j in 0..self.ncols {
                row.push((self.data[i][j].clone() - other.data[i][j].clone()).simplify_());
            }
            result_data.push(row);
        }

        ExprMatrix::new(result_data)
    }
}

// Matrix multiplication
impl Mul for ExprMatrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(
            self.ncols, other.nrows,
            "Matrix dimensions incompatible for multiplication"
        );

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row = Vec::with_capacity(other.ncols);
            for j in 0..other.ncols {
                let mut sum = Expr::Const(0.0);
                for k in 0..self.ncols {
                    sum = sum + (self.data[i][k].clone() * other.data[k][j].clone());
                }
                row.push(sum.simplify_());
            }
            result_data.push(row);
        }

        ExprMatrix::new(result_data)
    }
}

// Scalar multiplication (Expr * ExprMatrix)
impl Mul<ExprMatrix> for Expr {
    type Output = ExprMatrix;

    fn mul(self, matrix: ExprMatrix) -> Self::Output {
        matrix.map(|expr| (self.clone() * expr.clone()).simplify_())
    }
}

// Scalar multiplication (ExprMatrix * Expr)
impl Mul<Expr> for ExprMatrix {
    type Output = Self;

    fn mul(self, scalar: Expr) -> Self::Output {
        self.map(|expr| (scalar.clone() * expr.clone()).simplify_())
    }
}

// Matrix-vector multiplication
impl Mul<ExprVector> for ExprMatrix {
    type Output = ExprVector;

    fn mul(self, vector: ExprVector) -> Self::Output {
        self.mul_vector(&vector)
    }
}

// Additional useful operations
impl ExprMatrix {
    /// Kronecker product
    pub fn kronecker(&self, other: &ExprMatrix) -> ExprMatrix {
        let new_nrows = self.nrows * other.nrows;
        let new_ncols = self.ncols * other.ncols;
        let mut result_data = vec![vec![Expr::Const(0.0); new_ncols]; new_nrows];

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                for k in 0..other.nrows {
                    for l in 0..other.ncols {
                        let row_idx = i * other.nrows + k;
                        let col_idx = j * other.ncols + l;
                        result_data[row_idx][col_idx] =
                            (self.data[i][j].clone() * other.data[k][l].clone()).simplify_();
                    }
                }
            }
        }

        ExprMatrix::new(result_data)
    }

    /// Hadamard (element-wise) product
    pub fn hadamard(&self, other: &ExprMatrix) -> ExprMatrix {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row = Vec::with_capacity(self.ncols);
            for j in 0..self.ncols {
                row.push((self.data[i][j].clone() * other.data[i][j].clone()).simplify_());
            }
            result_data.push(row);
        }

        ExprMatrix::new(result_data)
    }

    /// Element-wise division
    pub fn hadamard_div(&self, other: &ExprMatrix) -> ExprMatrix {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        let mut result_data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row = Vec::with_capacity(self.ncols);
            for j in 0..self.ncols {
                row.push((self.data[i][j].clone() / other.data[i][j].clone()).simplify_());
            }
            result_data.push(row);
        }

        ExprMatrix::new(result_data)
    }

    /// Matrix power (for square matrices)
    pub fn pow(&self, n: usize) -> ExprMatrix {
        assert!(
            self.is_square(),
            "Matrix must be square for power operation"
        );

        if n == 0 {
            return ExprMatrix::identity(self.nrows);
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = result * self.clone();
        }
        result
    }

    /// Inverse for 2x2 matrices
    pub fn inverse_2x2(&self) -> Option<ExprMatrix> {
        if self.nrows != 2 || self.ncols != 2 {
            return None;
        }

        let det = self.determinant();
        if det.is_zero() {
            return None; // Singular matrix
        }

        let mut inv_data = vec![vec![Expr::Const(0.0); 2]; 2];
        inv_data[0][0] = (self[(1, 1)].clone() / det.clone()).simplify_();
        inv_data[0][1] = (-(self[(0, 1)].clone()) / det.clone()).simplify_();
        inv_data[1][0] = (-(self[(1, 0)].clone()) / det.clone()).simplify_();
        inv_data[1][1] = (self[(0, 0)].clone() / det.clone()).simplify_();

        Some(ExprMatrix::new(inv_data))
    }

    /// Create block matrix from 2x2 blocks
    pub fn block_matrix_2x2(
        a11: &ExprMatrix,
        a12: &ExprMatrix,
        a21: &ExprMatrix,
        a22: &ExprMatrix,
    ) -> ExprMatrix {
        assert_eq!(
            a11.nrows, a12.nrows,
            "Top blocks must have same number of rows"
        );
        assert_eq!(
            a21.nrows, a22.nrows,
            "Bottom blocks must have same number of rows"
        );
        assert_eq!(
            a11.ncols, a21.ncols,
            "Left blocks must have same number of columns"
        );
        assert_eq!(
            a12.ncols, a22.ncols,
            "Right blocks must have same number of columns"
        );

        let nrows = a11.nrows + a21.nrows;
        let ncols = a11.ncols + a12.ncols;
        let mut data = vec![vec![Expr::Const(0.0); ncols]; nrows];

        // Fill A11
        for i in 0..a11.nrows {
            for j in 0..a11.ncols {
                data[i][j] = a11[(i, j)].clone();
            }
        }

        // Fill A12
        for i in 0..a12.nrows {
            for j in 0..a12.ncols {
                data[i][j + a11.ncols] = a12[(i, j)].clone();
            }
        }

        // Fill A21
        for i in 0..a21.nrows {
            for j in 0..a21.ncols {
                data[i + a11.nrows][j] = a21[(i, j)].clone();
            }
        }

        // Fill A22
        for i in 0..a22.nrows {
            for j in 0..a22.ncols {
                data[i + a11.nrows][j + a11.ncols] = a22[(i, j)].clone();
            }
        }

        ExprMatrix::new(data)
    }

    /// Extract submatrix
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> ExprMatrix {
        assert!(row_start <= row_end && row_end <= self.nrows);
        assert!(col_start <= col_end && col_end <= self.ncols);

        let mut data = Vec::new();
        for i in row_start..row_end {
            let mut row = Vec::new();
            for j in col_start..col_end {
                row.push(self.data[i][j].clone());
            }
            data.push(row);
        }

        ExprMatrix::new(data)
    }

    /// Concatenate matrices horizontally
    pub fn hstack(matrices: &[ExprMatrix]) -> ExprMatrix {
        if matrices.is_empty() {
            return ExprMatrix::zeros(0, 0);
        }

        let nrows = matrices[0].nrows;
        for mat in matrices {
            assert_eq!(
                mat.nrows, nrows,
                "All matrices must have same number of rows"
            );
        }

        let total_cols: usize = matrices.iter().map(|m| m.ncols).sum();
        let mut data = vec![vec![Expr::Const(0.0); total_cols]; nrows];

        let mut col_offset = 0;
        for mat in matrices {
            for i in 0..nrows {
                for j in 0..mat.ncols {
                    data[i][col_offset + j] = mat[(i, j)].clone();
                }
            }
            col_offset += mat.ncols;
        }

        ExprMatrix::new(data)
    }

    /// Concatenate matrices vertically
    pub fn vstack(matrices: &[ExprMatrix]) -> ExprMatrix {
        if matrices.is_empty() {
            return ExprMatrix::zeros(0, 0);
        }

        let ncols = matrices[0].ncols;
        for mat in matrices {
            assert_eq!(
                mat.ncols, ncols,
                "All matrices must have same number of columns"
            );
        }

        let total_rows: usize = matrices.iter().map(|m| m.nrows).sum();
        let mut data = vec![vec![Expr::Const(0.0); ncols]; total_rows];

        let mut row_offset = 0;
        for mat in matrices {
            for i in 0..mat.nrows {
                for j in 0..ncols {
                    data[row_offset + i][j] = mat[(i, j)].clone();
                }
            }
            row_offset += mat.nrows;
        }

        ExprMatrix::new(data)
    }

    /// Norm operations
    pub fn frobenius_norm_squared(&self) -> Expr {
        let mut sum = Expr::Const(0.0);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                sum = sum + (self.data[i][j].clone().pow(Expr::Const(2.0)));
            }
        }
        sum.simplify_()
    }

    /// Sum of all elements
    pub fn sum(&self) -> Expr {
        let mut sum = Expr::Const(0.0);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                sum = sum + self.data[i][j].clone();
            }
        }
        sum.simplify_()
    }

    /// Row sums
    pub fn row_sums(&self) -> ExprVector {
        let mut sums = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let mut row_sum = Expr::Const(0.0);
            for j in 0..self.ncols {
                row_sum = row_sum + self.data[i][j].clone();
            }
            sums.push(row_sum.simplify_());
        }
        ExprVector::new(sums)
    }

    /// Column sums
    pub fn col_sums(&self) -> ExprVector {
        let mut sums = Vec::with_capacity(self.ncols);
        for j in 0..self.ncols {
            let mut col_sum = Expr::Const(0.0);
            for i in 0..self.nrows {
                col_sum = col_sum + self.data[i][j].clone();
            }
            sums.push(col_sum.simplify_());
        }
        ExprVector::new(sums)
    }

    pub fn show(&self) {
        for row in &self.data {
            let string_to_show = String::from_iter(row.iter().map(|x| format!("{}", x)));
            println!("{}", string_to_show);
        }
    }
}

// Display implementation
impl std::fmt::Display for ExprMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ExprMatrix {}x{} [", self.nrows, self.ncols)?;
        for i in 0..self.nrows {
            write!(f, "  [")?;
            for j in 0..self.ncols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[i][j])?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

//////////////////////////////////////////////////////////////////////////////////////
//  TESTS
//////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests_vector {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product() {
        let v1 = ExprVector::new(vec![
            Expr::Var("x".to_string()),
            Expr::Var("y".to_string()),
            Expr::Const(2.0),
        ]);

        let v2 = ExprVector::new(vec![
            Expr::Const(3.0),
            Expr::Var("z".to_string()),
            Expr::Var("x".to_string()),
        ]);

        let dot = v1.dot(&v2);

        // Expected: 3*x + y*z + 2*x = (3+2)*x + y*z = 5*x + y*z
        let expected = Expr::Var("x".to_string()) * Expr::Const(3.0)
            + Expr::Var("y".to_string()) * Expr::Var("z".to_string())
            + Expr::Const(2.0) * Expr::Var("x".to_string());

        assert_eq!(dot, expected);
    }

    #[test]
    fn test_dot_product_with_simplification() {
        let v1 = ExprVector::new(vec![
            Expr::Var("x".to_string()),
            Expr::Const(0.0), // Zero element
        ]);

        let v2 = ExprVector::new(vec![
            Expr::Const(2.0),
            Expr::Var("y".to_string()), // This should be eliminated
        ]);

        let dot = v1.dot(&v2);

        // Expected: 2*x + 0*y = 2*x
        let expected = Expr::Var("x".to_string()) * Expr::Const(2.0);

        assert_eq!(dot, expected);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match")]
    fn test_dot_product_dimension_mismatch() {
        let v1 = ExprVector::new(vec![Expr::Var("x".to_string())]);
        let v2 = ExprVector::new(vec![Expr::Var("x".to_string()), Expr::Var("y".to_string())]);

        v1.dot(&v2);
    }

    #[test]
    fn test_axpy_operation() {
        let mut y = ExprVector::new(vec![
            Expr::Var("y1".to_string()),
            Expr::Var("y2".to_string()),
        ]);

        let x = ExprVector::new(vec![
            Expr::Var("x1".to_string()),
            Expr::Var("x2".to_string()),
        ]);

        let a = Expr::Const(3.0);

        y.axpy(&a, &x, &Expr::Const(1.0));

        // Expected: y = 3*x + y
        let expected = ExprVector::new(vec![
            Expr::Const(3.0) * Expr::Var("x1".to_string()) + Expr::Var("y1".to_string()),
            Expr::Const(3.0) * Expr::Var("x2".to_string()) + Expr::Var("y2".to_string()),
        ]);

        assert_eq!(y, expected);
    }

    #[test]
    fn test_axpy_with_symbolic_scalar() {
        let mut y = ExprVector::new(vec![Expr::Var("y".to_string()), Expr::Const(1.0)]);

        let x = ExprVector::new(vec![Expr::Var("x".to_string()), Expr::Const(2.0)]);

        let a = Expr::Var("alpha".to_string());

        y.axpy(&a, &x, &Expr::Const(1.0));

        // Expected: y = alpha*x + y
        // [y, 1] + [alpha*x, alpha*2 + 1] = [alpha*x + y, alpha*2 + 1]
        let expected = ExprVector::new(vec![
            Expr::Var("alpha".to_string()) * Expr::Var("x".to_string())
                + Expr::Var("y".to_string()),
            Expr::Var("alpha".to_string()) * Expr::Const(2.0) + Expr::Const(1.0),
        ]);

        assert_eq!(y, expected);
    }

    #[test]
    fn test_scale_operation() {
        let mut v = ExprVector::new(vec![
            Expr::Var("x".to_string()),
            Expr::Var("y".to_string()) + Expr::Const(1.0),
            Expr::Const(3.0),
        ]);

        let scalar = Expr::Var("k".to_string());
        v.scale(&scalar);

        let expected = ExprVector::new(vec![
            Expr::Var("k".to_string()) * Expr::Var("x".to_string()),
            Expr::Var("k".to_string()) * (Expr::Var("y".to_string()) + Expr::Const(1.0)),
            Expr::Var("k".to_string()) * Expr::Const(3.0),
        ]);

        assert_eq!(v, expected);
    }

    #[test]
    fn test_evaluate_numerical() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()).pow(Expr::Const(2.0)),
            Expr::Var("x".to_string()) * Expr::Var("y".to_string()),
            Expr::Var("y".to_string()) + Expr::Const(1.0),
        ]);

        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];

        let result = v.evaluate(&vars, &values);

        // Expected: [2^2, 2*3, 3+1] = [4, 6, 4]
        let expected = nalgebra::DVector::from_vec(vec![4.0, 6.0, 4.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_variable() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()) + Expr::Var("y".to_string()),
            Expr::Var("x".to_string()).pow(Expr::Const(2.0)),
            Expr::Var("z".to_string()),
        ]);

        let substituted = v.substitute("x", &Expr::Const(5.0));

        let expected = ExprVector::new(vec![
            Expr::Const(5.0) + Expr::Var("y".to_string()),
            Expr::Const(5.0).pow(Expr::Const(2.0)),
            Expr::Var("z".to_string()),
        ]);

        assert_eq!(substituted, expected);
    }

    #[test]
    fn test_substitute_with_expression() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(2.0),
            Expr::Var("x".to_string()) + Expr::Var("y".to_string()),
        ]);

        let substitution = Expr::Var("t".to_string()).pow(Expr::Const(2.0));
        let substituted = v.substitute("x", &substitution);
        // [x*2, x+y] -> [t^2*2, t^2+y]
        let expected = ExprVector::new(vec![
            Expr::Var("t".to_string()).pow(Expr::Const(2.0)) * Expr::Const(2.0),
            Expr::Var("t".to_string()).pow(Expr::Const(2.0)) + Expr::Var("y".to_string()),
        ]);

        assert_eq!(substituted, expected);
    }

    #[test]
    fn test_differentiation() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()).pow(Expr::Const(2.0)),
            Expr::Var("x".to_string()) * Expr::Var("y".to_string()),
            Expr::Var("y".to_string()).pow(Expr::Const(3.0)),
            Expr::Const(5.0),
        ]);

        let grad_x = v.diff("x");

        // Expected derivatives: [2*x, y, 0, 0]
        let expected = ExprVector::new(vec![
            Expr::Const(2.0) * Expr::Var("x".to_string()),
            Expr::Var("y".to_string()),
            Expr::Const(0.0),
            Expr::Const(0.0),
        ]);

        assert_eq!(grad_x, expected);
    }

    #[test]
    fn test_simplify_complex() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(0.0), // Should become 0
            Expr::Var("x".to_string()) + Expr::Const(0.0), // Should become x
            Expr::Var("x".to_string()) * Expr::Const(1.0), // Should become x
            Expr::Const(2.0) + Expr::Const(3.0),           // Should become 5
        ]);

        let simplified = v.simplify();

        let expected = ExprVector::new(vec![
            Expr::Const(0.0),
            Expr::Var("x".to_string()),
            Expr::Var("x".to_string()),
            Expr::Const(5.0),
        ]);

        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_lambdify_function() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()) + Expr::Var("y".to_string()),
            Expr::Var("x".to_string()) * Expr::Var("y".to_string()),
            Expr::Var("x".to_string()).pow(Expr::Const(2.0)),
        ]);

        let vars = vec!["x", "y"];
        let func = v.lambdify(&vars);

        let result = func(&[2.0, 3.0]);

        // Expected: [2+3, 2*3, 2^2] = [5, 6, 4]
        let expected = nalgebra::DVector::from_vec(vec![5.0, 6.0, 4.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_lambdify_with_complex_expressions() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()).exp(),
            Expr::Var("y".to_string()).ln(),
            Expr::Var("x".to_string()).pow(Expr::Var("y".to_string())),
        ]);

        let vars = vec!["x", "y"];
        let func = v.lambdify(&vars);

        let result = func(&[1.0, std::f64::consts::E]);

        // Expected: [e^1, ln(e), 1^e] = [e, 1, 1]
        assert_relative_eq!(result[0], std::f64::consts::E, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_addition_complex() {
        let v1 = ExprVector::new(vec![
            Expr::Var("x".to_string()).pow(Expr::Const(2.0)),
            Expr::Var("y".to_string()) * Expr::Const(3.0),
        ]);

        let v2 = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(2.0),
            Expr::Var("y".to_string()),
        ]);

        let sum = v1 + v2;

        let expected = ExprVector::new(vec![
            Expr::Var("x".to_string()).pow(Expr::Const(2.0))
                + Expr::Var("x".to_string()) * Expr::Const(2.0),
            Expr::Var("y".to_string()) * Expr::Const(3.0) + Expr::Var("y".to_string()),
        ]);

        assert_eq!(sum, expected);
    }

    #[test]
    fn test_vector_subtraction_with_simplification() {
        let v1 = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(5.0),
            Expr::Var("y".to_string()) + Expr::Const(10.0),
        ]);

        let v2 = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(2.0),
            Expr::Const(5.0),
        ]);

        let diff = v1 - v2;

        let expected = ExprVector::new(vec![
            Expr::Var("x".to_string()) * Expr::Const(5.0)
                - Expr::Var("x".to_string()) * Expr::Const(2.0),
            Expr::Var("y".to_string()) + Expr::Const(10.0) - Expr::Const(5.0),
        ]);

        assert_eq!(diff, expected);
    }

    #[test]
    fn test_scalar_multiplication_both_directions() {
        let v = ExprVector::new(vec![
            Expr::Var("x".to_string()),
            Expr::Var("y".to_string()) + Expr::Const(1.0),
        ]);

        let scalar = Expr::Var("k".to_string());

        // Test both v * scalar and scalar * v
        let result1 = v.clone() * scalar.clone();
        let result2 = scalar.clone() * v;

        let expected = ExprVector::new(vec![
            Expr::Var("k".to_string()) * Expr::Var("x".to_string()),
            Expr::Var("k".to_string()) * (Expr::Var("y".to_string()) + Expr::Const(1.0)),
        ]);

        assert_eq!(result1, expected);
        assert_eq!(result2, expected);
    }

    #[test]
    fn test_from_variables_constructor() {
        let vars = vec!["x", "y", "z"];
        let v = ExprVector::from_variables(&vars);

        let expected = ExprVector::new(vec![
            Expr::Var("x".to_string()),
            Expr::Var("y".to_string()),
            Expr::Var("z".to_string()),
        ]);

        assert_eq!(v, expected);
    }
}

#[cfg(test)]
mod tests_exprmatrix {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_vectors::ExprVector;

    #[test]
    fn test_matrix_multiplication() {
        let m1 = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let m2 = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0)],
            vec![Expr::Const(3.0), Expr::Const(4.0)],
        ]);

        let result = m1 * m2;

        // Expected: [[a + 3b, 2a + 4b], [c + 3d, 2c + 4d]]
        let expected = ExprMatrix::new(vec![
            vec![
                Expr::Var("a".to_string()) + Expr::Var("b".to_string()) * Expr::Const(3.0),
                Expr::Var("a".to_string()) * Expr::Const(2.0)
                    + Expr::Var("b".to_string()) * Expr::Const(4.0),
            ],
            vec![
                Expr::Var("c".to_string()) + Expr::Var("d".to_string()) * Expr::Const(3.0),
                Expr::Var("c".to_string()) * Expr::Const(2.0)
                    + Expr::Var("d".to_string()) * Expr::Const(4.0),
            ],
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let v = ExprVector::new(vec![Expr::Var("x".to_string()), Expr::Var("y".to_string())]);

        let result = m.mul_vector(&v);

        // Expected: [ax + by, cx + dy]
        let expected = ExprVector::new(vec![
            Expr::Var("a".to_string()) * Expr::Var("x".to_string())
                + Expr::Var("b".to_string()) * Expr::Var("y".to_string()),
            Expr::Var("c".to_string()) * Expr::Var("x".to_string())
                + Expr::Var("d".to_string()) * Expr::Var("y".to_string()),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_transpose() {
        let m = ExprMatrix::new(vec![
            vec![
                Expr::Var("a".to_string()),
                Expr::Var("b".to_string()),
                Expr::Var("c".to_string()),
            ],
            vec![
                Expr::Var("d".to_string()),
                Expr::Var("e".to_string()),
                Expr::Var("f".to_string()),
            ],
        ]);

        let transposed = m.transpose();

        let expected = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("d".to_string())],
            vec![Expr::Var("b".to_string()), Expr::Var("e".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("f".to_string())],
        ]);

        assert_eq!(transposed, expected);
        assert_eq!(transposed.shape(), (3, 2));
    }

    #[test]
    fn test_determinant_2x2() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let det = m.determinant();

        // Expected: ad - bc
        let expected = Expr::Var("a".to_string()) * Expr::Var("d".to_string())
            - Expr::Var("b".to_string()) * Expr::Var("c".to_string());

        assert_eq!(det, expected);
    }

    #[allow(dead_code)]
    fn test_determinant_3x3() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0), Expr::Const(3.0)],
            vec![
                Expr::Const(4.0),
                Expr::Var("x".to_string()),
                Expr::Const(6.0),
            ],
            vec![Expr::Const(7.0), Expr::Const(8.0), Expr::Const(9.0)],
        ]);

        let det = m.determinant();

        // Manual calculation: 1*(x*9 - 6*8) - 2*(4*9 - 6*7) + 3*(4*8 - x*7)
        // = 1*(9x - 48) - 2*(36 - 42) + 3*(32 - 7x)
        // = 9x - 48 - 2*(-6) + 3*(32 - 7x)
        // = 9x - 48 + 12 + 96 - 21x
        // = -12x + 60
        let expected = Expr::Const(-12.0) * Expr::Var("x".to_string()) + Expr::Const(60.0);

        assert_eq!(det.simplify_(), expected);
    }

    #[test]
    fn test_trace() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let trace = m.trace();

        // Expected: a + d
        let expected = Expr::Var("a".to_string()) + Expr::Var("d".to_string());

        assert_eq!(trace, expected);
    }

    #[test]
    fn test_inverse_2x2() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0)],
            vec![Expr::Const(3.0), Expr::Const(4.0)],
        ]);

        let inv = m.inverse_2x2().unwrap();

        // det = 1*4 - 2*3 = -2
        // inv = (1/-2) * [[4, -2], [-3, 1]]
        let expected = ExprMatrix::new(vec![
            vec![Expr::Const(-2.0), Expr::Const(1.0)],
            vec![Expr::Const(1.5), Expr::Const(-0.5)],
        ]);

        assert_eq!(inv, expected);
    }

    #[test]
    fn test_inverse_2x2_symbolic() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let inv = m.inverse_2x2().unwrap();

        // det = ad - bc
        let det = Expr::Var("a".to_string()) * Expr::Var("d".to_string())
            - Expr::Var("b".to_string()) * Expr::Var("c".to_string());

        let expected = ExprMatrix::new(vec![
            vec![
                Expr::Var("d".to_string()) / det.clone(),
                -(Expr::Var("b".to_string())) / det.clone(),
            ],
            vec![
                -(Expr::Var("c".to_string())) / det.clone(),
                Expr::Var("a".to_string()) / det.clone(),
            ],
        ]);

        assert_eq!(inv, expected);
    }

    #[test]
    fn test_kronecker_product() {
        let a = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0)],
            vec![Expr::Const(3.0), Expr::Const(4.0)],
        ]);

        let b = ExprMatrix::new(vec![
            vec![Expr::Var("x".to_string()), Expr::Var("y".to_string())],
            vec![Expr::Var("z".to_string()), Expr::Var("w".to_string())],
        ]);

        let kron = a.kronecker(&b);

        // Expected 4x4 matrix
        let expected = ExprMatrix::new(vec![
            vec![
                Expr::Var("x".to_string()),
                Expr::Var("y".to_string()),
                Expr::Const(2.0) * Expr::Var("x".to_string()),
                Expr::Const(2.0) * Expr::Var("y".to_string()),
            ],
            vec![
                Expr::Var("z".to_string()),
                Expr::Var("w".to_string()),
                Expr::Const(2.0) * Expr::Var("z".to_string()),
                Expr::Const(2.0) * Expr::Var("w".to_string()),
            ],
            vec![
                Expr::Const(3.0) * Expr::Var("x".to_string()),
                Expr::Const(3.0) * Expr::Var("y".to_string()),
                Expr::Const(4.0) * Expr::Var("x".to_string()),
                Expr::Const(4.0) * Expr::Var("y".to_string()),
            ],
            vec![
                Expr::Const(3.0) * Expr::Var("z".to_string()),
                Expr::Const(3.0) * Expr::Var("w".to_string()),
                Expr::Const(4.0) * Expr::Var("z".to_string()),
                Expr::Const(4.0) * Expr::Var("w".to_string()),
            ],
        ]);

        assert_eq!(kron.shape(), (4, 4));
        assert_eq!(kron, expected);
    }

    #[test]
    fn test_hadamard_product() {
        let a = ExprMatrix::new(vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ]);

        let b = ExprMatrix::new(vec![
            vec![Expr::Const(2.0), Expr::Var("x".to_string())],
            vec![Expr::Var("y".to_string()), Expr::Const(3.0)],
        ]);

        let hadamard = a.hadamard(&b);

        let expected = ExprMatrix::new(vec![
            vec![
                Expr::Var("a".to_string()) * Expr::Const(2.0),
                Expr::Var("b".to_string()) * Expr::Var("x".to_string()),
            ],
            vec![
                Expr::Var("c".to_string()) * Expr::Var("y".to_string()),
                Expr::Var("d".to_string()) * Expr::Const(3.0),
            ],
        ]);

        assert_eq!(hadamard, expected);
    }

    #[test]
    fn test_matrix_power() {
        let m = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(1.0)],
            vec![Expr::Const(0.0), Expr::Const(1.0)],
        ]);

        let m_squared = m.pow(2);

        // Manual calculation: [[1,1],[0,1]] * [[1,1],[0,1]] = [[1,2],[0,1]]
        let expected = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0)],
            vec![Expr::Const(0.0), Expr::Const(1.0)],
        ]);

        assert_eq!(m_squared, expected);
    }

    #[test]
    fn test_block_matrix_2x2() {
        let a11 = ExprMatrix::new(vec![vec![Expr::Const(1.0)]]);
        let a12 = ExprMatrix::new(vec![vec![Expr::Const(2.0)]]);
        let a21 = ExprMatrix::new(vec![vec![Expr::Const(3.0)]]);
        let a22 = ExprMatrix::new(vec![vec![Expr::Const(4.0)]]);

        let b = ExprMatrix::block_matrix_2x2(&a11, &a12, &a21, &a22);

        let expected = ExprMatrix::new(vec![
            vec![Expr::Const(1.0), Expr::Const(2.0)],
            vec![Expr::Const(3.0), Expr::Const(4.0)],
        ]);

        assert_eq!(b, expected);
    }
}

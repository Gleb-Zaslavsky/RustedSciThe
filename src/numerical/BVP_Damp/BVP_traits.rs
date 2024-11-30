#![allow(non_camel_case_types)] //
use faer::mat::from_column_major_slice;
use faer::sparse::SparseColMat;
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector, Matrix};
use sprs::{CsMat, CsVec};
use std::any::Any;

use crate::somelinalg::some_matrix_inv::invers_csmat;
use crate::somelinalg::LUsolver::invers_Mat_LU;
use crate::somelinalg::Lx_eq_b::{solve_csmat, solve_sys_SparseColMat};
use faer::col::{from_slice, Col};
use faer::mat::Mat;
use faer::prelude::*;
use std::fmt::{self, Debug};
use std::ops::Sub;
type faer_mat = SparseColMat<usize, f64>;
type faer_col = Col<f64>; // Mat<f64>;
pub enum YEnum {
    Dense(DVector<f64>),
    Sparse_1(CsVec<f64>),
    Sparse_2(DVector<f64>),
    Sparse_3(faer_col),
}

pub trait VectorType: Any {
    fn as_any(&self) -> &dyn Any;
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType>;
    fn norm(&self) -> f64;
    fn to_DVectorType(&self) -> DVector<f64>;
    fn clone_box(&self) -> Box<dyn VectorType>;
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_>;
    fn get_val(&self, index: usize) -> f64;
    fn mul_float(&self, float: f64) -> Box<dyn VectorType>;
    fn len(&self) -> usize;
}

impl Sub for &dyn VectorType {
    type Output = Box<dyn VectorType>;
    fn sub(self, other: Self) -> Self::Output {
        self.subtract(other)
    }
}

impl Iterator for YEnum {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            YEnum::Dense(vec) => vec.iter().next().copied(),
            YEnum::Sparse_1(vec) => vec.iter().map(|x| *x.1).next(),
            YEnum::Sparse_2(vec) => vec.iter().next().copied(),
            YEnum::Sparse_3(vec) => vec.iter().next().copied(),
        }
    }
}

impl VectorType for YEnum {
    fn as_any(&self) -> &dyn Any {
        match self {
            YEnum::Dense(vec) => vec,
            YEnum::Sparse_1(vec) => vec,
            YEnum::Sparse_2(vec) => vec,
            YEnum::Sparse_3(vec) => vec,
        }
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => {
                if let Some(d_vec) = other.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(vec - d_vec)
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            YEnum::Sparse_1(vec) => {
                if let Some(c_vec) = other.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(vec - c_vec)
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            YEnum::Sparse_3(vec) => {
                if let Some(d_vec) = other.as_any().downcast_ref::<faer_col>() {
                    let subs = vec.sub(d_vec);
                    Box::new(subs)
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }

            _ => panic!("Type mismatch: expected DVector or CsVec"),
        }
    } // subtract
    fn norm(&self) -> f64 {
        match self {
            YEnum::Dense(vec) => Matrix::norm(vec),
            YEnum::Sparse_1(vec) => CsVec::l2_norm(vec),
            YEnum::Sparse_2(vec) => Matrix::norm(vec),
            YEnum::Sparse_3(vec) => vec.norm_l2(),
        }
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        match self {
            YEnum::Dense(vec) => vec.clone(),
            YEnum::Sparse_1(vec) => {
                let length = vec.dim();

                DVector::from_iterator(length, vec.iter().map(|x| *x.1))
            }
            YEnum::Sparse_2(vec) => vec.clone(),
            YEnum::Sparse_3(vec) => {
                let length = vec.nrows();

                //   DVector::from_iterator(length, vec.row_iter().map(|x| x[0]))
                DVector::from_iterator(length, vec.iter().map(|x| *x))
            }
        }
    } //to_iterator
    fn clone_box(&self) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.clone()),
            YEnum::Sparse_1(vec) => Box::new(vec.clone()),
            YEnum::Sparse_2(vec) => Box::new(vec.clone()),
            YEnum::Sparse_3(vec) => Box::new(vec.clone()),
        }
    }

    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Sparse_1(vec) => Box::new(vec.iter().map(|x| *x.1)),
            YEnum::Sparse_2(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Sparse_3(vec) => Box::new(vec.iter().map(|x| *x)),
        }
    }
    fn get_val(&self, index: usize) -> f64 {
        match self {
            YEnum::Dense(vec) => vec[index],
            YEnum::Sparse_1(vec) => vec[index],
            YEnum::Sparse_2(vec) => vec[index],
            YEnum::Sparse_3(vec) => vec[index],
        }
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec * float),
            YEnum::Sparse_1(vec) => Box::new(vec.map(|x| x * (float))),
            YEnum::Sparse_2(vec) => Box::new(vec * float),
            YEnum::Sparse_3(vec) => Box::new(vec * float),
        }
    }
    fn len(&self) -> usize {
        match self {
            YEnum::Dense(vec) => vec.len(),
            YEnum::Sparse_1(vec) => vec.len(),
            YEnum::Sparse_2(vec) => vec.len(),
            YEnum::Sparse_3(vec) => vec.len(),
        }
    }
}
impl VectorType for DVector<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = other.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self - d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn norm(&self) -> f64 {
        Matrix::norm(self)
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        self.clone()
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self * float)
    }
    fn len(&self) -> usize {
        self.len()
    }
}

impl VectorType for CsVec<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(c_vec) = other.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self - c_vec)
        } else {
            panic!("Type mismatch: expected CsVec")
        }
    }
    fn norm(&self) -> f64 {
        CsVec::l2_norm(self)
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        let length = self.dim();

        DVector::from_iterator(length, self.iter().map(|x| *x.1))
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x.1))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self.map(|x| x * (float)))
    }
    fn len(&self) -> usize {
        self.dim()
    }
}

impl VectorType for faer_col {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = other.as_any().downcast_ref::<faer_col>() {
            let subs = self.sub(d_vec);
            Box::new(subs)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn norm(&self) -> f64 {
        self.norm_l2()
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        let length = self.nrows();

        DVector::from_iterator(length, self.iter().map(|x| *x))
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self * float)
    }
    fn len(&self) -> usize {
        self.nrows()
    }
}
pub trait Fun {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType>;
}

pub enum FunEnum {
    Dense(Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>),
    Sparse_1(Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64>>),
    Sparse_2(Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>),
    Sparse_3(Box<dyn Fn(f64, &faer_col) -> faer_col>),
}

impl Fun for FunEnum {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        match self {
            FunEnum::Dense(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            FunEnum::Sparse_1(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            FunEnum::Sparse_2(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            FunEnum::Sparse_3(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected faer_col")
                }
            }
        }
    }
}

impl fmt::Display for YEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YEnum::Dense(vec) => write!(f, "Dense Vector: {:?}", vec),
            YEnum::Sparse_1(vec) => write!(f, "Sparse Vector 1: {:?}", vec),
            YEnum::Sparse_2(vec) => write!(f, "Sparse Vector 2: {:?}", vec),
            YEnum::Sparse_3(vec) => write!(f, "Sparse Vector 3: {:?}", vec),
        }
    }
}

impl fmt::Display for dyn VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DVector<f64>>() {
            write!(f, "{}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsVec<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(y_enum) = self.as_any().downcast_ref::<YEnum>() {
            write!(f, "{}", y_enum)
        } else {
            write!(f, "Unknown VectorType")
        }
    }
}
impl Debug for dyn VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DVector<f64>>() {
            write!(f, "{:?}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsVec<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(sparse) = self.as_any().downcast_ref::<faer_col>() {
            write!(f, "{:?}", sparse)
        } else {
            write!(f, "Unknown VectorType")
        }
    }
}

//_________________________________Jacobian______________________________
#[allow(dead_code)]
pub enum JacTypes {
    Dense(DMatrix<f64>),
    Sparse_1(CsMat<f64>),
    Sparse_2(CsMatrix<f64>),
    Sparse_3(faer_mat),
}
pub trait MatrixType: Any {
    fn as_any(&self) -> &dyn Any;
    fn inverse(self) -> Box<dyn MatrixType>;
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType>;
    fn clone_box(&self) -> Box<dyn MatrixType>;
    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType>;
    fn shape(&self) -> (usize, usize);
}
/*
impl Clone for dyn MatrixType {

    fn clone(&self) -> Self {
          self.as_any().downcast_ref::<Self>().clone()
    }
}
*/
impl MatrixType for DMatrix<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        let inverse = self.try_inverse();
        let inverse = inverse.unwrap();
        Box::new(inverse)
    }
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }

    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<DMatrix<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                if linear_sys_method.is_none() {
                    let lu = mat_.to_owned().lu();
                    let res = lu.solve(d_vec).unwrap();
                    Box::new(res)
                } else {
                    panic!("no such method for linear system")
                }
            } else {
                panic!("Type mismatch: expected DMatrix")
            }
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }

    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}

impl MatrixType for CsMat<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        let inv_mat = invers_csmat(self.clone(), 1e-8, 100).unwrap();
        Box::new(inv_mat)
    }
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }
    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<CsMat<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                if let Some(old_vec) = old_vec.as_any().downcast_ref::<CsVec<f64>>() {
                    let res = solve_csmat(mat_, d_vec, tol, max_iter, old_vec).unwrap();
                    Box::new(res)
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            } else {
                panic!("Type mismatch: expected DMatrix")
            }
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}

impl MatrixType for CsMatrix<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    } // NO IMPLEMENTATION
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(_d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            panic!("Type mismatch: expected DVector")
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }
    fn solve_sys(
        &self,
        _vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        panic!("method not written yet")
    }
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}
impl MatrixType for faer_mat {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    } //
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }
    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<faer_mat>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                assert_eq!(
                    mat_.ncols(),
                    d_vec.nrows(),
                    " matrix {} and  vector {} have different sizes",
                    mat_.ncols(),
                    d_vec.nrows(),
                );
                let d_vec: Mat<f64> =
                    from_column_major_slice::<f64>(d_vec.as_slice(), mat_.ncols(), 1).to_owned();
                //let LU0 = lu_in_place( mat_);
                "_gmres".to_owned();
                match linear_sys_method {
                    None => {
                        let LU = mat_.sp_lu().unwrap();
                        let res: Mat<f64> = LU.solve(d_vec);

                        let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                        let res = from_slice(res_vec.as_slice()).to_owned();
                        Box::new(res)
                    }
                    Some(_gmres) => {
                        if let Some(old_vec) = old_vec.as_any().downcast_ref::<faer_col>() {
                            let res =
                                solve_sys_SparseColMat(mat_.clone(), d_vec, tol, max_iter, old_vec)
                                    .unwrap();
                            Box::new(res)
                        } else {
                            panic!("old vec must be of type Col<f64>")
                        }
                    }
                    _ => {
                        panic!("no such method for linear system")
                    }
                }
            }
            //if let d_vec
            else {
                panic!("Type mismatch: expected Col")
            }
        }
        //if let mat_
        else {
            panic!("Type mismatch: expected faer_mat")
        }
    } //solve_sys
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
} //impl

impl Debug for dyn MatrixType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DMatrix<f64>>() {
            write!(f, "{:?}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsMat<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(cs_matrix) = self.as_any().downcast_ref::<CsMatrix<f64>>() {
            write!(f, "{:?}", cs_matrix)
        } else if let Some(faer_mat) = self.as_any().downcast_ref::<faer_mat>() {
            write!(f, "{:?}", faer_mat)
        } else {
            write!(f, "Unknown MatrixType")
        }
    }
}

pub trait Jac {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType>;
    fn inv(&mut self, matix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType>;
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType>;
}

pub enum JacEnum {
    Dense(Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>),
    Sparse_1(Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>>),
    Sparse_2(Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>>),
    Sparse_3(Box<dyn FnMut(f64, &faer_col) -> faer_mat>),
}

impl Jac for JacEnum {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType> {
        match self {
            JacEnum::Dense(jac) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(jac(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            JacEnum::Sparse_1(jac) => {
                if let Some(cs_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(jac(x, cs_vec))
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            JacEnum::Sparse_2(jac) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(jac(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            JacEnum::Sparse_3(jac) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                    Box::new(jac(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
        }
    } // call

    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        match self {
            JacEnum::Dense(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
                    Box::new(mat.to_owned().try_inverse().unwrap())
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            }
            JacEnum::Sparse_1(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<CsMat<f64>>() {
                    let inv_mat = invers_csmat(mat.to_owned(), tol, max_iter).unwrap();
                    Box::new(inv_mat)
                } else {
                    panic!("Type mismatch: expected CsMat")
                }
            }
            JacEnum::Sparse_2(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<CsMatrix<f64>>() {
                    //   let lower_triang = mat.to_owned().solve_lower_triangular_cs(bool::from(true));
                    // NO IMPLEMENTATION!
                    Box::new(mat.to_owned())
                } else {
                    panic!("Type mismatch: expected CsMat")
                }
            }
            JacEnum::Sparse_3(_jac) => {
                if let Some(mat_) = matrix.as_any().downcast_ref::<SparseColMat<usize, f64>>() {
                    //  let inv_mat = invers_Mat_mult(mat_.to_owned().expect("REASON"), tol, max_iter).unwrap();//why Result()?;
                    let inv_mat =
                        invers_Mat_LU(mat_.to_owned().expect("REASON"), tol, max_iter).unwrap();
                    //  mat_.to_owned().unwrap().sort_indices();
                    // let inv_mat =  solve_with_upper_triangular(mat_.to_owned().expect("REASON"), tol, max_iter).unwrap();
                    Box::new(inv_mat)
                } else {
                    panic!("Type mismatch: expected faer_mat")
                }
            }
        }
    }

    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        _tol: f64,
        _max_iter: usize,
    ) -> Box<dyn VectorType> {
        match self {
            JacEnum::Dense(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
                    if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                        let lu = mat.to_owned().lu();
                        let res = lu.solve(d_vec).unwrap();
                        Box::new(res)
                    } else {
                        panic!("Type mismatch: expected DVector")
                    }
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            } //dense
            JacEnum::Sparse_3(_jac) => {
                if let Some(mat_) = matrix.as_any().downcast_ref::<SparseColMat<usize, f64>>() {
                    if let Some(d_vec) = vec.as_any().downcast_ref::<Col<f64>>() {
                        let d_vec: Mat<f64> =
                            from_column_major_slice::<f64>(d_vec.as_slice(), mat_.ncols(), 1)
                                .to_owned();
                        let LU = mat_.sp_lu().unwrap();
                        let res: Mat<f64> = LU.solve(d_vec);

                        let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                        let res = from_slice(res_vec.as_slice()).to_owned();
                        Box::new(res)
                    } else {
                        panic!("Type mismatch: expected DVector")
                    }
                } else {
                    panic!("Type mismatch: expected faer_mat")
                }
            }
            _ => {
                panic!("Type mismatch: expected DMatrix")
            }
        }
    }
}

//_________________________________________________Y___________________________________________________________
pub trait Y: Any {
    fn as_any(&self) -> Box<&dyn Any>;
}

impl Y for DVector<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }
}

impl Y for CsVec<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }
}

impl Clone for Box<dyn Y> {
    fn clone(&self) -> Self {
        self.clone()
    }
}

//___________________________________________________________________
pub fn Vectors_type_casting(vec: &DVector<f64>, desired_type: String) -> Box<dyn VectorType> {
    let res: Box<dyn VectorType> = if desired_type == "Dense".to_string() {
        Box::new(YEnum::Dense(vec.clone()))
    } else if desired_type == "Sparse 1".to_string() {
        let mut ind = Vec::new();
        let mut val = Vec::new();
        vec.iter().enumerate().for_each(|(i, x)| {
            if x.abs() > 0.0 {
                ind.push(i);
                val.push(*x);
            }
        });
        Box::new(YEnum::Sparse_1(
            CsVec::new_from_unsorted(vec.len(), ind, val).expect("trouble with initial vector!"),
        ))
    } else if desired_type == "Sparse".to_string() {
        let Mat_vec = from_slice(vec.as_slice()).to_owned();

        Box::new(YEnum::Sparse_3(Mat_vec))
    } else {
        panic!("Unsupported vector type: {}", desired_type);
    };
    res
}
//________________________________________
#[allow(dead_code)]
pub fn jac_rowwise_printing(jac: &Box<dyn MatrixType>) {
    if let Some(dense) = jac.as_any().downcast_ref::<DMatrix<f64>>() {
        dense.row_iter().enumerate().for_each(|(i, row)| {
            let row: Vec<&f64> = row.iter().collect();
            println!("\n  {:?}-th row = {:?}", i, row);
        })
    } else if let Some(sparse) = jac.as_any().downcast_ref::<CsMat<f64>>() {
        println!("{:?}", sparse)
    } else if let Some(cs_matrix) = jac.as_any().downcast_ref::<CsMatrix<f64>>() {
        println!("{:?}", cs_matrix)
    } else if let Some(faer_mat) = jac.as_any().downcast_ref::<faer_mat>() {
        for i in 0..faer_mat.nrows() {
            let mut row_data: Vec<&f64> = Vec::new();
            for j in 0..faer_mat.ncols() {
                row_data.push(faer_mat.get(i, j).unwrap_or(&0.0));
            }
            println!("\n \n{:?}-th row = {:?}", i, row_data);
        }
    } else {
        println!("Unknown MatrixType")
    }
}

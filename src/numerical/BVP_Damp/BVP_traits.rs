#![allow(non_camel_case_types)] //
use crate::numerical::BVP_Damp::linear_sys_solvers_depot::nalgebra_solvers_depot;
use crate::somelinalg::iterative_solvers_cpu::LUsolver::invers_Mat_LU;
use crate::somelinalg::iterative_solvers_cpu::Lx_eq_b::{solve_csmat, solve_sys_SparseColMat};
use crate::somelinalg::iterative_solvers_cpu::some_matrix_inv::invers_csmat;

use faer::col::{Col, ColRef};
use faer::linalg::solvers::Solve;
use faer::mat::Mat;
use faer::mat::MatRef;

use faer::sparse::{SparseColMat, Triplet};
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector, Matrix};
use sprs::{CsMat, CsVec, TriMat};
use std::any::Any;
use std::fmt::{self, Debug};
use std::ops::Sub;
type faer_mat = SparseColMat<usize, f64>;
type faer_col = Col<f64>; // Mat<f64>;

/*
In RST BVP solvers there is an option to use different linear algebra crates
- Nalgebra
- SPRS
- FAER
- Nalgebra SPARSE
Generics in rust allow us to replace specific types with a placeholder that represents multiple types to remove code duplication.
So there are generic types:
 - VectorType (generic type that represents vectors of residuals and Newton steps)
 - MatrixType (generic type that represents matrices of jacobian)
 - Jac (generic type that represents functions for jacobian)
 - Fun (generic type that represents functions for vector)
 "method" variable (from the BVP task) is a keyword that is used to specify the crate that will be used for linear algebra operations
 "Dense" - nalgebra crate
 "Sparse" - faer crate
"Sparse_1" - sprs crate
"Sparse_2" - nalgebra sprs crate

*/
////////////////////////////////////////////////////////////////
//  VECTORTYPE - geneic type to store vectors of residuals and Newton steps
////////////////////////////////////////////////////////////////
pub enum YEnum {
    Dense(DVector<f64>),    // dense vector NALGEBRA CRATE
    Sparse_1(CsVec<f64>),   // sparse vector SPRS CRATE
    Sparse_2(DVector<f64>), // dense vector NALGEBRA SPARSE CRATE
    Sparse_3(faer_col),     // sparse vector  FAER CRATE
}
// basic funcionality for vectors
pub trait VectorType: Any {
    fn as_any(&self) -> &dyn Any;
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType>; //
    fn norm(&self) -> f64; // norm of vector
    fn to_DVectorType(&self) -> DVector<f64>; // convert to dense vector
    fn clone_box(&self) -> Box<dyn VectorType>; // cloning the vector into box
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_>; // iterating over vector
    fn get_val(&self, index: usize) -> f64; // get the value by index
    fn mul_float(&self, float: f64) -> Box<dyn VectorType>; // float multiplication
    fn len(&self) -> usize; // vector length
    fn zeros(&self, len: usize) -> Box<dyn VectorType>; // creating zero vector of length len
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType>; // assign value to index
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType>;
    fn vec_type(&self) -> String;
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
                    assert_eq!(vec.len(), d_vec.len());
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
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(_) => Box::new(DVector::zeros(len)),
            YEnum::Sparse_1(_) => Box::new(CsVec::empty(len)),
            YEnum::Sparse_2(_) => Box::new(DVector::zeros(len)),
            YEnum::Sparse_3(_) => Box::new(faer_col::zeros(len)), // faer_col::zeros(len)
        }
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(new_vec)
            }
            YEnum::Sparse_1(vec) => {
                let mut new_vec = vec.clone();
                new_vec.append(index, value);
                Box::new(new_vec)
            }
            YEnum::Sparse_2(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(new_vec)
            }
            YEnum::Sparse_3(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(new_vec)
            }
        }
    } // end assign
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        match self {
            YEnum::Dense(_) => {
                let new_matrix: DMatrix<f64> =
                    DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
                Box::new(new_matrix)
            }
            YEnum::Sparse_1(_) => {
                let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
                Box::new(new_matrix)
            }
            YEnum::Sparse_2(_) => {
                let new_matrix: CsMatrix<f64> =
                    DMatrix::from_row_slice(nrows, ncols, vec_with_zeros).into();
                Box::new(new_matrix)
            }
            YEnum::Sparse_3(_) => {
                let triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
                    .iter()
                    .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
                    .collect::<Vec<_>>();
                let new_matrix: SparseColMat<usize, f64> =
                    SparseColMat::try_new_from_triplets(nrows, ncols, triplet.as_slice()).unwrap();
                Box::new(new_matrix)
            }
        }
    } //from_vector
    fn vec_type(&self) -> String {
        match self {
            YEnum::Dense(_) => "Dense".to_string(),
            YEnum::Sparse_1(_) => "Sparse_1".to_string(),
            YEnum::Sparse_2(_) => "Sparse_2".to_string(),
            YEnum::Sparse_3(_) => "Sparse_3".to_string(),
        }
    }
} //end impl
////////////////////////////////////////////////////////////////
//           NALGEBRA CRATE
////////////////////////////////////////////////////////////////
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
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(DVector::zeros(len))
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut vec = self.clone();
            vec[index] = value;
            vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        _non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let new_matrix: DMatrix<f64> = DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
        Box::new(new_matrix)
    }
    fn vec_type(&self) -> String {
        "Dense".to_string()
    }
}
////////////////////////////////
//           SPRS CRATE
////////////////////////////////
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
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(CsVec::empty(len))
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut new_vec = self.clone();
            new_vec.append(index, value);
            new_vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        _vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
        Box::new(new_matrix)
    }
    fn vec_type(&self) -> String {
        "Sparse_1".to_string()
    }
}
////////////////////////////////////////////////////////////////////////////
//  FAER CRATE
////////////////////////////////////////////////////////////////////////////
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
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(faer_col::zeros(len)) //faer_col::zeros(len)
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut new_vec = self.clone();
            new_vec[index] = value;
            new_vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        _vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let non_zero_triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
            .iter()
            .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
            .collect::<Vec<_>>();
        let new_matrix: SparseColMat<usize, f64> =
            SparseColMat::try_new_from_triplets(nrows, ncols, non_zero_triplet.as_slice()).unwrap();
        Box::new(new_matrix)
    }
    fn vec_type(&self) -> String {
        "Sparse_3".to_string()
    }
}

/////////////////////////////////////////////////////////////
//      FUN - VECTOR-FUNCTION OF RESIDUALS
///////////////////////////////////////////////////////////////
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

impl Fun for dyn Fn(f64, &DVector<f64>) -> DVector<f64> {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
}
impl Fun for dyn Fn(f64, &faer_col) -> faer_col {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected faer_col")
        }
    }
}
impl Fun for dyn Fn(f64, &CsVec<f64>) -> CsVec<f64> {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected CsVec")
        }
    }
}
// First, create a wrapper struct for the function
pub struct FunctionWrapper(Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>>);

// Implement Fun for the wrapper
impl Fun for FunctionWrapper {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        (self.0)(x, vec)
    }
}

// Then create a conversion function
pub fn convert_to_fun(f: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>>) -> Box<dyn Fun> {
    Box::new(FunctionWrapper(f))
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

////////////////////////////////////////////////////////////////////////////
//_________________________________Jacobian______________________________
////////////////////////////////////////////////////////////////////////////
///  NUMERICAL REPRESENTATION OF JACOBIAN FOR DIFFERENT CRATES
#[allow(dead_code)]
pub enum JacTypes {
    Dense(DMatrix<f64>),
    Sparse_1(CsMat<f64>),
    Sparse_2(CsMatrix<f64>),
    Sparse_3(faer_mat),
}
pub trait MatrixType: Any {
    fn as_any(&self) -> &dyn Any;
    fn inverse(self) -> Box<dyn MatrixType>; // inverse
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType>; // multiplication of matrix and vector
    fn clone_box(&self) -> Box<dyn MatrixType>; // clone
    fn solve_sys(
        // solve linear system
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType>;
    fn shape(&self) -> (usize, usize); // shape of the matrix
    fn to_DMatrixType(&self) -> DMatrix<f64>; // convert to DMatrix
}
/*
impl Clone for dyn MatrixType {

    fn clone(&self) -> Self {
          self.as_any().downcast_ref::<Self>().clone()
    }
}
*/
////////////////////////////////////////////////////////////////
//           NALGEBRA CRATE
////////////////////////////////////////////////////////////////
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
        bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<DMatrix<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                if linear_sys_method.is_none() {
                    let res = nalgebra_solvers_depot(mat_, d_vec, linear_sys_method, bandwidth);
                    // let lu = mat_.to_owned().lu();
                    //  let res = lu.solve(d_vec).unwrap();
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

    fn to_DMatrixType(&self) -> DMatrix<f64> {
        self.to_owned()
    }
}
////////////////////////////////////////////////////////////////////////////////////
//      CRATE SPRS
////////////////////////////////////////////////////////////////////////////////////
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
        _bandwidth: (usize, usize),
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

    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let (nrows, ncols) = self.shape();
        let t = self.to_dense();
        let csmat = t.as_slice().unwrap();
        let dmatrix = DMatrix::from_row_slice(nrows, ncols, csmat);
        dmatrix
    }
}
////////////////////////////////////////////////////////////////
//           NALGEBRA SPARCE CRATE
////////////////////////////////////////////////////////////////
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
        _bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        panic!("method not written yet")
    }
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let t = self.to_owned();
        let dense: DMatrix<f64> = t.into();
        dense
    }
}
////////////////////////////////////////////////////////////////
//            FAER CRATE
////////////////////////////////////////////////////////////////
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
        _bandwidth: (usize, usize),
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

                assert_eq!(d_vec.len(), mat_.ncols());
                assert_eq!(d_vec.len(), mat_.nrows());
                //let LU0 = lu_in_place( mat_);
                //  "_gmres".to_owned();
                match linear_sys_method {
                    None => {
                        let lhs: MatRef<f64> = d_vec.as_mat();
                        let LU = mat_.sp_lu().unwrap();
                        let res: Mat<f64> = LU.solve(lhs);

                        let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                        let res = ColRef::from_slice(res_vec.as_slice()).to_owned(); // TODO! find more idiomatic 

                        Box::new(res)
                    }
                    Some(_gmres) => {
                        if let Some(old_vec) = old_vec.as_any().downcast_ref::<faer_col>() {
                            let lhs: MatRef<f64> = d_vec.as_mat();
                            let res =
                                solve_sys_SparseColMat(mat_, lhs, tol, max_iter, old_vec).unwrap();
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
    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let (nrows, ncols) = self.shape();
        let dense = self.to_dense();
        let mut dmatrix = DMatrix::zeros(nrows, ncols);
        for (i, col) in dense.col_iter().enumerate() {
            let col = col.to_owned().iter().map(|x| *x).collect::<Vec<f64>>();
            dmatrix.column_mut(i).copy_from(&DVector::from_vec(col));
        }

        dmatrix
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
////////////////////////////////////////////////////////////////////////
//  JACOBIAN MATRIX-FUNCTION
////////////////////////////////////////////////////////////////
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
                    let inv_mat = invers_Mat_LU(mat_.to_owned(), tol, max_iter).unwrap();
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
                        let lhs: MatRef<f64> = d_vec.as_mat();
                        let LU = mat_.sp_lu().unwrap();
                        let res: Mat<f64> = LU.solve(lhs);

                        let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                        let res = ColRef::from_slice(res_vec.as_slice()).to_owned();
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

pub struct JacWrapper(Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn MatrixType>>);

// Implement Fun for the wrapper
impl Jac for JacWrapper {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType> {
        (self.0)(x, vec)
    }
    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
            Box::new(mat.to_owned().try_inverse().unwrap())
        } else if let Some(mat) = matrix.as_any().downcast_ref::<CsMat<f64>>() {
            let inv_mat = invers_csmat(mat.to_owned(), tol, max_iter).unwrap();
            Box::new(inv_mat)
        } else if let Some(mat) = matrix.as_any().downcast_ref::<faer_mat>() {
            let inv_mat = invers_Mat_LU(mat.to_owned(), tol, max_iter).unwrap();
            Box::new(inv_mat)
        } else {
            panic!("Unsupported matrix type for inversion")
        }
    }
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        _tol: f64,
        _max_iter: usize,
    ) -> Box<dyn VectorType> {
        if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                let lu = mat.to_owned().lu();
                let res = lu.solve(d_vec).unwrap();
                Box::new(res)
            } else {
                panic!("Vector type mismatch for solving system")
            }
        } else if let Some(mat) = matrix.as_any().downcast_ref::<faer_mat>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                let lhs: MatRef<f64> = d_vec.as_mat();
                let LU = mat.sp_lu().unwrap();
                let res: Mat<f64> = LU.solve(lhs);

                let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                let res = ColRef::from_slice(res_vec.as_slice()).to_owned();

                Box::new(res)
            } else {
                panic!("Vector type mismatch for solving system")
            }
        } else {
            panic!("Unsupported matrix type for solving system")
        }
    }
}

// Then create a conversion function
pub fn convert_to_jac(f: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn MatrixType>>) -> Box<dyn Jac> {
    Box::new(JacWrapper(f))
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
///////////////////////////////
//     Miscellaneous
///////////////////////////////
//___________________________________________________________________
pub fn Vectors_type_casting(vec: &DVector<f64>, desired_type: String) -> Box<dyn VectorType> {
    let res: Box<dyn VectorType> = if desired_type == "Dense".to_string() {
        Box::new(YEnum::Dense(vec.clone()))
    } else if desired_type == "Sparse_1".to_string() {
        //sprs crate
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
        // faer crate
        let Mat_vec = ColRef::from_slice(vec.as_slice()).to_owned();

        Box::new(YEnum::Sparse_3(Mat_vec))
    } else if desired_type == "Sparse_2".to_string() {
        Box::new(YEnum::Sparse_2(vec.to_owned()))
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
                let element = faer_mat.as_dyn().get(i, j).unwrap_or(&0.0);
                row_data.push(element);
            }
            println!("\n \n{:?}-th row = {:?}", i, row_data);
        }
    } else {
        println!("Unknown MatrixType")
    }
}

pub fn from_vector_to_matrix(
    vec_type: String,
    nrows: usize,
    ncols: usize,
    vec_with_zeros: &Vec<f64>,
    non_zero_triplet: Vec<(usize, usize, f64)>,
) -> Box<dyn MatrixType> {
    match vec_type.as_str() {
        "Dense" => {
            let new_matrix: DMatrix<f64> = DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
            Box::new(new_matrix)
        }
        "Sparse_1" => {
            let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
            Box::new(new_matrix)
        }
        "Sparse_2" => {
            let new_matrix: CsMatrix<f64> =
                DMatrix::from_row_slice(nrows, ncols, vec_with_zeros).into();
            Box::new(new_matrix)
        }
        "Sparse_3" => {
            let non_zero_triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
                .iter()
                .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
                .collect::<Vec<_>>();

            let new_matrix: SparseColMat<usize, f64> =
                SparseColMat::try_new_from_triplets(nrows, ncols, non_zero_triplet.as_slice())
                    .unwrap();
            Box::new(new_matrix)
        }
        _ => panic!("Unsupported matrix type: {}", vec_type),
    }
} //from_vecto

fn sprs_triplet_to_csc(
    nrows: usize,
    ncols: usize,
    triplets: &Vec<(usize, usize, f64)>,
) -> CsMat<f64> {
    let mut triplet_matrix = TriMat::new((nrows, ncols));
    for (i, j, v) in triplets {
        triplet_matrix.add_triplet(*i, *j, *v);
    }

    // Convert the triplet matrix to a CSC matrix
    let csc_matrix: CsMat<f64> = triplet_matrix.to_csc();
    csc_matrix
}

pub enum YEnum {
    /// Dense Newton vector paired with a compact scalar-banded Jacobian.
    /// Keeping a distinct variant prevents the numeric Banded route from
    /// silently falling back to a dense Jacobian representation.
    Banded(DVector<f64>),
    Dense(DVector<f64>),  // dense vector NALGEBRA CRATE
    Sparse_1(CsVec<f64>), // sparse vector SPRS CRATE
    Sparse_3(faer_col),   // sparse vector  FAER CRATE
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
    /// Reports the dense representation without forcing external implementors
    /// to add a new required trait method. Built-in types override this with
    /// an allocation-free match; the compatibility default keeps the legacy
    /// `vec_type` semantics for downstream implementations.
    fn is_dense(&self) -> bool {
        self.vec_type() == "Dense"
    }
    fn vec_type(&self) -> String;
}

/// Borrows the dense Newton state for built-in vector representations.
///
/// The Banded Lambdify callback historically normalized every input through
/// `to_DVectorType`, which cloned the already-dense `DVector` hidden inside
/// both `DVector` and `YEnum::Banded`. External vector implementations still
/// use the compatibility allocation fallback.
#[inline]
pub(crate) fn dense_slice_or_copy<'a>(vector: &'a dyn VectorType) -> Cow<'a, [f64]> {
    if let Some(dense) = vector.as_any().downcast_ref::<DVector<f64>>() {
        Cow::Borrowed(dense.as_slice())
    } else {
        Cow::Owned(vector.to_DVectorType().as_slice().to_vec())
    }
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
            YEnum::Banded(vec) => vec.iter().next().copied(),
            YEnum::Sparse_1(vec) => vec.iter().map(|x| *x.1).next(),
            YEnum::Sparse_3(vec) => vec.iter().next().copied(),
        }
    }
}

impl VectorType for YEnum {
    fn as_any(&self) -> &dyn Any {
        match self {
            YEnum::Dense(vec) => vec,
            YEnum::Banded(vec) => vec,
            YEnum::Sparse_1(vec) => vec,
            YEnum::Sparse_3(vec) => vec,
        }
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => {
                let other_dense = other.to_DVectorType();
                Box::new(vec - other_dense)
            }
            YEnum::Banded(vec) => {
                let other_dense = other.to_DVectorType();
                Box::new(YEnum::Banded(vec - other_dense))
            }
            YEnum::Sparse_1(vec) => {
                if let Some(c_vec) = other.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(vec - c_vec)
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            YEnum::Sparse_3(vec) => {
                let other_dense = other.to_DVectorType();
                assert_eq!(vec.len(), other_dense.len());
                let other_faer = faer_col::from_fn(other_dense.len(), |i| other_dense[i]);
                let subs = vec.sub(&other_faer);
                Box::new(subs)
            }
        }
    } // subtract
    fn norm(&self) -> f64 {
        match self {
            YEnum::Dense(vec) => Matrix::norm(vec),
            YEnum::Banded(vec) => Matrix::norm(vec),
            YEnum::Sparse_1(vec) => CsVec::l2_norm(vec),
            YEnum::Sparse_3(vec) => vec.norm_l2(),
        }
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        match self {
            YEnum::Dense(vec) => vec.clone(),
            YEnum::Banded(vec) => vec.clone(),
            YEnum::Sparse_1(vec) => {
                let length = vec.dim();

                DVector::from_iterator(
                    length,
                    (0..length).map(|index| vec.get(index).copied().unwrap_or(0.0)),
                )
            }
            YEnum::Sparse_3(vec) => {
                let length = vec.nrows();

                DVector::from_iterator(length, (0..length).map(|index| vec[index]))
            }
        }
    } //to_iterator
    fn clone_box(&self) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.clone()),
            YEnum::Banded(vec) => Box::new(YEnum::Banded(vec.clone())),
            YEnum::Sparse_1(vec) => Box::new(vec.clone()),
            YEnum::Sparse_3(vec) => Box::new(vec.clone()),
        }
    }

    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Banded(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Sparse_1(vec) => Box::new(vec.iter().map(|x| *x.1)),
            YEnum::Sparse_3(vec) => Box::new(vec.iter().map(|x| *x)),
        }
    }
    fn get_val(&self, index: usize) -> f64 {
        match self {
            YEnum::Dense(vec) => vec[index],
            YEnum::Banded(vec) => vec[index],
            YEnum::Sparse_1(vec) => vec[index],
            YEnum::Sparse_3(vec) => vec[index],
        }
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec * float),
            YEnum::Banded(vec) => Box::new(YEnum::Banded(vec * float)),
            YEnum::Sparse_1(vec) => Box::new(vec.map(|x| x * (float))),
            YEnum::Sparse_3(vec) => Box::new(vec * float),
        }
    }
    fn len(&self) -> usize {
        match self {
            YEnum::Dense(vec) => vec.len(),
            YEnum::Banded(vec) => vec.len(),
            YEnum::Sparse_1(vec) => vec.len(),
            YEnum::Sparse_3(vec) => vec.len(),
        }
    }
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(_) => Box::new(DVector::zeros(len)),
            YEnum::Banded(_) => Box::new(YEnum::Banded(DVector::zeros(len))),
            YEnum::Sparse_1(_) => Box::new(CsVec::empty(len)),
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
            YEnum::Banded(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(YEnum::Banded(new_vec))
            }
            YEnum::Sparse_1(vec) => {
                let mut new_vec = vec.clone();
                new_vec.append(index, value);
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
            YEnum::Banded(_) => {
                assert_eq!(nrows, ncols, "banded Jacobian must be square");
                let mut lower = 0usize;
                let mut upper = 0usize;
                for &(row, col, value) in &non_zero_triplet {
                    if value == 0.0 {
                        continue;
                    }
                    assert!(
                        row < nrows && col < ncols,
                        "banded triplet is out of bounds"
                    );
                    lower = lower.max(row.saturating_sub(col));
                    upper = upper.max(col.saturating_sub(row));
                }

                let mut assembly = BandedAssembly::zeros(nrows, lower, upper)
                    .expect("numeric Banded Jacobian dimensions must be valid");
                for (row, col, value) in non_zero_triplet {
                    if value != 0.0 {
                        assembly
                            .set(row, col, value)
                            .expect("numeric Banded Jacobian triplet must fit its inferred band");
                    }
                }
                let layout = NodeMajorLayout::new(nrows, 1)
                    .expect("numeric Banded Jacobian layout must be non-empty");
                Box::new(BandedMatrixType::new_general_banded(
                    assembly,
                    layout,
                    LinearSolverConfig::faithful_banded(),
                ))
            }
            YEnum::Sparse_1(_) => {
                let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
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
    fn is_dense(&self) -> bool {
        matches!(self, YEnum::Dense(_))
    }
    fn vec_type(&self) -> String {
        match self {
            YEnum::Dense(_) => "Dense".to_string(),
            YEnum::Banded(_) => "Banded".to_string(),
            YEnum::Sparse_1(_) => "Sparse_1".to_string(),
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
    fn is_dense(&self) -> bool {
        true
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

        DVector::from_iterator(
            length,
            (0..length).map(|index| self.get(index).copied().unwrap_or(0.0)),
        )
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
    fn is_dense(&self) -> bool {
        false
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
        let other_dense = other.to_DVectorType();
        assert_eq!(self.nrows(), other_dense.len());
        let other_faer = faer_col::from_fn(other_dense.len(), |i| other_dense[i]);
        let subs = self.sub(&other_faer);
        Box::new(subs)
    }
    fn norm(&self) -> f64 {
        self.norm_l2()
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        let length = self.nrows();

        DVector::from_iterator(length, (0..length).map(|index| self[index]))
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
    fn is_dense(&self) -> bool {
        false
    }
    fn vec_type(&self) -> String {
        "Sparse_3".to_string()
    }
}

/////////////////////////////////////////////////////////////
//      FUN - VECTOR-FUNCTION OF RESIDUALS
///////////////////////////////////////////////////////////////

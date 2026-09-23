pub trait Fun {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType>;

    /// Fallible compatibility entry point for the typed solver path.
    fn try_call(
        &self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        catch_unwind(AssertUnwindSafe(|| self.call(x, vec))).map_err(|payload| {
            BvpCallbackError::Panic {
                message: panic_message(payload),
            }
        })
    }
}

pub enum FunEnum {
    Dense(Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>),
    Sparse_1(Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64>>),
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
            FunEnum::Sparse_3(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected faer_col")
                }
            }
        }
    }

    fn try_call(
        &self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        match self {
            FunEnum::Dense(fun) => {
                let d_vec = vec.as_any().downcast_ref::<DVector<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "DVector",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, d_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            FunEnum::Sparse_1(fun) => {
                let sparse_vec = vec.as_any().downcast_ref::<CsVec<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "CsVec",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, sparse_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            FunEnum::Sparse_3(fun) => {
                let faer_vec = vec.as_any().downcast_ref::<faer_col>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "faer_col",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, faer_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
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
            YEnum::Banded(vec) => write!(f, "Banded Vector: {:?}", vec),
            YEnum::Sparse_1(vec) => write!(f, "Sparse Vector 1: {:?}", vec),
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

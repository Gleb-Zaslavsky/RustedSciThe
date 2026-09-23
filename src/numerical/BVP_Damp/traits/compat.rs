pub trait Y: Any {
    fn as_any(&self) -> Box<&dyn Any>;
    fn clone_box(&self) -> Box<dyn Y>;
}

impl Y for DVector<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }

    fn clone_box(&self) -> Box<dyn Y> {
        Box::new(self.clone())
    }
}

impl Y for CsVec<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }

    fn clone_box(&self) -> Box<dyn Y> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Y> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
///////////////////////////////
//     Miscellaneous
///////////////////////////////
//___________________________________________________________________
pub fn Vectors_type_casting(vec: &DVector<f64>, desired_type: String) -> Box<dyn VectorType> {
    let res: Box<dyn VectorType> = if desired_type == "Dense".to_string() {
        Box::new(YEnum::Dense(vec.clone()))
    } else if desired_type == "Banded".to_string() {
        // Newton vectors remain dense, but the distinct variant ensures that
        // Jacobian assembly selects compact BandedAssembly instead of DMatrix.
        Box::new(YEnum::Banded(vec.clone()))
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
    } else {
        panic!("Unsupported vector type: {}", desired_type);
    };
    res
}

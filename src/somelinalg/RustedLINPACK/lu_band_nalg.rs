#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
use nalgebra::{DMatrix, DVector};
use std::mem;

/// Direct rewrite of LU decomposition with partial (row) pivoting from the nalgebra crate for
/// the case of a banded matrix
///  non-parallel version
/// faster than default lu() implementation by N/l, where N - is dimension of matrix ( nrows = ncols ), l - bandwidth
/// 

/// Computes the LU decomposition with partial (row) pivoting of `matrix`.
pub struct LU_nalgebra {
    A: DMatrix<f64>,
    lu: DMatrix<f64>,
    nrows: usize,
    p: Vec<usize>,
    l: DMatrix<f64>,
    u: DMatrix<f64>,
    kl: usize,
    ku: usize,
    y: DVector<f64>,
    band: DMatrix<f64>,
}
impl LU_nalgebra {
    pub fn new(matrix: DMatrix<f64>, bandwidth: Option<(usize, usize)>) -> LU_nalgebra {
     
        let (nrows, ncols) = matrix.shape();
   
    
        let (kl, ku) = if let Some((kl_, ku_)) = bandwidth { (kl_, ku_) } else { Self::find_bandwidths(&matrix) };
     
        println!("kl: {}, ku: {}", kl, ku);
       
        LU_nalgebra {
            A: matrix.clone(),
            lu: matrix,
            nrows: nrows,
            p: Vec::new(),
            l: DMatrix::zeros(nrows, ncols),
            u: DMatrix::zeros(nrows, ncols),
            kl: kl,
            ku: ku,
            y: DVector::zeros(nrows),
            band: DMatrix::zeros(nrows, ncols),
        }
       
    }
    pub fn new_matrix(&mut self, matrix: DMatrix<f64>, bandwidth: Option<(usize, usize)>) {
        let (kl, ku) = if let Some((kl_, ku_)) = bandwidth { (kl_, ku_) } else { Self::find_bandwidths(&matrix) };
        self.kl = kl;
        self.ku = ku;
        self.A = matrix.clone();
    }
    /// Computes the LU decomposition with partial (row) pivoting of the matrix `A`. The computed decomposition is stored in `lu` and the permutation vector is stored in `p`.
    /// The algorithm works by iterating over each column starting from the current column and finding the maximum absolute value in the current row.
    ///  This element  is used as a pivot and swapped with the current row if necessary. Then, the current column is subtracted from all rows
    /// below it using the pivot element as a scaling factor. This process is repeated for all columns. The resulting matrix is stored in `lu`
    ///  and the permutation vector is stored in `p`.
    pub fn LU(&mut self) {

        
    
        let kl = self.kl.clone();
        let ku = self.ku.clone();
        let mut matrix = &mut self.A.clone();
        let (nrows, ncols) = matrix.shape();
        self.nrows = nrows;
        if !matrix.is_square() {
            panic!("Matrix must be square");
        }
      
        // Create a permutation sequence p to represent the identity permutation.
        //  let mut p = PermutationSequence::identity(ncols);
        let mut p: Vec<usize> = (0..nrows).collect();
        // Start a loop that iterates over each column from 0 to the minimum number of rows and columns.
     
        for i in 0..ncols {
           
            let lower_border = std::cmp::min(ncols, i + kl + 1);

            //Find the pivot row by finding the maximum absolute value in the current row starting from the current column.
            let piv = matrix.view_range(i..lower_border, i).icamax() + i;
            //Extract the diagonal element diag from the pivot row and column.
            let diag = matrix[(piv, i)];
            //Check if the diagonal element is zero. If it is, continue to the next iteration (no non-zero entries on this column).
            if diag == 0.0 {
                // No non-zero entries on this column.
                continue;
            }

            //Check if the pivot row is different from the current row. If it is, append a permutation to the permutation sequence p to swap the
            // pivot row with the current row. Swap the pivot row with the current row in the matrix using
            
            if piv != i {
               
                //  p.append_permutation(i, piv);
                p.swap(i, piv);
                matrix.columns_range_mut(0..i).swap_rows(i, piv);
                Self::gauss_step_swap(&mut matrix, diag, i, piv, kl, ku, nrows);
            // Perform Gaussian elimination with swapping using the gauss_step_swap function.
               
                 
            } else {
                //  If the pivot row is the same as the current row, perform Gaussian elimination without swapping using the gauss_step function.
                
                Self::gauss_step(&mut matrix, diag, i, kl, ku, nrows);
               
            }
           
        }
       
    
        self.lu = matrix.clone();
        self.p = p;
    }

    /// Swaps the rows `i` with the row `piv` and executes one step of gaussian elimination on the i-th
    /// row and column of `matrix`. The diagonal element `matrix[(i, i)]` is provided as argument.
    pub fn gauss_step_swap(
        matrix: &mut DMatrix<f64>,
        diag: f64,
        i: usize,
        piv: usize,
        kl: usize,
        ku: usize,
        nrows: usize,
    ) {
       
        // This line calculates the relative pivot row index by subtracting the current row index i from the pivot row index piv.
        let piv = piv - i;
        //creates a mutable view of the submatrix starting from row i and column i to the end of the matrix.
        let lower_border = std::cmp::min(nrows, i + kl + 1);
        let right_border = std::cmp::min(nrows, i + ku + kl + 1);
        // let right_border = if (i as isize  - ku as isize - 1)<nrows as isize{i - ku - 1} else {nrows};
        let mut submat = matrix.view_range_mut(i..lower_border, i..right_border);

        let inv_diag = 1.0 / diag;
        //is line splits the submatrix into two parts: coeffs represents the first column (column i of the original matrix), and submat represents
        // the rest of the columns.
        let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);
        // swaps the first element of coeffs (which is the element at (0, 0)) with the element at (piv, 0) in coeffs.
        coeffs.swap((0, 0), (piv, 0));
        //takes a mutable view of coeffs excluding the first row.
        let right_border = std::cmp::min(coeffs.nrows(), kl + 1);

        let mut coeffs = coeffs.rows_range_mut(1..right_border);
      
        // multiplies all elements in coeffs by the inverse of the diagonal element inv_diag. This step calculates the multipliers for the Gaussian elimination.
        coeffs *= inv_diag;
        //splits the submatrix into two parts: pivot_row represents the first row (row i of the original matrix), and down represents the rest of the rows below
        //the pivot row.
        let (mut pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..right_border);

        // loop that iterates over each column in the pivot row.
    
        for k in 0..pivot_row.ncols() {
        
            //This line swaps the element at (k, k) in pivot_row with the element at (piv - 1, k) in down.
            mem::swap(&mut pivot_row[k], &mut down[(piv - 1, k)]);
            // performs an "axpy" operation on the corresponding column in down.
            down.column_mut(k) // The "axpy" operation is equivalent to down.column_mut(k) += -pivot_row[k] * coeffs.
                // This step updates the elements in down based on the multipliers calculated earlier.
                .axpy(-pivot_row[k], &coeffs, 1.0);
       
        }

    }

    pub fn gauss_step(
        matrix: &mut DMatrix<f64>,
        diag: f64,
        i: usize,
        kl: usize,
        ku: usize,
        nrows: usize,
    ) {
     
        // diag: The diagonal element at position (i, i) as an f64
        // Creates a mutable view of the submatrix starting from row i and column i to the end of the matrix.
        let lower_border = std::cmp::min(nrows, i + kl + 1);
        let right_border = std::cmp::min(nrows, i + ku + kl + 1);
        let mut submat = matrix.view_range_mut(i..lower_border, i..right_border); //
                                                                      //  Calculates the inverse of the diagonal element.
        let inv_diag = 1.0 / diag;
        //Splits the submatrix into two parts: coeffs: The first column (column i of the original matrix) submat: The rest of the columns
        let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);
        //Takes a mutable view of coeffs excluding the first row, then multiplies all elements by inv_diag.
        //This calculates the multipliers for the Gaussian elimination.
        let right_border = std::cmp::min(coeffs.nrows(), kl + 1);
        let mut coeffs = coeffs.rows_range_mut(1..right_border);
        coeffs *= inv_diag;
        // Splits submat into two parts: pivot_row: The first row (row i of the original matrix) down: The rest of the rows below the pivot row
        let (pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..right_border);
        //Starts a loop that iterates over each column in the pivot row
        for k in 0..pivot_row.ncols() {
       
            down.column_mut(k) // Performs an "axpy" operation on the corresponding column in down.
                .axpy(-pivot_row[k], &coeffs, 1.0); //
                                                    //"axpy" stands for "a * x plus y", where:
                                                    // a is -pivot_row[k] (negated value from the pivot row)
                                                    // x is coeffs (the multipliers calculated earlier)
                                                    // y is implicitly the column itself (modified in-place)
                                                    
        }
    } // fn gauss_step
    // more easy and faster LU decomposition solver
   pub fn LU2(
        &mut self
    )  {
        let kl = self.kl.clone();
        let ku = self.ku.clone();
        let A = self.A.clone();
        let n = A.nrows();
        let mut L = DMatrix::zeros(n, n);
        let mut U = A.clone();
        let mut P: Vec<usize> = (0..n).collect(); // Permutation vector
    
    
        for (k, _col_k) in A.column_iter().enumerate() {
    
            
            let low_border = std::cmp::min(n, k + kl + 1);
                  //Find the pivot row by finding the maximum absolute value in the current row starting from the current column.
            let piv = U.view_range(k..low_border, k).icamax() + k;
            //Extract the diagonal element diag from the pivot row and column.
            let diag = U[(piv, k)];
            //Check if the diagonal element is zero. If it is, continue to the next iteration (no non-zero entries on this column).
            if diag == 0.0 {
                // No non-zero entries on this column.
                continue;
            }
          //  println!("{} {}, {}", _col_k[piv], _col_k[k], _col_k);
    
            if piv != k {
                   
                //  p.append_permutation(i, piv);
                P.swap(k, piv);
               // Do for all rows below pivot:
                U.columns_range_mut(0..k).swap_rows(k, piv);
            }
    
         //   println!("{} ", U);
            L[(k, k)] = 1.0;
            for i in k + 1..low_border {
                L[(i, k)] = U[(i, k)] / U[(k, k)];
               // U[(i, k)]=0.0;
                let border = std::cmp::min(n, k + ku + 1);
                for j in k..border {
                    U[(i, j)] = U[(i, j)] - L[(i, k)] * U[(k, j)];
                }
            }
        }
                                        //unity matrix
        let LU:DMatrix<f64> = &L + &U - DMatrix::from_diagonal_element(n, n, 1.0);
        self.lu = LU;
        self.u = U;
        self.l = L;
        self.p = P;
    }

    pub fn l(&mut self) -> DMatrix<f64> {
        let mut m = self.lu.clone();

        m.fill_upper_triangle(0.0, 1);

        m.fill_diagonal(1.0);
        self.l = m.clone();
        m
    }

    pub fn u(&mut self) -> DMatrix<f64> {
        let m = self.lu.clone();
        let res = m.upper_triangle();
        self.u = res.clone();
        res
    }

    pub fn p(&self) -> Vec<usize> {
        self.p.clone()
    }

    pub fn solve_linear_system_easy(&mut self, b: &DVector<f64>) -> DVector<f64> {
        let P = self.p.clone();
        let n = b.len();
        let kl = self.kl;
        let ku = self.ku;
        let n = self.nrows;
        let l = self.l();
        let u = self.u();
        let mut y = DVector::zeros(n);
        let mut x = DVector::zeros(n);
        //println!("l: {}", l);
        //  println!("u: {}", u);
        // Apply permutation to b
        let mut Pb = DVector::zeros(n);
        for i in 0..n {
            Pb[i] = b[P[i]];
        }
     //   let lower_border = |i: usize| std::cmp::max(0, i as isize - kl as isize) as usize;
        // Forward substitution Ly = Pb
        for i in 0..n {
            y[i] = Pb[i];
            for j in 0..i {
                y[i] -= l[(i, j)] * y[j];
            }
        }
       // let upper_border = |i: usize| std::cmp::min(n, i + kl + 1);
        // Backward substitution Ux = y
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in i + 1..n {
                x[i] -= u[(i, j)] * x[j];
            }
            x[i] /= u[(i, i)];
        }
        self.y = x.clone();
        x
    }
    pub fn solve (&self, b: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = b.clone_owned();
        let P = self.p.clone();
       
        let n = b.len();
        let mut Pb = DVector::zeros(n);
        for i in 0..n {
            Pb[i] = res[P[i]];
        }
        let mut res = Pb.clone();
        if self.solve_mut(&mut res) {
            println!("b_b: {}", res);
            Some(res)
        } else {
            None
        }
    }

    
    fn solve_mut(&self,  b: &mut DVector<f64>)-> bool {
        

     
        assert_eq!(
            self.lu.nrows(),
            b.nrows(),
            "LU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "LU solve: unable to solve a non-square system."
        );
      

      
    
        let _ = self.lu.solve_lower_triangular_with_diag_mut(b, 1.0);
     
        self.lu.solve_upper_triangular_mut(b)
      
    }
    /*
    pub fn determinant(&self) -> T {
        let dim = self.lu.nrows();
        assert!(
            self.lu.is_square(),
            "LU determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { self.lu.get_unchecked((i, i)).clone() };
        }

        res * self.p.determinant()
    }
    */
    fn find_bandwidths(A: &DMatrix<f64>) -> (usize, usize) {
        let n = A.nrows();
        let mut kl = 0; // Number of subdiagonals
        let mut ku = 0; // Number of superdiagonals
                        /*
                            Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
                        Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
                        kl if this distance is greater than the current value of kl.
                        Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
                        and updates ku if this distance is greater than the current value of ku.
                            */
        for i in 0..n {
            for j in 0..n {
                if A[(i, j)] != 0.0 {
                    if j > i {
                        ku = std::cmp::max(ku, j - i);
                    } else if i > j {
                        kl = std::cmp::max(kl, i - j);
                    }
                }
            }
        }

        (kl, ku)
    }

    pub fn extract_band(&mut self) {
        let matrix = &self.A;
        let (nrows, ncols) = matrix.shape();
      //  println!("matrix shape: {}, {}", nrows, ncols);
        let mut subdiagonal_rows = 0;
        let mut superdiagonal_rows = 0;
        let mut band = Vec::new();
        let mut upper_diagonals = Vec::new();
        let mut lower_diagonals = Vec::new();
   
        for i in  0..nrows  {// iterating through diagonals
            
            let mut upper_diagonal:Vec<f64> = vec![0.0;nrows];
            let mut lower_diagonal:Vec<f64> = vec![0.0;nrows];
            let mut non_zero_element_upper_diagonal  = false;
            let mut non_zero_element_under_diagonal  = false;
          //  println!("i: {}", i);
            if ncols as i32-i as i32 - 1 >0 { 
               
                for j in 0..ncols-i-1 {// main diagonal is the longest,the far we go - shorter diagonal we have 
                     //  println!(" j: {}", i);
                        if matrix[( j+i+1, j)].abs() > 0.0 { 
                            non_zero_element_under_diagonal= true;

                            lower_diagonal[j+i+1] = matrix[( j+i+1, j)];
            

                        }
                        if matrix[(j, j+i+1)].abs() > 0.0 { 
                            non_zero_element_upper_diagonal= true;
                            upper_diagonal[j] = matrix[(j, j+i+1)];
                        }
                           
                }//for j in 0..ncols-i-2 {
            }// if ncols

            if non_zero_element_under_diagonal{
                subdiagonal_rows += 1;
            //    println!("lower_diagona: {:?}", lower_diagonal.clone());
                lower_diagonals.push(lower_diagonal.clone());
            }
            
          
            if non_zero_element_upper_diagonal{
                superdiagonal_rows += 1;
            //    println!("upper_diagona: {:?}", upper_diagonal.clone());
                upper_diagonals.push(upper_diagonal.clone());
            }
           // println!("band matrix: {:?}", band);
        }
        

   // println!("subdiagonal_rows: {}, superdiagonal_rows: {}", subdiagonal_rows, superdiagonal_rows);
    let main_diagonal = matrix.row_iter().enumerate().map(|(i, row_i)| row_i[i]).collect::<Vec<f64>>();
    
     band.extend(lower_diagonals);
     band.extend(vec![(main_diagonal)]);
     band.extend(upper_diagonals);
     
     
    let flat_band = band.clone().into_iter().flatten().collect::<Vec<f64>>();
   // println!("band matrix: {:?}", flat_band);
    let band = DMatrix::from_row_slice( band.len(), nrows, &flat_band).transpose();

    self.band = band;
    self.kl = subdiagonal_rows;
    self.ku = superdiagonal_rows;
    }
   
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;
    use nalgebra::DMatrix;

    #[test]
    fn test_LU() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let matrix = DMatrix::from_vec(4, 4, data);
        let lu_standard = matrix.clone().lu();

        let mut lu = LU_nalgebra::new(matrix.clone(), None);
        lu.LU();
        let l = lu.l();
        let u = lu.u();
        println!("L {:?}", l);
        println!("L standard: {:?}", lu_standard.l());
        println!("U: {:?}", u);
        println!("U standard: {:?}", lu_standard.u());
        let assertion = (&l * &u - &matrix).determinant();
        println!("assertion: {}", assertion);
        assert!(relative_eq!(assertion, 0.0, epsilon = 1e-5));
    }

    #[test]
    fn big_LU() {
        #[rustfmt::skip]
        let big = vec![
           -4.891568062814944, 7.260699648758951, 7.055590841198036, -4.34667361924181,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,-3.1937909960567668,-3.8204343410005093,-5.925296143888437,5.815824807825752,-6.238541007219842,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,-4.001633911642171,-7.402868450779367,3.9763710927602425,-1.843259939095736,-7.070623382567596,2.365872573301111,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,-1.7084822870536787,3.3087480575928794,6.00042273529203,5.622765331235771,
           -5.1645374848527315,8.623429310839246,-5.4514895704398425,0.0,0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6178459919000225,
           3.2737212617690403,-9.347990841068764, -0.09601374921562567,7.9429577660326025,-3.6794754667721863,5.4487763223636385,9.782582314040262,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-7.625559683088832, 3.2339981600457257, 4.316880297409437, 6.9822533359500625,
           -5.527051677742878,-1.2396836505394226,-5.294196661100012, -9.47953073826449,-2.2632063005734837,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.676799712052718, -9.207017072126762, 7.465261906149536, 6.313780651484969, -6.960118302618219,-9.43157106771415, -2.9017578139165723,
           -1.8250738962080693,-9.79260127852815,3.3446623776303497,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.894652699903581, -1.8460488628634764,
           -6.616175542484122, 7.177935069523009, -8.115748847049886, -3.75361133258175, -7.663040600719602,-3.504192654426763, 0.22932510516045923,
           -2.4715234963202226, -6.341195303276268,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,0.0,0.0,0.0, 8.791120207323978, 3.612917424886417,
           -4.454754858206305, -5.820093976613503, 6.787486227343408, 9.416448695194092, 9.158771738919341, -8.176780090739033, 2.286835563905761,
           -6.478539086605966, -5.231591721875097, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.833308115343261, -4.969651134533781, 6.165820461676105,
           2.6638369435206766,0.06878629767696509, -5.697898412855622, -2.1482242031804466, 5.461429112872519, 6.029154210050926, -1.8492678942051644,
           9.836846049089239,0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 9.612796337926838, 9.259351452553691, 0.07148685232728269, 7.379474282342041,
           -4.897644414095597, 0.9782135057764165, -1.9681350055801659, -3.0468339212012685, 4.913148913169572, 9.071154473356358, 8.160009179224502,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4250062332528692, -1.0353593358035376, 7.724394578782775, -3.2642437198078644, 6.064830026107661,
           9.163840161946375, -9.16420294297339, 1.0256957969155422, 4.627903850596326,9.119404443622876, 1.7516875971654287, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, -6.463177388802825, 9.039000451630187, 3.822869783728269,7.17113061563056, -1.9963788700931406,  -7.542918323395451,  6.007479423219408,
         7.357929490864613,-9.40533949345478,-4.642787299848501, 9.963777455622278, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3396284315959885,
           5.045176216140831,2.077467309123513, -1.1703436696861722, 0.8298386270925882,  5.263401683613317, 9.140087545745171, -0.1067554122155343,
           9.88378734345076, 3.8969399555854523, -8.670884005258902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.393187614914006, -2.5604125886410634,
           -9.053755277357208, -9.196209795482023, 0.3460496904643229, 4.484654516568751,  -7.536897422259958,  -7.552551952738851,  -4.737231458392692,
           8.292298113752196,0.7072284253450523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0478883082802595, -0.5318281567990333, -0.8553615411758244,
           -2.4786286252801837, -6.205559443074766, -6.894802496796997, 2.1443245368335617, 0.7734779896350865, -4.311895980308478, 3.5729837095311865,
           -9.142764585369022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.67888022429878, 0.9754870637913005, -2.2106925273066658, 2.25773422462189,
           8.035841502739597,  0.41122994279007585,  8.008300662115673,  -7.889184100300626,  1.362571060273794, 4.3623503873354785, 7.94463416153155,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.644398849051719, -0.6783199275713159, 3.2065184613680167, -3.6667750487062767, 1.6645432257128867,
        0.2851810259406484, 3.847769898134988, 0.804377292114534, 7.38381586679164, -7.59032349837319,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,
           -9.200358250412183,-6.8683997699727195, 5.196576589125881, -0.7858666699450687, -3.6303394514483234,  -7.345271002748732,  -5.420321921794349,
           8.475699481745977,1.653584649820603,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.604247613961981,  -2.3257850985546025,
           5.195182274462841,  0.09103197769932514,  4.925645800867194, 1.443285550654064, 6.141833267655457, -9.902404346186845,
       ];
        let kl = 3;
        let ku = 7;
        let matrix = DMatrix::from_vec(20, 20, big);
        let b = DVector::from_vec(vec![1.0; 20]);
        let mut lu = LU_nalgebra::new(matrix.clone(), None);
        lu.LU();
        let l = lu.l();
        let u = lu.u();
        let x = lu.solve_linear_system_easy(&b);

        let lu_standard = matrix.clone().lu();
        let x_standard = lu_standard.solve(&b).unwrap();
        assert!(relative_eq!((x.clone() - x_standard).norm(), 0.0, epsilon = 1e-5));
        //  println!("L {:?}", l);
        //  println!("L standard: {:?}", lu_standard.l());
        let dl = &l - &lu_standard.l();
        for l_i in dl.iter() {
            assert!(relative_eq!(*l_i, 0.0, epsilon = 1e-5));
        }
        //  println!("U: {:?}", u);
        //   println!("U standard: {:?}", lu_standard.u());
        let du = &u - &lu_standard.u();
        for u_i in du.iter() {
            assert!(relative_eq!(*u_i, 0.0, epsilon = 1e-5));
        }
        let assertion = (&l * &u - &matrix).determinant();
        assert!(relative_eq!(assertion, 0.0, epsilon = 1e-5));

        let x_1 = lu.solve(&b);
        assert!(relative_eq!((x - x_1.unwrap()).norm(), 0.0, epsilon = 1e-5));
    }
    #[test]
    fn real_jac(){
        #[rustfmt::skip]
        let vec = vec![-0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, 
        -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
         , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, 
         -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
           0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
           1.05, 0.0, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0, -0.05, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05, 0.0];
            let matrix = DMatrix::from_vec(20, 20, vec.clone());
            let mut lu = LU_nalgebra::new(matrix, None);
            lu.LU();
            let expected_lu = DMatrix::from_vec(20, 20, vec).lu();
            assert_eq!(lu.l(), expected_lu.l(), );
            assert_eq!(lu.u(), expected_lu.u());
           // assert_eq!(lu.p(), expected_lu.p());
            let b = vec![ -0.95, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 1.0];
            let b = DVector::from_vec(b);
            let x = lu.solve(&b).unwrap();
            let expected_x = expected_lu.solve(&b).unwrap();
            assert_eq!(x, expected_x);
           
    }
    #[test]
    fn test_extract_band() {
        // Test case 1: 5x5 banded matrix
        let matrix = DMatrix::from_row_slice(5, 5, &[
            1.0, 2.0, 0.0, 0.0, 0.0,
            3.0, 4.0, 5.0, 0.0, 0.0,
            0.0, 6.0, 7.0, 8.0, 0.0,
            0.0, 0.0, 9.0, 10.0, 11.0,
            0.0, 0.0, 0.0, 12.0, 13.0,
        ]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut bandec_instance = LU_nalgebra::new(matrix, None);
        bandec_instance.extract_band();
        let expected_band = DMatrix::from_row_slice(3, 5, &[
            0.0, 3.0, 6.0, 9.0, 12.0, 1.0, 4.0, 7.0, 10.0, 13.0, 2.0, 5.0, 8.0, 11.0, 0.
        ]).transpose();
        assert_eq!(bandec_instance.band, expected_band);


    }

}

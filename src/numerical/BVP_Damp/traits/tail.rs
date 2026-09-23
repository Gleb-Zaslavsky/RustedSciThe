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

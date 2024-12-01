use nalgebra::DMatrix;
use std::fs::File;
use std::io::{self, Write};

pub fn save_matrix_to_file(
    matrix: &DMatrix<f64>,
    headers: &Vec<String>,
    filename: &str,
) -> io::Result<()> {
    let mut file = File::create(filename)?;

    // Write headers
    writeln!(file, "{}", headers.join("\t"))?;

    // Write matrix data
    for row in matrix.row_iter() {
        let row_data: Vec<String> = row.iter().map(|&val| val.to_string()).collect();
        writeln!(file, "{}", row_data.join("\t"))?;
    }

    Ok(())
}

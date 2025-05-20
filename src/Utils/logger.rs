use csv::Writer;
use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::io::{self, Write};

pub fn save_matrix_to_file(
    matrix: &DMatrix<f64>,
    headers: &Vec<String>,
    filename: &str,
    x_mesh: &DVector<f64>,
    arg: &String,
) -> io::Result<()> {
    let mut file = File::create(filename)?;
    let mut headers_with_x = Vec::new();
    headers_with_x.push(arg.clone());
    headers_with_x.extend(headers.iter().cloned());
    // Write headers
    writeln!(file, "{}", headers_with_x.join("\t"))?;
    for (i, row) in matrix.row_iter().enumerate() {
        let mut row_data = Vec::new();
        row_data.push(x_mesh[i].to_string());
        row_data.extend(row.iter().map(|&val| val.to_string()));
        writeln!(file, "{}", row_data.join("\t"))?;
    }

    Ok(())
}
pub fn save_matrix_to_csv(
    matrix: &DMatrix<f64>,
    headers: &Vec<String>,
    filename: &str,
    x_mesh: &DVector<f64>,
    arg: &String,
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = Writer::from_writer(file);

    // Prepare and write headers
    let mut headers_with_x = Vec::new();
    headers_with_x.push(arg.clone());
    headers_with_x.extend(headers.iter().cloned());
    writer.write_record(&headers_with_x)?;

    // Write data rows
    for (i, row) in matrix.row_iter().enumerate() {
        let mut row_data = Vec::new();
        row_data.push(x_mesh[i].to_string());
        row_data.extend(row.iter().map(|&val| val.to_string()));
        writer.write_record(&row_data)?;
    }

    writer.flush()?;
    Ok(())
}

use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::io::{self, Write};
use csv::Writer;
use std::path::Path;
use std::env;

pub fn save_matrix_to_file(
    matrix: &DMatrix<f64>,
    headers: &Vec<String>,
    filename: &str,
    x_mesh: &DVector<f64>,
    arg: &String
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
    arg: &String
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


/* 
pub fn  save_matrix_to_csv(
    matrix: &DMatrix<f64>,
    values: &Vec<String>,
    filename: &str,
    x_mesh: &DVector<f64>,
    arg: &String) -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let path = Path::new(&current_dir); //.join("f:\\RUST\\RustProjects_\\RustedSciThe3\\src\\numerical\\results\\");
    let file_name = filename;// format!("{}+{}.csv", arg, values.join("+"));
    let full_path = path.join(file_name);

    let mut wtr = Writer::from_path(full_path)?;

    // Write column titles
    wtr.write_record(&[arg, values])?;

    // Write time column
    wtr.write_record(x_mesh.iter().map(|&x| x.to_string()))?;

    // Write y columns
    for (i, col) in matrix.column_iter().enumerate() {
        let col_name = format!("{}", &values[i]);
        wtr.write_record(&[
            &col_name,
            &col.iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ])?;
    }

    println!("result saved");
    wtr.flush()?;
    Ok(())
}
*/
use csv::Writer;
use nalgebra::{DMatrix, DVector};
use std::fs::{self, File};
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

///in my root folder with rust program src, target etc while code working appears many files like
/// log_2025-08-13_17-42-05.txt which format of name is log_date_time and this is function
///  to delete this files crested earlier then yesterday (so crate like chrono is needed).
/// this logs deleted only from root folder of projects not in the nested folders
pub fn delete_old_logs(root_path: &str) -> io::Result<()> {
    use chrono::Local;

    let yesterday = Local::now().naive_local().date() - chrono::Duration::days(1);

    for entry in fs::read_dir(root_path)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.starts_with("log_") && filename.ends_with(".txt") {
                if let Some(date_str) = filename
                    .strip_prefix("log_")
                    .and_then(|s| s.split('_').next())
                {
                    if let Ok(file_date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                        if file_date <= yesterday {
                            fs::remove_file(&path)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod old_logs {
    use super::*;
    #[test]
    fn call_delete_old_logs() {
        let binding = std::env::current_dir().unwrap();
        let path = binding.to_str().unwrap();
        println!("cleaning logs from{}", path);
        let _ = delete_old_logs(path);
    }
}

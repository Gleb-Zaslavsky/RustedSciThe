use nalgebra::{DMatrix, DVector};
 use textplots::{Chart, Plot, Shape};
pub fn plots(arg: String, values: Vec<String>, t_result: DVector<f64>, y_result: DMatrix<f64>) {
    use plotters::prelude::*;
    // Example data
    let x = t_result;
    let y = y_result;
    let x_min = x.min();
    let x_max = x.max();
    for col in 0..y.ncols() {
        let y_col = y.column(col);
        //   println!("{}" , y_col);
        let y_min = y_col.min();
        let y_max = y_col.max();
        let varname = values[col].clone();
        let filename = format!("{}.png", varname);
        let root_area = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        // Create a chart builder
        let mut chart = ChartBuilder::on(&root_area)
            .caption(format!("{}", varname), ("sans-serif", 50))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min * 0.95..x_max * 1.05, y_min * 0.95..y_max * 1.05)
            .unwrap();

        // Configure the mesh
        chart
            .configure_mesh()
            .x_desc(&arg)
            .y_desc(&varname)
            .draw()
            .unwrap();

        // Plot the variable
        let series: Vec<(f64, f64)> = x.iter().zip(y_col.iter()).map(|(&x, &y)| (x, y)).collect();
        // print!("\n \n series {:?} \n", series);
        chart
            .draw_series(LineSeries::new(series, &Palette99::pick(col)))
            .unwrap()
            .label(format!(" {}", varname))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], &Palette99::pick(col))
            });

        // Configure the legend
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }
}

use gnuplot::{AxesCommon, Caption, Color, Figure};
pub fn plots_gnulot(
    arg: String,
    values: Vec<String>,
    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
) {
    let x = t_result;
    //  println!("{:?}, {}", &x, &x.len());
    // println!("nrows: {:?}, ncols: {} \n ",  y_result.nrows(), y_result.ncols(),);
    //  println!("{:?} \n", &y_result);
    // Create a new figure for each y variable
    for col in 0..y_result.ncols() {
        let mut fg = Figure::new();
        let y_col: Vec<f64> = y_result.column(col).iter().copied().collect();
        let varname = &values[col];
        //  println!("\n {}, \n {:?},\n {} \n", varname, &y_col, &y_col.len());

        fg.axes2d()
            .set_title(&varname, &[])
            .set_x_label(&arg, &[])
            .set_y_label(&varname, &[])
            .lines(
                x.as_slice(),
                &y_col,
                &[Caption(&varname), Color(gnuplot::ColorType::Black)],
            );

        // Save the plot to a file
        let filename = format!("{}.png", varname);
        fg.save_to_png(&filename, 800, 600).unwrap();
    }
}

pub fn plots_terminal(arg: String, values: Vec<String>, t_result: DVector<f64>, y_result: DMatrix<f64>) {
    let x = &t_result;
    let x_min = x.min();
    let x_max = x.max();
    
    for col in 0..y_result.ncols() {
        let y_col = y_result.column(col);
        let y_min = y_col.min();
        let y_max = y_col.max();
        let varname = &values[col];
        
        // Convert data to f32 for textplots
        let points: Vec<(f32, f32)> = x.iter()
            .zip(y_col.iter())
            .map(|(&x_val, &y_val)| (x_val as f32, y_val as f32))
            .collect();
        
        println!("\n{} = {} vs {} (range: {:.3} to {:.3})", 
                 varname, varname, arg, y_min, y_max);
        println!("{}", "=".repeat(60));
        
        Chart::new(120, 30, x_min as f32, x_max as f32)
            .lineplot(&Shape::Lines(&points))
            .display();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_plots_terminal() {
        // Create test data: sine and cosine functions
        let n_points = 50;
        let t: Vec<f64> = (0..n_points).map(|i| i as f64 * 2.0 * PI / (n_points - 1) as f64).collect();
        let t_result = DVector::from_vec(t.clone());
        
        let sin_values: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let cos_values: Vec<f64> = t.iter().map(|&x| x.cos()).collect();
        
        let mut y_matrix = DMatrix::zeros(n_points, 2);
        for i in 0..n_points {
            y_matrix[(i, 0)] = sin_values[i];
            y_matrix[(i, 1)] = cos_values[i];
        }
        
        let arg = "t".to_string();
        let values = vec!["sin(t)".to_string(), "cos(t)".to_string()];
        
        println!("Testing terminal plots with sine and cosine functions:");
        plots_terminal(arg, values, t_result, y_matrix);
    }
}
use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;

pub fn plots(arg: String, values:Vec<String>, t_result: DVector<f64>, y_result: DMatrix<f64>) {
    // Example data
    let x = t_result;
    let y = y_result;
    let x_min =x[0];
    let x_max = x[x.len()-1];
    for col in 0..y.ncols() {
        let y_col = y.column(col);
      // println!("{}" , y_col);
        let y_min = y_col[0];
        let y_max = y_col[y_col.len()-1];
        let varname = values[col].clone();
        let filename = format!("{}.png", varname);
        let root_area = BitMapBackend::new(&filename, (800, 600))
            .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        // Create a chart builder
        let mut chart = ChartBuilder::on(&root_area)
            .caption(format!("{}", varname), ("sans-serif", 50))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        // Configure the mesh
        chart.configure_mesh()
            .x_desc(&arg)
            .y_desc(&varname)
            .draw()
            .unwrap();

        // Plot the variable
        let series: Vec<(f64, f64)> = x.iter().zip(y_col.iter()).map(|(&x, &y)| (x, y)).collect();
      //  print!("series {:?} \n", series);
        chart.draw_series(LineSeries::new(
            series,
            &Palette99::pick(col),
        )).unwrap()
        .label(format!(" {}", varname))
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &Palette99::pick(col)));

        // Configure the legend
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }





}





use RustedSciThe::numerical::BE::{
    BE, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

pub fn diffusion_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    DVector::from_fn(y.len(), |index, _| {
        let left = if index == 0 { 0.0 } else { y[index - 1] };
        let right = if index + 1 == y.len() {
            0.0
        } else {
            y[index + 1]
        };
        0.05 * (left - 2.0 * y[index] + right)
    })
}

pub fn diffusion_jacobian(_: f64, y: &DVector<f64>) -> DMatrix<f64> {
    let n = y.len();
    DMatrix::from_fn(n, n, |row, column| match row.abs_diff(column) {
        0 => -0.1,
        1 => 0.05,
        _ => 0.0,
    })
}

pub fn combustion_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    let [fuel, radical, _product] = [y[0], y[1], y[2]];
    let reaction = 0.1 * fuel * radical;
    let secondary = 0.2 * radical * radical;
    DVector::from_vec(vec![
        -0.3 * fuel + reaction,
        0.3 * fuel - reaction - secondary,
        secondary,
    ])
}

pub fn combustion_jacobian(_: f64, y: &DVector<f64>) -> DMatrix<f64> {
    let [fuel, radical, _product] = [y[0], y[1], y[2]];
    DMatrix::from_row_slice(
        3,
        3,
        &[
            -0.3 + 0.1 * radical,
            0.1 * fuel,
            0.0,
            0.3 - 0.1 * radical,
            -0.1 * fuel - 0.4 * radical,
            0.0,
            0.0,
            0.4 * radical,
            0.0,
        ],
    )
}

pub fn make_diffusion_solver(
    dimension: usize,
    analytic_jacobian: bool,
    telemetry: BeTelemetryMode,
    step_size: f64,
    final_time: f64,
) -> BE {
    let mut solver = BE::new();
    solver
        .try_set_telemetry_mode(telemetry)
        .expect("telemetry mode should be selectable before preparation");
    let values = (0..dimension).map(|i| format!("y{i}")).collect();
    let initial = DVector::from_fn(dimension, |i, _| 1.0 + i as f64 / dimension as f64);
    let jacobian =
        analytic_jacobian.then_some(diffusion_jacobian as fn(f64, &DVector<f64>) -> DMatrix<f64>);
    solver
        .try_set_native_initial(
            values,
            "t".to_string(),
            1e-10,
            16,
            Some(step_size),
            0.0,
            final_time,
            initial,
            diffusion_rhs,
            jacobian,
        )
        .expect("diffusion-chain native problem should validate");
    solver
}

pub fn make_combustion_solver(telemetry: BeTelemetryMode) -> BE {
    let mut solver = BE::new();
    solver
        .try_set_telemetry_mode(telemetry)
        .expect("telemetry mode should be selectable before preparation");
    solver
        .try_set_native_initial(
            vec!["fuel".into(), "radical".into(), "product".into()],
            "t".into(),
            1e-10,
            20,
            Some(0.001),
            0.0,
            0.01,
            DVector::from_vec(vec![0.9, 0.1, 0.0]),
            combustion_rhs,
            Some(combustion_jacobian),
        )
        .expect("combustion-like native problem should validate");
    solver
}

pub fn make_parameterized_decay_solver(
    rate: f64,
    initial_value: f64,
    initial_time: f64,
    final_time: f64,
    telemetry: BeTelemetryMode,
) -> BE {
    let mut solver = BE::new();
    solver
        .try_set_telemetry_mode(telemetry)
        .expect("telemetry mode should be selectable before preparation");
    solver
        .try_set_initial(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-12,
            20,
            Some(0.1),
            initial_time,
            final_time,
            DVector::from_vec(vec![initial_value]),
        )
        .expect("parameterized symbolic decay problem should validate");
    solver
        .try_set_equation_parameters(Some(&["rate"]))
        .expect("rate parameter schema should validate");
    solver
        .set_parameter_values(DVector::from_vec(vec![rate]))
        .expect("rate should be bound");
    solver
}

pub fn make_parameterized_combustion_solver(
    rate: f64,
    telemetry: BeTelemetryMode,
    assembly: BeSymbolicAssemblyBackend,
) -> BE {
    let equations = vec![
        Expr::parse_expression("-0.3*fuel+rate*fuel*radical"),
        Expr::parse_expression("0.3*fuel-rate*fuel*radical-0.2*radical^2"),
        Expr::parse_expression("0.2*radical^2"),
    ];
    let options = BeSolverOptions::new(
        equations,
        vec!["fuel".into(), "radical".into(), "product".into()],
        "t".into(),
        1e-10,
        16,
        Some(0.001),
        0.0,
        0.04,
        DVector::from_vec(vec![0.9, 0.1, 0.0]),
    )
    .with_telemetry_mode(telemetry)
    .with_symbolic_assembly_backend(assembly);
    let mut solver = BE::try_new_with_options(options)
        .expect("parameterized combustion-like BE problem should validate");
    solver
        .try_set_equation_parameters(Some(&["rate"]))
        .expect("combustion rate schema should validate");
    solver
        .set_parameter_values(DVector::from_vec(vec![rate]))
        .expect("combustion rate should bind");
    solver
}

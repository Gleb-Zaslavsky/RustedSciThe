//! Fit once, then evaluate, extrapolate, and reconstruct explicitly known gaps.
//!
//! Run with `cargo run --example universal_fitting_post_fit`.

use RustedSciThe::numerical::optimization::{
    MissingInterval, UniversalFitting, UniversalFittingResult,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![0.0, 1.0, 2.0, 4.0, 5.0, 6.0];
    let y = x.iter().map(|&x| 2.5 * x + 1.0).collect();
    let fit = UniversalFitting::new()
        .with_data(x, y)
        .with_equation_str("a*x+b")
        .with_arg("x")
        .with_unknowns(["a", "b"])
        .with_initial_guess(vec![1.0, 0.0])
        .build()?;

    let values = fit.evaluate_uniform_range(2.0, 4.0, 5)?;
    println!(
        "Interpolated values: {:?}",
        values.values.column(0).iter().copied().collect::<Vec<_>>()
    );

    let extrapolated = fit.extrapolate_uniform_range(6.5, 8.5, 3)?;
    println!(
        "Right extrapolation: {:?}",
        extrapolated
            .values
            .column(0)
            .iter()
            .copied()
            .collect::<Vec<_>>()
    );

    let gaps = fit.fill_missing_intervals(&[MissingInterval {
        start: 2.0,
        end: 4.0,
        missing_points: 3,
    }])?;
    println!("Restored gap x: {:?}", gaps[0].x);
    println!(
        "Restored gap y: {:?}",
        gaps[0].values.column(0).iter().copied().collect::<Vec<_>>()
    );

    let UniversalFittingResult::LM(fit_result) = fit else {
        unreachable!("LM was selected by default");
    };
    println!("Fitted coefficients: {:?}", fit_result.solution_map());
    Ok(())
}

use splines::{Interpolation, Key, Spline};

//==============================================================
#[derive(Clone, Copy, Debug)]
pub enum SplineKind {
    Linear,
    Cosine,
    Bezier(f64),
}

impl SplineKind {
    pub fn to_interpolation(self) -> Interpolation<f64, f64> {
        match self {
            SplineKind::Linear => Interpolation::Linear,
            SplineKind::Cosine => Interpolation::Cosine,
            SplineKind::Bezier(tension) => Interpolation::Bezier(tension),
        }
    }
}

pub fn spline_domain(x: &[f64], kind: SplineKind) -> Result<(f64, f64), String> {
    if x.len() < 2 {
        return Err("Need at least 2 points".into());
    }

    match kind {
        SplineKind::Linear => Ok((x[0], x[x.len() - 1])),
        SplineKind::Cosine | SplineKind::Bezier(_) => {
            if x.len() < 4 {
                return Err("Cubic spline requires at least 4 points".into());
            }
            Ok((x[0], x[x.len() - 1]))
        }
    }
}
/*
3. Шаг 1 — генерация нового грида
Контракт

вход: t_old, n_points
выход: равномерный t_new
без привязки к данным
*/
pub fn uniform_grid_from(t: &[f64], n_points: usize, kind: SplineKind) -> Result<Vec<f64>, String> {
    let (t_min, t_max) = spline_domain(t, kind)?;

    if n_points < 2 {
        return Err("n_points must be >= 2".into());
    }

    // Add a larger epsilon to avoid boundary issues with spline evaluation
    // Cubic splines need more margin than linear splines
    let eps = match kind {
        SplineKind::Linear => (t_max - t_min) * 1e-12,
        SplineKind::Cosine | SplineKind::Bezier(_) => (t_max - t_min) * 1e-12,
    };

    let adjusted_min = t_min + eps;
    let adjusted_max = t_max - eps;

    let dt = (adjusted_max - adjusted_min) / (n_points as f64 - 1.0);

    let mut grid = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let v = if i == 0 {
            adjusted_min
        } else if i == n_points - 1 {
            adjusted_max
        } else {
            adjusted_min + i as f64 * dt
        };
        grid.push(v);
    }

    Ok(grid)
}
/*
4. Шаг 2 — построение ключей сплайна
Контракт
вход: x, y
выход: Vec<Key<_, _>>
без вычислений

*/
pub fn spline_keys(x: &[f64], y: &[f64], kind: SplineKind) -> Result<Vec<Key<f64, f64>>, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if x.len() < 2 {
        return Err("Need at least 2 points for spline".into());
    }

    let interp = kind.to_interpolation();

    Ok(x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| Key::new(xi, yi, interp))
        .collect())
}
/*
Шаг 3 — вычисление значений по сплайну
Контракт
вход: keys, x_new
выход: y_new
чистая функция
*/

pub fn spline_eval(keys: Vec<Key<f64, f64>>, x_new: &[f64]) -> Result<Vec<f64>, String> {
    let spline = Spline::from_vec(keys);

    x_new
        .iter()
        .map(|&x| {
            spline
                .sample(x)
                .ok_or_else(|| format!("Spline evaluation failed at x={}", x))
        })
        .collect()
}

pub fn spline_resample(
    x: &[f64],
    y: &[f64],
    n_points: usize,
    kind: SplineKind,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let x_new = uniform_grid_from(x, n_points, kind)?;
    let keys = spline_keys(x, y, kind)?;
    let y_new = spline_eval(keys, &x_new)?;

    Ok((x_new, y_new))
}

// Quality metrics
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mse: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64;
    mse.sqrt()
}

pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64
}

pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let y_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    1.0 - (ss_res / ss_tot)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spline_resample_linear() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

        let (xr, yr) = spline_resample(&x, &y, 50, SplineKind::Linear).unwrap();

        for (xv, yv) in xr.iter().zip(yr.iter()) {
            assert!((yv - (2.0 * xv + 1.0)).abs() < 1e-8);
        }
    }
    #[test]
    fn spline_resample_exp() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&t| (-t).exp()).collect();

        let (xr, yr) = spline_resample(&x, &y, 200, SplineKind::Cosine).unwrap();

        assert_eq!(xr.len(), 200);
        assert_eq!(yr.len(), 200);
        assert!(yr[0] > yr[yr.len() - 1]);
    }

    #[test]
    fn spline_bezier() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v).collect();

        let (xr, yr) = spline_resample(&x, &y, 50, SplineKind::Bezier(0.5)).unwrap();
        assert_eq!(xr.len(), 50);
        assert_eq!(yr.len(), 50);
    }

    #[test]
    fn spline_quadratic() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v + 2.0 * v + 1.0).collect();

        let (xr, yr) = spline_resample(&x, &y, 100, SplineKind::Linear).unwrap();
        let y_true: Vec<f64> = xr.iter().map(|&v| v * v + 2.0 * v + 1.0).collect();

        let error = rmse(&y_true, &yr);
        assert!(error < 1.0);
    }

    #[test]
    fn spline_exponential() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|&t| (2.0 * t).exp()).collect();

        let (xr, yr) = spline_resample(&x, &y, 150, SplineKind::Cosine).unwrap();
        let y_true: Vec<f64> = xr.iter().map(|&t| (2.0 * t).exp()).collect();

        let r2 = r_squared(&y_true, &yr);
        assert!(r2 > 0.9);
    }

    #[test]
    fn spline_sine() {
        let x: Vec<f64> = (0..5000).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();

        let (xr, yr) = spline_resample(&x, &y, 200, SplineKind::Bezier(0.3)).unwrap();
        let y_true: Vec<f64> = xr
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();

        let mae_val = mae(&y_true, &yr);
        println!("mae {}", mae_val);
        assert!(mae_val < 0.2);
    }

    #[test]
    fn quality_metrics() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let rmse_val = rmse(&y_true, &y_pred);
        let mae_val = mae(&y_true, &y_pred);
        let r2_val = r_squared(&y_true, &y_pred);

        assert!(rmse_val < 0.2);
        assert!(mae_val < 0.2);
        assert!(r2_val > 0.9);
    }

    #[test]
    fn spline_grid_monotonic() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let grid = uniform_grid_from(&x, 100, SplineKind::Cosine).unwrap();

        for i in 1..grid.len() {
            assert!(grid[i] > grid[i - 1]);
        }
    }
    #[test]
    fn spline_domain_error() {
        let x = vec![0.0, 1.0, 2.0];
        let err = spline_domain(&x, SplineKind::Cosine).unwrap_err();
        assert!(err.contains("at least 4"));
    }

    #[test]
    fn uniform_grid_linear_basic() {
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let n = 11;

        let grid = uniform_grid_from(&t, n, SplineKind::Linear).unwrap();

        assert_eq!(grid.len(), n);

        // monotonic
        for w in grid.windows(2) {
            assert!(w[1] > w[0]);
        }

        // inside domain
        assert!(grid[0] > 0.0);
        assert!(grid[n - 1] < 4.0);

        // uniform step
        let dt = grid[1] - grid[0];
        for i in 1..grid.len() - 1 {
            let dti = grid[i + 1] - grid[i];
            assert!((dti - dt).abs() < 1e-12);
        }
    }
    #[test]
    fn uniform_grid_cubic_requires_enough_points() {
        let t_short = vec![0.0, 1.0, 2.0];

        let err = uniform_grid_from(&t_short, 10, SplineKind::Cosine).unwrap_err();
        assert!(err.contains("at least 4 points"));
    }
    #[test]
    fn uniform_grid_invalid_npoints() {
        let t = vec![0.0, 1.0];

        let err = uniform_grid_from(&t, 1, SplineKind::Linear).unwrap_err();
        assert!(err.contains("n_points"));
    }
}

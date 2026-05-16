//=================================================================================================
// TESTING
//=================================================================================================
#[cfg(test)]
mod tests {
    use crate::numerical::data_processing::LSQSplines::{
        SolverKind, SymmetricBanded, basis_functions, find_span, make_lsq_spline,
        make_lsq_univariate_spline, rms,
    };
    use std::time::Instant;

    use ndarray::Array1;
    use ndarray::ArrayBase;
    use ndarray::s;
    //
    #[test]
    fn test_lsq_basic_fit() {
        use ndarray::Array1;

        let x: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>, f64> =
            Array1::linspace(-3., 3., 200);
        let y = x.mapv(|v| (-v * v).exp());

        let internal = Array1::linspace(-2., 2., 10);

        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let xs = Array1::linspace(-3., 3., 100);

        let ys = xs.mapv(|v| spline.evaluate(v));

        let mse = (&ys - &xs.mapv(|v| (-v * v).exp()))
            .mapv(|v| v * v)
            .mean()
            .unwrap();

        assert!(mse < 1e-3);
    }

    //
    #[test]
    fn test_sw_violation() {
        use ndarray::array;

        let x = array![0., 1., 2., 3.];
        let y = x.clone();

        let internal = array![10.]; // outside

        let result = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR);

        assert!(result.is_err());
    }

    #[test]
    fn test_exact_cubic_reproduction() {
        use ndarray::Array1;

        let x = Array1::linspace(-2., 2., 200);
        let y = x.mapv(|v| 3.0 * v * v * v - 2.0 * v * v + v - 5.0);

        let internal = Array1::linspace(-1.5, 1.5, 8);

        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let xs = Array1::linspace(-2., 2., 100);

        let ys: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>, f64> =
            xs.mapv(|v| spline.evaluate(v));

        let true_vals: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>, f64> =
            xs.mapv(|v| 3.0 * v * v * v - 2.0 * v * v + v - 5.0);

        let max_err: f64 = (&ys - &true_vals)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        assert!(max_err < 1e-8);
    }

    #[test]
    fn test_large_dataset_stability() {
        use ndarray::Array1;

        let x = Array1::linspace(0., 1000., 100_000);
        let y: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>, f64> =
            x.mapv(|v| (v / 50.0_f64).sin());

        // Internal knots should be strictly between xmin and xmax
        let internal = Array1::linspace(10., 990., 50);

        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let val = spline.evaluate(123.456);

        assert!(val.is_finite());
    }

    // banded matrix tests
    #[test]
    fn test_banded_indexing() {
        let mut A = SymmetricBanded::new(5, 2);

        A.add(1, 4, 10.0); // outside half-bandwidth (|4-1| = 3 > 2)
        A.add(1, 2, 5.0); //

        assert_eq!(A.get(1, 2), 5.0);
        assert_eq!(A.get(2, 1), 5.0);
        assert_eq!(A.get(1, 4), 0.0);
    }

    #[test]
    fn test_banded_cholesky_small() {
        let mut A = SymmetricBanded::new(3, 1);

        A.add(0, 0, 4.0);
        A.add(0, 1, 1.0);
        A.add(1, 1, 3.0);
        A.add(1, 2, 1.0);
        A.add(2, 2, 2.0);

        A.cholesky_in_place().unwrap();

        // РџСЂРѕРІРµСЂСЏРµРј СЂРµС€РµРЅРёРµ A x = b
        let mut rhs = vec![1.0, 2.0, 3.0];

        A.solve_spd_in_place(&mut rhs).unwrap();

        //
        let dense =
            nalgebra::DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);

        let sol = dense
            .lu()
            .solve(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]))
            .unwrap();

        for i in 0..3 {
            assert!((rhs[i] - sol[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_partition_of_unity() {
        let knots = Array1::from(vec![0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.]);

        let x = 1.5;
        let degree = 3;
        let n = knots.len() - degree - 1;

        let span = find_span(n - 1, degree, x, &knots);
        let basis = basis_functions(span, x, degree, &knots);

        let sum: f64 = basis.iter().sum();

        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dense_vs_banded_large() {
        let x = Array1::linspace(-5_f64, 5_f64, 5000);
        let y = x.mapv(|v| (-(v * v)).exp());

        let internal = Array1::linspace(-4., 4., 30);

        let s_dense =
            make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let s_band = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::Banded).unwrap();

        let diff = (&s_dense.coeffs - &s_band.coeffs)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        assert!(diff < 1e-8);
    }

    #[test]
    fn test_schoenberg_whitney_violation() {
        let x = Array1::linspace(0., 1., 10);

        //
        let internal = Array1::linspace(0.1, 0.9, 8);

        let res = make_lsq_univariate_spline(&x, &x, &internal, 3, SolverKind::DenseQR);

        assert!(res.is_err());
    }

    #[test]
    fn test_dense_vs_banded() {
        let x: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>, f64> =
            Array1::linspace(-3., 3., 2000);
        let y = x.mapv(|v| (-v * v).exp());

        let internal = Array1::linspace(-2., 2., 20);

        let spline_dense =
            make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let spline_banded =
            make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::Banded).unwrap();

        let diff = (&spline_dense.coeffs - &spline_banded.coeffs)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        assert!(diff < 1e-8);
    }

    #[test]
    fn test_dense_vs_banded_weighted() {
        let x = Array1::linspace(-2.5_f64, 2.5_f64, 1200);
        let y = x.mapv(|v| (2.0 * v).sin() + 0.15 * v * v);
        let w = x.mapv(|v| 1.0 + 0.5 * (v * v) / 6.25);
        let internal = Array1::linspace(-2.0, 2.0, 16);

        let degree = 3;
        let xmin = x[0];
        let xmax = x[x.len() - 1];

        let mut knots = Vec::new();
        for _ in 0..=degree {
            knots.push(xmin);
        }
        for &k in internal.iter() {
            knots.push(k);
        }
        for _ in 0..=degree {
            knots.push(xmax);
        }
        let knots = Array1::from(knots);

        let dense = make_lsq_spline(&x, &y, &knots, degree, Some(&w), SolverKind::DenseQR).unwrap();
        let banded = make_lsq_spline(&x, &y, &knots, degree, Some(&w), SolverKind::Banded).unwrap();

        let diff = (&dense.coeffs - &banded.coeffs)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        assert!(diff < 1e-8);
    }

    #[test]
    fn test_banded_cholesky_non_spd_fails() {
        let mut a = SymmetricBanded::new(2, 1);
        a.add(0, 0, 1.0);
        a.add(0, 1, 2.0);
        a.add(1, 1, 1.0);

        assert!(a.cholesky_in_place().is_err());
    }

    #[test]
    fn test_banded_stability_medium_large() {
        let x = Array1::linspace(-20.0_f64, 20.0_f64, 25_000);
        let y = x.mapv(|v| (0.4 * v).sin() + 0.05 * v * v - 0.002 * v * v * v);
        let internal = Array1::linspace(-18.0, 18.0, 70);

        let dense = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();
        let banded = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::Banded).unwrap();

        let coeff_diff = (&dense.coeffs - &banded.coeffs)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        let probe = Array1::linspace(-20.0, 20.0, 200);
        let dense_eval = probe.mapv(|v| dense.evaluate(v));
        let banded_eval = probe.mapv(|v| banded.evaluate(v));
        let eval_diff = (&dense_eval - &banded_eval)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(*b));

        assert!(coeff_diff < 1e-7);
        assert!(eval_diff < 1e-7);
    }

    #[test]
    // #[ignore = "stress/perf test; run manually with: cargo test test_banded_stress_very_large -- --ignored --nocapture"]
    fn test_banded_stress_very_large() {
        let instant = Instant::now();
        unsafe { std::env::set_var("LSQ_SPLINES_TIMING", "on") };
        let x = Array1::linspace(0.0_f64, 5000.0_f64, 3000_000);
        let y = x.mapv(|v| (v / 120.0).sin() + 0.02 * (v / 60.0).cos());
        let internal = Array1::linspace(20.0, 4980.0, 120);

        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::Banded).unwrap();

        assert!(spline.coeffs.iter().all(|c| c.is_finite()));
        assert!(spline.evaluate(1234.5).is_finite());
        assert!(spline.evaluate(4321.0).is_finite());
        println!(
            "processing time: {}",
            instant.elapsed().as_secs_f64() * 1_000.0
        );
    }
    //====================================================================================
    // SUBJECT AREA TESTS
    //====================================================================================
    #[test]
    fn test_clean_exponential_preserved() {
        let k = 3e-6;
        let t = Array1::linspace(0f64, 1e6_f64, 200_000);

        let y_true = t.mapv(|ti| 1.0 - (-k * ti).exp());

        let internal = Array1::linspace(0., 1e6, 40);

        let spline = make_lsq_univariate_spline(
            &t,
            &y_true,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            3,
            SolverKind::Banded,
        )
        .unwrap();

        // редуцируем до 1000 точек
        let t_reduced = Array1::linspace(0., 1e6, 1000);
        let y_fit = t_reduced.mapv(|ti| spline.evaluate(ti));

        let y_true_reduced = t_reduced.mapv(|ti| 1.0 - (-k * ti).exp());

        let err = rms(&y_fit, &y_true_reduced);

        assert!(err < 1e-4);
    }
    #[test]
    fn test_noise_suppression() {
        let k = 3e-6;
        let sigma = 5e-4;

        let t = Array1::linspace(0_f64, 1e6_f64, 500_000);

        let mut y = t.mapv(|ti| 1.0 - (-k * ti).exp());

        // добавляем белый шум
        for yi in y.iter_mut() {
            *yi += sigma * rand::random::<f64>() - sigma / 2.0;
        }

        let internal = Array1::linspace(0., 1e6, 40);

        let spline = make_lsq_univariate_spline(
            &t,
            &y,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            3,
            SolverKind::Banded,
        )
        .unwrap();

        let t_reduced = Array1::linspace(0., 1e6, 1000);

        let y_fit = t_reduced.mapv(|ti| spline.evaluate(ti));
        let y_true = t_reduced.mapv(|ti| 1.0 - (-k * ti).exp());

        let raw_rms = rms(&y, &t.mapv(|ti| 1.0 - (-k * ti).exp()));
        let fit_rms = rms(&y_fit, &y_true);

        assert!(fit_rms < raw_rms);
    }

    #[test]
    fn test_physical_constraints() {
        let k = 3e-6;
        let t = Array1::linspace(0_f64, 1e6_f64, 200_000);

        let y = t.mapv(|ti| 1.0 - (-k * ti).exp());

        let internal = Array1::linspace(0., 1e6, 40);

        let spline = make_lsq_univariate_spline(
            &t,
            &y,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            3,
            SolverKind::Banded,
        )
        .unwrap();

        let t_reduced = Array1::linspace(0., 1e6, 1000);

        let y_fit = t_reduced.mapv(|ti| spline.evaluate(ti));

        let min = y_fit.fold(f64::INFINITY, |a, b| a.min(*b));
        let max = y_fit.fold(f64::NEG_INFINITY, |a, b| a.max(*b));

        assert!(min >= -1e-6);
        assert!(max <= 1.001);
    }

    #[test]
    fn test_recover_rate_constant() {
        let k_true = 2e-6;
        let t = Array1::linspace(0_f64, 8e5_f64, 200_000);

        let y = t.mapv(|ti| 1.0 - (-k_true * ti).exp());

        let internal = Array1::linspace(0., 8e5, 40);
        let now_spline = Instant::now();
        println!("start spline");
        let spline = make_lsq_univariate_spline(
            &t,
            &y,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            3,
            SolverKind::Banded,
        )
        .unwrap();
        println!("end spline {}", now_spline.elapsed().as_secs());
        let now_spline_calc = Instant::now();
        println!("start calculating spline");
        // берём редуцированный сигнал
        let t_small = Array1::linspace(0., 8e5, 1000);
        let now_spline_eval = Instant::now();
        println!("start evaluating spline");
        println!("{:?}, {}", &spline.coeffs, spline.coeffs.len());
        let y_fit = spline.evaluate_batch_array(&t_small); //t_small.mapv(|ti| spline.evaluate(ti));

        println!(
            "end evaluating spline {}",
            now_spline_eval.elapsed().as_secs()
        );
        // отбрасываем значения близкие к 1
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        for i in 0..t_small.len() {
            if y_fit[i] < 0.98 {
                x_vals.push(t_small[i]);
                y_vals.push((1.0 - y_fit[i]).ln());
            }
        }
        println!(
            "end calculating spline {}",
            now_spline_calc.elapsed().as_secs()
        );
        println!("start linear regression");
        // линейная регрессия
        let x = Array1::from(x_vals);
        let ylog = Array1::from(y_vals);

        let slope = {
            let xm = x.mean().unwrap();
            let ym = ylog.mean().unwrap();

            let num = (&x - xm) * (&ylog - ym);
            let den = (&x - xm).mapv(|v| v * v);

            num.sum() / den.sum()
        };

        let k_est = -slope;

        let rel_error = (k_est - k_true).abs() / k_true;
        println!("end linear regression");
        assert!(rel_error < 0.02); // 2% точность
    }

    //====================================================================================
    // PARALLEL DEBOOR TESTS
    //====================================================================================

    #[test]
    fn test_deboor_parallel_correctness() {
        // Создаём простой кубический сплайн
        let x = Array1::linspace(-2.0_f64, 2.0, 500);
        let y = x.mapv(|v| (v * v).exp() - 1.0);

        let internal = Array1::linspace(-1.5, 1.5, 10);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        // Тестируем batch evaluation
        let test_points = vec![-1.8, -1.2, -0.5, 0.0, 0.5, 1.2, 1.8];

        // Последовательная оценка
        let sequential: Vec<f64> = test_points.iter().map(|&xi| spline.evaluate(xi)).collect();

        // Parallel batch evaluation
        let parallel = spline.evaluate_batch(&test_points);

        // Результаты должны быть идентичны
        for (s, p) in sequential.iter().zip(parallel.iter()) {
            assert!(
                (s - p).abs() < 1e-14,
                "Sequential vs parallel mismatch: {} vs {}",
                s,
                p
            );
        }
    }

    #[test]
    fn test_deboor_parallel_batch_array1() {
        let x = Array1::linspace(-3.0_f64, 3.0, 300);
        let y = x.mapv(|v| v.sin());

        let internal = Array1::linspace(-2.5, 2.5, 8);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        let test_x = Array1::linspace(-3.0, 3.0, 1000);

        // Последовательная оценка
        let sequential = test_x.mapv(|v| spline.evaluate(v));

        // Parallel Array1 batch
        let parallel = spline.evaluate_batch_array(&test_x);

        // Проверяем совпадение
        let diff = (&sequential - &parallel)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |max, &v| max.max(v));
        assert!(diff < 1e-13, "Max difference: {}", diff);
    }

    #[test]
    fn test_deboor_parallel_vs_sequential_large() {
        // Большой набор данных для проверки производительности
        let x = Array1::linspace(0.0_f64, 100.0, 5000);
        let y = x.mapv(|v| (0.1 * v).sin() + 0.02 * v);

        let internal = Array1::linspace(10.0, 90.0, 20);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::Banded).unwrap();

        // Генерируем много точек для оценки
        let eval_points = Array1::linspace(0.0, 100.0, 10000);

        // Последовательная оценка с таймером
        let seq_start = Instant::now();
        let sequential = eval_points.mapv(|v| spline.evaluate(v));
        let seq_time = seq_start.elapsed();

        // Parallel batch оценка
        let par_start = Instant::now();
        let parallel = spline.evaluate_batch_array(&eval_points);
        let par_time = par_start.elapsed();

        // Проверяем результаты одинаковые
        let max_diff = (&sequential - &parallel)
            .mapv(|v| v.abs())
            .fold(0.0_f64, |max, &v| max.max(v));
        assert!(max_diff < 1e-12, "Results differ: max diff = {}", max_diff);

        // Parallel версия должна быть быстрее (особенно на больших наборах)
        eprintln!("Sequential time: {:?}", seq_time);
        eprintln!("Parallel time: {:?}", par_time);
        eprintln!(
            "Speedup: {:.2}x",
            seq_time.as_secs_f64() / par_time.as_secs_f64()
        );
    }

    #[test]
    fn test_deboor_batch_with_slice() {
        let x = Array1::linspace(-1.0_f64, 1.0, 200);
        let y = x.mapv(|v| v.powi(3) - v);

        let internal = Array1::linspace(-0.8, 0.8, 6);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        // Тест с обычным slice
        let points = [-0.5, -0.2, 0.0, 0.2, 0.5];
        let results = spline.evaluate_batch(&points);

        assert_eq!(results.len(), 5);

        // Проверяем каждую точку отдельно
        for (i, &p) in points.iter().enumerate() {
            let single = spline.evaluate(p);
            assert!((results[i] - single).abs() < 1e-14);
        }
    }

    #[test]
    fn test_deboor_parallel_edge_cases() {
        let x = Array1::linspace(0.0_f64, 10.0, 100);
        let y = x.mapv(|v| (0.5 * v).cos());

        let internal = Array1::linspace(1.0, 9.0, 5);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        // Граничные точки
        let edge_points = vec![0.0, 10.0, 5.0];

        let results = spline.evaluate_batch(&edge_points);
        assert_eq!(results.len(), 3);

        // Проверяем каждую с последовательной версией
        for (i, &p) in edge_points.iter().enumerate() {
            let expected = spline.evaluate(p);
            assert!((results[i] - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_deboor_parallel_empty_batch() {
        let x = Array1::linspace(-1.0, 1.0, 50);
        let y = x.mapv(|v| v * v);

        let internal = Array1::linspace(-0.5, 0.5, 3);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        // Пустой batch
        let empty_batch: Vec<f64> = vec![];
        let results = spline.evaluate_batch(&empty_batch);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_deboor_parallel_single_point() {
        let x = Array1::linspace(-2.0_f64, 2.0, 100);
        let y = x.mapv(|v| (-v * v).exp());

        let internal = Array1::linspace(-1.5, 1.5, 5);
        let spline = make_lsq_univariate_spline(&x, &y, &internal, 3, SolverKind::DenseQR).unwrap();

        // Batch с одной точкой должен совпадать с single evaluate
        let point = 0.7;
        let batch_result = spline.evaluate_batch(&[point]);

        assert_eq!(batch_result.len(), 1);
        assert!((batch_result[0] - spline.evaluate(point)).abs() < 1e-14);
    }

    #[test]
    //#[ignore]  // запускать вручную: cargo test -- --ignored
    fn stress_test_million_points() {
        unsafe { std::env::set_var("LSQ_SPLINES_TIMING", "on") };
        let k = 2e-6;
        let sigma = 5e-4;

        let t = Array1::linspace(0_f64, 1e6, 1_000_000);

        let mut y = t.mapv(|ti| 1.0 - (-k * ti).exp());

        // добавляем шум
        for yi in y.iter_mut() {
            *yi += sigma * (rand::random::<f64>() - 0.5);
        }

        let internal = Array1::linspace(0_f64, 1e6, 50);

        let start = std::time::Instant::now();

        let spline = make_lsq_univariate_spline(
            &t,
            &y,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            3,
            SolverKind::Banded,
        )
        .unwrap();

        let elapsed = start.elapsed();

        println!("Time elapsed: {:?}", elapsed);

        // Проверка редукции
        let t_small = Array1::linspace(0., 1e6, 1000);
        let y_fit = spline.evaluate_batch_array(&t_small);

        // Проверка на NaN
        assert!(y_fit.iter().all(|v| v.is_finite()));

        // Проверка подавления шума
        let y_true = t_small.mapv(|ti| 1.0 - (-k * ti).exp());
        let fit_rms = rms(&y_fit, &y_true);

        assert!(fit_rms < sigma);

        // В production ориентир:
        // < 100 ms на современной машине
    }
}

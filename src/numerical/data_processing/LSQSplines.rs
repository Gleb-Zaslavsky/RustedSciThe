//! # LSQ Splines: инженерный разбор алгоритма
//!
//! Этот модуль решает задачу аппроксимации данных `y(x)` B-сплайном по методу
//! наименьших квадратов (Least Squares).
//!
//! ## 1) Что минимизируется
//! Ищем коэффициенты `c`, минимизирующие `||W^(1/2)(A c - y)||_2^2`, где
//! `A` — матрица значений базиса, `W` — диагональная матрица весов.
//!
//! ## 2) Как строится матрица A
//! Для каждой точки `x_i` ненулевы только `k+1` соседних B-сплайнов степени `k`.
//! Поэтому каждая строка `A` локальна (редкая структура).
//!
//! ## 3) Два численных пути
//! - DenseQR: формируем плотную `A` и решаем LS через QR.
//! - Banded: собираем `G = A^T W A` и `rhs = A^T W y`; так как `G` полосатая SPD,
//!   применяем полосатый Cholesky.
//!
//! ## 4) Инженерный смысл
//! Banded-подход особенно выгоден на больших задачах: меньше памяти, меньше
//! операций, так как работаем только в полосе ширины порядка `2k`.
/*
================================================================================
Least-Squares B-Spline Fitting (Dense QR and Banded Normal Equations)
================================================================================

Этот модуль реализует аппроксимацию одномерных данных с помощью B-сплайна
методом наименьших квадратов.

Задача
------

Имеются данные:

    x_i  — точки измерения (узлы сетки)
    y_i  — измеренные значения

Требуется построить сглаживающий B-сплайн S(x) степени k, такой что:

    S(x) = Σ_j c_j * N_j,k(x)

где:
    N_j,k(x) — базисные B-сплайны степени k
    c_j      — неизвестные коэффициенты

Коэффициенты c_j находятся из условия минимизации суммы квадратов:

    Σ_i w_i ( y_i - S(x_i) )² → min

где w_i — веса (если не заданы, принимаются равными 1).

================================================================================
1. ЛИНЕЙНАЯ АЛГЕБРА ЗАДАЧИ
================================================================================

Подставляя выражение для S(x_i):

    y_i ≈ Σ_j A_ij c_j

где:

    A_ij = N_j,k(x_i)

Получаем линейную систему в матричной форме:

    A c ≈ y

Это переопределённая система (обычно строк больше, чем столбцов).

Решение в смысле наименьших квадратов:

    min || W (A c - y) ||²

где W = diag(w_i).

Существует два способа решения:

------------------------------------------------------------------------------
1) Dense QR (численно устойчивый)
------------------------------------------------------------------------------
Решается напрямую:

    A c ≈ y

через QR-разложение:

    A = Q R

Тогда:

    R c = Qᵀ y

Это устойчивый метод, но требует хранения всей матрицы A
размера (m × n), что дорого при больших m.

------------------------------------------------------------------------------
2) Banded Normal Equations (оптимизированный метод)
------------------------------------------------------------------------------
Используется тот факт, что B-сплайн имеет локальную поддержку:

Каждый базис N_j,k(x) ненулевой только на ограниченном интервале.

Следовательно:

• В каждой строке A ненулевых элементов не более (k+1)
• Матрица A разреженная
• Матрица G = Aᵀ W² A имеет ленточную структуру

Мы формируем нормальные уравнения:

    G c = rhs

где:

    G = Aᵀ W² A
    rhs = Aᵀ W² y

Размер G — (n × n), где n — число коэффициентов.
Матрица G симметричная и положительно определённая (если выполнено
условие Schoenberg–Whitney).

================================================================================
2. ЧТО ТАКОЕ ЛЕНТОЧНАЯ (BANDED) МАТРИЦА?
================================================================================

Ленточная матрица — это матрица, у которой ненулевые элементы
расположены только вблизи главной диагонали.

Например (ширина ленты = 2):

    x x x 0 0
    x x x x 0
    x x x x x
    0 x x x x
    0 0 x x x

В нашем случае:

• Ширина ленты G равна 2k
• Матрица симметрична
• Мы храним только верхнюю половину

Хранение организовано компактно:

    band = j - i
    data[band * n + i]

band = 0  → главная диагональ
band = 1  → первая наддиагональ
...
band = k  → k-тая наддиагональ

Это позволяет хранить матрицу размером:

    O(n * k)

вместо O(n²).

================================================================================
3. IN-PLACE BANDED CHOLESKY
================================================================================

Поскольку G симметрична и положительно определена,
мы можем разложить её:

    G = Rᵀ R

где R — верхняя треугольная ленточная матрица.

Алгоритм Cholesky выполняется in-place:

Для каждого i:

1) Обновляется диагональ:

   G_ii ← G_ii - Σ R_{i-p,i}²

2) Берётся квадратный корень:

   R_ii = sqrt(G_ii)

3) Обновляются элементы над диагональю:

   G_ij ← (G_ij - Σ R_{i-p,i} R_{i-p,j}) / R_ii

Вся операция выполняется в пределах ленты.

Сложность:

    O(n k²)

Для cubic spline (k=3):

    k_eff = 6
    сложность ≈ 36 n

Это чрезвычайно быстро даже для сотен коэффициентов.

================================================================================
4. РЕШЕНИЕ СИСТЕМЫ
================================================================================

После разложения:

    Rᵀ R c = rhs

Решаем в два шага:

1) Rᵀ z = rhs   (прямая подстановка)
2) R c = z      (обратная подстановка)

Обе операции выполняются только внутри ленты.

================================================================================
5. УСЛОВИЕ SCHOENBERG–WHITNEY
================================================================================

Чтобы система имела единственное решение, необходимо,
чтобы в каждом интервале поддержки базиса существовала
хотя бы одна точка x_i:

    t_j < x_i < t_{j+k+1}

Это гарантирует невырожденность G.

================================================================================
6. СРАВНЕНИЕ МЕТОДОВ
================================================================================

Dense QR:
    + численно устойчив
    - требует O(m n) памяти
    - медленный при больших m

Banded:
    + требует O(n k) памяти
    + очень быстрый
    + идеально подходит для больших наборов данных
    - основан на нормальных уравнениях (может усиливать плохую обусловленность)

В задачах аппроксимации B-сплайнами с разумным числом узлов
метод banded является практическим стандартом.

================================================================================
7. ОБЛАСТЬ ПРИМЕНЕНИЯ
================================================================================

Алгоритм оптимизирован для:

• сотен тысяч – миллионов точек измерения
• небольшого числа коэффициентов (обычно 20–200)
• степени сплайна 2–5 (чаще всего 3)

Типичный сценарий — анализ TGA/DSC данных,
где требуется сглаживание больших массивов.

================================================================================
*/
use nalgebra::{DMatrix, DVector};
use ndarray::s;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::time::Instant;
use thiserror::Error;
// use ndarray_linalg::QRInto;
// bspline.rs

#[derive(Clone, Debug)]
/// Представление одномерного B-сплайна: узлы, коэффициенты, степень.
pub struct BSpline {
    /// Полный вектор узлов (включая кратные узлы на концах).
    pub knots: Array1<f64>,
    /// Вектор коэффициентов при базисе.
    pub coeffs: Array1<f64>,
    /// Степень сплайна `k` (например, `k=3` для кубического).
    pub degree: usize,
}
impl BSpline {
    /// Вычисляет `s(x)` по алгоритму де Бура (single point).
    pub fn evaluate(&self, x: f64) -> f64 {
        deboor(x, &self.knots, &self.coeffs, self.degree)
    }

    /// **Parallel batch evaluation** — вычисляет сплайн для множества точек одновременно.
    /// Рекомендуется для N > 100 точек. Использует все ядра процессора.
    ///
    /// # Example
    /// ```ignore
    /// let x_vals = vec![0.1, 0.2, 0.3, ..., 0.9];  // 1000+ точек
    /// let y_vals = spline.evaluate_batch(&x_vals);  // вычисляет параллельно
    /// ```
    pub fn evaluate_batch(&self, x: &[f64]) -> Vec<f64> {
        deboor_batch(x, &self.knots, &self.coeffs, self.degree)
    }

    /// **Parallel batch evaluation для Array1** — удобно для ndarray.
    pub fn evaluate_batch_array(&self, x: &Array1<f64>) -> Array1<f64> {
        deboor_batch_array1(x, &self.knots, &self.coeffs, self.degree)
    }
}
//=================================================================================
//          Basis + de Boor (production safe)
//======================================================================================
/// Находит индекс интервала узлов (`span`), которому принадлежит точка `x`.
/// Это ключ к локальности: дальше нужны только `k+1` соседних базисных функций.
pub fn find_span(n: usize, k: usize, x: f64, t: &Array1<f64>) -> usize {
    // Clamp outside support to a valid edge span. Without this, binary search may not shrink.
    if x <= t[k] {
        return k;
    }
    if x >= t[n + 1] {
        return n;
    }

    let mut low = k;
    let mut high = n + 1;
    let mut mid = (low + high) / 2;

    while x < t[mid] || x >= t[mid + 1] {
        if x < t[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    mid
}
/// Значения `k+1` ненулевых базисных функций B-сплайна в точке `x`.
/// Это локальный базис вокруг найденного `span`.
pub fn basis_functions(span: usize, x: f64, k: usize, t: &Array1<f64>) -> Vec<f64> {
    let mut left = vec![0.0; k + 1];
    let mut right = vec![0.0; k + 1];
    let mut N = vec![0.0; k + 1];

    N[0] = 1.0;

    for j in 1..=k {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;

        let mut saved = 0.0;

        for r in 0..j {
            let temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        N[j] = saved;
    }

    N
}
/// Устойчивое вычисление значения сплайна `s(x)` по схеме де Бура.
/// **Original sequential version** — используйте для единичных точек.
pub fn deboor(x: f64, t: &Array1<f64>, c: &Array1<f64>, k: usize) -> f64 {
    let n = c.len() - 1;
    let span = find_span(n, k, x, t);

    let mut d = vec![0.0; k + 1];

    // Берем только локальное окно коэффициентов, влияющее на точку x.
    for j in 0..=k {
        d[j] = c[span - k + j];
    }

    // Треугольная рекурсия: последовательная линейная интерполяция.
    for r in 1..=k {
        for j in (r..=k).rev() {
            let i = span - k + j;
            let denom = t[i + k + 1 - r] - t[i];
            if denom.abs() < 1e-14 {
                // Защита от почти совпавших узлов.
                continue;
            }
            let alpha = (x - t[i]) / denom;
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    d[k]
}

/// Внутренняя функция де Бура для известного `span`.
/// Избегает переповторного поиска интервала при оценке нескольких точек.
#[inline]
fn deboor_with_span(x: f64, t: &Array1<f64>, c: &Array1<f64>, k: usize, span: usize) -> f64 {
    let mut d = vec![0.0; k + 1];

    // Берем только локальное окно коэффициентов, влияющее на точку x.
    for j in 0..=k {
        d[j] = c[span - k + j];
    }

    // Треугольная рекурсия: последовательная линейная интерполяция.
    for r in 1..=k {
        for j in (r..=k).rev() {
            let i = span - k + j;
            let denom = t[i + k + 1 - r] - t[i];
            if denom.abs() < 1e-14 {
                continue;
            }
            let alpha = (x - t[i]) / denom;
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    d[k]
}

/// **Parallel batch evaluation** — вычисляет B-сплайн для множества точек в параллели.
/// Намного быстрее, чем вызов `deboor` в цикле для больших наборов данных (N > 100).
///
/// # Arguments
/// * `x` — массив точек, где нужно вычислить значение сплайна
/// * `t` — вектор узлов сплайна
/// * `c` — коэффициенты сплайна
/// * `k` — степень сплайна
///
/// # Returns
/// Вектор значений сплайна в точках `x`.
///
/// # Example
/// ```ignore
/// let x = Array1::linspace(0., 1., 10000);
/// let result = deboor_batch(&x, &knots, &coeffs, 3);  // параллельно вычисляет 10000 точек
/// ```
pub fn deboor_batch(x: &[f64], t: &Array1<f64>, c: &Array1<f64>, k: usize) -> Vec<f64> {
    let n = c.len() - 1;

    x.par_iter()
        .map(|&xi| {
            let span = find_span(n, k, xi, t);
            deboor_with_span(xi, t, c, k, span)
        })
        .collect()
}

/// **Parallel batch evaluation для Array1** — удобный wrapper для ndarray.
/// Преобразует Array1 в slice, вычисляет в параллели и возвращает Array1.
pub fn deboor_batch_array1(
    x: &Array1<f64>,
    t: &Array1<f64>,
    c: &Array1<f64>,
    k: usize,
) -> Array1<f64> {
    Array1::from(deboor_batch(x.as_slice().unwrap(), t, c, k))
}

//=======================================================================================
//             Design Matrix (Dense)
//=======================================================================================
/// Формирует плотную матрицу дизайна `A` размера `(m x n)`,
/// где `m = len(x)`, `n = число коэффициентов`.
pub fn design_matrix_dense(x: &Array1<f64>, knots: &Array1<f64>, degree: usize) -> Array2<f64> {
    let m = x.len();
    let n = knots.len() - degree - 1;

    let mut A = Array2::<f64>::zeros((m, n));

    for (i, &xi) in x.iter().enumerate() {
        let span = find_span(n - 1, degree, xi, knots);
        let basis = basis_functions(span, xi, degree, knots);

        // В строке i заполняем только локальные degree+1 позиций.
        for j in 0..=degree {
            A[[i, span - degree + j]] = basis[j];
        }
    }

    A
}

/// Parallel variant of `design_matrix_dense`.
/// Computes per-row local basis contributions in parallel and assembles dense matrix.
pub fn design_matrix_dense_parallel(
    x: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Array2<f64> {
    let m = x.len();
    let n = knots.len() - degree - 1;

    let row_entries: Vec<Vec<(usize, f64)>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let xi = x[i];
            let span = find_span(n - 1, degree, xi, knots);
            let basis = basis_functions(span, xi, degree, knots);

            (0..=degree)
                .map(|j| (span - degree + j, basis[j]))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut A = Array2::<f64>::zeros((m, n));
    for (i, entries) in row_entries.into_iter().enumerate() {
        for (col, val) in entries {
            A[[i, col]] = val;
        }
    }

    A
}
//============================================================================
// Solver enum (production API)
//==============================================================================

#[derive(Debug, Error)]
/// Ошибки LSQ-решателя.
pub enum LssError {
    #[error("Linear algebra failure")]
    Linalg,
    #[error("Invalid spline input")]
    InvalidInput,
}

#[derive(Clone, Copy)]
/// Выбор численного метода решения LSQ-задачи.
pub enum SolverKind {
    DenseQR,
    Banded, // future
}

#[inline]
/// Проверяет флаг тайминга `LSQ_SPLINES_TIMING`.
fn timing_enabled() -> bool {
    std::env::var("LSQ_SPLINES_TIMING")
        .map(|v| {
            let normalized = v.trim().to_ascii_lowercase();
            !(normalized.is_empty()
                || normalized == "0"
                || normalized == "false"
                || normalized == "off")
        })
        .unwrap_or(false)
}

#[inline]
/// Логирует длительность этапа (в миллисекундах).
fn log_timing(stage: &str, started: Instant) {
    if timing_enabled() {
        eprintln!(
            "[LSQSplines][timing] {:<28} {:>10.3} ms",
            stage,
            started.elapsed().as_secs_f64() * 1_000.0
        );
    }
}

/// Единая точка входа решения LSQ-задачи:
/// выбирает между плотным QR и полосатым решателем.
pub fn solve_lsq(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    solver: SolverKind,
) -> Result<Array1<f64>, LssError> {
    let total_started = Instant::now();
    match solver {
        SolverKind::DenseQR => {
            let design_started = Instant::now();
            let A = design_matrix_dense(x, knots, degree);
            log_timing("dense.design_matrix", design_started);

            let solve_started = Instant::now();
            let result = dense_qr_nalgebra(A, y, w);
            log_timing("dense.solve_qr", solve_started);
            log_timing("dense.total", total_started);
            result
        }
        SolverKind::Banded => {
            // Для больших задач обычно выгоднее полосатый путь.
            let result = banded_solver(x, y, w, knots, degree);
            log_timing("banded.total", total_started);
            result
        }
    }
}

/// Parallel variant of `solve_lsq`.
pub fn solve_lsq_parallel(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    solver: SolverKind,
) -> Result<Array1<f64>, LssError> {
    let total_started = Instant::now();
    match solver {
        SolverKind::DenseQR => {
            let design_started = Instant::now();
            let A = design_matrix_dense_parallel(x, knots, degree);
            log_timing("dense.parallel.design_matrix", design_started);

            let solve_started = Instant::now();
            let result = dense_qr_nalgebra_parallel(A, y, w);
            log_timing("dense.parallel.solve_qr", solve_started);
            log_timing("dense.parallel.total", total_started);
            result
        }
        SolverKind::Banded => {
            let result = banded_solver_parallel(x, y, w, knots, degree);
            log_timing("banded.parallel.total", total_started);
            result
        }
    }
}
//===================================================================================
// DENSE APPROACH (not for large systems - for testing purpose)
//====================================================================================
/// Плотное решение через QR-факторизацию.
/// Решает: `min ||W^(1/2)(A c - y)||_2`.
fn dense_qr_nalgebra(
    A: Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<Array1<f64>, LssError> {
    let (m, n) = A.dim();

    if m < n {
        return Err(LssError::InvalidInput);
    }

    let weighted_started = Instant::now();
    // Приводим взвешенную задачу к невзвешенной:
    // A_w = W^(1/2) A, y_w = W^(1/2) y.
    let mut mat = DMatrix::<f64>::zeros(m, n);
    let mut vec = DVector::<f64>::zeros(m);

    for i in 0..m {
        let wi = w[i].sqrt();
        vec[i] = y[i] * wi;

        for j in 0..n {
            mat[(i, j)] = A[[i, j]] * wi;
        }
    }
    log_timing("dense.weighted_assembly", weighted_started);

    // QR: A_w = Q R.
    let qr_started = Instant::now();
    let qr = mat.qr();
    let q = qr.q();
    let r = qr.r();
    log_timing("dense.qr_factorization", qr_started);

    // Получаем правую часть треугольной системы: Q^T y_w.
    let solve_started = Instant::now();
    let qt_b = q.transpose() * vec;

    // Берем квадратный верхнетреугольный блок R(n x n)
    // и решаем обратным ходом.
    let r_square = r.view((0, 0), (n, n));
    let qt_b_trunc = qt_b.rows(0, n);

    let solution = r_square
        .solve_upper_triangular(&qt_b_trunc)
        .ok_or(LssError::Linalg)?;
    log_timing("dense.back_substitution", solve_started);

    Ok(Array1::from(solution.data.as_vec().clone()))
}

/// Parallel variant of `dense_qr_nalgebra`.
/// QR factorization itself remains sequential in nalgebra; weighted assembly is parallelized.
fn dense_qr_nalgebra_parallel(
    A: Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<Array1<f64>, LssError> {
    let (m, n) = A.dim();

    if m < n {
        return Err(LssError::InvalidInput);
    }

    let weighted_started = Instant::now();
    let weighted_rows: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let wi = w[i].sqrt();
            (0..n).map(|j| A[[i, j]] * wi).collect::<Vec<_>>()
        })
        .collect();
    let flat_weighted: Vec<f64> = weighted_rows.into_iter().flatten().collect();
    let mat = DMatrix::<f64>::from_row_slice(m, n, &flat_weighted);

    let vec_data: Vec<f64> = (0..m).into_par_iter().map(|i| y[i] * w[i].sqrt()).collect();
    let vec = DVector::<f64>::from_vec(vec_data);
    log_timing("dense.parallel.weighted_assembly", weighted_started);

    let qr_started = Instant::now();
    let qr = mat.qr();
    let q = qr.q();
    let r = qr.r();
    log_timing("dense.parallel.qr_factorization", qr_started);

    let solve_started = Instant::now();
    let qt_b = q.transpose() * vec;

    let r_square = r.view((0, 0), (n, n));
    let qt_b_trunc = qt_b.rows(0, n);

    let solution = r_square
        .solve_upper_triangular(&qt_b_trunc)
        .ok_or(LssError::Linalg)?;
    log_timing("dense.parallel.back_substitution", solve_started);

    Ok(Array1::from(solution.data.as_vec().clone()))
}
//РЈСЃР»РѕРІРёРµ:
//Р”Р»СЏ j = 0..(n-k-2):
//t[j] < x[i_j] < t[j+k+1]
//РњС‹ СЂРµР°Р»РёР·СѓРµРј РїСЂРѕРІРµСЂРєСѓ СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёСЏ С…РѕС‚СЏ Р±С‹ РѕРґРЅРѕРіРѕ x РІ РєР°Р¶РґРѕРј РёРЅС‚РµСЂРІР°Р»Рµ.

//====================================================================================================
// BANDED MATRIX APPROACH
//BandedMatrix
/// Простейшее хранение верхней полосы (legacy-вариант, оставлен для отладки).
pub struct BandedMatrix {
    pub data: Vec<Vec<f64>>, // [band][col]
    pub n: usize,
    pub bandwidth: usize,
}

impl BandedMatrix {
    /// Создает полосатую матрицу размера `n x n` с заданной шириной верхней полосы.
    pub fn new(n: usize, bandwidth: usize) -> Self {
        Self {
            data: vec![vec![0.0; n]; bandwidth],
            n,
            bandwidth,
        }
    }

    // add value to (i, j), assuming j >= i
    pub fn add(&mut self, i: usize, j: usize, val: f64) {
        let band = j - i;
        if band < self.bandwidth {
            self.data[band][i] += val;
        }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        let band = j - i;
        if band < self.bandwidth {
            self.data[band][i]
        } else {
            0.0
        }
    }
}
/*
fn build_normal_equations_banded(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> (BandedMatrix, Vec<f64>) {
    let n = knots.len() - degree - 1;
    let bandwidth = 2 * degree + 1;

    let mut G = BandedMatrix::new(n, bandwidth);
    let mut g = vec![0.0; n];

    for i in 0..x.len() {
        let xi = x[i];
        let wi2 = w[i] * w[i];
        let yi = y[i];

        let span = find_span(n - 1, degree, xi, knots);
        let basis = basis_functions(span, xi, degree, knots);

        for a in 0..=degree {
            let row = span - degree + a;
            let va = basis[a];

            g[row] += wi2 * va * yi;

            for b in a..=degree {
                let col = span - degree + b;
                let vb = basis[b];

                G.add(row, col, wi * va * vb);
            }
        }
    }

    (G, g)
}
     */
/// Legacy-реализация полосатого Cholesky для `BandedMatrix`.
/// В текущем решателе не используется, но оставлена как reference/отладка.
fn cholesky_banded(G: &mut BandedMatrix) -> Result<(), LssError> {
    let n = G.n;
    let bw = G.bandwidth;

    for i in 0..n {
        // Диагональ: G_ii - сумма уже вычисленных квадратов.
        let mut sum = 0.0;
        for k in 1..bw {
            if i >= k {
                let val = G.get(i - k, i);
                sum += val * val;
            }
        }

        let diag = G.get(i, i) - sum;

        if diag <= 0.0 {
            return Err(LssError::Linalg);
        }

        G.data[0][i] = diag.sqrt();

        // Элементы выше диагонали.
        for j_offset in 1..bw {
            let j = i + j_offset;
            if j >= n {
                break;
            }

            let mut sum = 0.0;
            for k in 1..bw {
                if i >= k && j >= k {
                    sum += G.get(i - k, i) * G.get(j - k, i);
                }
            }

            // Формула полосатого Cholesky для верхнетреугольного фактора.
            let val = (G.get(i, j) - sum) / G.data[0][i];
            G.data[j_offset][i] = val;
        }
    }

    Ok(())
}
//Production Struct
#[derive(Debug, Clone)]
/// Симметричная полосатая матрица в компактном формате.
/// Храним верхний треугольник по полосам:
/// - `band = 0` — диагональ,
/// - `band = 1` — первая наддиагональ и т.д.
pub struct SymmetricBanded {
    n: usize,
    k: usize,       // half-bandwidth
    data: Vec<f64>, // flat storage
}

impl SymmetricBanded {
    /// `n` — размер матрицы, `k` — половина ширины полосы.
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            n,
            k,
            data: vec![0.0; (k + 1) * n],
        }
    }

    #[inline(always)]
    fn index(&self, band: usize, col: usize) -> usize {
        band * self.n + col
    }

    #[inline(always)]
    fn in_band(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        if i > j {
            return self.in_band(j, i);
        }

        let band = j - i;

        if band <= self.k {
            Some((band, i))
        } else {
            None
        }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if let Some((band, col)) = self.in_band(i, j) {
            self.data[self.index(band, col)]
        } else {
            0.0
        }
    }

    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        if let Some((band, col)) = self.in_band(i, j) {
            let idx = self.index(band, col);
            self.data[idx] += value;
        }
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if let Some((band, col)) = self.in_band(i, j) {
            let idx = self.index(band, col);
            self.data[idx] = value;
        }
    }

    #[inline(always)]
    pub fn diag(&self, i: usize) -> f64 {
        self.data[self.index(0, i)]
    }

    #[inline(always)]
    pub fn diag_mut(&mut self, i: usize) -> &mut f64 {
        let idx = self.index(0, i);
        &mut self.data[idx]
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn bandwidth(&self) -> usize {
        self.k
    }

    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut mat = nalgebra::DMatrix::zeros(self.n, self.n);

        for i in 0..self.n {
            for j in i..=usize::min(self.n - 1, i + self.k) {
                let val = self.get(i, j);
                mat[(i, j)] = val;
                mat[(j, i)] = val;
            }
        }

        mat
    }
    /// Полосатый Cholesky "на месте":
    /// после вызова в `self` хранится верхнетреугольный фактор `R`,
    /// такой что исходная матрица `A = R^T R`.
    pub fn cholesky_in_place(&mut self) -> Result<(), &'static str> {
        let n = self.n;
        let k = self.k;

        for i in 0..n {
            // ---- 1. Обновление диагонали ----
            let mut sum = 0.0;

            for p in 1..=k {
                if i < p {
                    break;
                }

                // Вклад уже вычисленных элементов столбца i фактора R.
                let val = self.get(i - p, i);
                sum += val * val;
            }

            let diag_idx = self.index(0, i);
            let updated = self.data[diag_idx] - sum;

            if updated <= 0.0 {
                return Err("Matrix not positive definite");
            }

            let rii = updated.sqrt();
            self.data[diag_idx] = rii;

            // ---- 2. Обновление элементов выше диагонали ----
            let max_j = usize::min(n - 1, i + k);

            for j in (i + 1)..=max_j {
                let mut sum = 0.0;

                for p in 1..=k {
                    if i < p {
                        break;
                    }

                    let s = i - p;
                    if j - s > k {
                        // Элемент R_{s,j} вне полосы, его вклад равен 0.
                        continue;
                    }

                    let rik = self.get(s, i);
                    let rsj = self.get(s, j);

                    sum += rik * rsj;
                }

                if let Some((band, col)) = self.in_band(i, j) {
                    let idx = self.index(band, col);
                    // Формула: R_{ij} = (A_{ij} - сумма) / R_{ii}.
                    let val = (self.data[idx] - sum) / rii;
                    self.data[idx] = val;
                }
            }
        }

        Ok(())
    }

    /// Решает систему `A x = rhs`, предполагая что в `self` уже хранится фактор `R`
    /// после `cholesky_in_place` и `A = R^T R`.
    pub fn solve_spd_in_place(&self, rhs: &mut [f64]) -> Result<(), &'static str> {
        let n = self.n;
        let k = self.k;

        if rhs.len() != n {
            return Err("Dimension mismatch");
        }

        // ---- Прямой проход: R^T z = b ----
        for i in 0..n {
            let mut sum = rhs[i];

            for p in 1..=k {
                if i < p {
                    break;
                }

                // Используем элементы полосы верхнего фактора R.
                let val = self.get(i - p, i); // R_{i-p, i}
                sum -= val * rhs[i - p];
            }

            let rii = self.get(i, i);
            rhs[i] = sum / rii;
        }

        // ---- Обратный проход: R x = z ----
        for i in (0..n).rev() {
            let mut sum = rhs[i];

            let max_j = usize::min(n - 1, i + k);

            for j in (i + 1)..=max_j {
                let val = self.get(i, j);
                sum -= val * rhs[j];
            }

            let rii = self.get(i, i);
            rhs[i] = sum / rii;
        }

        Ok(())
    }
}
/// Строит нормальные уравнения в banded формате:
/// `G = A^T W A`, `rhs = A^T W y`.
fn build_normal_equations_banded(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> (SymmetricBanded, Vec<f64>) {
    let n = knots.len() - degree - 1;

    // Для B-сплайнов степени k у G полуширина полосы равна 2k.
    let half_bandwidth = 2 * degree;

    let mut G = SymmetricBanded::new(n, half_bandwidth);
    let mut rhs = vec![0.0; n];

    for i in 0..x.len() {
        let xi = x[i];
        let wi = w[i];
        let yi = y[i];

        let span = find_span(n - 1, degree, xi, knots);
        let basis = basis_functions(span, xi, degree, knots);

        // Собираем вклад i-й точки в rhs и G.
        for a in 0..=degree {
            let row = span - degree + a;
            let va = basis[a];

            rhs[row] += wi * va * yi;

            for b in a..=degree {
                let col = span - degree + b;
                let vb = basis[b];

                G.add(row, col, wi * va * vb);
            }
        }
    }

    (G, rhs)
}

/// Parallel variant of `build_normal_equations_banded`.
fn build_normal_equations_banded_parallel(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> (SymmetricBanded, Vec<f64>) {
    let n = knots.len() - degree - 1;
    let half_bandwidth = 2 * degree;

    (0..x.len())
        .into_par_iter()
        .fold(
            || (SymmetricBanded::new(n, half_bandwidth), vec![0.0; n]),
            |(mut g_local, mut rhs_local), i| {
                let xi = x[i];
                let wi = w[i];
                let yi = y[i];

                let span = find_span(n - 1, degree, xi, knots);
                let basis = basis_functions(span, xi, degree, knots);

                for a in 0..=degree {
                    let row = span - degree + a;
                    let va = basis[a];
                    rhs_local[row] += wi * va * yi;

                    for b in a..=degree {
                        let col = span - degree + b;
                        let vb = basis[b];
                        g_local.add(row, col, wi * va * vb);
                    }
                }

                (g_local, rhs_local)
            },
        )
        .reduce(
            || (SymmetricBanded::new(n, half_bandwidth), vec![0.0; n]),
            |(mut g_acc, mut rhs_acc), (g_part, rhs_part)| {
                for (acc, part) in g_acc.data.iter_mut().zip(g_part.data.iter()) {
                    *acc += *part;
                }
                for (acc, part) in rhs_acc.iter_mut().zip(rhs_part.iter()) {
                    *acc += *part;
                }
                (g_acc, rhs_acc)
            },
        )
}

/// Banded solver:
/// 1) сборка нормальных уравнений,
/// 2) факторизация Холецкого,
/// 3) два треугольных прохода.
fn banded_solver(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array1<f64>, LssError> {
    let build_started = Instant::now();
    let (mut G, mut rhs) = build_normal_equations_banded(x, y, w, knots, degree);
    log_timing("banded.build_normal_eq", build_started);

    let cholesky_started = Instant::now();
    G.cholesky_in_place().map_err(|_| LssError::Linalg)?;
    log_timing("banded.cholesky", cholesky_started);

    let solve_started = Instant::now();
    G.solve_spd_in_place(&mut rhs)
        .map_err(|_| LssError::Linalg)?;
    log_timing("banded.triangular_solve", solve_started);

    Ok(Array1::from(rhs))
}

/// Parallel variant of `banded_solver`.
fn banded_solver_parallel(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array1<f64>, LssError> {
    let build_started = Instant::now();
    let (mut G, mut rhs) = build_normal_equations_banded_parallel(x, y, w, knots, degree);
    log_timing("banded.parallel.build_normal_eq", build_started);

    let cholesky_started = Instant::now();
    G.cholesky_in_place().map_err(|_| LssError::Linalg)?;
    log_timing("banded.parallel.cholesky", cholesky_started);

    let solve_started = Instant::now();
    G.solve_spd_in_place(&mut rhs)
        .map_err(|_| LssError::Linalg)?;
    log_timing("banded.parallel.triangular_solve", solve_started);

    Ok(Array1::from(rhs))
}
//=================================================================
// schoenberg whitney
/// Проверка условия Шенберга-Уитни.
/// Интуитивно: в каждом "окне" базисной функции должна быть хотя бы одна точка данных.
pub fn check_schoenberg_whitney(
    x: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<(), &'static str> {
    let n = knots.len() - degree - 1;

    for j in 0..n {
        let left = knots[j];
        let right = knots[j + degree + 1];

        let exists = x.iter().any(|&xi| xi > left && xi < right);

        if !exists {
            return Err("SchoenbergвЂ“Whitney condition violated");
        }
    }

    Ok(())
}

/// Parallel variant of `check_schoenberg_whitney`.
pub fn check_schoenberg_whitney_parallel(
    x: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<(), &'static str> {
    let n = knots.len() - degree - 1;

    (0..n).into_par_iter().try_for_each(|j| {
        let left = knots[j];
        let right = knots[j + degree + 1];

        let exists = x.iter().any(|&xi| xi > left && xi < right);
        if exists {
            Ok(())
        } else {
            Err("SchoenbergРІР‚вЂњWhitney condition violated")
        }
    })
}
//==============================================================================
//                High-level API
//====================================================================================

/// Низкоуровневое построение LSQ-сплайна по полному вектору узлов.
pub fn make_lsq_spline(
    x: &Array1<f64>,
    y: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    weights: Option<&Array1<f64>>,
    solver: SolverKind,
) -> Result<BSpline, LssError> {
    let n_coeffs = knots.len() - degree - 1;

    if x.len() < n_coeffs {
        return Err(LssError::InvalidInput);
    }

    let w = weights.cloned().unwrap_or_else(|| Array1::ones(x.len()));

    // Решаем линейную LSQ-задачу выбранным способом.
    let coeffs = solve_lsq(x, y, &w, knots, degree, solver)?;

    Ok(BSpline {
        knots: knots.clone(),
        coeffs,
        degree,
    })
}

/// Parallel variant of `make_lsq_spline`.
pub fn make_lsq_spline_parallel(
    x: &Array1<f64>,
    y: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    weights: Option<&Array1<f64>>,
    solver: SolverKind,
) -> Result<BSpline, LssError> {
    let n_coeffs = knots.len() - degree - 1;

    if x.len() < n_coeffs {
        return Err(LssError::InvalidInput);
    }

    let w = weights.cloned().unwrap_or_else(|| Array1::ones(x.len()));
    let coeffs = solve_lsq_parallel(x, y, &w, knots, degree, solver)?;

    Ok(BSpline {
        knots: knots.clone(),
        coeffs,
        degree,
    })
}
/// Удобный API: принимает только внутренние узлы и сам достраивает полный узловой вектор.
pub fn make_lsq_univariate_spline_nonparallel(
    x: &Array1<f64>,
    y: &Array1<f64>,
    internal_knots: &Array1<f64>,
    degree: usize,
    solver: SolverKind,
) -> Result<BSpline, LssError> {
    let xmin = x[0];
    let xmax = x[x.len() - 1];

    // Строим полный узловой вектор с кратностью degree+1 на концах.
    let mut knots = Vec::new();

    for _ in 0..=degree {
        knots.push(xmin);
    }

    for &k in internal_knots {
        knots.push(k);
    }

    for _ in 0..=degree {
        knots.push(xmax);
    }

    let knots = Array1::from(knots);

    check_schoenberg_whitney(x, &knots, degree).map_err(|_| LssError::InvalidInput)?;

    make_lsq_spline(x, y, &knots, degree, None, solver)
}

/// Parallel variant of `make_lsq_univariate_spline`.
pub fn make_lsq_univariate_spline(
    x: &Array1<f64>,
    y: &Array1<f64>,
    internal_knots: &Array1<f64>,
    degree: usize,
    solver: SolverKind,
) -> Result<BSpline, LssError> {
    let xmin = x[0];
    let xmax = x[x.len() - 1];

    let mut knots = Vec::with_capacity(internal_knots.len() + 2 * (degree + 1));
    knots.extend(std::iter::repeat_n(xmin, degree + 1));
    knots.extend(internal_knots.iter().copied());
    knots.extend(std::iter::repeat_n(xmax, degree + 1));
    let knots = Array1::from(knots);

    check_schoenberg_whitney_parallel(x, &knots, degree).map_err(|_| LssError::InvalidInput)?;
    make_lsq_spline_parallel(x, y, &knots, degree, None, solver)
}
//Идея:
//Мы знаем уровень шума sigm.а в данных (например, от характеристик прибора или от анализа остатков при большом числе узлов).
//Выбираем минимальное число узлов n такое, что:
//RMS(residual)≈σ
//Алгоритм:
//Начинаем с малого числа узлов (например 10)
//Фитим
//Считаем RMS остатка
//Если RMS >> σ → увеличиваем узлы
//Если RMS ≈ σ → остановка
pub fn choose_knots_by_noise(
    x: &Array1<f64>,
    y: &Array1<f64>,
    sigma: f64,
    degree: usize,
    min_knots: usize,
    max_knots: usize,
) -> usize {
    let xmin = x[0];
    let xmax = x[x.len() - 1];

    for n in min_knots..=max_knots {
        let internal = Array1::linspace(xmin, xmax, n);

        let spline = make_lsq_univariate_spline(
            x,
            y,
            &internal.slice(s![1..internal.len() - 1]).to_owned(),
            degree,
            SolverKind::Banded,
        );

        if spline.is_err() {
            continue;
        }

        let spline = spline.unwrap();

        let y_fit = x.mapv(|xi| spline.evaluate(xi));

        let residual = rms(&y_fit, y);

        if residual <= 1.2 * sigma {
            return n;
        }
    }

    max_knots
}

pub fn rms(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff = a - b;
    let mse = diff.mapv(|v| v * v).mean().unwrap();
    mse.sqrt()
}

pub fn rms_zero_mean(a: &Array1<f64>) -> f64 {
    let mean = a.mean().unwrap();
    let centered = a.mapv(|v| v - mean);
    centered.mapv(|v| v * v).mean().unwrap().sqrt()
}

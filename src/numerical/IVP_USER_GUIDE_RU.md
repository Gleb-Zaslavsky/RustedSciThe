# IVP User Guide (RustedSciThe)

Этот гайд рассчитан на разработчика на Rust, который хочет уверенно решать задачи Коши в RustedSciThe и не тратить время на «археологию» по исходникам. Здесь важна практическая логика: какой метод выбрать для конкретного типа задачи, как backend-маршрут влияет на поведение и производительность, и как собрать конфиг, который сначала корректен, а уже потом оптимален по времени.

Если вы уже используете LSODE2, сразу зафиксируем рамку. У LSODE2 есть отдельное подробное руководство и отдельная алгоритмическая история (mirroring LSODE/LSODA, переключение Adams/BDF, parity-наборы). В этом документе мы сознательно разбираем остальные IVP-пути: Native BDF, Radau, Backward Euler и нежесткую explicit-семью через универсальный фасад.

## 1. Картина методов в этом гайде

В RustedSciThe есть два уровня IVP API.

Первый уровень — специализированные API конкретных солверов. Вы работаете напрямую с `BDF::ODEsolver`, `Radau` или `BE`, детально настраиваете их опции и явно вызываете нужный метод. Это дает максимум контроля и часто удобнее в production-коде, где поведение должно быть максимально прозрачным.

Второй уровень — `UniversalODESolver` из [`ODE_api2.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/ODE_api2.rs). Это единый фасад, который позволяет менять метод без переписывания всей обвязки. Такой путь особенно полезен для сравнительных прогонов, регрессионных тестов и оркестрации, где выбор метода задается данными, а не руками.

В этом гайде обсуждаются:

- жесткие методы: BDF, Radau IIA, Backward Euler;
- нежесткие explicit-методы: RK45, Dormand–Prince (`DOPRI`), Adams–Bashforth 4 (`AB4`).

Комментарий про RK4: отдельной публичной точки входа «классический фиксированный RK4» в `UniversalODESolver` сейчас нет. На практике его нишу закрывают RK45/DOPRI, а в multistep-сценариях — AB4.

## 2. Жесткость задачи и выбор семейства

Во многих реальных задачах самое сложное — не записать уравнения, а правильно выбрать семейство интегратора до того, как начнутся численные проблемы.

Если в модели есть сильно разделенные масштабы времени, быстрые релаксации, химическая кинетика или выраженная диссипация, стартовать разумно со stiff-метода. BDF обычно выступает рабочим «дефолтом» для больших жестких систем, Radau хорош, когда нужна устойчивая высокопорядковая implicit Runge–Kutta схема, а Backward Euler полезен как максимально консервативный baseline.

Если траектория гладкая и stiffness-индикаторы не проявляются, explicit-семейство почти всегда дешевле по запуску и проще в настройке. Для общего случая в этой группе обычно берут RK45, DOPRI используют в той же нише, а AB4 — когда нужен именно явный многошаговый характер.

Если вы не уверены, универсальный API позволяет быстро прогнать одну и ту же постановку разными методами и сравнить финальную ошибку, число вызовов и встроенную статистику.

## 3. Философия backend-маршрутов: Numerical, Lambdify, AOT

Для stiff-солверов в RustedSciThe метод интегрирования и способ вычисления residual/Jacobian — это разные оси настройки. Это ключевая идея.

`Numerical` означает, что вы передаете callbacks напрямую: residual `f(t, y)` и, опционально, Jacobian `df/dy`. Если Jacobian не задан, используется конечная разность. Такой путь удобен, когда модель уже реализована в Rust и символьный этап не нужен.

`Lambdify` означает, что уравнения задаются как `Expr`, а затем при подготовке солвера превращаются в исполняемые closures. Этот путь практически не требует внешней инфраструктуры и обычно удобен как базовый сценарий «сначала корректность».

`AOT` (ahead-of-time) означает, что символьные выражения превращаются в сгенерированный код, который компилируется и линкуется в артефакт. Вы платите upfront-цену за сборку, зато снижаете стоимость вызовов внутри solve-loop. Это инструмент производительности, а не просто другой синтаксис.

Важно понимать, что математика метода при этом не меняется. Меняется только инфраструктура вычислителей.

## 4. UniversalODESolver: один фасад, несколько методов

`UniversalODESolver` позволяет держать единый каркас задачи и менять метод/маршрут с минимальными правками.

### 4.1 Минимальный explicit-пример (нежесткий)

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

let eq = vec![Expr::parse_expression("-y")];
let vars = vec!["y".to_string()];

let mut solver = UniversalODESolver::rk45(
    eq,
    vars,
    "t".to_string(),
    0.0,
    DVector::from_vec(vec![1.0]),
    1.0,
    1e-4,
);

solver.solve();
let (t, y) = solver.get_result();
println!("status = {:?}", solver.get_status());
println!("grid points = {}", t.unwrap().len());
println!("y(1) = {}", y.unwrap()[(y.unwrap().nrows() - 1, 0)]);
```

### 4.2 Native stiff-путь с пользовательским Jacobian (BDF)

```rust
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

let eq = vec![
    Expr::parse_expression("-1000.0*x0"),
    Expr::parse_expression("-x1"),
];
let vars = vec!["x0".to_string(), "x1".to_string()];

let mut solver = UniversalODESolver::bdf(
    eq,
    vars,
    "t".to_string(),
    0.0,
    DVector::from_vec(vec![1.0, 1.0]),
    0.02,
    1e-4,
    1e-7,
    1e-9,
)
.with_native_ode_callbacks(
    |_t, y| DVector::from_vec(vec![-1000.0 * y[0], -y[1]]),
    Some(|_t, _y| DMatrix::from_row_slice(2, 2, &[-1000.0, 0.0, 0.0, -1.0])),
);

solver.solve();
if let Some(stats) = solver.get_statistics() {
    println!("{}", stats.table_report());
}
```

### 4.3 Native stiff-путь с FD Jacobian fallback

Меняется только аргумент Jacobian:

```rust
.with_native_ode_callbacks(
    |_t, y| DVector::from_vec(vec![-1000.0 * y[0], -y[1]]),
    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
);
```

Это удобно на этапе черновой отладки модели, но для тяжелых stiff-прогонов аналитический Jacobian обычно лучше.

## 5. Собственные API солверов (прямой контроль)

Во многих production-проектах предпочитают прямые API, потому что в ревью и сопровождении сразу видно, какой именно метод используется.

### 5.1 Native BDF API

Основной модуль: [`BDF_api.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/BDF/BDF_api.rs).

Типичный сценарий:

1. собрать `BdfSolverOptions`;
2. создать солвер через `ODEsolver::new_with_options(...)`;
3. при необходимости поставить native callbacks через `set_native_ode_callbacks(...)`;
4. запустить solve и снять статистику.

`BDF` поддерживает и symbolic-маршруты, и pure numerical callbacks. Если native Jacobian не задан, автоматически используется конечная разность.

### 5.2 Radau API

Основной модуль: [`Radau_main.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/Radau/Radau_main.rs).

Radau работает через `RadauSolverOptions` с выбором порядка (`Order3/Order5/Order7`), tolerances и generated backend config. Native callbacks включаются через `set_native_ode_callbacks(...)`; при отсутствии Jacobian применяется FD fallback.

У Radau подробная встроенная статистика (Newton solves, Jacobian calls, LU usage), поэтому метод удобен не только как интегратор, но и как диагностический инструмент.

### 5.3 Backward Euler API

Основной модуль: [`BE.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/BE.rs).

`BE` — самый консервативный stiff baseline в этой группе. Он поддерживает generated backends и pure numerical callbacks, включая FD fallback для Jacobian. Это полезный вариант, когда важны предсказуемость и прозрачность Newton-итераций.

## 6. Как выбирать backend в stiff-методах

### 6.1 Lambdify как baseline

Lambdify разумно брать первым шагом, когда вы валидируете новую модель, хотите минимум инфраструктурных переменных и хотите быстро получить корректный референс.

### 6.2 AOT как инженерный ускоритель

AOT имеет смысл, когда модель решается многократно или solve-loop очень длинный, так что compile/link-затраты окупаются. Для dense IVP-маршрутов в RustedSciThe сейчас практически доступны четыре цели: `C+tcc`, `C+gcc`, `Zig` и `Rust`. Для быстрой локальной итерации часто удобнее `C+tcc`, а для повторяемых production-прогонов нередко выгоднее `C+gcc` из-за лучшей скорости в steady-state. `Zig` и `Rust` — полноценные рабочие варианты, когда они лучше подходят вашей инфраструктуре.

Жизненный цикл артефакта задается через `DenseIvpGeneratedBackendMode` и нижележащий `SymbolicIvpAotBuildPolicy`. На практике чаще всего используют:

- `BuildIfMissingRelease`: собрать при отсутствии, затем переиспользовать;
- `RequirePrebuilt`: требовать заранее собранный артефакт и падать при его отсутствии;
- `Defaults`/`UseIfAvailable`: взять готовый артефакт, а при его отсутствии остаться на lambdify-пути.

Быстрый вариант настройки — через fluent API солвера:

```rust
let solver = ODEsolver::new_with_options(opts)
    .with_dense_generated_backend_c_tcc("target/generated-ivp-tests")
    .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);
```

Если нужен явный контроль над политикой сборки, backend-типом и директорией артефактов, лучше задавать `SymbolicIvpGeneratedBackendConfig` напрямую:

```rust
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpGeneratedBackendConfig,
};

let gen_cfg = SymbolicIvpGeneratedBackendConfig::from_mode(
    DenseIvpGeneratedBackendMode::BuildIfMissingRelease
)
.with_c_gcc()
.with_output_parent_dir(Some("target/generated-ivp-tests".into()));
```

Параллельная декомпозиция в AOT задается через `SymbolicIvpAotOptions`: отдельно для residual и для dense Jacobian. Это тот слой, где вы управляете chunking-стратегией:

```rust
use RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};

let chunked = SymbolicIvpAotOptions {
    residual_strategy: ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 },
    jacobian_strategy: DenseJacobianChunkingStrategy::ByTargetChunkCount { target_chunks: 2 },
};

let gen_cfg = gen_cfg.with_aot_options(chunked);
```

Практическая рекомендация: разделяйте warm-up и чистый solve-benchmark. Сначала прогоните build/prepare, затем отдельно измеряйте steady-state solve, иначе шум сборки скроет реальную разницу между backend-маршрутами.

### 6.3 Pure numerical путь

Native callbacks выбирают, когда модель уже реализована в Rust, символьная подготовка не нужна, и требуется полный контроль над Jacobian-логикой. Для честных бенчмарков против внешних реализаций этот путь обычно самый чистый.

## 7. Статистика — это часть результата, а не «опция»

Сильная сторона текущей архитектуры — встроенная статистика этапов и вызовов. Вместо обвязки кустарными таймерами лучше использовать нативный инструмент.

Для `UniversalODESolver`:

```rust
if let Some(stats) = solver.get_statistics() {
    println!("{}", stats.table_report());
}
```

Для прямых API солверов используйте их собственные `get_statistics()` (где доступны). В stiff-анализе обычно важны как минимум residual calls, Jacobian calls, LU counts, nonlinear iterations и суммарные времена setup/solve.

## 8. End-to-end примеры в репозитории

Если нужен не сниппет, а готовый исполняемый сценарий, начинайте с:

- [`bdf_solver_examples.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/bdf_solver_examples.rs): набор BDF-примеров, включая stiff и нелинейные системы;
- [`universal_ode_example.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/universal_ode_example.rs): использование RK45, DOPRI, Radau, BDF и BE через единый фасад;
- [`radau_backends_guide.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/radau_backends_guide.rs): практические паттерны backend-настройки Radau.

Для LSODE2-сценариев используйте отдельные руководства:

- [`LSODE2_USER_GUIDE_EN.md`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/LSODE2/LSODE2_USER_GUIDE_EN.md)
- [`LSODE2_USER_GUIDE_RU.md`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/LSODE2/LSODE2_USER_GUIDE_RU.md)

## 9. Рабочая последовательность выбора

Практический устойчивый сценарий обычно выглядит так. Сначала вы добиваетесь корректности на Lambdify с аккуратными tolerances. Затем проверяете, действительно ли задача stiff/non-stiff в ваших данных, и читаете статистику. Потом, если есть реальный bottleneck, переходите к native callbacks или AOT. После этого фиксируете выбранную политику в тестах, чтобы будущие изменения не сдвигали поведение молча.

Эта последовательность может показаться «слишком дисциплинированной», но именно она экономит больше всего времени на длинной дистанции.

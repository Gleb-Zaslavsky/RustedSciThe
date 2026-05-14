# LSODE2 в RustedSciThe: руководство пользователя

Этот текст написан для разработчика на Rust, который открывает LSODE2 впервые и хочет быстро перейти от «что это вообще такое» к рабочей, осмысленной конфигурации. Мы будем говорить не только о том, какие флаги существуют, но и о том, почему они вообще появились, какие решения стоят за ними и в какой момент выбирать один маршрут вместо другого.

Важно сразу договориться о фокусе. LSODE2 в текущей архитектуре RustedSciThe одновременно решает две задачи: воспроизводит математическую философию семейства ODEPACK (в особенности LSODA-подобный автопереход Adams/BDF), и встраивается в современный pipeline символики, codegen и линейной алгебры RustedSciThe. Поэтому у него есть «алгоритмическая часть» и «инфраструктурная часть». Если смешивать их в голове, конфиг быстро превращается в хаос; если разделять, всё становится логичным.

## С чего начать: две оси выбора

Практически любую постановку в LSODE2 удобно описывать по двум осям. Первая ось отвечает за математику шага: фиксированный BDF, фиксированный Adams или автоматическое переключение семейства, как в LSODA. Вторая ось отвечает за то, как считаются residual/Jacobian и как решается линейная система Ньютона.

Именно вторая ось порождает три знакомых маршрута: чисто численный (`Numerical`), символический с lambdify (`Lambdify`) и символический с ahead-of-time генерацией (`AOT`). На этом месте полезно держать простую мысль: это не «три разных солвера», это три способа кормить один и тот же алгоритм качественными функциями и матрицами.

## Что принимает `Lsode2ProblemConfig::new` и почему это «ядро синтаксиса»

Конструктор `Lsode2ProblemConfig::new(...)` — действительно центральная точка входа. Его сигнатура:

```rust
pub fn new(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    max_step: f64,
    rtol: f64,
    atol: f64,
) -> Self
```

По смыслу это читается как: «вот система уравнений, вот имена переменных, вот независимая переменная, вот старт, стартовое состояние, конечное время и базовые настройки точности/шага».

`eq_system` — правая часть ОДУ в символическом виде (`Expr`).  
`values` — имена компонент вектора состояния в том же порядке, в каком компоненты лежат в `y0`.  
`arg` — имя независимой переменной, обычно `"t"`.  
`t0`, `y0`, `t_bound` — стандартная тройка начальные условия + горизонт интегрирования.  
`max_step`, `rtol`, `atol` — верхний потолок шага и относительная/абсолютная точности.

Даже если вы потом пойдёте в чисто аналитические callbacks, этот конструктор остаётся основой: он задаёт общую геометрию задачи и политику интегратора.

## Dense, Sparse, Banded: не «вкусовщина», а модель стоимости

Выбор структуры Якобиана — это выбор вычислительной экономики.

`Dense` оправдан для маленьких систем, для отладки и для случаев, где главная ценность — простота трассировки.  
`Sparse` — обычный рабочий default для больших неструктурированных систем: меньше память, дешевле факторизация.  
`Banded` — лучший путь там, где матрица действительно ленточная, например в дискретизированных задачах переноса/диффузии и многих BVP/PDE-подобных постановках.

Если структура выбрана неверно, солвер не обязательно «упадёт», но почти гарантированно потеряет производительность и иногда устойчивость. Поэтому сначала стоит смотреть на структуру Якобиана, а уже потом на тюнинг толерансов.

## LSODE и LSODA: исторический контекст и как это отражено в LSODE2

В классическом Fortran-мире LSODE и LSODA — не синонимы. LSODE предполагает, что семейство метода выбирается пользователем (Adams или BDF) и дальше не прыгает само по себе. LSODA добавляет автоматику: оценивает «жёсткость» и стоимость, и при необходимости переключает семью.

LSODE2 в RustedSciThe использует эту философию буквально. Если вы задаёте ручной контроллер (`bdf_only` или `adams_only`), это LSODE-стиль. Если выбираете `automatic_adams_bdf`, это LSODA-стиль поведения в рамках единого API.

## Ключевые типы конфигурации: что за что отвечает

### `Lsode2BackendConfig`

Это «контейнер» низкоуровневых backend-настроек. Внутри три важные составляющие: какой backend Якобиана (`jacobian_backend`), какой линейный backend (`linear_solver_backend`) и как организован generated backend для символики/AOT (`generated_backend`).

В обычной практике вы редко собираете его «с нуля руками»: чаще используете готовые конструкторы вроде `native_sparse_faer()`, `native_banded_faithful()`, `dense_aot_c_gcc(...)` и затем при необходимости аккуратно донастраиваете.

### `Lsode2JacobianBackend`

`SymbolicGenerated` означает, что Якобиан идёт через символический pipeline (Lambdify или AOT).  
`AnalyticClosure` означает пользовательский аналитический callback.  
`FiniteDifference` — встроенный конечно-разностный Якобиан.

Важно: в текущем состоянии API `AnalyticClosure` и `FiniteDifference` валидны именно для native solve path; bridge-режим под это не предназначен.

### `Lsode2LinearSolverBackend`

`Dense` — плотный LU,  
`SparseFaer` — sparse LU через `faer`,  
`BandedFaithful` — faithful LAPACK-style banded LU.

Это технический слой. На пользовательском уровне чаще удобнее задавать структуру (`Dense/Sparse/Banded`) и политику (`Auto/Force`), а не backend напрямую.

## Политика выбора линейного решателя: Auto против Force

`Lsode2LinearSolverPolicy::Auto` — это не магия и не «угадайка». Выбор делается детерминированно из `Lsode2LinearSystemStructure`: dense ведёт к `DenseLu`, sparse к `FaerSparseLu`, banded к `LapackFaithfulBandedLu`.

`Force(...)` нужен, когда вы делаете исследовательский прогон, parity-сравнение или сознательно проверяете гипотезу о деградации конкретного backend-а.

Именно эта схема «сначала структура, затем Auto» обычно даёт самый устойчивый и воспроизводимый production-конфиг.

## Символика: `ExprLegacy` и `AtomView`

`Lsode2SymbolicAssemblyBackend` имеет два варианта: `ExprLegacy` и `AtomView`.

`ExprLegacy` — консервативный baseline.  
`AtomView` — более современное упакованное представление (внутренний IR для символики), которое в ряде задач ускоряет подготовку и/или выполнение.

Здесь термин IR стоит проговорить явно. IR (intermediate representation, промежуточное представление) — это внутренний формат между «математическим выражением» и «исполняемым кодом». В LSODE2 IR-слой позволяет не переписывать математику заново при смене backend-а: вы меняете исполнение, но не формулы.

## Lambdify и AOT: что это такое на практике

Lambdify в контексте LSODE2 — это построение исполняемых Rust-замыканий из символики во время подготовки solve. Без внешнего компилятора, с быстрым стартом, с хорошей переносимостью.

AOT (ahead-of-time) — это другой компромисс. Вы платите upfront-цену: codegen, компиляция, линковка артефакта. Зато на длинной дистанции (много вызовов residual/Jacobian, много повторных запусков) runtime-часть обычно выигрывает.

С практической стороны AOT в RustedSciThe поддерживает несколько тулчейнов: C через `gcc`, C через `tcc`, `zig`, а также Rust toolchain. Это означает, что соответствующие компиляторы должны быть доступны в окружении запуска.

`Debug` и `Release` в AOT-профиле — стандартный компромисс. Debug обычно быстрее собирается, но медленнее работает на solve-этапе. Release дольше собирается, но снижает стоимость вычислений в цикле интегрирования.

## Параллелизм/чанкинг в AOT: как это связано с производительностью

В generated backend есть `aot_options`, где задаётся стратегия «нарезки» residual/Jacobian на куски. Это влияет на размер функций, стоимость компиляции и поведение runtime-плана. Для residual используются стратегии вроде `Whole`, `ByTargetChunkCount`, `ByOutputCount`; для dense Jacobian — `Whole`, `ByTargetChunkCount`, `ByRowCount`.

На уровне LSODE2 это применяется через `SymbolicIvpGeneratedBackendConfig`:

```rust
use RustedSciThe::numerical::LSODE2::{Lsode2BackendConfig, Lsode2ProblemConfig};
use RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};

let aot_options = SymbolicIvpAotOptions {
    residual_strategy: ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 },
    jacobian_strategy: DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 32 },
};

let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-aot")
    .with_c_gcc()
    .with_aot_options(aot_options);

let cfg = Lsode2ProblemConfig::new(/* ... */)
    .with_backend(
        Lsode2BackendConfig::native_sparse_faer_with_generated_backend(generated)
    );
```

Обычно это уже advanced-тюнинг: сначала стоит добиться корректного baseline, и только потом играть стратегиями чанкинга.

## `with_stop_condition_*`: зачем три варианта

Ранний останов в LSODE2 полезен не только для удобства, но и для физической корректности в задачах с естественным «концом процесса».

`with_stop_condition(...)` — shorthand для условия `variable >= target`.  
`with_stop_condition_ge(...)` — явный вариант `>=`.  
`with_stop_condition_le(...)` — явный вариант `<=`.  
`with_stop_condition_abs(...)` — останов по близости к цели `|variable - target| <= tolerance`.

В задачах горения это особенно полезно: например, можно остановить интегрирование, когда степень превращения достигает 0.999, а не продолжать шагать в «формально разрешённую, но физически бессмысленную» область.

## Что такое task document и как LSODE2 запускается через парсер

В RustedSciThe есть командный интерпретатор, который разбирает человекочитаемый документ задачи и превращает его в типизированный `IvpTaskSpec`, а затем в `UniversalODESolver`.

Это и есть task document путь: вы описываете задачу текстом (`task`, `equations`, `initial_conditions`, `solver_options`, `postprocessing`), а парсер нормализует и валидирует поля. Для LSODE2 доступны поля `lsode2_symbolic_assembly`, `lsode2_symbolic_execution`, `lsode2_aot_toolchain`, `lsode2_aot_profile`, `lsode2_linear_structure`, `lsode2_linear_solver_policy`, `lsode2_native_execution` и другие.

С точки зрения результата сейчас встроенный postprocessing-конвейер намеренно консервативен: CSV-выгрузка поддерживается напрямую, а флаг `plot` парсится и сохраняется в спецификации для внешних обёрток/фронтендов, но не запускает графику автоматически в ядре парсера.

## Полный пример из Rust-кода (Lambdify, sparse, faithful BDF)

Ниже законченный минимальный сценарий: от задания уравнений до вывода статуса и статистики.

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-10.0*y1 + 9.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();

    let status = solver.get_status().unwrap_or_else(|| "unknown".to_string());
    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|tv| tv[tv.len() - 1]).unwrap_or(f64::NAN);
    let final_y1 = y.as_ref().map(|m| m[(m.nrows() - 1, 0)]).unwrap_or(f64::NAN);
    let final_y2 = y.as_ref().map(|m| m[(m.nrows() - 1, 1)]).unwrap_or(f64::NAN);

    println!("status  = {status}");
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");

    if let Some(stats) = solver.get_statistics() {
        println!("{}", stats.table_report());
    }
}
```

## Полный пример task document

Этот формат полезен, когда задача приезжает из CLI, скрипта или внешнего сервиса:

```text
task
solver: IVP
method: LSODE2

equations
arg: t
y1: -10.0*y1 + 9.0*y2
y2: y1 - y2

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0, 0.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: ExprLegacy
lsode2_symbolic_execution: AOT
lsode2_aot_toolchain: c_gcc
lsode2_aot_profile: release
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
lsode2_native_execution: faithful_bdf_solve

postprocessing
save_csv: true
csv_path: lsode2_result.csv
plot: false
```

## Практическая стратегия выбора конфигурации

Если система небольшая и вы в фазе отладки, начинайте с Dense + Lambdify. Если система большая и разреженная, начинайте со Sparse + Lambdify, а затем переходите в AOT и сравнивайте stage-метрики (`prepare` против `solve`) в multi-run story tests. Для действительно ленточных систем выбирайте Banded + faithful backend и внимательно задавайте `kl/ku`.

Смысл такой последовательности прост: сначала подтверждаем корректность математики на самом прозрачном маршруте, затем переносим ту же математику в более агрессивный backend и проверяем эквивалентность, а уже потом оптимизируем compile/runtime баланс.

## Где смотреть дальше в репозитории

За быстрыми живыми примерами идите в `examples/lsode2_numerical_guide.rs`, `examples/lsode2_lambdify_guide.rs`, `examples/lsode2_aot_guide.rs`, `examples/lsode2_manual_bdf_guide.rs`, `examples/lsode2_manual_adams_guide.rs` и `examples/lsode2_task_shell_guide.rs`.

Если нужен профиль производительности и корректности на уровне сценариев, смотрите `story_tests.rs` и `story_tests2.rs`. Если нужна математика parity относительно ODEPACK-логики, смотрите `parity_micro.rs`, `stiff_parity_tests.rs`, `nonstiff_parity_tests.rs` и `MIRRORING_CHECKLIST.md`.

Когда эти уровни разделены (guide для эксплуатации, story для поведения end-to-end, parity для математической эквивалентности), LSODE2 перестаёт выглядеть сложным. Он становится предсказуемым инженерным инструментом, который можно осознанно настраивать под конкретную задачу.
## Update: AOT/Lambdify parallel chunking

Практический минимум для параллельного chunking в LSODE2:

```rust
let config = config
    .with_aot_parallel_chunking(2)
    .with_aot_target_chunks(8, 8);
```

Точечная настройка sparse-чанкинга:

```rust
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
let config = config.with_aot_sparse_chunking_strategy(
    SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }
);
```

Для Lambdify можно также задать generated runtime chunking:

```rust
let config = config.with_backend(
    Lsode2BackendConfig::native_sparse_faer()
        .with_generated_backend_target_chunks(4, 4),
);
```

# BVP Damped/Frozen в RustedSciThe: руководство пользователя

Этот текст написан для разработчика на Rust, который впервые открывает BVP-часть RustedSciThe и хочет не гадать по тестам, а сразу понять рабочую модель: какие данные передаются в солвер, чем отличаются Damped и Frozen, когда выбирать Sparse или Banded, что реально означают Lambdify, AOT и Numerical path, и какие выводы дают наши story/performance-прогоны.

В RustedSciThe BVP-солверы решают краевые задачи (boundary value problems) для систем обыкновенных дифференциальных уравнений. Пользователь задает непрерывную систему первого порядка, сетку, начальное приближение и граничные условия. Внутри система дискретизуется, превращается в большую нелинейную систему алгебраических уравнений, а затем решается модифицированным методом Ньютона. Это важно держать в голове: вы настраиваете не только формулы правых частей, но и весь путь от дискретизации до сборки якобиана и решения линейной системы на каждом шаге Ньютона.

## 1. Damped и Frozen: две стратегии Ньютона

Основной production-путь - `Damped` Newton, реализованный в [`NR_Damp_solver_damped.rs`](NR_Damp_solver_damped.rs). Он использует демпфирование шага Ньютона, bounds, повторную сборку якобиана, диагностику и современный backend-handoff. Если вы не знаете заранее, что вам нужна Frozen-стратегия, начинайте с Damped.

`Frozen` Newton, реализованный в [`NR_Damp_solver_frozen.rs`](NR_Damp_solver_frozen.rs), оставлен как специализированная стратегия с "замороженной" линеаризацией. Он полезен, когда вы намеренно хотите переиспользовать структуру/якобиан или сравнить поведение с более простым frozen-процессом. Это не более "новый" и не более "быстрый" путь по умолчанию. Практическое правило простое: сначала добейтесь сходимости и правильности на Damped, затем используйте Frozen как осознанную оптимизацию или исследовательский режим.

Оба солвера поддерживают символическую постановку через `Expr` и generated backend configuration. Чистый численный путь с Rust-замыканием RHS сейчас полноценно реализован для Damped.

## 2. Что именно передается в BVP-солвер

Предпочтительный конструктор для нового кода - `NRBVP::new_with_options(...)`. Его сигнатура для Damped выглядит так:

```rust
pub fn new_with_options(
    eq_system: Vec<Expr>,
    initial_guess: DMatrix<f64>,
    values: Vec<String>,
    arg: String,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    t0: f64,
    t_end: f64,
    n_steps: usize,
    options: DampedSolverOptions,
) -> NRBVP
```

`eq_system` - это правая часть системы первого порядка. Например, уравнение `y'' + y = 0` обычно записывают как `y' = z`, `z' = -y`, значит `eq_system = ["z", "-y"]`, а `values = ["y", "z"]`. Порядок в `values` задает порядок компонент состояния и должен совпадать с порядком правых частей.

`arg` - имя независимой переменной, обычно `"x"`. Пара `t0`, `t_end` задает отрезок интегрирования/краевой задачи, а `n_steps` - число интервалов сетки.

`initial_guess` - начальное приближение для неизвестных значений на сетке. Важная синтаксическая особенность текущего BVP Damped/Frozen API: начальное приближение обычно имеет размер `values.len() x n_steps`, а не `values.len() x (n_steps + 1)`. Граничные значения задаются отдельно в `border_conditions` и затем встраиваются в полное сеточное решение. В тестах это обычно выглядит так:

```rust
let initial_guess = DMatrix::from_vec(values.len(), n_steps, guess_values);
```

`border_conditions` задаются как `HashMap<String, Vec<(usize, f64)>>`. В текущем соглашении индекс `0` означает левую границу, индекс `1` - правую границу. Например, `y(0)=0` и `z(L)=1` записываются так:

```rust
let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(1usize, 1.0)]),
]);
```

Damped numeric route (`NumericOnly`) ожидает корректно поставленную систему первого порядка: общее число фиксированных граничных условий должно равняться `values.len()`. Эти условия не обязаны распределяться строго "одно на переменную". Например, осциллятор `y' = z`, `z' = -y` можно задать как численную краевую задачу с условиями `y(0)=0` и `y(pi/2)=1`, оставляя производную переменную `z` свободной на обеих границах. Frozen численный route не использует.

## 3. `DampedSolverOptions`: основной синтаксис настройки

`DampedSolverOptions` группирует настройки, которые раньше передавались длинным списком позиционных аргументов. Его полный конструктор:

```rust
pub fn new(
    scheme: String,
    strategy: String,
    strategy_params: Option<SolverParams>,
    linear_sys_method: Option<String>,
    method: String,
    abs_tolerance: f64,
    rel_tolerance: Option<HashMap<String, f64>>,
    max_iterations: usize,
    bounds: Option<HashMap<String, (f64, f64)>>,
    loglevel: Option<String>,
) -> Self
```

На практике чаще используются пресеты:

```rust
let options = DampedSolverOptions::sparse_damped();
let options = DampedSolverOptions::banded_damped();
let options = DampedSolverOptions::dense_damped();
```

После пресета обычно уточняют tolerance, bounds, nonlinear strategy и backend:

```rust
let strategy = SolverParams {
    max_jac: Some(6),
    max_damp_iter: Some(6),
    damp_factor: Some(0.5),
    adaptive: None,
};

let options = DampedSolverOptions::banded_damped()
    .with_strategy_params(Some(strategy))
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(HashMap::from([
        ("y".to_string(), 1e-8),
        ("z".to_string(), 1e-8),
    ]))
    .with_bounds(HashMap::from([
        ("y".to_string(), (-2.0, 2.0)),
        ("z".to_string(), (-2.0, 2.0)),
    ]))
    .with_max_iterations(40)
    .with_loglevel(Some("none".to_string()));
```

`scheme` выбирает дискретизационную схему, чаще всего используется `"forward"` или варианты, уже закрепленные в тестах. `strategy` обычно `"Damped"`. `linear_sys_method` лучше оставлять `None`, если вы выбираете Sparse/Banded через `method` и generated backend config. `method` - это крупный выбор матричного пути: `"Dense"`, `"Sparse"` или `"Banded"`.

`FrozenSolverOptions` устроен похожим образом, но проще: там нет bounds и per-variable relative tolerance в том же виде, а главный пресет выглядит так:

```rust
let options = FrozenSolverOptions::banded_frozen()
    .with_tolerance(1e-8)
    .with_max_iterations(40);
```

## 4. Dense, Sparse, Banded: это не стиль, а модель стоимости

`Dense` - проверочный и малый путь. Он удобен для маленьких систем, отладки и случаев, когда структура матрицы не важна. Для больших BVP Dense быстро становится дорогим по памяти и времени.

`Sparse` - универсальный production-путь для больших задач, где якобиан разреженный, но структура не обязательно узкополосная. Он хорошо подходит для химических сеток, систем с нерегулярной связностью и задач, где вы не хотите заранее доказывать banded-структуру.

`Banded` - лучший выбор для задач, где ненулевые элементы якобиана лежат около диагонали. Это типично для BVP после конечно-разностной дискретизации систем первого порядка: соседние узлы связаны, дальние узлы почти никогда не связаны напрямую. В RustedSciThe Banded route использует LAPACK-style banded LU, то есть матрица хранится и факторизуется в полосовом виде. На наших combustion BVP story-тестах Banded обычно дает заметно меньшее время линейного solve, чем Sparse, особенно когда система действительно имеет узкую полосу.

Главное предупреждение: Banded хорош тогда, когда структура правда banded. Если задача не полосовая, принудительный Banded route может ухудшить устойчивость и производительность. В этом смысле Banded - как спортивная резина: прекрасна на своем покрытии и бессмысленна на болоте.

## 5. Три backend-пути: Numerical, Lambdify, AOT

В BVP Damped/Frozen полезно разделять математический метод и backend. Метод Ньютона и дискретизация остаются теми же, но backend определяет, как вычисляется невязка, как собирается якобиан и какой линейный solve используется.

### Numerical path

Numerical path в Damped - это путь без символической машинерии. Вы передаете Rust-замыкание правой части `f(x, y, params)`, а отдельный модуль [`numeric_discretization.rs`](numeric_discretization.rs) строит дискретизованную невязку напрямую из этого замыкания. Это путь для моделей, где source of truth уже является Rust-кодом, а не `Expr`.

Теперь у Damped есть два явных численных конструктора. `NRBVP::new_numeric_fd_with_options(...)` принимает RHS-замыкание и явно выбирает конечно-разностный якобиан Ньютона. `NRBVP::new_numeric_with_jacobian_options(...)` принимает RHS-замыкание и маленькое непрерывное замыкание якобиана `df/dy`; большой дискретизованный BVP-якобиан строится внутри солвера с учетом сетки, схемы дискретизации и граничных условий. Это важное отличие от старого низкоуровневого паттерна: пользователю больше не нужно передавать пустой `eq_system`, чтобы сказать "это чисто численная задача".

Граничные условия в численном пути следуют тому же правилу для системы первого порядка: нужно задать ровно столько фиксированных значений на концах отрезка, сколько переменных состояния находится в `values`. Они могут относиться к разным переменным или к одной переменной на двух концах, если итоговая редуцированная система остается квадратной. Типичный двухточечный синтаксис выглядит так:

```rust
let border_conditions = HashMap::from([(
    "y".to_string(),
    vec![(0usize, 0.0), (1usize, 1.0)],
)]);
```

Здесь `z = y'` остается частью вектора состояния и по-прежнему участвует в RHS-замыкании и замыкании якобиана, но прямого граничного значения для `z` нет.

Для типичного sparse-сценария есть более короткие обертки: `NRBVP::new_numeric_fd(...)` и `NRBVP::new_numeric_with_jacobian(...)`. Они все равно принимают `bounds` и per-variable `rel_tolerance`, потому что в демпфированном Ньютоне это не декоративные параметры: bounds ограничивают допустимый шаг, а tolerances задают взвешенный критерий сходимости. Если нужен Banded, особые параметры нелинейной стратегии или логирование, используйте конструкторы `*_with_options`.

Frozen устроен строже. `BackendSelectionPolicy::NumericOnly` для Frozen намеренно отвергается, а не превращается молча в Lambdify или FD. Frozen сейчас ожидает symbolic Lambdify/AOT маршрут с подготовленным callback-якобианом.

### Lambdify

Lambdify означает: вы задаете систему через `Expr`, RustedSciThe выполняет символическую сборку дискретизованной BVP-системы и превращает выражения невязки/якобиана в исполняемые Rust callbacks без внешней компиляции. Это лучший baseline для correctness. У него низкий bootstrap-риск: не нужны `gcc`, `tcc`, `zig` или загрузка динамических библиотек.

Lambdify остается лучшим первым запуском новой постановки: в этот момент важнее всего проверить уравнения, граничные условия и начальное приближение без зависимости от внешнего toolchain. Для малых одноразовых расчетов он часто выигрывает end-to-end. Однако после оптимизации `AtomView` это уже не следует переносить на большие задачи: на measured combustion BVP с `n_steps = 1000` cold `tcc` AOT оказался быстрее Lambdify и в Sparse, и в Banded маршруте.

### AOT

AOT (ahead-of-time) - это compiled backend. Символические выражения проходят через intermediate representation, затем генерируется код, собирается отдельный артефакт и подключается как runtime callback. Философия простая: заплатить за подготовку заранее, чтобы последующие вызовы невязки и якобиана были дешевле.

RustedSciThe поддерживает несколько codegen/toolchain-вариантов: C через `gcc` или `tcc`, Zig, а также Rust backend в codegen-слое. Для C/Zig путей соответствующие компиляторы должны быть установлены и доступны в `PATH`. По текущим Windows release-прогонам `tcc` оказался не просто самым дешевым способом собрать артефакт: в больших `AtomView`-задачах он вполне конкурентен Lambdify по полному времени от старта до результата и в отдельных опытах превосходит его уже при cold build. `gcc` может давать сильный runtime-код, но стартовая цена обычно выше, а Zig иногда имеет заметную и нестабильную стоимость сборки. Это измеренная рекомендация, а не жесткое правило: на другой машине и для другой структуры уравнений ranking следует перепроверить.

AOT безусловно имеет смысл, когда вы решаете одну и ту же большую задачу много раз, меняете параметры, прогоняете continuation, sensitivity-like сценарии или запускаете тяжелый production batch. Но граница оказалась приятнее, чем ожидалось: с `AtomView` и `tcc` AOT может быть разумным выбором уже для одного крупного расчета, особенно в Banded-маршруте. Для маленького BVP цена подготовки артефакта по-прежнему обычно не окупается.

## 6. ExprLegacy и AtomView

В символическом BVP pipeline есть два пути сборки выражений:

`ExprLegacy` - старый, проверенный путь через классические `Expr`-структуры. Он полезен как compatibility baseline и страховка.

`AtomView` - более современный путь, рассчитанный на эффективную сборку и codegen. Отдельные process-isolated release-измерения на семействе combustion BVP показали, что он снимает особенно дорогую стоимость построения символьного якобиана и в разреженном (Sparse), и в полосовом (Banded) пути, сохраняя решение в ожидаемой численной точности. Поэтому production-пресеты Sparse и Banded и для Damped, и для Frozen солвера теперь выбирают `AtomView` по умолчанию.

Для полосового решения через Lambdify production-синтаксис теперь намеренно короткий:

```rust
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;

let options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify();

// Compatibility/control route для воспроизведения ExprLegacy-прогона:
let legacy_options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify()
    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
```

Тот же default наследуется в `FrozenSolverOptions::{sparse_frozen, banded_frozen}()` и в Sparse/Banded AOT-пресетах. `ExprLegacy` остается поддерживаемым явным override для проверки совместимости, контрольных сравнений и исследования frontend-зависимого поведения.

## 7. Generated backend config: как выбирать Lambdify, AOT и build policy

`GeneratedBackendConfig` - центральный объект для generated backend route. Он хранит policy выбора backend, build policy, codegen backend, C-компилятор, symbolic assembly backend, matrix override и chunking policy.

Высокоуровневые режимы:

```rust
SparseGeneratedBackendMode::Defaults
SparseGeneratedBackendMode::RequirePrebuilt
SparseGeneratedBackendMode::BuildIfMissingRelease

BandedGeneratedBackendMode::Defaults
BandedGeneratedBackendMode::Lambdify
BandedGeneratedBackendMode::BuildIfMissingRelease
```

`Defaults` обычно означает "предпочесть AOT, если он уже доступен, иначе fallback на Lambdify". Для production-режимов Sparse и Banded это теперь дополнительно означает символьную сборку через `AtomView`; Banded сверх того выбирает faithful LAPACK-style полосовой линейный решатель. `RequirePrebuilt` полезен в production, где отсутствие артефакта должно быть ошибкой, а не скрытой компиляцией. `BuildIfMissingRelease` удобен в интерактивном workflow: если артефакта нет, RustedSciThe соберет его на первом запуске.

Пресеты для repeated solves:

```rust
let sparse_aot = GeneratedBackendConfig::sparse_atomview_for_repeated_solves();
let banded_aot = GeneratedBackendConfig::banded_atomview_for_repeated_solves();
```

Эти пресеты сейчас используют AtomView и `tcc`-compiled C AOT как практический компромисс между скоростью bootstrap и runtime. Для явного выбора toolchain:

```rust
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc();
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig();
```

Скомпилированные артефакты можно использовать повторно. В этом и состоит смысл `BuildIfMissing`: первый запуск платит цену генерации/сборки, последующие запуски могут идти через уже собранный runtime backend. Если вам нужно жестко гарантировать, что production-ран не начнет компиляцию, используйте `RequirePrebuilt`.

Выбор для новой большой символической задачи удобно делать в два хода. Сначала запустите `AtomView + Lambdify`: это простой контроль решения и структуры матрицы. Затем сравните его с `AtomView + tcc AOT` на том же `Sparse` или `Banded` маршруте. В сохраненном честном cold-прогоне `combustion-1000` AOT `tcc/whole` дал около `2.46 s` против `2.84 s` у Banded Lambdify и около `2.68 s` против `4.33 s` у Sparse Lambdify. Это достаточно сильный сигнал, чтобы тестировать TCC AOT рано, а не оставлять его только для поздней оптимизации repeated solves.

Теперь есть и контролируемое warm-измерение production lifecycle. В story-тесте
Damped Banded AtomView `combustion-1000` один setup-запуск собрал `tcc`-артефакт,
после чего пять строгих решений `RequirePrebuilt` сравнивались с пятью решениями
Lambdify при чередовании порядка и одинаковой пятисекундной паузе. Warm AOT дал
`431.4 +/- 10.9 ms` против `468.3 +/- 12.5 ms` у Lambdify, то есть небольшой,
но устойчивый выигрыш примерно `7.9%`. С учетом цены первой сборки этот выигрыш
в зафиксированном опыте окупается приблизительно после восьми последующих
решений. Именно так следует понимать repeated-solve preset: не как обещание
победы в одиночном запуске, а как compiled route для повторного решения одной
крупной модели.

## 8. Chunking и parallel execution: честный вывод из измерений

AOT route умеет разбивать вычисление невязки и sparse Jacobian values на чанки. В терминах API это `AotChunkingPolicy`, `ResidualChunkingStrategy` и `SparseChunkingStrategy`. Runtime execution дополнительно управляется `AotExecutionPolicy` и `ParallelExecutorConfig`.

Пример явной настройки:

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use RustedSciThe::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;

let chunking = AotChunkingPolicy::with_parts(
    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
);

let exec = ParallelExecutorConfig {
    jobs_per_worker: 1,
    max_residual_jobs: None,
    max_sparse_jobs: None,
    fallback_policy: ParallelFallbackPolicy::Auto,
};

let cfg = GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
    .with_aot_chunking_policy(chunking)
    .with_aot_execution_policy(AotExecutionPolicy::Parallel(exec));
```

По нашим текущим BVP combustion story-тестам вывод такой: chunking корректен и реальный parallel path существует, но его польза зависит от масштаба и от того, какая фаза доминирует. На средних задачах `n_steps = 200` и `n_steps = 1000` вычисление callback values часто занимает лишь миллисекунды, поэтому распараллеливание мало двигает полный wall clock. На большой Banded `AtomView` задаче с `n_steps = 3000` путь `tcc/chunk4` действительно выполнил четыре residual и четыре Jacobian jobs без fallback и сократил hot callback intervals: residual values примерно с `13.8` до `7.1 ms`, Jacobian values примерно с `6.7` до `1.7 ms`.

Даже этот большой прогон не делает forced chunking универсальным default: полное cold-время `tcc/whole` и `tcc/chunk4` оказалось статистически близким, потому что hot callback - только часть расчета. Именно поэтому policy `Auto` важнее, чем "включить parallel всегда". Auto может fallback-нуться в sequential path, если chunk/job workload слишком мал; это защита от overhead, а не скрытая неисправность. Если вы исследуете break-even на своей машине, используйте diagnostic tests и печать `actual_jobs`, `fallback`, `residual_ms`, `jacobian_ms`, `linear_ms`. Для production оставляйте `Auto`, а forced parallel используйте как измеряемую настройку конкретной тяжелой модели.

### Как сравнивать cold и warm расчеты

Число `total_ms` имеет смысл только если ясно, что именно было включено в запуск. В cold wall-clock тесте новый процесс начинает с построения symbolic frontend, генерации кода, компиляции, линковки и Newton solve; такой опыт отвечает на вопрос "сколько ждать от нажатия кнопки до результата". В warm или `RequirePrebuilt` режиме готовый артефакт уже существует, и сравнение отвечает на другой вопрос: "сколько стоит очередное решение в серии".

Для выбора пользовательского маршрута нужны оба взгляда. Cold-проверка защищает от AOT pipeline, который великолепен в callback, но слишком дорог до первого ответа. Warm/prebuilt-проверка показывает отдачу в parameter sweep, continuation и batch workflow. Не сравнивайте cold Lambdify с warm AOT и не делайте вывод по таблице, где одна строка могла переиспользовать artifact, а другая его строила.

`RequirePrebuilt` при этом является еще и договором о корректности lifecycle,
а не просто performance-флагом. В текущих story-тестах Damped Sparse/Banded и
Frozen Sparse/Banded строгий reuse сохраняет `AotCompiled`, не показывает
стадию compiler/linker и совпадает с Lambdify на уровне округления. Если
prebuilt-строка вновь начнет компиляцию или молча перейдет на Lambdify, это
регрессия. В частности, lifecycle Frozen combustion-1000 теперь подтвержден в
release для обоих матричных маршрутов; на Sparse пути reuse-строки совпали с
Lambdify до `4.44e-16`.

## 9. End-to-end пример: Symbolic Lambdify + Banded

Ниже пример гармонического осциллятора как BVP: `y' = z`, `z' = -y`, `y(0)=0`, `z(0)=1`. Решение `y=sin(x)`, `z=cos(x)`, что удобно для sanity check.

```rust
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let n_steps = 80;
let values = vec!["y".to_string(), "z".to_string()];
let eq_system = vec![
    Expr::parse_expression("z"),
    Expr::parse_expression("-y"),
];

let t0 = 0.0;
let t_end = std::f64::consts::FRAC_PI_2;
let h = (t_end - t0) / n_steps as f64;
let mut guess = Vec::with_capacity(values.len() * n_steps);
for i in 0..n_steps {
    let x = t0 + i as f64 * h;
    guess.push(x.sin());
    guess.push(x.cos());
}
let initial_guess = DMatrix::from_column_slice(
    values.len(),
    n_steps,
    DVector::from_vec(guess).as_slice(),
);

let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(0usize, 1.0)]),
]);

let bounds = HashMap::from([
    ("y".to_string(), (-2.0, 2.0)),
    ("z".to_string(), (-2.0, 2.0)),
]);

let rel_tol = HashMap::from([
    ("y".to_string(), 1e-8),
    ("z".to_string(), 1e-8),
]);

let options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify()
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(rel_tol)
    .with_bounds(bounds)
    .with_max_iterations(40)
    .with_loglevel(Some("none".to_string()));

let mut solver = NRBVP::new_with_options(
    eq_system,
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
);

solver.dont_save_log(true);
solver.try_solve()?;
let result = solver.get_result().expect("BVP result should be available");
let stats = solver.get_statistics();

println!("solution shape = {} x {}", result.nrows(), result.ncols());
println!("timers = {:?}", stats.timers);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Этот пример хорош как стартовый: он использует символические уравнения, но не требует внешнего компилятора.

## 10. End-to-end пример: Numerical path

Pure Numerical path полезен, когда source of truth - ваше Rust-замыкание, а не `Expr`. Конструктор ниже вообще не принимает символические уравнения. Невязка строится численной дискретизацией RHS-замыкания, а выбор якобиана явный: либо конечно-разностный путь через `new_numeric_fd_with_options`, либо пользовательский непрерывный якобиан через `new_numeric_with_jacobian_options`.

```rust
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP,
};

let n_steps = 80;
let values = vec!["y".to_string(), "z".to_string()];
let t0 = 0.0;
let t_end = std::f64::consts::FRAC_PI_2;
let h = (t_end - t0) / n_steps as f64;

let mut guess = Vec::with_capacity(values.len() * n_steps);
for i in 0..n_steps {
    let x = t0 + i as f64 * h;
    guess.push(x.sin());
    guess.push(x.cos());
}
let initial_guess = DMatrix::from_column_slice(
    values.len(),
    n_steps,
    DVector::from_vec(guess).as_slice(),
);

let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(0usize, 1.0)]),
]);

let bounds = HashMap::from([
    ("y".to_string(), (-2.0, 2.0)),
    ("z".to_string(), (-2.0, 2.0)),
]);

let rel_tol = HashMap::from([
    ("y".to_string(), 1e-8),
    ("z".to_string(), 1e-8),
]);

let options = DampedSolverOptions::sparse_damped()
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(rel_tol)
    .with_bounds(bounds)
    .with_loglevel(Some("none".to_string()));

let mut solver = NRBVP::new_numeric_fd_with_options(
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
    |_x, y, _params| {
    DVector::from_vec(vec![y[1], -y[0]])
    },
);

solver.dont_save_log(true);
solver.try_solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Если у вас есть аналитический локальный якобиан, используйте второй численный конструктор. Замыкание возвращает непрерывный `2 x 2` якобиан правой части в одной точке сетки; глобальную BVP-матрицу солвер соберет сам.

```rust
let mut solver = NRBVP::new_numeric_with_jacobian_options(
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
    |_x, y, _params| DVector::from_vec(vec![y[1], -y[0]]),
    |_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]),
);
```

Практический смысл этого пути - исключить символический слой. Если такой тест падает, вы локализуете проблему в численной дискретизации, Newton loop или линейной алгебре, а не в symbolic/AOT.

## 11. End-to-end пример: AtomView + AOT + Banded

Для большой структурированной BVP-задачи, которую вы будете решать много раз, типичный production-кандидат - Banded + AtomView + AOT. Первый запуск может быть дорогим из-за сборки артефакта, но повторные solve получают compiled callbacks и быстрый banded linear solve.

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;

let generated = GeneratedBackendConfig::banded_atomview_for_repeated_solves();

let options = DampedSolverOptions::banded_damped()
    .with_generated_backend_config(generated)
    .with_abs_tolerance(1e-8)
    .with_max_iterations(60)
    .with_loglevel(Some("none".to_string()));
```

Если вы хотите явно выбрать компилятор:

```rust
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc();
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig();
```

Для CI или production, где артефакт должен быть подготовлен заранее:

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, GeneratedBackendConfig,
};

let generated = GeneratedBackendConfig::banded_atomview_for_repeated_solves()
    .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
```

`RequirePrebuilt` намеренно падает typed error, если артефакт отсутствует. Это лучше, чем неожиданно начать компиляцию внутри production run.

## 12. Практические рекомендации по выбору

Если вы пишете новую постановку BVP, начните с Lambdify. Он проще, прозрачнее и быстрее дает ответ на вопрос "правильны ли уравнения и граничные условия?".

Если у вас есть готовая Rust-модель в виде RHS-замыкания и вы не хотите использовать символику, выбирайте Numerical path. Используйте `new_numeric_fd_with_options`, когда поддерживать аналитический якобиан невыгодно, и `new_numeric_with_jacobian_options`, когда непрерывный `df/dy` доступен и хочется избежать шума конечных разностей. Для очень жестких или плохо масштабированных символических моделей Lambdify/AOT может быть предпочтительнее, потому что якобиан механически выводится из `Expr`-источника истины.

Если задача большая и символическая, проверяйте AOT с `tcc` одновременно с Lambdify, а не только после того, как возникнет repeated-solve сценарий. В исправленном `AtomView` пути TCC AOT уже может не уступать или превосходить Lambdify в честном cold end-to-end расчете. Для BVP/PDE-like задач с локальным stencil почти всегда стоит проверить Banded; для неструктурированных sparse-задач выбирайте Sparse. Для маленького solve по-прежнему начинайте с Lambdify, потому что build/bootstrap легко съедает весь runtime-выигрыш.

Если одну и ту же модель предстоит решать многократно, разделяйте создание и
использование артефакта явно: создавайте либо проверяйте его через
`BuildIfMissing`, а в измеряемом или production-цикле используйте
`RequirePrebuilt`. В измеренном Damped Banded случае combustion-1000 такой
warm-путь оказался примерно в `1.09x` быстрее Lambdify на одно решение, однако
потребовал нескольких повторений для компенсации первой сборки.

Если вы выбираете между Sparse и Banded, сначала спросите себя, откуда берется якобиан. Если переменная на узле связана только с соседними узлами, Banded почти наверняка кандидат номер один. Если связность дальняя или нерегулярная, Sparse безопаснее.

Если вы выбираете toolchain для AOT, первым кандидатом сейчас разумно делать `tcc`: он дал лучший practical cold profile в наших больших BVP-прогонах. `gcc`, Zig и Rust backend остаются важными контрольными и платформенно-зависимыми альтернативами, но их не следует молча считать default для performance.

Если вы выбираете chunking, не включайте parallel "на веру". Используйте story/perf tests, смотрите `callback residual values`, `callback jacobian values`, `linear_ms`, `solver_total_ms`, `actual_jobs` и `fallback`. Наши текущие измерения показывают одновременно две вещи: реальный параллелизм доказан, и на Banded `n_steps = 3000` он заметно ускоряет callback values; однако полный cold wall clock может остаться почти тем же. Поэтому для библиотечного default подходит `Auto`, а не forced `chunk4`.

## 13. Как читать statistics и story-таблицы

`solver.get_statistics()` возвращает счетчики и таймеры и для Damped, и для Frozen солвера. Frozen остается другой нелинейной стратегией, однако теперь показывает те же факты о generated backend, которые нужны для честного сравнения Lambdify/AOT: выбранный backend, символьный frontend, стадии handoff-подготовки, callback-диагностику, итерации Ньютона, пересборки якобиана и решения линейной системы. В BVP story-тестах особенно полезны:

`linear_ms` - время решения линейных систем Ньютона.  
`jac_ms` - время подготовки/вычисления якобиана на уровне solver statistics.  
`fun_ms` - время вычисления невязки.  
`Callback Residual Values` и `Callback Jacobian Values` - более низкоуровневые таймеры compiled/lambdified callbacks.  
`Callback Jacobian Matrix Assembly` - сборка матрицы из values, часто важнее самой арифметики values.  
`iters`, `linsys`, `jac_re` - число итераций Ньютона, линейных solve и пересборок якобиана.

Для generated AOT route тот же объект statistics содержит карту `diagnostics`. Самые полезные ключи: `generated.selected_backend`, `generated.handoff.initial_generate_wall_ms`, `generated.handoff.build_policy_wall_ms`, `generated.handoff.post_build_rebind_wall_ms`, `aot.runtime.execution_policy`, `aot.runtime.parallel_requested`, `aot.runtime.residual.actual_jobs`, `aot.runtime.residual.fallback_reason`, `aot.runtime.sparse_jacobian.actual_jobs`, `aot.runtime.sparse_jacobian.fallback_reason`, а также соответствующие `work_per_job` и `work_per_chunk`. Это не просто повтор пользовательской настройки, а фактическая runtime-диагностика: если `Auto` или threshold policy ушли в sequential fallback, statistics покажет это напрямую.

В старых таблицах этот подготовительный этап может называться `Symbolic Operations`; новый код дополнительно экспортирует тот же таймер как `Backend Preparation`. Alias оставлен намеренно. В Lambdify/AOT это действительно символическая/codegen подготовка, а в Numerical path - численная дискретизация и backend handoff, а не символическая работа.

По результатам текущих story-тестов для combustion BVP складывается следующая рабочая картина. Для `n_steps = 200` cold AOT еще легко теряет время на подготовке. Для `n_steps = 1000` после перехода на `AtomView` cold TCC AOT уже обогнал Lambdify в зафиксированном сравнении и на Sparse, и на Banded пути; в отдельном warm Damped Banded прогоне со строгим `RequirePrebuilt` и паузами TCC также оказался быстрее Lambdify примерно на `7.9%`. Для Banded `n_steps = 3000` production-frontend `AtomView` удержал решение на уровне roundoff и показал TCC AOT примерно на одном или лучшем уровне полного cold wall clock, хотя Lambdify-строка в этом опыте была шумной. Banded остается естественным выбором для узкой полосы, а chunking следует включать через `Auto` либо после отдельного измерения: он ускоряет hot callback work, но не гарантирует пропорционального ускорения всего solve.

Это не маркетинговые обещания, а engineering conclusions. Запускайте story-тесты на своей машине, особенно если меняете toolchain, число потоков, размер сетки или структуру уравнений.

## 14. Запускаемые guide-examples

В каталоге `examples/` теперь есть небольшие исполняемые приложения к этой главе. Каждое решает краевую задачу с известным аналитическим решением, поэтому показывает не только синтаксис настройки, но и немедленную проверку корректности. Для Damped представлены чисто численный путь (pure numerical path), символьная ламбдификация (Lambdify) и AOT; для Frozen представлены поддерживаемые символьные пути Lambdify и AOT. Отдельный пример про numerical route для Frozen поясняет, почему задачу, заданную замыканиями, следует отправлять в Damped, а не маскировать пустой символьной заглушкой.

```powershell
cargo run --example bvp_damped_numerical_guide
cargo run --example bvp_damped_lambdify_guide
cargo run --example bvp_damped_aot_guide
cargo run --example bvp_frozen_lambdify_guide
cargo run --example bvp_frozen_aot_guide
cargo run --example bvp_frozen_numerical_route_guide
```

В AOT-примерах намеренно показаны две фазы жизни артефакта: первое решение с `BuildIfMissing`, а затем решение с `RequirePrebuilt`. Для их запуска нужен `tcc` в `PATH`; если компилятор не установлен, пример сообщает о пропуске AOT-демонстрации, а не выдает отсутствие toolchain за ошибку солвера.

## 15. Где смотреть примеры и регрессии

Базовые correctness-тесты находятся в `BVP_Damp_tests.rs`. Более тяжелые story/performance-прогоны - в `BVP_Damp_tests3.rs` и `BVP_Damp_tests4.rs`. Сводка гипотез, команд и результатов ведется в [`BVP_DAMP_STORY_TESTS.md`](BVP_DAMP_STORY_TESTS.md).

Низкоуровневый codegen/performance слой дополнительно документируется в [`../../symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md`](../../symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md). Это полезно, когда нужно понять, проблема в самом BVP solver loop или в generated callback backend.

В рабочем коде лучше использовать `try_solve()` вместо `solve()`: первый возвращает typed errors, второй оставлен как compatibility wrapper и паникует при ошибках. Для production BVP-задач typed error почти всегда лучше, чем красивый panic с грустной музыкой.

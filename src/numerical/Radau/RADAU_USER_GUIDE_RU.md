# Руководство по Radau

Это руководство описывает поддерживаемую реализацию Radau в
`src/numerical/Radau/new`. Архивная реализация сохранена только как reference
и не является стандартным production-маршрутом.

## 1. Выбор API

Есть два публичных уровня API.

`UniversalODESolver::radau(...)` — короткий конструктор для общего фасада. Он
использует поддерживаемый Radau фиксированного порядка с symbolic Lambdify,
ExprLegacy и плотным хранением. Это удобный вариант, когда нужно единое API
для нескольких солверов без Radau-специфичных настроек.

Для выбора frontend, layout, parameter rebinding, output, telemetry, execution policy
или AOT следует использовать прямой API:

Если используется общий фасад, production-маршрут Radau оставляет telemetry
выключенной (`Off`) по умолчанию. Включить её можно через
`UniversalODESolver::with_radau_telemetry_mode(...)`, выбрав один из режимов
`RadauTelemetryMode::Counters`, `Timings` или `Detailed`. Режим `Off` явно
отключает сбор данных.

```rust
use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauProblem, RadauSolver,
    RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let problem = RadauProblem::new(
    vec![Expr::parse_expression("-k*y")],
    vec!["y".to_owned()],
    "t",
)
.with_parameter_names(vec!["k".to_owned()])
.with_jacobian(vec![Expr::parse_expression("-k")]);

let config = RadauConfig {
    frontend: RadauFrontend::ExprLegacy,
    matrix_layout: RadauMatrixLayout::Dense,
    telemetry: RadauTelemetryMode::Timings,
    t_bound: 1.0,
    ..RadauConfig::default()
};

let mut solver = RadauSolver::prepare(problem, config)?;
let solution = solver.try_solve_with_parameter_values(&[1.0], &[2.0])?;
println!("t={} y={:?}", solution.t, solution.y);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Подготовка отделена от решения и возвращает `Result`. Подготовленную модель
можно переиспользовать для серии значений параметров.

## 2. Выбор символьного frontend

`RadauFrontend::ExprLegacy` оставляет подготовку в legacy-представлении
выражений. Это хороший baseline для небольших задач и независимая reference
ветка для проверки correctness.

`RadauFrontend::AtomViewNative` один раз переводит модель в AtomView и далее
использует native-путь подготовки residual/Jacobian и callback-ов. Для больших
символьных систем это первый маршрут, который стоит проверить, но
универсального победителя нет.

Frontend и runtime выбираются независимо:

- `Lambdify` строит in-process closure без внешнего компилятора;
- `Aot` генерирует, компилирует, линкует и переиспользует native artifact.

Обе ветки используют одно численное ядро Radau. Сравнивать их нужно на одной
задаче, с одинаковыми `y0`, tolerance, layout и output policy.

## 3. Выбор матричного layout

Layout выбирается при подготовке:

```rust
let dense = RadauMatrixLayout::Dense;
let sparse = RadauMatrixLayout::Sparse;
let banded = RadauMatrixLayout::BandedAuto;
```

`Dense` выбирайте для полностью связанных Jacobian и небольших или средних
систем. `BandedAuto` выводит полосу из символьных зависимостей автоматически;
явный `Banded { lower, upper }` остается необязательной подсказкой. `Sparse` подходит для
больших задач со стабильным sparse-pattern, если экономия на хранении и
линейной алгебре окупает служебные расходы.

Серебряной пули нет. Diffusion-chain часто выигрывает от Banded или Sparse,
тогда как небольшая combustion или полностью связанная kinetics-задача может
быть быстрее с Dense. Нельзя выбирать structured layout только потому, что
сам residual дешёвый.

После подготовки backend не должен скрытно конвертировать Dense, Sparse и
Banded на каждой scalar-операции: выбранный layout владеет своим callback и
linear workspace.

## 4. Контракт Jacobian

Для symbolic Radau по возможности задавайте явный Jacobian. Вектор имеет
row-major порядок. Для `n` уравнений и `n` переменных идут сначала все
производные первой строки, затем второй и так далее:

```rust
let problem = RadauProblem::new(
    vec![Expr::parse_expression("-k*y")],
    vec!["y".to_owned()],
    "t",
)
.with_parameter_names(vec!["k".to_owned()])
.with_jacobian(vec![Expr::parse_expression("-k")]);
```

`RadauJacobianSource::Analytic` — default и рекомендуемый вариант для stiff
production-задач. `FiniteDifference` удобен для диагностики и fallback, но
добавляет вызовы residual. Для AOT и structured layout сначала проверяйте
явный Jacobian: так контракт генерации и проверка pattern остаются
однозначными.

Если выбран ExprLegacy Lambdify, а аналитический Jacobian не передан, Radau
получит его из residual во время подготовки. Построитель дифференцирует только
state-переменные, от которых зависит каждое уравнение, и хранит результат в
координатном row-major виде. Поэтому Sparse и Banded не создают промежуточную
плотную символьную матрицу. Вызов `.with_jacobian(...)` по-прежнему использует
переданные пользователем выражения. Для AOT явный Jacobian остаётся обязательным.

### Native callback API

Если модель уже реализована как Rust-замыкания, используйте
`RadauNativeSolver`. Обязателен callback невязки. Если передан callback
Якобиана, используется аналитический путь. Если передать `None`, Radau
использует типизированный конечно-разностный fallback и повторно вызывает
невязку для построения столбцов Якобиана.

Этот native callback API сейчас поддерживает только плотный (`Dense`)
Якобиан. Для `Sparse` и `Banded` следует использовать symbolic API
`RadauProblem`: выбранный structured backend сохраняется без скрытой
конвертации плотного callback во время решения.

При включённой telemetry число внутренних проб конечно-разностного метода
доступно в счётчике `finite_difference_probes`. Это внутренние вызовы для FD,
а не дополнительные пользовательские вызовы solve-level residual callback.

Пример `examples/radau_native_callbacks.rs` показывает оба режима.

## 5. Tolerance и шаги

Главные численные параметры задаются так:

```rust
let config = RadauConfig {
    rtol: 1e-6,
    atol: 1e-9,
    first_step: None,
    max_step: f64::INFINITY,
    max_steps: 1_000_000,
    max_newton_iterations: 6,
    max_retries: 12,
    ..RadauConfig::default()
};
```

Начинайте с default и затем ужесточайте `rtol` и `atol` вместе с проверкой,
масштабированной под задачу. `max_step` задавайте при известном физическом
масштабе или быстро меняющемся forcing. `first_step` нужен только если
автоматическая оценка начального шага не подходит. Увеличение Newton/retry
budget не должно маскировать плохой Jacobian или плохое масштабирование.

## 6. Повторные решения с переназначением параметров

Для модели без параметров:

```rust
let solution = solver.solve(&[1.0])?;
```

Если модель объявлена через `with_parameter_names`, значения связываются явно:

```rust
for k in [0.5, 1.0, 2.0, 4.0] {
    let solution = solver.try_solve_with_parameter_values(&[1.0], &[k])?;
    println!("k={k}: y(1)={:.9e}", solution.y[0]);
}
```

Каждый вызов переиспользует подготовленную symbolic model: изменение значений
параметров не повторяет symbolic differentiation и frontend preparation. Это
независимые решения, потому что каждому явно передаётся `y0`; parameter
continuation вместо этого использовала бы предыдущее решение как начальное
приближение. В benchmark подготовку нужно показывать отдельно от повторной серии.

Для нового интервала с сохранением подготовленных callback-ов:

```rust
solver.restart(0.0, 2.0)?;
let solution = solver.try_solve_with_parameter_values(&[1.0], &[2.0])?;
```

`restart` меняет только интервал, а новый `y0` передаётся следующему solve;
само изменение границ не является parameter continuation.

## 7. Output и post-processing

Политика хранения результата задаётся явно:

```rust
use RustedSciThe::numerical::Radau::RadauOutputPolicy;

let final_only = RadauOutputPolicy::FinalOnly;
let dense_output = RadauOutputPolicy::Dense;
let selected_times = RadauOutputPolicy::Sampled(vec![0.25, 0.5, 0.75, 1.0]);
```

`FinalOnly` требует минимум памяти. `Dense` сохраняет continuous-output
segments и позволяет вызвать `solution.sample(t)` или
`solution.sample_many(&times)`. `Sampled` вычисляет только запрошенные точки.
Для большой production-задачи не включайте Dense без необходимости.

## 8. Telemetry

Telemetry по умолчанию выключена и не должна оплачивать работу обычного пути:

```rust
let config = RadauConfig {
    telemetry: RadauTelemetryMode::Counters,
    ..RadauConfig::default()
};
```

`Counters` собирает residual/Jacobian calls, accepted/rejected steps, Newton
iterations, factorizations и linear solves без чтения часов. `Timings` собирает
только длительности, а `Detailed` — и счётчики, и времена. Preparation и runtime
scopes разделены; parent и child timings диагностические и не должны складываться.

Важные группы telemetry: preparation, residual/Jacobian, Newton, linear
factorization/solve, output, parameter rebind и AOT build/link/publication.
В отчёте рядом с числами всегда фиксируйте frontend, execution, layout,
output policy и фазу parameter series.

## Логирование

Логирование солвера по умолчанию выключено независимо от телеметрии. Задайте
общий `SolverLogLevel` в `RadauConfig` или через `with_logging_level(...)`;
это не включает и не меняет телеметрию. При включении Radau записывает начало,
успешное завершение и отказ solve через общий scoped timestamped file logger.

## 9. AOT lifecycle

AOT выбирается явно и требует конфигурации:

```rust
use RustedSciThe::numerical::Radau::{RadauAotConfig, RadauExecution};

let config = RadauConfig {
    execution: RadauExecution::Aot,
    frontend: RadauFrontend::AtomViewNative,
    matrix_layout: RadauMatrixLayout::Dense,
    aot: Some(RadauAotConfig::build_if_missing_release(
        "target/radau-artifacts",
    )),
    ..RadauConfig::default()
};
```

Режимы:

- `build_if_missing_release(path)` — собрать только при отсутствии artifact;
- `require_prebuilt()` — не запускать compiler и завершиться ошибкой без artifact;
- `rebuild_always_release(path)` — принудительно пересобрать для cold-gate.

Для разработки используйте BuildIfMissing, для deployment и process-isolated
consumer — RequirePrebuilt, а RebuildAlways оставляйте для cold measurement.
Build/link время нельзя смешивать с warm solve временем.

## 10. Sequential, Parallel и Auto

Безопасный default — последовательное выполнение:

```rust
use RustedSciThe::numerical::Radau::RadauExecutionPolicy;

let sequential = RadauExecutionPolicy::Sequential;
let forced = RadauExecutionPolicy::Parallel { min_work: 256 };
let automatic = RadauExecutionPolicy::Auto { min_work: 256 };
```

`Parallel` включайте после измерения workload. `Auto` должен избегать dispatch,
если стоимость worker overhead не окупается. Callback-only и full-solve
break-even нужно измерять отдельно: выигрыш callback может исчезнуть в linear
algebra или controller overhead.

## 11. Ошибки и готовые examples

Preparation и solve возвращают `Result<_, RadauError>`. В прикладном коде
проверяйте тип ошибки, а не разбирайте display-строку. Отдельно различаются
ошибки configuration, shape, callback, unsupported route, Newton failure,
step-budget exhaustion и output range.

Готовые examples:

- `examples/radau_lambdify_single.rs` — одиночный Lambdify solve;
- `examples/radau_lambdify_continuation.rs` — повторные решения с новыми значениями параметров;
- `examples/radau_public_api_guide.rs` — AtomView, counters и dense output;
- `examples/radau_backends_guide.rs` — сравнение frontend/layout;
- `examples/radau_aot_single.rs` и `examples/radau_aot_parameter_sweep.rs` — AOT и повторная привязка параметров между fresh solve;
- `examples/universal_ode_example.rs` и `examples/rus/universal_ode_example.rs` —
  короткий universal facade.

Сначала подтверждайте correctness на одной и той же задаче. Затем отдельно
сравнивайте preparation, callback-only, full solve и continuation. Быстрый
результат на одном layout не является общим утверждением о Radau.

## 12. Текстовые task documents

Текстовый маршрут — компактный воспроизводимый адаптер поддерживаемого Radau
для examples и batch-запусков. Если нужны полный выбор frontend, layout, AOT
или настройка parallel executor, используйте нативные `RadauConfig` и
`RadauProblem`.

```text
task
solver: IVP
method: Radau5

equations
arg: t
parameters: k
parameter_values: 2.0
y: -k*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
first_step: 1e-3

postprocessing
output_policy: none
```

Парсер направляет `solver: IVP` вместе с `method: Radau5` в native Radau
adapter. `rtol` и `atol` независимы; `tolerance` — удобное общее значение,
если оба допуска должны совпадать. `max_step`, `first_step` и лимиты итераций/
шагов проверяются до создания solver-а. `Radau3` не является поддержанным
методом task parser: текущая реализация использует фиксированный Radau IIA
пятого порядка.

В task documents сейчас доступен компактный symbolic adapter с его
поддерживаемыми настройками по умолчанию. Выбор `ExprLegacy` или
`AtomViewNative`, `Dense`/`Sparse`/`Banded`, AOT build policy, telemetry и
кастомной политики `Parallel`/`Auto` выполняется через типизированный Radau
API. Так advanced option не может быть молча проигнорирована текстовым
форматом.

Continuation переиспользует распарсенную символьную модель, если это допускает
native contract:

```text
continuation
parameter: k
values: 1.0, 2.0, 4.0
mode: prepared
restart_each: false
```

`fresh` заново подготавливает каждый сегмент, `warm` переиспользует callbacks с
предыдущим принятым состоянием, а `prepared` запрашивает максимально сильный
маршрут reuse. Намеренный restart может задавать для сегмента собственные
`y0`, `t0` и `t_end`. В отчётах разделяйте preparation, callback, full-solve и
continuation timings. `output_policy` принимает `none`, `terminal`, `plotters`
или `gnuplot`; `plot` оставлен только как compatibility alias. Комментарии с
`#` игнорируются. Общая грамматика и batch-runner описаны в
[`TASK_DOCS_GUIDE_EN.md`](../../command_interpreter/TASK_DOCS_GUIDE_EN.md).

## 13. Сверка с SciPy и измерения производительности

Численное ядро проверяется по официальной реализации Radau из SciPy
`v1.18.1`. Минимальный набор исходников и сведения об их происхождении
находятся в `src/numerical/ivp_scipy_reference`; в parity-отчётах сохраняются
версии Python, NumPy и SciPy, а также хеши исходников. Сопоставимые проверки
охватывают stiff scalar, Robertson, combustion-like и HIRES, конечное решение
и общие точки dense output, трассы контроллера, обратное интегрирование,
автоматический и заданный `first_step`, некорректную конфигурацию, underflow и
исчерпание бюджета шагов. Дополнительно есть проверки по точным решениям и
инвариантам, а Robertson независимо сверяется с LSODE2/BDF.

Цель — faithful-поведение математического метода и контроллера, а не
побитовое совпадение каждого счётчика или детали реализации. Поэтапно
проверяются Radau IIA пятого порядка, преобразованная collocation-система,
modified Newton, оценка локальной ошибки, адаптивный контроллер и
полином dense output. Оставшиеся наблюдаемые различия объяснены:

- Rust лениво готовит callbacks при решении, тогда как SciPy вычисляет
  начальные RHS/Jacobian при создании solver-а. Это различие жизненного цикла
  API.
- Rust переиспользует буферы workspace и не нуждается в дополнительном
  вычислении SciPy для конечного RHS; поэтому число residual-вызовов может
  быть на один меньше.
- Реализации плотного LU различаются: Rust использует собственный in-place
  real/complex backend, SciPy — LAPACK. Вблизи порога сходимости Newton это
  может изменить решение о пересчёте Jacobian или повторе шага, как в базовом
  Robertson-сценарии, при этом траектория остаётся в заданном допуске.
  Проверка с соседним tolerance подтверждает, что ветвление зависит от
  численной границы.
- При step underflow Rust возвращает типизированную ошибку до бесперспективной
  Newton-попытки; SciPy может выполнить и отклонить одну такую попытку раньше.
  Оба варианта завершаются без принятых шагов. Бюджет принятых шагов Rust —
  расширение API; для сверки ему соответствует явно обозначенный внешний
  лимит driver-а SciPy.

Это осознанные различия жизненного цикла, backend-а и API, а не случайные
отклонения численного алгоритма. Подробная fidelity-матрица и компактные
результаты находятся в `src/numerical/ivp_scipy_reference/TODO.md` и
`src/numerical/Radau/RADAU_STORY_TESTS.md`.

### Снимок производительности численного ядра

В одном release-эксперименте сравнивались решение Rust Radau с заранее
подготовленными native callbacks и создание solver-а SciPy вместе с решением
и вычислением тех же трёх точек результата. Использовались одинаковые
уравнения, аналитические Jacobian, интервалы, tolerance и ограничения шага;
телеметрия была отключена. Полученные отношения времени SciPy/Rust:

| Задача | Решение Rust, медиана | Измеренный scope SciPy, медиана | Отношение |
|---|---:|---:|---:|
| Stiff scalar | 0.0065 ms | 0.4633 ms | 71.28x |
| Robertson | 0.0216 ms | 2.0500 ms | 94.91x |
| Combustion-like | 0.0073 ms | 0.4982 ms | 68.25x |

Это веское свидетельство высокой скорости численного пути Rust на этих
показательных малых stiff-задачах, но не универсальное обещание ускорения.
Измеренные scopes не полностью одинаковы: Rust не включает уже завершённую
подготовку callbacks, тогда как измерение Python включает создание solver-а.
Кроме того, проверены только три задачи на одной машине; другие уравнения,
размеры, layout и окружения могут дать иные результаты. Исходный отчёт:
`test_reports/Radau_SciPy_Parity/release/numerical__Radau__tests__scipy_parity__radau_scipy_release_core_timing_benchmark_story.md`.

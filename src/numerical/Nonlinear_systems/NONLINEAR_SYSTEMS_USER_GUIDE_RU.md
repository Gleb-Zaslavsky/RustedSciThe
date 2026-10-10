# Руководство по нелинейным системам

## Рекомендуемый типизированный путь

В новом коде лучше один раз подготовить символьную систему и отделить
численные значения параметров от этой подготовки:

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let prepared = PreparedSymbolicNonlinearProblem::from_strings(
    vec!["a*x + y - 3".into(), "x - y".into()],
    SymbolicProblemOptions::new()
        .with_variables(vec!["x".into(), "y".into()])
        .with_equation_parameters(vec!["a".into()]),
)?;

let bound = prepared.bind_values(DVector::from_vec(vec![2.0]))?;
let options = SolveOptions {
    tolerance: 1e-10,
    max_iterations: 64,
    diagnostics: DiagnosticsOptions {
        collect_history: false,
        collect_statistics: true,
        ..DiagnosticsOptions::default()
    },
    ..SolveOptions::default()
};
let result = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
    .solve(&bound, DVector::from_vec(vec![1.0, 1.0]), options)?;
```

`PreparedSymbolicNonlinearProblem` владеет разобранными уравнениями, порядком
переменных, подготовкой символьного якобиана и вызываемым backend. Метод
`bind_values` проверяет численный вектор по объявленной схеме параметров и
возвращает заимствованный bound-view. Перебор других значений не повторяет
символьную подготовку: один prepared-объект можно использовать для
continuation-подобного перебора и разных начальных приближений.

`SymbolicProblemOptions::with_lambdify_backend()` является явным выбором
внутреннего символьного пути по умолчанию. Для AOT действует общий жизненный
цикл: артефакт нужно материализовать, собрать, зарегистрировать и связать с
процессом до запуска AOT-only задачи. `BuildIfMissing` и `RequirePrebuilt` это
политики жизненного цикла артефакта, а не два разных численных метода; детали
проверенного контракта приведены в `STORY_TESTS.md`.

### Обновление параметров и структурная пересборка

Порядок имен, переданный в `with_equation_parameters`, является ABI параметров.
Вызов `bind_values` (или совместимый `set_parameter_values`) меняет только
численные значения уже существующей схемы. Он не разбирает уравнения заново,
не дифференцирует и не ламбдифицирует их, не пересобирает AOT-артефакт и не
меняет порядок переменных. Изменение уравнений, переменных, имен или порядка
параметров, а также generated chunking является структурным изменением и
требует подготовки нового объекта.

### Диагностика отдельных попыток

При включенной статистике `SolveStatistics::attempts` содержит по одной
записи на каждую внешнюю нелинейную итерацию после начального вычисления
состояния. Счетчики имеют тот же solver-level смысл, что и агрегированные
поля: одно вычисление невязки или якобиана означает один полный запрос к
провайдеру, даже если AOT-провайдер внутри выполняет несколько generated jobs.
Запись содержит также refresh/reuse якобиана, факторизации, линейные решения,
принятые и отвергнутые trial-шаги и суммарные времена стадий:

```rust
for attempt in &result.statistics.attempts {
    println!(
        "iteration={} R/J={}/{} refresh/reuse={}/{} factor/linear={}/{}",
        attempt.iteration,
        attempt.residual_evaluations,
        attempt.jacobian_evaluations,
        attempt.jacobian_refreshes,
        attempt.jacobian_reuses,
        attempt.linear_factorizations,
        attempt.linear_solves,
    );
}
```

Начальное вычисление невязки и якобиана входит в агрегированную статистику,
но не в запись отдельной попытки. `termination_retries` явно равен `None` в
generic engine, поскольку у него нет отдельной границы повторов завершения,
специфичной для метода. Внутренние generated job/chunk принадлежат
`SymbolicPreparationReport`, а не solver-счетчикам. При отключенной статистике
`attempts` пуст, а поля счетчиков и времени не являются измерениями.

### Подробная телеметрия подготовки

Статистика солвера описывает численные итерации. Если нужно отдельно измерять
стоимость построения подготовленной символьной задачи, включите опциональную
телеметрию подготовки:

```rust
let prepared = PreparedSymbolicNonlinearProblem::from_strings(
    equations,
    SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_preparation_telemetry(PreparationTelemetryMode::Collect),
)?;

if let Some(report) = prepared.preparation_report().detailed.as_ref() {
    println!(
        "preparation total={:?}, input={:?}, mode={:?}",
        report.total_wall_time, report.input_kind, report.execution_mode,
    );
    for stage in &report.stages {
        println!("{:?}: {:?}", stage.stage, stage.wall_time);
    }
}
```

Агрегированный `SymbolicPreparationReport` доступен и при отключенной
подробной телеметрии. Время стадии является исключительным, если реализация
может выделить его отдельно; `None` означает, что стадия неприменима или ее
нельзя надежно изолировать, а не нулевое время. Поле
`unattributed_wall_time` содержит остаток времени подготовки. Обновление
значений параметров не повторяет подготовку и не создает новый отчет о
подготовке.

Для диагностического контура, которому нужно сохранить границу ошибки,
используйте подробный конструктор. Он возвращает `SymbolicPreparationFailure`,
содержащий исходный `SolveError` и завершенные к моменту ошибки стадии:

```rust
let attempt = SymbolicNonlinearProblem::from_strings_with_options_detailed(
    equations,
    SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_preparation_telemetry(PreparationTelemetryMode::Collect),
);
if let Err(failure) = attempt {
    eprintln!("{}; telemetry={:?}", failure, failure.telemetry);
}
```

Совместимые конструкторы не меняются и по-прежнему возвращают непосредственно
`SolveError`. Failure telemetry предназначена для отчетов и отладки и не
превращает неудачную подготовку в пригодную для решения задачу.

### Опциональное логирование солвера

Логирование солвера по умолчанию отключено. Настройка действует только в
рамках одного запуска `SolverEngine` и не меняет глобальный logger процесса:

```rust
let quiet = SolveOptions::default();
let verbose = SolveOptions::default().with_logging(EngineLogLevel::Debug);
let quiet_again = verbose.without_logging();
```

Та же политика доступна у `DiagnosticsOptions` через методы
`with_logging(...)` и `without_logging()`. Диагностика runtime-методов
использует этот публичный параметр и не пишет напрямую в stdout. Поэтому
обычный путь не выполняет форматирование и отправку log-записей, а приложение
с установленным фасадом `log` может явно включить уровни `Info`, `Warn` или
`Debug`.

### Жизненный цикл AOT-артефакта

Для холодного production-подобного запуска задайте каталог вывода и
используйте `BuildIfMissing`:

```rust
let cold = SymbolicNonlinearProblem::from_strings_with_generated_backend(
    equations.clone(),
    problem_options.clone(),
    SymbolicGeneratedBackendConfig::build_if_missing_release(&output_dir),
)?;
let resolver = cold.updated_resolver.clone();
println!(
    "cold: backend={:?}, action={:?}, manifest_key={:?}, lifecycle_key={:?}, build={:?}",
    cold.selected_backend,
    cold.preparation_report.artifact_action,
    cold.preparation_report.artifact_key,
    cold.preparation_report.artifact_lifecycle_key,
    cold.preparation_report.build_duration,
);
```

После публикации артефакта строгий теплый запуск использует `RequirePrebuilt`
и ничего не компилирует:

```rust
let warm = SymbolicNonlinearProblem::from_strings_with_generated_backend(
    equations,
    problem_options,
    SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(resolver),
)?;
assert!(warm.build_result.is_none());
assert_eq!(warm.preparation_report.artifact_action, SymbolicArtifactAction::Reused);
```

`RequirePrebuilt` возвращает типизированную ошибку для отсутствующего,
несовместимого или не связанного артефакта и не делает молчаливый fallback на
Lambdify. `resolver` является снимком внутри процесса; новый процесс должен
найти опубликованный совместимый артефакт в настроенном каталоге. После любой
из этих стадий обновление параметров остается дешевой численной операцией.

## Выбор политики выполнения и backend

### Последовательный и параллельный Lambdify

Политику выполнения callback можно задать при подготовке символьной задачи:

```rust
let sequential = SymbolicProblemOptions::new()
    .with_variables(variables.clone())
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential);

let parallel = SymbolicProblemOptions::new()
    .with_variables(variables)
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel {
        min_work: 1_500,
    });
```

`Sequential` является режимом по умолчанию и первым выбором для небольших
или разреженных систем. `Parallel { min_work }` включается явно и запускает
распараллеливание только когда подготовленный объем работы достигает порога.
`min_work` является правилом runtime-диспетчеризации, а не параметром
сходимости или точности. Начните с `Sequential`, затем измерьте настоящую
структуру невязки и якобиана на целевой машине и только после этого выбирайте
`min_work`.

Это подтверждается release-корпусом. При размерности `512` подготовленный
Sequential был самым быстрым полным Newton-маршрутом во всех шести строках
большого корпуса. Parallel был полезен для широкого `band-five` callback и
оказался быстрее legacy-маршрута, но в полном решении все еще уступал
подготовленному Sequential. В sweep по порогу `Parallel { min_work: 1 }`
был примерно на `9-10%` медленнее Sequential, а включение Parallel ровно на
границе structural `nnz` замедляло путь еще сильнее. Доказательства приведены
в Sections 44 и 45 `STORY_TESTS.md`.

### Почему новый якобиан не использует Mutex

Подготовленный якобиан уже знает независимую структуру вычисления ненулевых
элементов. Новый parallel-путь распределяет независимые строки/элементы по
worker-потокам и пишет непосредственно в принадлежащую вызывающей стороне
матрицу `DMatrix`. Поэтому ему не нужны общий аккумулятор, поэлементный
`Mutex` или промежуточная dense-матрица. Legacy compatibility-путь сохраняет
старую семантику allocated-return и mutex/Rayon dispatch, чтобы не ломать
существующих пользователей.

Это одновременно архитектурное и производительное решение. На release
корпусе `band-five`, размерность `512`, legacy Jacobian занимал `1230.7 us`,
подготовленный Sequential `397.15 us`, а новый mutex-free Parallel `426.31 us`.
То есть Parallel был примерно в `2.89x` быстрее legacy, хотя Sequential был
еще быстрее. Parallel не обязан быть быстрее Sequential: для дешевых
элементов стоимость запуска worker-потоков перекрывает выигрыш. Корректность
и детерминированность разреженной раскладки проверяются в Sections 41 и 42.

### Lambdify и AOT: когда наступает break-even

Выбирайте Lambdify, если важны короткий запуск, переносимость и небольшое
число решаемых задач. Выбирайте AOT, если доступен toolchain, артефакт можно
сохранить и переиспользовать, а выигрыш callback подтвержден именно для вашей
задачи. Первый AOT-запуск включает materialization, compilation, linking и
publication; эту стоимость нельзя смешивать с warm solve.

Грубая оценка числа warm-решений до окупаемости:

```text
break_even_solves ~=
    (AOT preparation + build - Lambdify preparation)
    / (Lambdify warm solve - AOT warm solve)
```

Знаменатель должен быть положительным. Если AOT в warm-режиме медленнее,
для данной конфигурации break-even отсутствует. В release large-story при
`n=512` warm total составил `17.242 ms` у Lambdify и `17.835 ms` у AOT;
якобиан AOT занимал `2.913 ms` против `2.296 ms`, а невязка и linear stage
были близки. При `n=128` AOT был быстрее (`0.397 ms` против `0.516 ms`), но
стоимость build около `365 ms` означает тысячи повторных решений до
амортизации. Поэтому текущие данные не дают универсального утверждения
«AOT быстрее». Они подтверждают AOT как deployment/lifecycle-вариант с
зависящим от workload break-even. См. Sections 53, 64 и 65.

### Архитектурные решения и их обоснование

- **Разделение prepared и bound:** parsing, differentiation и построение
  callback выполняются один раз; новая привязка параметров создает дешевое
  представление для solver-а. Поэтому parameter sweep должен переиспользовать
  prepared-объект.
- **Разделение solver-level и generated-job счетчиков:** один счетчик
  residual/Jacobian означает один полный запрос к provider. Внутренние AOT
  chunks/jobs являются backend detail, поэтому Lambdify и AOT сопоставимы.
- **Разделение manifest и lifecycle identity:** математический artifact key
  используется resolver-ом, а profile/compiler settings входят в on-disk
  lifecycle key. Debug-marker не может молча удовлетворить другому build.
- **Fail-closed RequirePrebuilt:** отсутствующий, устаревший, несовместимый или
  не связанный артефакт дает typed error; молчаливого fallback на Lambdify нет.
  Это делает deployment воспроизводимым.
- **Whole для compact dense AOT по умолчанию:** layout-profile при `n=512`
  показал, что `Whole` быстрее row chunks 32 и 64. Caller-side adaptation
  занимала `0.216 ms` из `0.301 ms` полного вызова якобиана. Это вывод для
  измеренного dense-пути, а не правило для sparse или banded задач.

### Стоимость телеметрии

Solver statistics и preparation telemetry отвечают на разные вопросы.
`collect_statistics` дает solver-level counters и stage timers, а
`collect_history` дополнительно сохраняет снимки итераций и может увеличить
расход памяти. Подробная телеметрия подготовки отдельно включается через
`PreparationTelemetryMode::Collect`. Release-аудит подготовки показал лишь
сотые доли миллисекунды разницы между disabled и enabled режимами при
размерностях `40`, `128` и `512`; диапазоны повторов перекрывались. Это не
является универсальным процентом стоимости warm solve. Для чистого hot-path
benchmark отключайте history/statistics, если сами диагностические данные не
являются предметом эксперимента.

## Ограничения и диагностика

Для решения в прямоугольной области используйте `SolveOptions::bounds`:

```rust
let bounds = Bounds::new(vec![(0.0, 3.0), (0.0, 2.0)])?;
let options = SolveOptions {
    bounds: Some(bounds),
    ..SolveOptions::default()
};
```

При `collect_statistics: true` поле `SolveResult::statistics` содержит
счетчики вызовов невязки, якобиана и линейного этапа, принятых и отвергнутых
шагов, а также суммарные времена стадий. `collect_history` включается
отдельно и может увеличивать память и количество аллокаций. Для измерения
горячего пути без истории диагностику следует отключать, если история не
является предметом эксперимента.

Для полностью численных задач реализуйте `NonlinearProblem` и
`JacobianProvider`, используя принадлежащие вызывающей стороне
`DVector`/`DMatrix`. Дополнительные методы `residual_into` и `jacobian_into`
позволяют заполнять буферы солвера и являются правильным путем для задач, где
аллокации важны. Математика метода при этом не меняется.

## Выбор метода

Обычный Newton обычно быстрее всего вблизи хорошо масштабированного решения,
но чувствительнее к плохому начальному приближению и вырожденному якобиану.
Damped Newton удобен, когда нужны backtracking или ограничения. Варианты
Levenberg-Marquardt и trust-region полезны для трудных, плохо масштабированных
или least-squares-подобных систем. Одно число wall-clock не ранжирует все
методы: сравнивайте сходимость, качество невязки, отвергнутые шаги и времена
стадий на том классе задач, который важен именно вам.

### Levenberg-Marquardt с backtracking

`BacktrackingLevenbergMarquardt` является самостоятельным вариантом наравне
с классическим LM, MINPACK, Nielsen и trust-region методами. Он решает
регуляризованную систему с единичным демпфированием:

`(J^T J + lambda I) delta = -J^T residual`

Затем проверяются допустимые пробные точки: сначала `alpha = 1`, после чего
`alpha` уменьшается вдвое, пока норма невязки строго не уменьшится или не
будет достигнуто `alpha_min`. После принятого шага `lambda` умножается на
`lambda_decrease`, а после полностью неудачного backtracking-поиска — на
`lambda_increase`. Значения по умолчанию равны `0.3` и `10.0`. Общий движок
по-прежнему объявляет `Converged` только при достижении заданной точности
невязки: принятый шаг сам по себе не является доказательством сходимости.

```rust
let method = NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
    BacktrackingLevenbergMarquardtMethod::default(),
);
let result = method.solve(&problem, initial_guess, SolveOptions::default())?;
```

Выбирайте этот вариант, когда нужна именно политика единичного демпфирования
и строгого backtracking с учетом границ. Для классической настраиваемой
политики масштабирования используйте `LevenbergMarquardt`; варианты намеренно
разделены и не подменяют друг друга неявно.

## Аудит аллокаций

Запуск release-аудита на уровне процесса:

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

Аудит охватывает размерности 32, 128 и 512, принятые решения без истории,
сравнение с историей на минимальной и максимальной размерностях, reusable
callback-буферы и отдельный rejection-heavy контроль. Он печатает число
аллокаций/деаллокаций и объем байт за полный lifetime результата решения.
Counting allocator меняет поведение процесса, поэтому его времена нужны только
для диагностики и не должны напрямую сравниваться с обычным Criterion или
прикладным wall-clock. Аудит не определяет каждый отдельный copy и не заменяет
измерение peak resident memory; это материал для гипотезы оптимизации, после
которой обязательны correctness-тесты.

Современный executable-пример:

```text
cargo run --example rus_nonlinear_systems_modern_guide
```

Примеры совместимости и AOT lifecycle:

```text
cargo run --example nonlinear_lambdify_legacy_guide
cargo run --example nonlinear_lambdify_prepared_guide
cargo run --example rus_nonlinear_aot_lifecycle_guide
```

Старый `nonlinear_systems_guide` оставлен как compatibility-пример legacy
LM-wrapper.

### Выбор символьного frontend

Плотный Lambdify backend поддерживает два frontend-пути:

```rust
let expr_legacy = SymbolicProblemOptions::new()
    .with_lambdify_frontend(SymbolicLambdifyFrontend::ExprLegacy);
let atom_native = SymbolicProblemOptions::new()
    .with_atom_native_frontend();
```

`ExprLegacy` остается совместимым выбором по умолчанию. `AtomViewNative` один
раз преобразует публичные `Expr` в Atom, выполняет дифференцирование
Якобиана на Atom-представлении и записывает невязку и Якобиан в плотные буферы
вызывающей стороны. Плотный AOT поддерживает оба frontend-а: `ExprLegacy` и
`AtomViewNative`; во втором случае генерация невязки и Якобиана остается на
Atom-представлении и использует тот же dense ABI. Для AOT сейчас требуется
цельная (`Whole`) стратегия chunking.

Этот решатель намеренно работает с плотными матрицами. Для Sparse и Banded
следует использовать ODE-солверы, например LSODE2, где layout-specific
линейные backend-ы являются частью контракта.

## Прямоугольные задачи least-squares

Канонический least-squares-солвер минимизирует вектор невязок `r(x)`. Он
подходит для переопределённых и недоопределённых моделей, шумных наблюдений и
задач, где невозможно одновременно обратить все невязки в ноль. Это другой
контракт, чем поиск корня: root solver ищет `r(x) = 0`, а LM минимизирует
`0.5 * ||r(x)||^2` и возвращает итоговую целевую функцию. Для несогласованной
аппроксимации ненулевая итоговая невязка ожидаема.

Общий селектор `NonlinearSolver` содержит отдельные варианты `Root` и
`LeastSquares`. Least-squares **намеренно не является** вариантом
`NonlinearSolverMethod`: этот enum используется движком поиска корня для
квадратной системы и возвращает `SolveResult`; прямоугольный least-squares
принимает собственный контракт задачи и возвращает `MinimizationReport`.

### Численные callbacks невязки и Якобиана

Реализуйте `LeastSquaresProblem` или используйте
`ClosureLeastSquaresProblem`, задав текущий вектор параметров, невязку и её
Якобиан. Ниже аппроксимируется прямая по трём наблюдениям. Данные
несогласованы, поэтому в оптимуме останется небольшая ненулевая невязка:

```rust
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let problem = ClosureLeastSquaresProblem::new(
    DVector::from_vec(vec![0.0, 0.0]), // [свободный член, наклон]
    |p| DVector::from_vec(vec![p[0] - 1.0, p[0] + p[1] - 2.0, p[0] + 2.0 * p[1] - 2.9]),
    |_| DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]),
);

let method = NonlinearSolver::LeastSquares(
    LeastSquaresLevenbergMarquardt::new().with_tol(1e-10),
);
let (problem, report) = method.try_minimize_least_squares(problem)?;
assert!(report.termination.was_successful());
println!("fit={:?}, objective={:e}", problem.params(), report.objective_function);
```

`try_minimize_least_squares` сохраняет типизированные ошибки численного
метода и callbacks. Сходимость и другие допустимые исходы записаны в
`report.termination`; проверяйте `was_successful()`, а не считайте любой
возвращённый отчёт успешной подгонкой. Runtime-телеметрия по умолчанию
отключена; её можно включить через `LeastSquaresTelemetryMode::Counters` или
`Detailed` в настройках LM.

### Символьный least-squares builder

Для символьных невязок `SymbolicLeastSquaresSolver` подготавливает невязку и
Якобиан, после чего использует тот же канонический LM core. Builder принимает
начальное приближение, имена неизвестных, параметризованные уравнения,
настройку телеметрии и явное объявление положительных переменных:

```rust
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let mut solver = SymbolicLeastSquaresSolver::new()
    .with_equations_str(vec![
        "a - 1".into(),
        "a + b - 2".into(),
        "a + 2*b - 2.9".into(),
    ])
    .with_unknowns(vec!["a".into(), "b".into()])
    .with_initial_guess(vec![0.0, 0.0])
    .with_tolerance(1e-10)
    .with_telemetry(LeastSquaresTelemetryMode::Detailed);

let report = solver.try_solve()?;
if report.termination.was_successful() {
    let coefficients = solver.map_of_solutions.as_ref().expect("успешная аппроксимация");
    println!("{coefficients:?}; objective={:e}", report.objective_function);
}
```

Для типизированной обработки ошибок используйте `try_solve` и
`try_minimize_least_squares`. `set_positive_variables` — явная политика
пользователя для любой задачи, в которой выбранные переменные должны
оставаться строго положительными; солвер сам не выводит ограничения из
предметной области. Это может быть полезно для логарифмов и других выражений
с ограниченной областью определения, но не ограничивается химическими
задачами.

### Повторное использование параметризованной символьной модели

Объявляйте параметры уравнений отдельно от неизвестных. Каждый вызов
`try_solve_with_params` привязывает новые численные значения и повторно
использует подготовленные символьные невязку и Якобиан, не выполняя символьную
подготовку заново. Wrapper использует заданное в нём начальное приближение для
каждого вызова, поэтому это отдельные fresh-решения, а не warm-start
continuation от предыдущего результата:

```rust
let mut solver = SymbolicLeastSquaresSolver::new()
    .with_equations_str(vec!["a - target".into(), "2*a - 2*target".into()])
    .with_unknowns(vec!["a".into()])
    .with_parameters(vec!["target".into()])
    .with_initial_guess(vec![0.0]);

for target in [1.0, 2.0, 3.0] {
    let report = solver.try_solve_with_params(vec![target])?;
    assert!(report.termination.was_successful());
    println!("target={target}, fit={:?}", solver.map_of_solutions);
}
```

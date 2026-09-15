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
    "cold: backend={:?}, action={:?}, key={:?}, build={:?}",
    cold.selected_backend,
    cold.preparation_report.artifact_action,
    cold.preparation_report.artifact_key,
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

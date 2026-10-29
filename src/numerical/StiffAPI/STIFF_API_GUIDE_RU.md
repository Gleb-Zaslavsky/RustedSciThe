# StiffAPI: единый фасад для stiff IVP

StiffAPI позволяет приложению выбрать production-солвер и его настройки один
раз, а затем подготовить и запустить его без собственного `match` между BDF,
Radau, Backward Euler и LSODE2. Численный алгоритм, символьная подготовка,
AOT, логирование, телеметрия и владение результатом остаются у native-солвера.

## Выбор маршрута и запуск

`StiffSolverRequest` хранит типизированную конфигурацию выбранного солвера.
Общее приложение может хранить этот request в своей конфигурации, подготовить
его один раз, а выполнение делегировать `PreparedStiffSolver::try_solve()`:

```rust,ignore
use RustedSciThe::numerical::StiffAPI::{
    StiffSolveOutcome, StiffSolverRequest, StiffPreset,
};
use RustedSciThe::numerical::Radau::api::{RadauConfig, RadauProblem};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let request = StiffSolverRequest::Radau {
    problem: RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]),
    config: RadauConfig::default(),
    initial_state: vec![1.0],
    parameters: vec![0.7],
};

let mut solver = request.prepare(Some(&StiffPreset::expr_dense()))?;
let outcome = solver.try_solve()?;
if let StiffSolveOutcome::Radau(solution) = outcome {
    println!("final state: {:?}", solution.y);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`try_solve()` выполняет dispatch внутри фасада. Приложению не нужно отдельно
вызывать `as_bdf_mut().try_solve()`, `as_radau_mut().solve(...)` и так далее.
`StiffSolverRequest` намеренно использует enum с native option types, а не
непрозрачный набор строковых ключей или общий плоский список `Option`.
Например, `BdfSolverOptions`, `RadauProblem`/`RadauConfig`, `BeSolverOptions` и
`Lsode2ProblemConfig` сохраняют свои специфичные настройки.

## Hybrid-адаптер настроек солвера

Приложение вроде KiThe может самостоятельно разбирать физические данные и
строить численную модель, а RST при этом читает из того же документа только
выбор солвера и его настройки. Для этого используется
`command_interpreter::task_parser_ivp_hybrid::parse_hybrid_stiff_settings_from_document`.
Он возвращает `HybridStiffSettings`; затем host-приложение строит native-задачу
и передаёт типизированный `StiffSolverRequest` в
`HybridStiffSettings::prepare`. Адаптер проверяет соответствие семейства и
делегирует проверку capabilities и подготовку в StiffAPI.

Адаптер не интерпретирует и не присваивает себе domain-секции, уравнения или
начальные условия. Дополнительный сериализатор solver-фрагмента возвращает
только выбор солвера и его опции; сохранение и объединение физического
документа остаётся ответственностью host-приложения. Полнотекстовый IVP parser
по-прежнему существует как отдельный end-to-end путь RST для документов, где
уравнения и начальные условия также принадлежат RST. Ошибки разбора сохраняют
тип `IvpTaskError`, а подготовка запроса возвращает typed `StiffApiError` и
исходные ошибки native-солвера.

## Новые расчёты после rebind

После подготовки маршрута `try_solve_with_parameter_values(y0, values)` запускает
новое решение с начальными условиями и параметрами, сохраняя настроенные `t0`
и `t_bound`. Метод `try_solve_with_parameter_values_in_interval(t0, y0,
t_bound, values)` также принимает новый временной интервал. Оба метода повторно
используют подготовленный symbolic/AOT маршрут, но сбрасывают численное состояние
интегратора. Это не continuation с предыдущим решением как начальным
приближением и не продолжение интегрирования по времени.

```rust,ignore
let outcome = prepared.try_solve_with_parameter_values(
    nalgebra::DVector::from_vec(vec![1.0]),
    nalgebra::DVector::from_vec(vec![1.2]),
)?;
let outcome = prepared.try_solve_with_parameter_values_in_interval(
    0.0,
    nalgebra::DVector::from_vec(vec![0.8]),
    2.0,
    nalgebra::DVector::from_vec(vec![0.9]),
)?;
```

Начальные данные Radau находятся в request (`initial_state` и `parameters`),
потому что native Radau принимает их при каждом `solve`; другие три маршрута
хранят начальное состояние в native-конфигурации. Число параметров Radau должно
совпадать с декларацией `RadauProblem::with_parameter_names`.

## Результаты и ошибки

`StiffSolveOutcome` не маскирует различия владения результатом:

- Radau возвращает native `RadauSolution`, включая dense-output sampling и
  telemetry report.
- LSODE2 возвращает native `Lsode2SolveSummary`; полная траектория остаётся
  доступна у подготовленного solver.
- BDF и Backward Euler сохраняют результат у solver и возвращают маркер
  успешного выполнения в outcome.

После ошибки `PreparedStiffSolver` остаётся у вызывающей стороны. Ошибка
`StiffSolveError` содержит исходный тип ошибки выбранного solver-а и реализует
`Error::source()`. Можно проверить `solver_kind()`, затем inspect native status,
частичный результат и применить native restart/recovery через соответствующий
типизированный accessor. Частичный `RadauSolution` не обещается: native Radau
возвращает его только при успешном solve.

Общий API намеренно не выдумывает одинаковый restart. BDF, BE и LSODE2 имеют
разные правила продолжения/перезапуска, а Radau принимает новые начальные
значения при повторном solve. Используйте native методы через `as_bdf_mut()`,
`as_backward_euler_mut()`, `as_lsode2_mut()` или `as_radau_mut()`; это escape
hatch для lifecycle, не место для повторного solver dispatch.

## Пресеты и observability

`StiffPreset` задаёт только явно перечисленные общие оси. Например,
`expr_dense()` выбирает ExprLegacy и Dense, не меняя tolerance, шаги или
controller. Неподдерживаемая комбинация возвращает `StiffApiError`, а не
молча преобразуется в другой backend. Для полного списка применимости и
ограничений см. [матрицу контрактов](CONTRACT_AUDIT.md).

Телеметрия остаётся native и solver-specific. `telemetry_mode()` сообщает
выбранный режим, `telemetry_snapshot()` доступен там, где solver хранит snapshot;
у Radau он извлекается из `RadauSolution`. StiffAPI не объединяет счётчики и не
складывает вложенные времена. Logging настраивается пресетом, если выбранный
native solver поддерживает его.

## Postprocessing и lifecycle

Для BDF/BE/LSODE2 `trajectory_view()` возвращает borrowed view с формой
`(time_points, state_variables)`. Его можно явно преобразовать в общий
`PostprocessDataset` и выполнить `PostprocessPlan`; преобразование копирует
массивы. Для Radau передайте `RadauSolution` в `radau_postprocess_dataset()` и
задайте времена выборки явно. Dense output должен быть запрошен в Radau config.

Export, plotting, restart, continuation и пере-bind параметров не запускаются
неявно и не сведены к искусственно одинаковому API. Используйте native
accessor для функций, контракт которых различается между solver-ами. Подробные
route/default/error различия перечислены в [contract audit](CONTRACT_AUDIT.md).
# Логирование Radau

Для Radau, как и для остальных stiff-маршрутов, используется общий
`SolverLogLevel`. По умолчанию уровень `Off`; логирование не включает и не
меняет телеметрию. Уровень можно задать через preset StiffAPI либо напрямую в
`RadauConfig::logging_level` / `RadauConfig::with_logging_level(...)`.
Сообщения проходят через общий scoped file logger библиотеки. Общий тип уровня
доступен как `numerical::StiffAPI::SolverLogLevel`.

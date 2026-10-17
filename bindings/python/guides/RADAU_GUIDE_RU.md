# Radau5: руководство для Python

Это Python-версия руководства поддерживаемого Rust-солдера Radau. Используется
фиксированный метод Radau IIA пятого порядка. Численный алгоритм выполняется в
Rust; Python задаёт выражения и настройки и получает отделённые от workspace
результаты.

## Когда выбирать Radau

Radau5 — неявный stiff IVP-метод. Он подходит для сильно жёсткой кинетики и
задач, где важнее надёжность на жёстком режиме, чем минимальная стоимость
шага. Для гладкой нежёсткой задачи он не обязан быть быстрее LSODE2 Adams/Auto;
при выборе сравнивайте солверы на одной и той же модели.

## Однократный и подготовленный запуск

`solve_ivp("radau5", ...)` удобен для одного решения. `create_ivp` или
`RadauSolver` нужен для явной подготовки, переназначения параметров, restart и
специфичных настроек. Прямой wrapper принимает `y0` при `solve`:

```python
from rustedscithe import Expr, RadauSolver, SolverOptions

y = Expr.variable("y")
k = Expr.variable("k")
solver = RadauSolver(
    [-k * y], ["y"], "t", 0.0, 1.0, 0.05,
    parameters=["k"], options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
    rtol=1e-7, atol=1e-9, output="dense",
)
solver.prepare()
first = solver.solve([1.0], [0.7])
continued = solver.continue_with_parameters([first.final_state[0]], [1.2])
```

Continuation повторно использует подготовленные символьные коллбэки, но
продолжает от предыдущей конечной точки. Для независимого нового начального
состояния/интервала используйте `restart(...)` и новый `solve(y0, ...)`.

## Frontend, execution и матричный layout

`expr_legacy`/`atom_native` выбирают символьную подготовку;
`lambdify`/`aot` — исполнение; `dense`/`sparse`/`banded` — представление
линейной системы. Проверяйте доступные комбинации через
`capabilities("radau5")`. Серебряной пули для frontend или layout нет.

Для структурированных layout нужен символьный якобиан. Если он не задан,
fallback с конечными разностями доступен только для Dense. Callback-wrapper —
отдельный Dense/sequential маршрут; аналитический якобиан необязателен, иначе
используются native конечные разности. Неподдержанные комбинации дают typed
`UnsupportedOptionError`, а не скрытую конвертацию.

## Настройки, output и диагностика

`rtol`/`atol` управляют локальной ошибкой, `first_step` задаёт начальный шаг,
`max_step` ограничивает шаг сверху, а `max_steps`, `max_newton_iterations` и
`max_retries` ограничивают вычислительный бюджет. `output="dense"` сохраняет
непрерывный интерполянт. NumPy-массивы результата — собственные `float64`
данные, не представления изменяемого Rust-workspace.

В `SolverOptions` можно выбрать `telemetry="counters"`, `"timings"` или
`"detailed"`: `counters` собирает только дискретные счётчики, `timings` — только
времена, `detailed` — оба набора. Затем читайте `telemetry_counters()`,
`telemetry_timings_ms()` и `telemetry_report()`. Ноль и
неизмеренная стадия — разные вещи. Ошибки конфигурации, подготовки, коллбэка,
AOT и исчерпания бюджета типизированы; у native-ошибок есть stage/solver/route,
если эти данные доступны.

## Примеры и evidence

`examples/radau_basic.py` показывает минимальную задачу,
`examples/radau_robertson_routes.py` — matched-сравнение frontend/layout на
одной Robertson-системе, а `examples/radau_robertson_aot.py` требует доступного
компилятора. Данные о fidelity с SciPy находятся в руководстве и story-отчётах
основного crate; Python-примеры не являются бенчмарками.

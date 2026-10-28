# LSODE2: руководство для Python

LSODE2 — универсальный IVP-солвер для нежёстких и жёстких задач, а также для
задач, где важна структура Sparse/Banded матриц. Автоматический контроллер
может переключаться между семействами Adams и BDF. Это отдельный, более
настраиваемый солвер, а не вариант небольших BDF/Radau wrappers.

## Семейство солвера и контроллер

В качестве семейства указывайте `"lsode2"`, а метод задавайте отдельно:
`controller="adams"`, `"bdf"` или `"auto"`. `"lsoda"` — alias семейства для
автоматического контроллера Adams/BDF. Не передавайте `"adams"` как имя
семейства солвера.

```python
from rustedscithe import Expr, SolverOptions, create_ivp

y = Expr.variable("y")
rate = Expr.variable("rate")
solver = create_ivp(
    "lsode2", [-rate * y], ["y"], "t", [1.0], 0.0, 2.0, 0.05,
    parameters=["rate"], parameter_values=[0.7], controller="auto",
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", execution_policy="sequential",
        telemetry="timings",
    ),
    rtol=1e-8, atol=1e-10,
)
solution = solver.solve()
print(solution.controller_mode, solution.final_state_array())
```

## Layout и символьные маршруты

LSODE2 предоставляет подготовку ExprLegacy/AtomNative, Lambdify/AOT и
линейные маршруты Dense/Sparse/Banded. Эти измерения независимы. Dense —
естественный baseline для малых полностью связанных систем, Sparse или Banded
могут лучше подойти при соответствующей структуре якобиана. Универсального
победителя нет: сравнивайте одинаковые задачи и смотрите provenance и
выбранный линейный решатель.

Для символьных выражений `matrix_layout="banded"` автоматически определяет
ширину полосы якобиана, если обе подсказки ширины опущены. Передавайте
`band_lower` и `band_upper` вместе, только если нужно явно задать или ограничить
структуру; `(0, 0)` означает диагональный якобиан, а не автоматический режим.
Для численных Python callbacks нет символьной структуры для анализа, поэтому
в Banded-маршруте обе ширины обязательны.

| Настройка | Python API | Примечание |
| --- | --- | --- |
| Контроллер | `create_ivp(..., controller=...)` | `"bdf"`, `"adams"` или `"auto"`; для `"lsoda"` по умолчанию выбирается `"auto"` |
| Допуски | `rtol`, `atol` | Локальный контроль ошибки IVP |
| Параметры | `parameters`, `parameter_values` | Имена и порядок совпадают с выражениями |
| Ширины полосы | `band_lower`, `band_upper` | Для символьного Banded можно опустить обе; для численных callbacks обязательны обе |
| Callbacks | `residual_callback`, `jacobian_callback` | Без Jacobian callback применяется FD |
| Исполнение | `SolverOptions(execution=...)` | `lambdify` или `aot` |
| Планирование | `SolverOptions(execution_policy=...)` | `sequential`, `parallel`, `auto` там, где поддержано |

`Lsode2Solver` принимает контроллер через аргумент конструктора `method`;
общая фабрика `create_ivp` называет эту настройку `controller`, потому что её
аргумент `method` выбирает семейство солвера. Сам `Lsode2Solver` принимает
`band_lower`/`band_upper` как аргументы конструктора или через
`SolverOptions.set_banded_width(lower, upper)`. Если заданы обе формы ширин,
значения должны совпадать, иначе возвращается типизированная ошибка
конфигурации. Ширины отклоняются для layout, отличного от Banded; передача
только одной ширины также считается ошибкой.

Wrapper также предоставляет native analytical residual/Jacobian callbacks
для поддержанных layout. Они пересекают Python-границу и удерживают GIL;
настройки worker-ов Rust не превращают Python callback в Rust-native код и не
обещают масштабирование Python-потоков. Для горячего численного пути по
возможности предпочитайте символьную подготовку.

## Подготовка, новые решения и временные сегменты

`create_ivp` даёт явный lifecycle. `prepare()` один раз строит выбранный
символьный маршрут. `solve_with_parameter_values(y0, values)` запускает новое
решение на настроенном интервале; вариант
`solve_with_parameter_values_in_interval(t0, y0, t_end, values)` задаёт также
новый интервал. `continue_with_parameters(...)` отдельно продолжает временной
сегмент от последней успешной точки, а `restart(...)` задаёт новую точку и
интервал.

## Телеметрия и AOT

В `SolverOptions` доступны режимы `telemetry="off"`, `"counters"`,
`"timings"` и `"detailed"`. `counters` собирает только дискретные счётчики,
`timings` — только длительности, а `detailed` — оба набора. Читайте snapshot
через `solution.telemetry_counters()`, `solution.telemetry_timings_ms()` и
`solution.telemetry_report()`. Временные scopes могут перекрываться и не всегда
аддитивны. AOT поддерживает общий lifecycle BuildIfMissing, RequirePrebuilt и
RebuildAlways, handoff между процессами и provenance. Холодная компиляция
окупается лишь при повторной работе; измеряйте подготовку, callback и full
solve отдельно.

## Большие примеры

`examples/lsode2_controller_matrix.py` сравнивает контроллеры на неизменной
задаче. `examples/lsode2_lorenz.py` показывает нелинейную хаотическую систему
из трёх переменных; `examples/lsode2_three_body.py` переносит 12-компонентную
задачу гравитации из основного crate, включая символьные aliases расстояний.
Это численные примеры, а не контролируемые бенчмарки.

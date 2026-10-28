# Stiff API

`StiffSolverRequest` — Python-фасад над Rust StiffAPI. Он централизует выбор
общих маршрутов и native-диспетчеризацию, но не превращает четыре семейства
солверов в урезанный общий интерфейс. Запрос создаётся с настройками
конкретного солвера, затем подготавливается и выполняется в Rust. Результат
сохраняет обычный Python-тип выбранного солвера.

## Общие настройки и пресеты

`SolverOptions` задаёт общие оси: символьный frontend, Lambdify/AOT,
представление матрицы, политику callbacks, телеметрию и логгирование.
Переиспользуемый профиль можно задать через `StiffPreset`: он изменяет только
явно перечисленные общие настройки. Сочетаемость пресетов проверяет Rust core;
неподдерживаемая комбинация вызывает typed exception, а не молча
переписывается.

Каталог встроенных профилей доступен через `StiffPreset.catalog()`, а точные
изменения и применимость составленного профиля — через `preset.describe()`.
Список `fields_changed` показывает, какие оси профиль действительно меняет;
неуказанные настройки остаются native defaults выбранного солвера. Составление
двух профилей, задающих одну ось, отклоняется, а не использует скрытое правило
«последний победил».

```python
from rustedscithe import Expr, SolverOptions, StiffPreset, StiffSolverRequest

request = StiffSolverRequest.bdf(
    [Expr.parse("-rate*y")],
    ["y"],
    "t",
    [1.0],
    0.0,
    1.0,
    max_step=0.05,
    parameters=["rate"],
    parameter_values=[0.7],
    options=SolverOptions(telemetry="detailed"),
)
prepared = request.prepare(StiffPreset.solve_1d())
solution = prepared.solve()
print(solution.states()[-1])
print(solution.telemetry_report())
```

`prepare()` потребляет запрос ровно один раз. Возвращённый объект хранит
подготовленный native solver; `solve()` выполняет native-диспетчеризацию и
возвращает detached-результат выбранного solver-класса. В приложении с
конфигурацией выбирайте фабрику нужного солвера при создании расчёта, а затем
скрывайте prepared object за границей приложения. Python не реализует заново
численные шаги или планирование callbacks.

`telemetry_snapshot()` возвращает структуру с явной меткой solver: поля
`solver` и `mode` находятся сверху, а native-счётчики, тайминги, отчёт и
данные AOT вложены под ключом выбранного солвера. Имена метрик не объединяются
между семействами. Для Radau телеметрия берётся из `RadauSolution`, который
возвращает `solve()`.

`restart(t0, y0, t_end)` запускает новый сегмент на том же подготовленном
солвере и возвращает обычный solver-specific result. Символьная подготовка,
AOT-артефакты и общая политика остаются привязаны к native-экземпляру.

После решения `PreparedStiffSolver.status` возвращает native-статус выбранного
солвера, если такой статус предусмотрен его API. `trajectory_arrays()` явно
копирует сохранённую траекторию BDF, Backward Euler или LSODE2 в собственные
NumPy-массивы формы `(число_точек,)` и `(число_точек, число_переменных)`.
Результаты Radau остаются в `RadauSolution` и запрашиваются через его методы
выборки. Если траектория ещё не создана, метод выдаёт типизированную ошибку.

## Настройки семейств

- `StiffSolverRequest.bdf(...)` сохраняет настройки порядка BDF, первого и
  максимального шага, а также лимита работы. Результат — `BdfSolution`.
- `StiffSolverRequest.radau(...)` принимает начальное состояние как часть
  запроса, поддерживает optional analytic Jacobian, `output="dense"` для
  произвольной интерполяции и `output="sampled", sample_times=[...]` для
  сохранения только заданных узлов. При sampled output обращаться можно только
  к этим временам. Результат — `RadauSolution`; логгирование Radau пока не
  поддерживает, поэтому запрос ненулевого logging level будет отклонён.
- `StiffSolverRequest.backward_euler(...)` сохраняет фиксированный шаг `h` и
  настройки Newton. Если `h` не задан, остаётся native эвристика солвера.
  Результат — `BackwardEulerSolution`.
- `StiffSolverRequest.lsode2(...)` не смешивает выбор метода
  (`"bdf"`, `"adams"`, `"auto"`) с выбором исполнения
  (`"native"`, `"bridge"`, `"probe_before_bridge"`). Результат —
  `Lsode2Solution`. Фабрика принимает либо символьную систему, либо пару
  `residual_callback(time, state)` и `jacobian_callback(time, state)`. Python
  callbacks работают только с Lambdify: AOT не может встроить Python-функции,
  а автоопределение ширины полосы требует символьных уравнений. Для banded
  callback-маршрута задайте оба параметра `band_lower` и `band_upper` в
  `SolverOptions`. Исключение из callback возвращается как `CallbackError`, а
  исходное Python-исключение сохраняется в `__cause__`.

Символьные фабрики проверяют имена state/time/parameter до native подготовки.
Если solver требует начальные значения параметров, число конечных значений
должно соответствовать объявленным именам. AOT toolchain, политика артефакта и
каталог сборки задаются явно в `SolverOptions`; подготовка AOT может запустить
внешний компилятор.

## Телеметрия и ошибки

`prepared.telemetry_mode` показывает эффективный native-режим сбора.
Detached-результат предоставляет счётчики, времена, AOT provenance и
диагностику через API своего солвера. Эти данные намеренно не сведены к
искусственной общей схеме: счётчики и иерархия временных scopes различаются
между реализациями. Typed native errors переводятся в существующую иерархию
Python exceptions с контекстом solver/route.

`StiffPostprocessPlan` — тонкий Python builder над Rust-планом экспорта:
CSV/текст, краткий отчёт, Plotters/Gnuplot или график в терминале.
`prepared.postprocess(plan, axis_name, variable_names)` применяет его к
сохранённой траектории BDF, Backward Euler или LSODE2. Для Radau используйте
`RadauSolution.postprocess(plan, sample_times, axis_name, variable_names)`:
времена выборки задаёт вызывающий код, поскольку Radau хранит результат
отдельно. Возвращаемая структура содержит статус каждого действия, путь и
необязательное сообщение; отсутствие Gnuplot отражается как `skipped`.

`solve_with_parameter_values(y0, values)` и
`solve_with_parameter_values_in_interval(t0, y0, t_end, values)` запускают
новую задачу, сохраняя подготовленные символьные/AOT ресурсы. Первый метод
оставляет настроенный интервал, второй принимает новый. `restart()` меняет
интервал и начальное состояние без переназначения параметров.

Продолжение интегрирования во времени остаётся solver-specific:

- `continue_bdf_with_parameters(values, t_end)` продолжает BDF от последней
  принятой точки и сбрасывает численную историю multistep-метода.
- `continue_backward_euler(t_end, parameter_values=None)` продлевает текущую
  траекторию и при необходимости обновляет параметры.
- `continue_lsode2_with_parameters(values, t_end)` начинает LSODE2-сегмент с
  последней успешной точки, обновляет параметры и использует native restart.

Эти методы возвращают обычный тип результата своего солвера. Вызов метода
продолжения на несовместимом prepared solver выдаёт typed unsupported-route
ошибку. Для Radau используется общий fresh-solve API.

Примеры `examples/stiff_api_solver_selection.py` и
`examples/stiff_api_restart.py` показывают выбор семейства решателя, общий
план вывода и повторное использование подготовленного BDF.
`examples/stiff_api_solver_lifecycle.py` показывает продолжение во времени для
BDF/BE/LSODE2 и новый parameterized solve для Radau, а
`examples/stiff_api_reaction_chain.py` — выбор solver для двухкомпонентной
реакционной цепочки. Примеры не являются бенчмарками.

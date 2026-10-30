# BVP_Damp: руководство для Python

BVP_Damp решает системы ОДУ первого порядка как краевую задачу с демпфированной
нелинейной итерацией и политиками уточнения сетки. Его постановка отличается
от BVP_sci: границы задаются mapping-ом имён состояний в наблюдаемые значения
на концах интервала, при необходимости с bounds. Python wrapper предоставляет
отдельные демпфированный, fixed-mesh Naive и fixed-mesh Frozen маршруты. Это
разные native-алгоритмы, а не псевдонимы одного маршрута.

Для обоих fixed-mesh маршрутов начальное приближение задаётся node-major по
сокращённым неизвестным Ньютона: одна строка на интервал. Возвращаемый профиль
восстанавливает фиксированные значения на границах и содержит строку на каждый узел сетки.

## Модель и граничные наблюдения

Уравнения задают первые производные состояний. Python-адаптер сейчас требует
одно суммарное граничное наблюдение на неизвестную; это отличается от полного
вектора левых/правых residual-условий BVP_sci. Например, для `y'=0` и
`y(0)=1` точное решение — постоянный профиль `y=1`. Начальное состояние —
node-major приближение; bounds ограничивают итерации и не заменяют границы.

```python
import numpy as np
from rustedscithe import Expr, SolverOptions, solve_bvp

mesh = np.linspace(0.0, 1.0, 16)
options = SolverOptions(
    frontend="atom_native", execution="lambdify",
    matrix_layout="banded", telemetry="timings",
)
options.set_banded_width(1, 1)
solution = solve_bvp(
    "bvp_damp", [Expr.constant(0.0)], ["y"],
    {"y": [(0, 1.0)]}, mesh, np.full(mesh.size, 0.6),
    options=options,
    bounds={"y": (-0.1, 1.2)}, tolerance=1e-7,
    max_iterations=40, max_refinements=4,
    logging_mode="off", loglevel="error",
)
```

Boundary API принимает условия в терминах состояний, а не список residual
expressions как BVP_sci. `values_array()` сохраняет native-ориентацию
`(state, mesh)`; всегда проверяйте форму, не предполагая node-major формат
BVP_sci.

## Политики солвера

`BvpDampNaiveSolver` — отдельный fixed-mesh маршрут native реализации Naive;
он не перенаправляется в демпфированный или Frozen солвер. Доступны те же
символьные frontend и выбор Lambdify/AOT, layouts Dense/Sparse/Banded,
телеметрия и prepared-перепривязка численных параметров. Сетка должна быть
равномерной. Damping controls, bounds, adaptive refinement и явная ширина
Banded не относятся к этому native маршруту и не принимаются как будто влияют
на решение. Prepared API поддерживает перепривязку параметров, но пока не
замену начального приближения или интервала на месте.
Одно сочетание намеренно отклоняется: fixed-mesh Naive/Frozen с
`frontend="atom_native"`, `execution="aot"` и `matrix_layout="dense"`.
Сейчас этот native-маршрут доходит до несовместимого vector ABI и вызывает
panic, поэтому Python-адаптер заранее выдаёт `UnsupportedOptionError`.
ExprLegacy Dense AOT и AtomNative Sparse/Banded AOT проверены отдельными
route-тестами. Fixed-mesh AOT предоставляет численную телеметрию, но не общие
AOT lifecycle-счётчики; выбранный маршрут проверяйте через `provenance()`.

```python
from rustedscithe import BvpDampNaiveSolver, Expr, SolverOptions

solver = BvpDampNaiveSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 4,
    parameters=["rate"], parameter_values=[1.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="timings"),
)
solver.prepare()
first = solver.solve_prepared()
second = solver.solve_with_parameter_values([2.0])
```

`BvpDampFrozenSolver` использует замороженный якобиан с выбранной пользователем
native-политикой его повторного использования. Сетка фиксированная и
равномерная; после `prepare()` можно менять fixed inputs. Метод
`set_prepared_iterate(...)` отдельно задаёт новое приближение на той же сетке,
не пересобирая callbacks; он проверяет форму данных и сбрасывает численные
факторы. Интервал остаётся фиксированным. Поддерживаются политики
`Frozen_naive` (повторное использование после первого якобиана), `every_m`
(положительный целый период), `at_high_morm` (положительный порог нормы),
`at_low_speed` (порог скорости не больше 1) и `complex`
(`[period, norm_threshold, speed_rate]`). Для `Frozen_naive` параметр
`policy_parameters` не задаётся; для остальных он соответствует указанной
форме. Выбранная политика доступна через `provenance()`, а результат содержит
тот же detached telemetry contract, что и другие BVP_Damp-маршруты. Публичный
ключ `at_high_morm` сохраняет написание нативного API; `morm` здесь не отдельная
величина или стратегия.

```python
from rustedscithe import BvpDampFrozenSolver, Expr, SolverOptions

solver = BvpDampFrozenSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 4,
    parameters=["rate"], parameter_values=[1.0],
    policy="every_m", policy_parameters=[2.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="counters"),
)
solver.prepare()
for rate in (1.0, 2.0):
    solution = solver.solve_with_parameter_values([rate])
    print(rate, solution.values()[0][-1], solution.telemetry_counters())
```

Для заданной вызывающим стороной последовательности векторов фиксированных
параметров Frozen предоставляет `solve_parameter_loop([[...], [...]])`. Каждая
успешная точка становится начальным приближением следующей. Невязка, не
дошедшая до сходимости, или ошибка native solve сохраняются в отчёте точки и не
останавливают последующие расчёты; `point.error` остаётся типизированным Python
исключением с обычным контекстом solver/stage/route/native_kind. Для каждой
точки доступны нативные счётчики, времена стадий, scopes, callback- и
preparation-телеметрия согласно выбранному режиму. Пример success-failure-success находится в
`examples/bvp_damp_frozen_parameter_loop.py`. API пилота использует фиксированную
сетку, как и сам Frozen/Naive маршрут.

Для адаптивного `BvpDampSolver` действует тот же ordered-loop контракт, но
checkpoint включает пару «уточнённая сетка + успешный профиль». Следующая
точка начинается с этой пары; если ей требуется дальнейшее уточнение, нативный
solver пересобирает зависящие от сетки символьные и численные структуры. Для
каждой точки заново доступен её лимит refinement. Ошибка точки сохраняется в
отчёте и не заменяет checkpoint. См. `examples/bvp_damp_parameter_loop.py`:
там есть success-failure-success проход и чтение нативных счётчиков.

Для `Naive` не применяются параметры демпфирования, bounds и адаптивное
уточнение сетки; они доступны только соответствующему damped-пути.

ExprLegacy/AtomNative и Lambdify/AOT выбираются независимо. Dense, Sparse,
Banded, damping, обновление якобиана, grid refinement, linear-system policy,
iterative refinement и bounds передаются прямым wrapper-ом в поддержанных
маршрутах. Проверяйте `capabilities("bvp_damp")`: convenience API не должен
молча заменять специфичные настройки дефолтами. Универсального layout нет;
сравнивайте одинаковые уравнения и mesh size с учётом структуры глобального
якобиана.

Прямой конструктор сохраняет основные численные и структурные настройки:

| Область | Python keyword(s) | Примечание |
| --- | --- | --- |
| Дискретизация | `scheme="forward"` или `"trapezoid"` | Выбор явной схемы Эйлера или трапециевидного правила; у них разные свойства точности и устойчивости |
| Стратегия | `BvpDampSolver`, `BvpDampNaiveSolver` или `BvpDampFrozenSolver` | Каждый выбирает отдельный native-алгоритм; один не подменяется другим |
| Нелинейный шаг | `tolerance`, `max_iterations`, `max_jacobian_refreshes`, `max_damping_iterations`, `damping_factor` | Newton/damping policy |
| Сетка | `max_refinements`, `grid_method`, `grid_parameters` | Число параметров зависит от выбранного метода |
| Линейная система | `linear_sys_method`, `banded_linear_solver`, `banded_fallback`, `iterative_refinement_steps` | Настройки native backend |
| Bounds/logging | `bounds`, `logging_mode`, `loglevel`, `logging_max_events` | Logging выключен по умолчанию; bounds задаёт пользователь |
| Banded layout | `SolverOptions.set_banded_width(lower, upper)` | Для damped-маршрута ширина обязательна; fixed-mesh Naive/Frozen выводят структуру и отклоняют явную ширину |

## Lifecycle, логирование, телеметрия

`prepare()` отделяет символьную подготовку от повторных численных запусков.
`set_parameter_values()` только меняет fixed inputs, не меняя символьную схему;
`set_prepared_iterate()` только устанавливает warm-start профиль на прежней
сетке. `solve_with_parameter_values()` объединяет эти операции и использует
последний успешный профиль как начальное приближение. Это удобный маршрут
продолжения по параметру для Damped, Naive и Frozen.
`restart(...)` заменяет обычное начальное приближение и может инвалидировать
подготовку в зависимости от маршрута. Логирование выключено, пока его явно не
включили; режим
телеметрии задаётся отдельно через `SolverOptions` и входит в detached result.

Режимы telemetry не смешиваются: `off` отключает счетчики и таймеры;
`counters` собирает дискретные события, не запуская часы; `timings` измеряет
время без дискретных счетчиков; `detailed` включает оба вида измерений.
Выбранный режим доступен как `solution.telemetry_mode`; accessor-ы счетчиков
и таймингов возвращают пустые словари для отключенной части.

Изучайте counters и stage timings раздельно. Проверяйте status, итоговую сетку,
невязки/итерации и bounds. Обрабатывайте типизированные configuration,
preparation, linear-backend и convergence ошибки, не анализируя текст логов.

## Примеры

`examples/bvp_damp_observations.py` — короткий одиночный запуск;
`examples/bvp_damp_layout_matrix.py` сравнивает все три layout и оба frontend
на точном постоянном профиле. Native BVP_Damp guides остаются источником
полного описания схемы и настроек, пока не выставленных Python-адаптером.

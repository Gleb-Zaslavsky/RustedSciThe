# BVP_Damp: руководство для Python

BVP_Damp решает системы ОДУ первого порядка как краевую задачу с демпфированной
нелинейной итерацией и политиками уточнения сетки. Его постановка отличается
от BVP_sci: границы задаются mapping-ом имён состояний в наблюдаемые значения
на концах интервала, при необходимости с bounds. Python wrapper предоставляет
отдельные демпфированный, fixed-mesh Naive и fixed-mesh Frozen маршруты. Это
разные native-алгоритмы, а не псевдонимы одного маршрута.

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

```python
from rustedscithe import BvpDampNaiveSolver, Expr, SolverOptions

solver = BvpDampNaiveSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
    parameters=["rate"], parameter_values=[1.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="timings"),
)
solver.prepare()
first = solver.solve_prepared()
solver.set_parameters([2.0])
second = solver.solve_prepared()
```

`BvpDampFrozenSolver` использует замороженный якобиан с выбранной пользователем
native-политикой его повторного использования. Сетка фиксированная и
равномерная; после `prepare()` можно перепривязывать параметры, но нельзя на
месте заменить начальное приближение или интервал. Поддерживаются политики
`Frozen_naive` (повторное использование после первого якобиана), `every_m`
(положительный целый период), `at_high_morm` (положительный порог нормы),
`at_low_speed` (порог скорости не больше 1) и `complex`
(`[period, norm_threshold, speed_rate]`). Для `Frozen_naive` параметр
`policy_parameters` не задаётся; для остальных он соответствует указанной
форме. Выбранная политика доступна через `provenance()`, а результат содержит
тот же detached telemetry contract, что и другие BVP_Damp-маршруты.

```python
from rustedscithe import BvpDampFrozenSolver, Expr, SolverOptions

solver = BvpDampFrozenSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
    parameters=["rate"], parameter_values=[1.0],
    policy="every_m", policy_parameters=[2.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="counters"),
)
solver.prepare()
for rate in (1.0, 2.0):
    solver.set_parameters([rate])
    solution = solver.solve_prepared()
    print(rate, solution.values()[0][-1], solution.telemetry_counters())
```

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
| Дискретизация | `scheme="forward"` или `"trapezoid"` | Выбор схемы первого порядка |
| Стратегия | `BvpDampSolver`, `BvpDampNaiveSolver` или `BvpDampFrozenSolver` | Каждый выбирает отдельный native-алгоритм; один не подменяется другим |
| Нелинейный шаг | `tolerance`, `max_iterations`, `max_jacobian_refreshes`, `max_damping_iterations`, `damping_factor` | Newton/damping policy |
| Сетка | `max_refinements`, `grid_method`, `grid_parameters` | Число параметров зависит от выбранного метода |
| Линейная система | `linear_sys_method`, `banded_linear_solver`, `banded_fallback`, `iterative_refinement_steps` | Настройки native backend |
| Bounds/logging | `bounds`, `logging_mode`, `loglevel`, `logging_max_events` | Logging выключен по умолчанию; bounds задаёт пользователь |
| Banded layout | `SolverOptions.set_banded_width(lower, upper)` | Для damped-маршрута ширина обязательна; fixed-mesh Naive/Frozen выводят структуру и отклоняют явную ширину |

## Lifecycle, логирование, телеметрия

`prepare()` отделяет символьную подготовку от повторных численных запусков.
Для подготовленного приближения используйте `solve_prepared()`, а для замены
приближения — `restart(...)`. Параметры меняются через `set_parameters(...)`;
подготовленная символьная работа должна переиспользоваться, если это позволяет
native lifecycle. Логирование выключено, пока его явно не включили; режим
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

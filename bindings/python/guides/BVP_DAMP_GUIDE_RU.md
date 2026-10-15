# BVP_Damp: руководство для Python

BVP_Damp решает системы ОДУ первого порядка как краевую задачу с демпфированной
нелинейной итерацией и политиками уточнения сетки. Его постановка отличается
от BVP_sci: границы задаются mapping-ом имён состояний в наблюдаемые значения
на концах интервала, при необходимости с bounds. Frozen-маршруты в Python
wrapper отложены; наличие их в native Rust crate не означает поддержку здесь.

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
| Стратегия | `strategy="damped"` | `"frozen"` в этом wrapper явно не поддерживается |
| Нелинейный шаг | `tolerance`, `max_iterations`, `max_jacobian_refreshes`, `max_damping_iterations`, `damping_factor` | Newton/damping policy |
| Сетка | `max_refinements`, `grid_method`, `grid_parameters` | Число параметров зависит от выбранного метода |
| Линейная система | `linear_sys_method`, `banded_linear_solver`, `banded_fallback`, `iterative_refinement_steps` | Настройки native backend |
| Bounds/logging | `bounds`, `logging_mode`, `loglevel`, `logging_max_events` | Logging выключен по умолчанию; bounds задаёт пользователь |
| Banded layout | `SolverOptions.set_banded_width(lower, upper)` | Ширина обязательна и не выводится из RHS |

## Lifecycle, логирование, телеметрия

`prepare()` отделяет символьную подготовку от повторных численных запусков.
Для подготовленного приближения используйте `solve_prepared()`, а для замены
приближения — `restart(...)`. Параметры меняются через `set_parameters(...)`;
подготовленная символьная работа должна переиспользоваться, если это позволяет
native lifecycle. Логирование выключено, пока его явно не включили; режим
телеметрии задаётся отдельно через `SolverOptions` и входит в detached result.

Изучайте counters и stage timings раздельно. Проверяйте status, итоговую сетку,
невязки/итерации и bounds. Обрабатывайте типизированные configuration,
preparation, linear-backend и convergence ошибки, не анализируя текст логов.

## Примеры

`examples/bvp_damp_observations.py` — короткий одиночный запуск;
`examples/bvp_damp_layout_matrix.py` сравнивает все три layout и оба frontend
на точном постоянном профиле. Native BVP_Damp guides остаются источником
полного описания схемы и настроек, пока не выставленных Python-адаптером.

# BDF: руководство для Python

Python-адаптер BDF намеренно соответствует области плотного солвера crate.
Sparse и Banded для BDF не предоставляются; для этих IVP-layout используйте
LSODE2. Численное ядро написано на Rust, здесь native-руководство переведено на
Python API.

## Выбор маршрута

`BdfSolver` — полный символьный Python API. `expr_legacy` или `atom_native`
выбираются независимо от `lambdify` или `aot` через `SolverOptions`. Layout
только Dense. Отдельный `BdfCallbackSolver` принимает численные callback-и,
тоже только Dense/sequential; если передана лишь невязка, якобиан вычисляется
native конечными разностями.

```python
from rustedscithe import BdfSolver, Expr, SolverOptions

y = Expr.variable("y")
rate = Expr.variable("rate")
solver = BdfSolver(
    [-rate * y], ["y"], "t", [1.0], 0.0, 0.5, 0.05,
    parameters=["rate"], parameter_values=[0.7], rtol=1e-8, atol=1e-10,
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
)
solver.prepare()
result = solver.solve()
print(result.final_state_array())
```

## Управление интегрированием и continuation

`rtol`/`atol` задают допуски; `first_step`, `max_step`, `max_bdf_order` и
`max_steps` управляют шагами. Всегда проверяйте status/diagnostics, а не
делайте вывод об успехе лишь по конечному числу.

`continue_with_parameters(values, t_end)` повторно использует подготовленный
символьный маршрут и последнюю успешную точку. `restart(t0, y0, t_end)` меняет
начальное состояние и интервал, сохраняя символьные callback-и. При изменении
структуры уравнений или размерности создавайте новый solver.

## AOT, телеметрия и ошибки

Для одного холодного решения стоимость AOT-компиляции может не окупиться.
Общий lifecycle поддерживает BuildIfMissing, RequirePrebuilt и RebuildAlways;
компилятор, профиль и директория артефактов задаются через `SolverOptions`.
Проверяйте provenance и счетчики, прежде чем утверждать, что артефакт был
переиспользован.

Телеметрия отделена от изменяемого workspace. Выбирайте `off`, `counters`,
`timings` или `detailed`: счетчики и времена собираются раздельно, а
`detailed` включает оба набора. Сравнивайте подготовку, callback и численное
решение раздельно. Python callback API не обещает
параллельного ускорения. Ошибочные настройки и исчерпание бюджета возвращают
типизированные исключения.

## Примеры

`examples/bdf_continuation.py` показывает параметрическую серию,
`examples/bdf_aot_build.py` — опциональный compiler lifecycle,
`examples/bdf_robertson.py` — жёсткую химическую IVP. Алгоритмические детали и
ограничение Dense объясняются в native BDF guide; structured IVP-маршруты
относятся к LSODE2.

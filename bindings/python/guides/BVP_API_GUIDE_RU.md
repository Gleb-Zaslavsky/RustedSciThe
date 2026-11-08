# Типизированный BVP API

`BvpSolverRequest` — прикладной селектор поверх Rust-фасада `BVP_api`. Он
позволяет выбирать семейство солвера в конфигурации математической задачи,
а оркестрацию оставляет Rust. Это не замена прямым API `BvpDampSolver` и
`BvpSciSolver`, если нужны специализированные диагностики выбранного метода.

```python
import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions

y = Expr.variable("y")
mesh = [i / 20 for i in range(21)]
guess = [[math.exp(-x)] for x in mesh]  # [узел][переменная], включая края

request = BvpSolverRequest.sci(
    expressions=[-y],
    state_variables=["y"],
    independent_variable="x",
    mesh=mesh,
    boundary_conditions=[(0, "left", 1.0)],
    initial_guess=guess,
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="banded",  # полоса выводится из символьной структуры
        telemetry="detailed",
    ),
    tolerance=1e-5,
)
solver = request.prepare()
solution = solver.solve()
print(solution.values_array()[-1, 0])
print(solver.telemetry_snapshot())
```

Фабрики фасада выбирают семейство native-солвера: `damped`, `frozen`, `sci`
или `correction_newton`. Маршрут `correction_newton` реализует fixed-grid trapezoid/Newton для
равномерной сетки и поддерживает ExprLegacy или нативный AtomView, Lambdify
или AOT, а также Dense, Sparse и автоматически определяемый Banded backend.
Начальное приближение содержит одну строку на интервал; граничные значения
задаются отдельно. Grid refinement этим fixed-mesh маршрутом не поддерживается.
Фиксированные символьные входы можно передать через
`external_parameters` и `external_parameter_values`; вызов
`solve_with_parameter_values(values)` повторно использует подготовленные
Lambdify/AOT callbacks. `solve_parameter_loop(parameter_vectors)` выполняет
продолжение по порядку: первое решение использует исходное приближение, а
следующие — последний успешный профиль. Ошибка точки сохраняется как typed
error с доступной telemetry, но не прерывает последующие точки и не заменяет
последнее успешное решение. Простой пример находится в
`examples/bvp_api_correction_newton.py`, нетривиальные sweeps — в
`examples/bvp_api_correction_newton_combustion_continuation.py` и
`examples/bvp_api_correction_newton_bratu_continuation.py`.

## Маршруты и формы данных

Фабрики помечены native-семейством: `damped`, `frozen` или `sci`. `mesh`
должна быть конечной и строго возрастающей. Граничные условия задаются
тройками `(индекс_переменной, "left" | "right", значение)`; индекс относится
к порядку `state_variables`. Число уравнений и переменных должно совпадать,
для каждой переменной требуется одно скалярное граничное условие.

Начальное приближение передаётся строками в порядке Python/NumPy `[узел][переменная]`.
Damped и Frozen Newton ожидают строку на каждый интервал: их native iterates
содержат сокращённые неизвестные. Оба маршрута возвращают полный профиль с
граничными значениями на endpoint-ах. BVP_sci ожидает строку на каждый узел.
Damped и Frozen требуют равномерную сетку, поскольку их конструкторы принимают
границы и число интервалов, а не произвольные координаты. BVP_sci допускает неравномерную
сетку.

Общие frontend, execution, layout, callback policy, telemetry, logging и AOT
настройки задаются через `SolverOptions`; алгоритмические параметры доступны
в соответствующей фабрике. Неподдерживаемые комбинации вызывают типизированное
исключение с контекстом `stage`, `solver`, `route`, `native_kind`; скрытого
fallback нет. Generic facade поддерживает символические Lambdify и AOT.
Численные Python-callback маршруты остаются в прямом API `BvpSciSolver`.

`PreparedBvpSolver.set_boundary_values(values)` меняет только числовые значения
исходных граничных слотов в исходном порядке; `solve_with_boundary_values(values)`
также сразу запускает solve. Схему состояний и сторон границ менять нельзя.
Переиспользование callbacks реализовано для символьных маршрутов Sci,
CorrectionNewton, Damped и Frozen. Для Damped/Frozen поддерживаются подготовленные
ExprLegacy или AtomView с Lambdify либо AOT, но не численные callback-маршруты.
Перепривязка сбрасывает принятый результат/сохранённое приближение, сохраняя
подготовленные residual/Jacobian callbacks и AOT artifact. Изменение схемы
состояний или endpoint-ов требует подготовки нового solver.
Исполняемый пример повторного решения с новым числовым значением границы:
`examples/bvp_api_boundary_rebind.py`.

Для свободных параметров BVP_sci передаются вместе
`free_parameter_names`, `free_parameter_guess` и `boundary_residual`. Callback
невязки получает `(ya, yb, parameters)` и возвращает дополнительную граничную
невязку для каждого неизвестного. Необязательный `boundary_jacobian` возвращает
`(dR/dya, dR/dyb, dR/dp)` как вложенные списки в row-major порядке. Эти
callback-и выполняются на стороне Python и проходят через GIL; для символьных
выражений граничных условий используйте прямой API BVP_sci.

`BvpApiSolution` всегда содержит detached mesh и значения в node-major форме.
`sci_details` сохраняет полный native результат BVP_sci, включая статус и
запрошенный dense output. `telemetry_snapshot()` возвращает счётчики, времена и
scope metadata исходного маршрута без их суммирования. Отсутствующая запись
означает, что измерение не выполнялось; измеренный ноль сохраняется. Rust-логи
пишутся нативным логгером и не возвращаются как Python-данные.

Для нативных CSV/text, Markdown, Plotters/Gnuplot или терминального вывода
используйте `BvpPostprocessPlan` вместе с `execute_postprocessing()`. Для
каждого действия возвращается статус `done`, `skipped` или `failed`. Общий
postprocessing использует значения на сохранённой сетке; sampling dense output
BVP_sci доступен через `solution.sci_details`.

Для BVP_sci фиксированные входные параметры задаются отдельно от свободных
Newton-неизвестных: передайте `external_parameters` и начальные
`external_parameter_values` при создании запроса. Символьные callback-и
подготавливаются один раз. Вызов `solve_parameter_loop([[...], ...])` перед
каждым решением задаёт следующий вектор значений и использует последнее
успешное решение как начальное приближение. Ошибка отдельной точки остаётся в
route-specific отчёте и не прерывает цикл, если удалось восстановить checkpoint.
Для BVP_sci возвращается `BvpSciParameterLoopReport`, для BVP_Damp —
`BvpDampParameterLoopReport`; структура телеметрии и типизированные ошибки
сохраняют native-семантику. Пример: `python examples/bvp_api_parameter_loop.py`.
Для более крупного примера с шестью уравнениями горения и десятью значениями
тепловыделения см. `examples/bvp_api_combustion_parameter_loop.py`.

Единый текстовый уровень логирования задаётся через общий
`options.set_logging_level("info")` для Damped, Frozen/Naive и BVP_sci.
Параметр `logging_mode` у Damped/Frozen управляет отдельным ограниченным
decision trace и не заменяет текстовый лог.

Пример выбора solver-а: `python examples/bvp_api_solver_selection.py`.

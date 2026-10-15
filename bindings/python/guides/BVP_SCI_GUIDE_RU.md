# BVP_sci: руководство для Python

BVP_sci — collocation-солвер краевой задачи, численный метод которого
ориентирован на семейство SciPy `solve_bvp`. Rust-реализация добавляет
символьные frontend-ы, execution-маршруты, layout, телеметрию и типизированные
ошибки. Граничные условия задаются уравнениями-невязками, а не mapping-ом
состояния в наблюдения.

## Размерность задачи и минимальный запуск

Для задачи первого порядка `y'=-y` можно задать одну левую граничную невязку.
Начальные mesh/state — это начальное приближение, не сами граничные условия.

```python
import numpy as np
from rustedscithe import Expr, SolverOptions, solve_bvp

y = Expr.variable("y")
x = np.linspace(0.0, 1.0, 12)
guess = np.exp(-x)  # node-major: (nodes, state)
solution = solve_bvp(
    "bvp_sci", [-y], ["y"], [Expr.parse("ya_y - 1")], x, guess,
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="sparse", telemetry="timings",
    ),
    output="dense",
)
print(solution.sample(1.0), "точно:", np.exp(-1.0))
```

Результат содержит уточнённую сетку и node-major NumPy-массив формы
`(nodes, states)`. Для непрерывного решения используйте `sample(x)`, а не
предполагайте, что каждая исходная точка останется после refinement.

## Frontend-ы, layout и AOT

`expr_legacy` или `atom_native` выбирают символьную подготовку независимо от
`lambdify` или `aot`. Linear layout может быть Dense, Sparse или Banded. Это
самостоятельные маршруты солвера, а не конвертации между форматами. Dense
подходит для малых полностью связанных систем, Sparse/Banded могут выиграть,
если глобальный collocation-якобиан имеет подходящую структуру. Универсального
победителя нет.

Прямой Python-конструктор не прячет важные численные настройки:

| Настройка | Python keyword | Значение |
| --- | --- | --- |
| Допуск Ньютона | `tolerance`, `boundary_tolerance` | Критерии внутренних и граничных residual |
| Бюджеты | `max_nodes`, `max_newton_iterations`, `max_mesh_refinements`, `max_backtracking_steps` | Явные лимиты работы |
| Output | `output="final"`, `"mesh"`, `"dense"` | Итог, значения на mesh или непрерывный интерполянт |
| Boundary Jacobian | `boundary_jacobian="symbolic"` или `"finite_difference"` | Для символьных boundary expressions |
| Singular term | `singular_term_a`, `singular_term_matrix` | Передаются парой, `a` — левый конец интервала |
| Параметры | `parameters`, `parameter_values`, `set_parameters(...)` | Численный rebind без изменения структуры модели |
| Restart | `restart(mesh, initial_state)` | Заменяет сетку и численное начальное приближение |

Для Banded задайте через `SolverOptions.set_banded_width(lower, upper)`
реальную полуширину полосы collocation Jacobian. Не делайте вывод о layout
по неудачному решению с неверно заданной узкой полосой.

Структурированному layout нужна информация о якобиане, совместимая с маршрутом.
Проверяйте `capabilities("bvp_sci")`; не считайте, что callback fallback
доступен для любого layout. Опциональный SuperLU — opt-in backend sparse
факторизации, он не требуется базовому wheel. Если он недоступен, изучите
fallback/provenance и не считайте такие времена свидетельством скорости SuperLU.

## Параметры, restart и singular terms

Прямой wrapper поддерживает параметры и `set_parameters`, restart с новой
сеткой/начальным приближением, управление mesh refinement, dense output и
singular-term пару `(singular_term_a, singular_term_matrix)` в стиле SciPy.
Изменение параметров не означает повторную символьную подготовку. Restart
сбрасывает численное состояние траектории, сохраняя подготовленный маршрут,
если это поддержано. Граничные невязки и singular terms выбирают из модели,
а не для подгонки под backend.

## Телеметрия и ошибки

Включайте counters/timings через `SolverOptions`; отдельно анализируйте
подготовку, collocation, факторизацию, решение системы и refinement. Scopes
времени могут перекрываться. Результат также сообщает layout и provenance
frontend/execution и sparse-факторизации. Неверная форма границ, non-finite
данные, неподдержанный маршрут и численные ошибки представлены typed
исключениями.

Для численных boundary residual/Jacobian callbacks есть отдельный более узкий
маршрут; смотрите `capabilities` и документацию конструктора. Если уравнения
заданы как `Expr`, обычно выбирают полный символьный wrapper.

`examples/bvp_sci_sparse.py` показывает базовый маршрут,
`examples/bvp_sci_frontend_layouts.py` — matched-матрицу точной линейной BVP.
`examples/bvp_sci_aot.py` сравнивает Lambdify и AOT на одной Sparse/AtomNative
задаче; для компиляции требуется `tcc` или `gcc` в `PATH`.
Численные parity/performance claims относятся к тестовым отчётам основного
crate, а не к выводу демонстрационных скриптов.

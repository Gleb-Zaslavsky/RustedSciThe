# Backward Euler: руководство для Python

Backward Euler — неявный интегратор первого порядка. Он полезен как простой
надёжный baseline, для фиксированного шага и демонстрации влияния шага на
точность. Это не адаптивные высокопорядковые BDF или Radau.

## Минимальный запуск

```python
from rustedscithe import BackwardEulerSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BackwardEulerSolver(
    [-y], ["y"], "t", [1.0], 0.0, 1.0, 0.01,
    tolerance=1e-9,
    options=SolverOptions(
        frontend="expr_legacy", execution="lambdify",
        matrix_layout="dense", telemetry="counters",
    ),
)
solution = solver.solve()
```

Для `y'=-y` точное значение в конце равно `exp(-1)`. Метод имеет первый
порядок: при уменьшении шага вдвое глобальная ошибка в асимптотическом режиме
уменьшается примерно линейно. Пример `backward_euler_basic.py` это проверяет;
его времена не являются бенчмарком.

## Маршрут и lifecycle

Wrapper поддерживает Dense, символьную подготовку ExprLegacy/AtomNative и
исполнение Lambdify/AOT. Sparse, Banded и неподдержанная parallel-политика
явно отклоняются. Сейчас `solve()` выполняет ленивую native-подготовку; API
не обещает отдельный prepared lifecycle, пока его не предоставляет ядро.
Continuation и restart доступны в пределах native-контракта; проверьте,
требуется ли задаче сохранить фиксированный шаг или изменить интервал.

`h` — фиксированный шаг интегрирования, `tolerance` — допуск нелинейного
решения, `max_iterations` ограничивает работу. Допуск Ньютона не является
глобальной ошибкой интегрирования. Результаты возвращаются в собственных
NumPy-массивах.
`continue_to(t_end, parameter_values=None)` продолжает от последнего принятого
состояния с тем же фиксированным шагом; это тёплый сегмент, а не независимый
запуск. `restart(t0, y0, t_end)` заменяет состояние и интервал, сохраняя
подготовленные callback-и и шаг. Чтобы изменить `h` или структуру уравнений,
создайте новый solver.

## AOT и диагностика

AOT использует общий lifecycle сборки и cache. Компилятор не нужен для
Lambdify и требуется только при выборе AOT. Сохраняйте выбранный execution
route и provenance артефакта. Режим телеметрии задаётся через
`SolverOptions(telemetry=...)`: `off`, `counters`, `timings` или `detailed`.
`timings` возвращает длительности без дискретных счётчиков, а `detailed` — оба
набора данных. Ошибки подготовки и исчерпания бюджета обрабатывайте по типам
исключений, а не по строке сообщения.

См. `examples/backward_euler_basic.py` для уточнения шага и
`examples/backward_euler_routes.py` для matched-сравнения ExprLegacy/AtomNative.

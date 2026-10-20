# Блочно-трёхдиагональная LU-факторизация

`rustedscithe` предоставляет проверенный маршрут `somelinalg` как
`BlockTridiagonalLuConsistent`. Он предназначен для систем, у которых на
главной, нижней и верхней блочных диагоналях стоят плотные квадратные блоки.
Факторизация остаётся структурированной: Python API принимает три компактных
массива и не создаёт глобальную плотную матрицу `n x n`.

## Компактное представление NumPy

Для `n_blocks` блоков размера `block_size` передаются:

- `diagonal`: форма `(n_blocks, block_size, block_size)`;
- `lower`: форма `(n_blocks - 1, block_size, block_size)`;
- `upper`: форма `(n_blocks - 1, block_size, block_size)`.

Массивы копируются при создании и должны содержать конечные значения
`float64`. Конструктор сразу выполняет нативную факторизацию, поэтому объект
готов к повторному использованию:

```python
import numpy as np
from rustedscithe import BlockTridiagonalLuConsistent

lower = np.asarray([[[0.25, 0.0], [0.0, 0.25]]], dtype=np.float64)
diagonal = np.asarray(
    [
        [[4.0, 1.0], [2.0, 3.0]],
        [[5.0, 1.0], [1.0, 4.0]],
    ],
    dtype=np.float64,
)
upper = np.asarray([[[0.5, 0.0], [0.0, 0.5]]], dtype=np.float64)
lu = BlockTridiagonalLuConsistent(lower, diagonal, upper)
```

Изменение `lower`, `diagonal` или `upper` после создания не меняет нативную
факторизацию. Свойства `storage_bytes`, `n_blocks`, `block_size`, `n` и
`is_factorized` делают владение памятью и lifecycle видимыми без раскрытия
внутренних векторов.

## Решение и диагностика

`solve(rhs)` принимает и возвращает вектор формы `(n,)`. В
`solve_multiple(rhs)` используется естественная форма NumPy `(n, nrhs)`;
преобразуется только layout правых частей, коэффициентная матрица остаётся
структурированной. Для готовых contiguous-workspace доступны
`solve_in_place` и `solve_multiple_in_place`. В padded-варианте для каждого RHS
задаётся блок `stride`, а padding не изменяется.

Факторизация предоставляет `factor_residual_relative`,
`condition_diagnostics()` и `pivot_diagnostics()`. Это диагностические
величины, а не обещание автоматически отвергать всякую плохо обусловленную
матрицу. Для контролируемого уточнения используйте
`solve_with_refinement(rhs, refinement_steps)`, который возвращает отдельный
`BlockTridiagonalRefinementResult`.

## Ошибки и границы API

Неправильная форма блоков, переполнение размерности, NaN/Inf в матрице или RHS
и неправильный layout RHS дают `InvalidConfigurationError`. Сингулярный pivot,
нечисловые факторы или решение дают `SolverError`. Исключения сохраняют
`stage`, `solver`, `route` и явный `native_kind`; прикладной код не должен
разбирать текст ошибки.

У standalone-ядра нет собственного контракта logging/telemetry, поэтому
`capabilities("block_tridiagonal")` сообщает `telemetry_modes == ["off"]`.
Счётчики и времена внешнего BVP/IVP-солвера остаются его ответственностью.
Старая block-LU реализация и bordered block solver намеренно не включены в
Python API. Рабочий пример:
[`somelinalg_block_tridiagonal.py`](../examples/somelinalg_block_tridiagonal.py).

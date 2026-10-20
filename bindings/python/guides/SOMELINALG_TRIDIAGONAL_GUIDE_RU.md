# Pivoted tridiagonal LU

`rustedscithe` предоставляет стабильный pivoted tridiagonal LU-солвер из
`somelinalg` через `PivotedTridiagonalLu`. Это небольшой API численной
линейной алгебры, а не символьный солвер. Матрица хранится в трех одномерных
массивах `float64`; плотная NumPy-матрица размера `n x n` не создается.

## Компактное хранение и pivoting

Для матрицы `n x n` передаются `lower` и `upper` формы `(n - 1,)`, а
`diagonal` формы `(n,)`. При создании native-факторизация копирует эти
массивы. Поэтому последующее изменение исходного NumPy-массива не меняет уже
созданную факторизацию.

```python
import numpy as np
from rustedscithe import PivotedTridiagonalLu

lower = np.asarray([2.0, 1.0, 1.0], dtype=np.float64)
diagonal = np.asarray([1.0e-12, 2.0, 2.0, 2.0], dtype=np.float64)
upper = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
lu = PivotedTridiagonalLu(lower, diagonal, upper)
```

Здесь первая диагональная величина намеренно мала. Частичный pivoting
использует соседний элемент нижней диагонали и не предполагает, что главный
диагональный элемент всегда является хорошим pivot. Свойство `ipiv` содержит
решения native-алгоритма о перестановках. Параметр `pivot_epsilon` задает
абсолютный порог принятия pivot; он должен быть конечным и неотрицательным.

## Один и несколько RHS

`solve()` возвращает собственный вектор формы `(n,)`. `solve_in_place()` можно
использовать для contiguous writable-массива `float64`, когда важна экономия
аллокации результата. Для нескольких правых частей используется естественная
форма NumPy `(n, nrhs)`:

```python
rhs = np.asarray(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    dtype=np.float64,
)
solutions = lu.solve_multiple(rhs)
```

Низкоуровневый `solve_multiple_in_place()` принимает плоский буфер, число RHS
и `stride`. Он изменяет только первые `n` элементов каждого блока и сохраняет
padding. Обычный `solve_multiple()` возвращает стандартный массив `(n, nrhs)`;
преобразование layout выполняется на границе Python, а native-факторизация
остается структурированной.

## Ошибки и границы API

Неверные длины диагоналей, `NaN`/`inf`, некорректный tolerance, короткий RHS
или ошибочный padded layout дают `InvalidConfigurationError`. Нулевой pivot,
нечисловой фактор или нечисловое решение дают `SolverError`. Оба класса
сохраняют `stage`, `solver`, `route` и явный `native_kind`; прикладному коду
следует проверять эти поля, а не разбирать текст сообщения.

У kernel нет собственного logging/telemetry-контракта. Поэтому
`capabilities("pivoted_tridiagonal").telemetry_modes` равен `["off"]`, а
внешние IVP/BVP-солверы по-прежнему отвечают за свои стадии и счетчики. В
публичный Python API включен только `float64`-маршрут с последовательным
исполнением. Незавершенная непивотированная реализация намеренно не открыта.

Рабочий пример находится в
[`somelinalg_tridiagonal.py`](../examples/somelinalg_tridiagonal.py).

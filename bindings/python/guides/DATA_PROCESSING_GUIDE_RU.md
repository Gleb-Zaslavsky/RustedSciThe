# Руководство по обработке данных

Python API обработки данных вызывает нативные Rust-утилиты из
`numerical::data_processing`, а не реализует алгоритмы повторно на Python.
Входные NumPy-массивы `float64` копируются перед передачей в Rust; выходы
владеют своей памятью. Поддерживаются и невыровненные/непрерывные срезы NumPy.
Ошибочные формы, нечисловые значения и неподдерживаемые настройки приводят к
типизированным исключениям RustedSciThe.

## Сглаживание и изменение сетки сигнала

```python
import numpy as np
from rustedscithe import fit_quality, lowess_fit, savgol_filter, spline_resample

x = np.linspace(0.0, 4.0 * np.pi, 81, dtype=np.float64)
signal = np.sin(x) + 0.05 * np.sin(17.0 * x)

filtered = savgol_filter(signal, window_length=9, polyorder=3)
fit = lowess_fit(x, signal, fraction=0.2, iterations=2, delta=0.0)
x_fine, y_fine = spline_resample(fit.x, fit.y, 241, kind="cosine")
rmse, mae, r_squared = fit_quality(signal, filtered)
```

`lowess_fit` возвращает отделённый от внутренних буферов `LowessResult` с
NumPy-массивами `x` и `y`, фактической `fraction_used` и числом итераций, если
его сообщает нативный backend. Если нужны только сглаженные ординаты, используйте
`lowess_smooth`. Сейчас доступен CPU-маршрут; GPU-вариант отклоняется, а не
подменяется незаметно вычислением на CPU.

Производные `savgol_filter` используют единичный шаг между отсчётами; API не
заявляет совместимость с настраиваемым `delta` из SciPy. `spline_resample`
поддерживает `linear`, `cosine` и `bezier`, возвращая равномерную сетку внутри
исходного диапазона. `tension` применяется только для Bezier. `fit_quality`
возвращает `(RMSE, MAE, R-squared)`; для постоянного эталонного сигнала
коэффициент R-squared возвращается как `None`.

## B-сплайны

`BSplineInterpolator` представляет переиспользуемый скалярный интерполятор.
Можно передать полный вектор узлов или запросить нативную конструкцию с
равномерными зажатыми узлами:

```python
from rustedscithe import BSplineInterpolator, least_squares_bspline

x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
signal = np.sin(2.0 * np.pi * x)
control = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
curve = BSplineInterpolator.with_uniform_knots(3, control)
samples = np.linspace(*curve.knot_domain, 100, dtype=np.float64)
values = curve.interpolate_batch(samples)

knots = np.array(
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=np.float64
)
fit = least_squares_bspline(x, signal, knots, degree=3, solver="banded")
```

Перед вызовом legacy-конструктора нативного интерполятора (который может
паниковать на некорректных узлах) binding проверяет длину и порядок узлов,
кратность и диапазон. Вычисления ограничены knot-domain и возвращают
самостоятельные массивы. `least_squares_bspline` аппроксимирует наблюдения и
позволяет явно выбрать `dense` (QR) или `banded`; скрытого переключения между
решателями нет. Это разные задачи: интерполяция использует заданные контрольные
точки, а LSQ-вариант оценивает коэффициенты по наблюдениям.

Для траекторий солверов предпочтительнее их собственные dense-output и sampling
API. Общие утилиты обработки данных предназначены для подготовки и
post-processing, а не для замены интерполянта с контролем ошибки.

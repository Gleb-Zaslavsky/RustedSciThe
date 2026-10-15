# Нелинейные системы и fitting: руководство для Python

В Python-пакете доступны разные контракты задач. `NonlinearSolver` решает
квадратную систему невязок `F(x)=0`; `FittingSolver` — символьную least-squares
задачу; `GavinFittingSolver` — fitting наблюдений с семантикой Gavin для
модели/весов/bounds; `VarProSolver` — variable projection. Они используют
общее Rust-символьное ядро, но не являются взаимозаменяемыми objective и
типами результата.

## Квадратная нелинейная система

```python
from rustedscithe import Expr, SymbolicSystem, create_nonlinear

x = Expr.variable("x")
system = SymbolicSystem([x * x - Expr.constant(4.0)], ["x"])
solver = create_nonlinear(system, method="newton", telemetry="timings")
solver.prepare()
root = solver.solve([1.0])
print(root.solution_array())  # при этом старте корень около +2
```

API предоставляет Newton, damped Newton, семейство LM и trust-region методы,
если они объявлены в `capabilities("nonlinear")`. Опции метода передаются
через соответствующий native mapping; неизвестные ключи отклоняются, а не
игнорируются. ExprLegacy/AtomNative выбирают символьную подготовку. AOT не
заявляется, пока capability metadata не сообщает конкретный compiled-маршрут.

## Least-squares fitting

`FittingSolver` минимизирует вектор символьных residual equations. Схема
параметров и опциональная политика положительности относятся к fitting-задаче,
а не к API решения корней. Подготовленные параметры можно переназначать через
`solve_with_parameters(...)`.

```python
from rustedscithe import Expr, FittingSolver

p = Expr.variable("p")
fit = FittingSolver([p - Expr.constant(3.0)], [0.0], variables=["p"])
fit.prepare()
result = fit.solve()
print(result.solution_array(), result.termination_reason)
```

## Gavin и VarPro

`GavinFittingSolver` принимает выборки `x_data`, наблюдения, имя параметра
выборки, неизвестные параметры и начальное приближение. Это отдельный
one-shot контракт, не prepared API `FittingSolver`. Положительные ограничения
параметров выбирает пользователь; они не выводятся автоматически из
химической модели. `VarProSolver` также имеет собственный one-shot lifecycle
и статистики. Не предполагайте у них continuation/rebind семантику canonical
fitting солвера.

## Frontend-ы, логирование и результаты

В поддержанных маршрутах можно выбрать ExprLegacy/AtomNative и sequential/
parallel вычисление символьных выражений. Точка окупаемости параллелизма
зависит от задачи и не означает параллельное выполнение Python callbacks.
Логирование выключено по умолчанию и включается явно; телеметрия настраивается
отдельно и не разделяется с изменяемым состоянием солвера.

Результаты содержат найденные значения, причину остановки/цель и специфичные
диагностики. Covariance, R² и статистики fitted output осмысленны только там,
где контракт содержит нужные наблюдения и веса. Ошибки типизированы;
исчерпание бюджета следует отличать от неверной конфигурации и ошибки
символьной подготовки.

`examples/nonlinear_and_fitting.py` даёт короткий обзор,
`examples/gavin_observation_fit.py` показывает fitting наблюдений с bounds.
Алгоритмические детали см. в native guides нелинейных систем и optimization.

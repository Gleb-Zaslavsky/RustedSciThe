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
игнорируются. ExprLegacy/AtomNative выбирают символьную подготовку. Доступен
Dense AOT через `execution="aot"`; lifecycle настраивается параметрами
`aot_policy`, `aot_profile` и `aot_output_dir`. По умолчанию используется
Lambdify, а для политик сборки требуется явная выходная папка.

```python
from pathlib import Path
from rustedscithe import Expr, NonlinearSolver, SymbolicSystem

x = Expr.variable("x")
solver = NonlinearSolver(
    SymbolicSystem([x * x - 2.0], ["x"]),
    execution="aot",
    aot_policy="build_if_missing",
    aot_profile="release",
    aot_output_dir=str(Path("target") / "rst-nonlinear-aot"),
)
root = solver.solve([1.0])
```

`NonlinearSolver` охватывает все 11 native-методов, оба фронтенда ExprLegacy и
AtomNative, sequential/parallel/thresholded-auto execution, явный повторно
безопасный `prepare()`, parameter rebind, логгирование и opt-in counters либо
detailed timings. Числовые и boolean-настройки передаются через
`method_options` и `method_flags`. `method_vectors={"diag": [...]}` доступен
только MINPACK и требует `mode=2`; `method_choices` задаёт advanced Nielsen
`scaling_method` (`levenberg`, `marquardt`, `more`) и `reduction_ratio_method`
(`nielsen`, `more`). Именованные частичные box bounds допускают `None` на любой
границе. Отдельная positive-domain политика и Python callbacks для этого
square-system API не поддерживаются и не подменяются скрыто другими режимами.

## Least-squares fitting

`FittingSolver` минимизирует вектор символьных residual equations. Схема
параметров и опциональная политика положительности относятся к fitting-задаче,
а не к API решения корней. Подготовленные параметры можно переназначать через
`solve_with_parameters(...)`.

```python
from rustedscithe import Expr, FittingSolver

p = Expr.variable("p")
fit = FittingSolver([p - Expr.constant(3.0)], [0.0], variables=["p"])
result = fit.solve()
print(result.solution_array(), result.termination)
```

## Gavin и VarPro

`GavinFittingSolver` принимает выборки `x_data`, наблюдения, имя параметра
выборки, неизвестные параметры и начальное приближение. Это отдельный
one-shot контракт, не prepared API `FittingSolver`. Положительные ограничения
параметров выбирает пользователь; они не выводятся автоматически из
химической модели. Телеметрия solve у Gavin по умолчанию выключена и следует
общему четырёхрежимному контракту: `off`, `counters`, `timings` или `detailed`
(оба канала). Передайте `telemetry_mode` в `GavinFittingSolver` либо вызовите
`with_gavin_telemetry_mode(...)` у `UniversalFitting`. Старый флаг
`telemetry=True` оставлен как сокращение для `detailed`. Это отдельно от
`preparation_telemetry`, измеряющей символьную подготовку. `VarProSolver` также имеет собственный one-shot lifecycle
и статистики. Не предполагайте у них continuation/rebind семантику canonical
fitting солвера.

`VarProSolver` предоставляет настройки native LM: tolerances, лимиты итераций
и вычислений функции, step bound, diagonal scaling и telemetry через
`with_tolerances`, `with_max_iterations`, `with_max_function_evaluations`,
`with_step_bound`, `with_scale_diag`, `with_telemetry`. Это ExprLegacy-only,
последовательный one-shot маршрут без логирования и отдельной стадии подготовки.

`UniversalFitting` — отдельный facade над native Rust method selector, а не
псевдоним одного из standalone-солверов. Выберите `"lm"`, `"gavin"` или
`"varpro"`, затем задайте общие данные/неизвестные и equations либо basis
функции. `build()` потребляет builder и возвращает общий результат с `method`,
`solution_map()` и `r_squared`; для route-specific telemetry/covariance следует
использовать standalone-класс. Неподдерживаемые сочетания отклоняются явно:
VarPro не поддерживает AtomNative и parallel, Gavin не поддерживает
`f_tolerance` и `scale_diag`. Общий многоуровневый `with_telemetry("counters"|"detailed")`
применяется к выбранному маршруту, включая Gavin. `with_gavin_telemetry(True)`
сохранён как совместимое сокращение для включения `detailed`.

```python
from rustedscithe import UniversalFitting

fit = UniversalFitting("lm")
fit.with_data([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
fit.with_arg("x")
fit.with_equation_str("a*x+b")
fit.with_unknowns(["a", "b"])
fit.with_initial_guess([0.0, 0.0])
solution = fit.build()
print(solution.method, solution.solution_map())
```

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

# Символьные подстановки до подготовки решателя

Именованные подвыражения можно задать через `substitutions` при создании
символьного BDF, Backward Euler, LSODE2 или Radau solver-а, либо передать
запросу BVP API через `with_substitutions`. Нелинейные solver-ы принимают ту же
таблицу при создании, а `UniversalFitting` предоставляет её как builder-метод.
Во всех случаях подстановки рекурсивно раскрываются до
численной подготовки:

```python
from rustedscithe import BdfSolver, Expr

solver = BdfSolver(
    [Expr.parse("loss")], ["y"], "t", [1.0], 0.0, 1.0, 0.01,
    parameters=["rate"], parameter_values=[2.0],
    substitutions={"loss": Expr.parse("-rate*y")},
)
solution = solver.solve()
```

Имена `rate` и другие свободные символы остаются обычными переменными/параметрами
задачи; подстановка не является bind/rebind и не запускает расчёт. Для общего
фасада можно вызвать `request.with_substitutions({...})` у
`StiffSolverRequest` или `BvpSolverRequest` до `prepare()`.

Например, BVP RHS может ссылаться на alias:

```python
request = BvpSolverRequest.sci(
    [Expr.parse("decay")], ["y"], "x", mesh,
    [(0, "left", 1.0)], initial_guess,
)
request.with_substitutions({"decay": Expr.parse("-y")})
solution = request.prepare().solve()
```

В нелинейной задаче alias раскрывается до вывода неизвестных, поэтому его имя
само неизвестной не становится:

```python
system = SymbolicSystem([Expr.parse("force - 2")])
solver = NonlinearSolver(
    system,
    parameters=["rate"],
    parameter_values=[4.0],
    substitutions={"force": Expr.parse("rate*x")},
)
result = solver.solve([1.0])
```

`UniversalFitting.with_substitutions({...})` применяет alias к уравнениям
LM/Gavin и к basis/invariant-выражениям VarPro до подготовки fitting-модели.

Пустые, повторные, циклические и конфликтующие с именами состояния, времени или
параметров определения отклоняются типизированной configuration error. Для
LSODE2 подстановки не сочетаются с пользовательскими numerical callbacks:
callbacks непрозрачны для символического resolver-а, поэтому такой случай явно
отклоняется.

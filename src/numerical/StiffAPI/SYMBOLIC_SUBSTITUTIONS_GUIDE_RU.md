# Символьные подстановки перед подготовкой IVP

`try_with_substitutions` — удобная операция над исходными `Expr`, которая
раскрывает именованные подвыражения **до** подготовки solver-а. Она не меняет
численный алгоритм и не является parameter binding/rebinding или continuation.
Сами уравнения остаются единственным источником RHS; добавленные определения
раскрываются рекурсивно, включая цепочки alias.

```rust
let request = StiffSolverRequest::Bdf { options }
    .try_with_substitutions([("loss", Expr::parse_expression("-rate*y"))])?;
let mut solver = request.prepare(None)?;
solver.try_solve()?;
```

`rate` здесь остаётся обычным runtime-параметром: его имя и значение задаются
штатным parameter API. Имена alias не могут совпадать с переменными состояния,
независимой переменной или объявленными параметрами. Дубликаты и циклические
определения дают typed `SymbolicSubstitutionError` до подготовки callback-ов.

Поддерживается на BDF, Backward Euler, LSODE2 и Radau, а также через общий
`StiffSolverRequest`. В native API options/config имеют тот же
`try_with_substitutions`. Прямые Python-классы принимают
`substitutions={"loss": Expr.parse("-rate*y")}`; `StiffSolverRequest` получает
их через `request.with_substitutions({...})`.

Подстановка применима к символическим `Expr` задачам. Она не может переписать
пользовательские numerical callbacks; LSODE2 отвергает такую комбинацию typed
ошибкой, вместо молчаливого игнорирования alias.

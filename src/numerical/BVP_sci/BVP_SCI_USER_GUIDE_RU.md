# Руководство BVP_sci

`BVP_sci` — поддерживаемый коллокационный BVP solver в стиле SciPy. Для
Lambdify он поддерживает символьные frontend-ы `ExprLegacy` и `AtomViewNative`,
а линейные layout-ы `Dense`, `Sparse` и `Banded`. AOT является отдельным
маршрутом с generated callbacks, а не неявным fallback.

## Маршрут task parser

`solver: BVP_sci` выбирает именно этот solver. Комментарии, начинающиеся с
`#`, parser игнорирует, поэтому reference-файлы могут подробно объяснять DSL
прямо внутри задания.

```text
# BVP_sci и BVP_Damp — разные solver families.
task
solver: BVP_sci

# ExprLegacy/AtomViewNative — символьные frontend-ы.
frontend: ExprLegacy
# Lambdify/AOT — способы исполнения подготовленных callbacks.
execution: Lambdify
# Dense/Sparse/Banded — layout линейной алгебры.
method: Sparse

equations
arg: x
unknowns: y
rhs: -rate*y
# В BVP_sci параметры — свободные Newton-неизвестные, а не fixed inputs.
# Это начальное приближение; на каждую неизвестную нужно доп. условие.
parameters: rate
parameter_values: 0.7

boundary_conditions
# Для BVP_sci нужен полный набор scalar boundary conditions: обычно по одному
# условию на каждое состояние, распределённому между левым и правым концом.
y_left: 1.0
y_right: 0.36787944117144233

mesh
t0: 0.0
t_end: 1.0
n_steps: 16

initial_guess
y: 0.0

solver_options
tolerance: 1e-6
max_iterations: 40

postprocessing
# Современная политика вывода; plot — только compatibility alias.
output_policy: none
```

В `solver_options` также доступны `matrix_backend` и generated-backend
preset-ы. `method` — удобное task-level имя layout. Проверка полноты boundary
conditions выполняется до создания solver.

Универсально самого быстрого layout не существует. `Dense` естественен для
малых полностью связанных систем, а `Sparse` или `Banded` могут выигрывать,
если глобальный collocation Jacobian имеет используемую структуру. Сравнивайте
одинаковые задачи, а не разные уравнения. Полное руководство с прямым API и
численными настройками находится в
[`BVP_SOLVER_USER_GUIDE.md`](BVP_SOLVER_USER_GUIDE.md).

## Frontend и execution — разные оси

Не смешивайте эти понятия:

- `frontend: ExprLegacy` использует совместимое expression graph;
- `frontend: AtomViewNative` подготавливает и дифференцирует Atom view;
- `execution: Lambdify` создаёт in-process callbacks;
- `execution: AOT` материализует generated callbacks и проходит AOT lifecycle.

Пример AtomView + Lambdify:

```text
solver: BVP_sci
frontend: AtomViewNative
execution: Lambdify
method: Sparse
```

Тот же frontend можно направить в AOT, изменив `execution` и добавив настройки
generated backend.

## AOT и handoff между процессами

```text
solver_options
generated_backend: sparse_aot_tcc
symbolic_backend: atomview
aot_build_policy: build_if_missing
aot_build_profile: release
aot_output_dir: test_reports/bvp_sci/aot
aot_handoff_path: test_reports/bvp_sci/aot/handoff.txt
aot_execution_policy: sequential
```

Политики имеют разный смысл:

- `use_if_available` — использовать доступный linked artifact;
- `build_if_missing` — собрать и опубликовать только отсутствующий artifact;
- `require_prebuilt` — только использовать опубликованный artifact, без compile;
- `rebuild_always` — принудительно выполнить materialize/build/link заново.

Producer записывает manifest и provenance в handoff. Consumer в другом процессе
читает его при `require_prebuilt`; telemetry должна показать ноль build attempts,
сохранив artifact key и link/reconnect provenance. Missing/stale/malformed
artifact и compiler/linker/timeout failures возвращаются как typed errors.
Dense AOT намеренно не является production generated route; для generated
режима используйте поддержанные Sparse/Banded preset-ы или Lambdify Dense.

## Continuation и restart

```text
continuation
# Параметр должен быть объявлен в equations.parameters.
parameter: rate
values: 0.5, 1.0, 2.0
# fresh — новая подготовка; warm — reuse с предыдущим состоянием;
# prepared — явный lifecycle подготовленной модели.
mode: prepared
# false продолжает от предыдущего принятого решения.
restart_each: false
```

Для restart используются поддержанные схемой `y0_values`, `t0_values` и
`t_end_values`. Изменение mesh или состояния инвалидирует затронутое numeric
state, но не позволяет молча использовать несовместимый callback/factor.
Результат содержит сегменты и continuation telemetry, поэтому длинная серия
различает fresh preparation, prepared reuse и restart cost.

## Вывод и postprocessing

```text
postprocessing
output_policy: none       # none | plotters | gnuplot | terminal
write_report: true
report_path: reports/bvp_sci.md
```

CSV, TXT и Markdown — независимые действия. `plot: true/false` поддерживается
для старых файлов, но в новых документах используйте `output_policy`.

Прямой типизированный Rust API BVP_sci нужен для пользовательских callbacks,
тонкой настройки solver, dense-output control и программного восстановления
после ошибок.

## Телеметрия

Уровень сбора задаётся через `BvpSciTelemetryMode`: `Off` отключает сбор,
`Counters` собирает дискретные счётчики без чтения часов, `Timings` измеряет
время без дискретных счётчиков. `Detailed` объединяет оба вида данных и
сохраняет подробную историю Newton и проходов по сетке. Временные scopes могут
перекрываться: не складывайте родительские и дочерние длительности.

## Упорядоченный цикл по фиксированным параметрам

`parameter_names` объявляет неизвестные, которые BVP_sci решает совместно с
профилем. Для фиксированных входов модели используйте отдельную схему
`with_external_parameter_names` / `with_external_parameter_values`. Эти значения
передаются в callbacks, но не добавляют столбцов в систему Newton. Подготовленные
ExprLegacy, AtomViewNative или AOT evaluators переиспользуются между точками.

Вызовите `solve_parameter_loop(&[vec![...], ...])`, чтобы обработать наборы
последовательно. После успеха mesh, профиль и найденные free-параметры становятся
начальным приближением для следующей точки. Отказ одной точки сохраняется вместе
с typed error и snapshot телеметрии; solver восстанавливает последнее успешное
состояние (или исходное, если первый solve не удался) и продолжает цикл. Snapshot-ы
телеметрии накопительные для solver на момент каждой точки, а не отдельные дельты.
Численные callbacks пока не принимают именованные внешние параметры: для них ещё
не определён отдельный канал фиксированных входов.


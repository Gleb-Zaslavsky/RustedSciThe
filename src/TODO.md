# RustedSciThe: технический долг и дальнейший план

Дата ревизии: 2026-05-27.

Этот файл не заменяет подробные story-ledger и solver-specific checklist. Его
назначение - держать в одном месте следующий рабочий фронт после большой серии
изменений в `LSODE2`, BVP Damped/Frozen, generated backends и task parser.
Подробные сырые результаты измерений по-прежнему следует хранить в
`numerical/BVP_Damp/BVP_DAMP_STORY_TESTS.md` и
`symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md`.

## Что уже не является TODO

Ниже перечислены вещи, которые важно не открывать заново без новой регрессии
или нового контрпримера.

- У `BVP Damped` есть настоящий Numerical path с Rust-замыканием правой части:
  отдельно поддерживаются конечно-разностный якобиан и пользовательский
  непрерывный якобиан `df/dy`. Фиктивный пустой вектор символических уравнений
  для этого маршрута больше не является пользовательским API.
- Numerical discretization принимает корректные краевые условия, в которых
  несколько условий могут относиться к одной переменной; требуется лишь
  правильное общее число условий для системы первого порядка.
- `Frozen NumericOnly` намеренно не поддерживается: Frozen сейчас является
  symbolic/generated маршрутом с готовым callback-якобианом, а не FD-only
  вариантом Damped.
- `AtomView` реализован для Sparse и Banded символических BVP-маршрутов и
  является production frontend по умолчанию; `ExprLegacy` остается контрольным
  и compatibility-путем.
- Ошибка chunked AOT Jacobian исправлена; runtime parallel binding и прямое
  заполнение disjoint output slices защищены correctness/concurrency tests.
- Для Damped и Frozen `AtomView + tcc` lifecycle
  `BuildIfMissing -> RequirePrebuilt` подтвержден release-прогонами на Sparse
  и Banded маршрутах.
- Честное warm-сравнение Damped Banded combustion-1000 показало, что strict
  prebuilt `tcc` быстрее Lambdify примерно на `7.9%`, однако первая сборка
  окупается только после нескольких повторных решений.
- Основные явно дорогие этапы BVP AOT materialization/lowering/packaging уже
  диагностированы и существенно удешевлены; дальнейшие изменения в этой зоне
  следует подтверждать stage tables, а не предположениями.

## P0: ошибки и надежность

### Потенциальная рекурсия в `Clone` BVP trait objects

При `cargo test --lib --no-run` компилятор сообщает о безусловной рекурсии:

- `src/numerical/BVP_Damp/BVP_traits.rs`: `impl Clone for Box<dyn Y>`.
- `src/numerical/BVP_Damp/BVP_traits2.rs`: `impl Clone for Box<dyn Y>`.

Тело вида `self.clone()` внутри `Clone for Box<dyn Y>` выглядит как вызов
самого себя и может привести к переполнению стека при реально достигнутом пути.

- [x] Проверить, вызываются ли эти `Clone` реализации в Damped/Frozen/BVP_sci.
- [x] Заменить рекурсивную реализацию на объектно-безопасный `clone_box`/явное
  клонирование конкретного payload либо удалить недостижимую abstraction.
- [x] Добавить unit test, который действительно клонирует соответствующий
  trait object и проверяет содержимое результата.

### AOT artifact lifecycle как публичный контракт

`BuildIfMissing` и `RequirePrebuilt` уже работают в solver lifecycle, но
пользователю пока не хватает ясного API для эксплуатации артефактов между
запусками приложения.

- [ ] Документировать место хранения артефактов, ключ проблемы и связь ключа с
  сеткой, frontend, matrix backend, toolchain и chunking policy.
- [ ] Предусмотреть безопасную публичную операцию очистки артефактов по ключу
  либо documented artifact-directory workflow.
- [ ] Определить поведение при stale artifact: проверка manifest/hash/ABI перед
  binding, а не загадочный runtime failure.
- [ ] Добавить acceptance tests на missing, stale и intentionally reused
  artifact для `RequirePrebuilt`.

### AOT toolchain robustness

Внешние компиляторы и динамическая загрузка остаются источником
платформенно-зависимых сбоев.

- [ ] Добавить понятные diagnostics для отсутствующих `tcc`, `gcc` и `zig`,
  failed spawn, failed compile и failed dynamic load.
- [ ] Решить, нужен ли ограниченный retry только для transient file-lock/load
  failures; не ретраить детерминированные compile errors.
- [ ] Проверить очистку временных файлов и lock-поведение на Windows после
  неуспешной сборки/загрузки.
- [ ] Добавить determinism/reproducibility check для AOT manifest/artifact hash.

## P1: BVP Damped/Frozen - оставшиеся production gates

### Финальная согласованность Sparse и Banded stories

После исправлений frontend/handoff важно один раз подтвердить, что старые
heavy comparisons по-прежнему рассказывают ту же историю, что новые lifecycle
tests.

- [ ] Сопоставить актуальные результаты
  `combustion_1000_aot_sparse_vs_banded_end_to_end_race` и
  `combustion_1000_end_to_end_banded_lapack_refine_statistics`.
- [ ] Проверить, что Banded advantage наблюдается в linear-system части для
  задач с узкой полосой и не объясняется случайным cold/warm смешением.
- [ ] Пометить устаревшие либо дублирующие story tests как `superseded`, чтобы
  будущий читатель не пытался интерпретировать старые аномальные таблицы.

### Auto chunking как production gate

Forced `chunk4` нужен для диагностики, но production-default должен опираться
на `Auto` и native runtime diagnostics.

- [ ] Добавить один heavy end-to-end test для `Auto` на большой задаче,
  проверяющий correctness и печатающий planned/actual jobs, fallback reason и
  work per job.
- [ ] Зафиксировать, что `Auto` вправе выбрать whole для малой нагрузки и
  chunked execution для достаточно тяжелой; тест не должен требовать
  конкретного выбора на любой машине без учета diagnostics.
- [ ] Обновить гайды конкретным примером чтения `aot.auto.*` и `aot.runtime.*`.

### Расширение Frozen coverage

Sparse и Banded combustion-1000 уже закрывают artifact lifecycle, а Banded
дополнительно закрывает тяжелую whole/chunk4 symbolic/AOT основу. Дальше
достаточно добавить не матрицу похожих тестов, а один качественно другой
сценарий.

- [ ] Выбрать hard BVP, отличный от combustion, например задачу с boundary
  layer или сильной нелинейностью.
- [ ] Проверить Frozen Lambdify против `AtomView + tcc AOT` на correctness и
  lifecycle reuse.

## P1: BVP generated code и performance methodology

### Codegen story ledger

`symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md` содержит мощную
инфраструктуру, но ряд секций все еще помечен `TODO`, хотя часть решений уже
была получена в более поздних прогонах.

- [ ] Пройти секции ledger и удалить либо заполнить устаревшие пустые
  `TODO: paste latest ...`, если их вывод уже перекрыт новыми stage tables.
- [ ] Явно помечать каждый тест как `correctness gate`, `diagnostic`,
  `cold benchmark` или `warm/prebuilt benchmark`.
- [ ] Не сравнивать `total_ms` между таблицами с различным lifecycle.

### Break-even и выбор toolchain

Текущие результаты делают `tcc` разумным первым кандидатом, но ranking зависит
от машины и структуры уравнений.

- [ ] Довести таблицы break-even для Lambdify против `tcc` по нескольким
  размерам и числу повторных solves.
- [ ] Зафиксировать отдельно cold startup и warm callback economics для
  `tcc`, `gcc`, Zig и Rust AOT.
- [ ] Исследовать масштабируемость больших Rust AOT cdylib: ранее большие
  generated crates могли падать при компиляции до runtime evaluation.

## P1: LSODE2

Status update 2026-05-31:

- [x] Added LSODE2 story registry executive summary with source-of-truth
  conclusions for AtomView, `tcc`, warm/cold AOT, Banded/Sparse and chunking.
- [x] Added LSODE2 `BuildIfMissing -> RequirePrebuilt` story test and release
  result notes.
- [x] Added LSODE2 warm `Lambdify` vs strict `tcc RequirePrebuilt` story test
  and release result notes.
- [x] Added LSODE2 large synthetic chain chunking story harness
  (`lsode2_large_chain_tcc_chunking_sparse_banded_warm_story`); release
  break-even data is still pending.
- [x] Moved LSODE2 backend API syntax notes out of story tests and into
  `IVP_USER_GUIDE_EN.md` / `IVP_USER_GUIDE_RU.md`.
- [x] Added LSODE2 non-stiff Adams corpus dashboard covering fixed Adams and
  automatic Adams/BDF routes on Sparse/Banded native paths.
- [x] Added LSODE2 symbolic-vs-pure-numerical closure dashboard covering
  Lambdify/AtomView, user analytical Jacobian closures, and FD Jacobian
  closures on Sparse/Banded native paths.
- [x] Added LSODE2 mixed-regime diagnostic story and native acceptance gate.
  `NativeSolve` now re-evaluates Adams/BDF during the full solve and observes
  real Adams -> BDF execution on the mixed-regime ramp case.

`LSODE2` уже сильно встроен в окружение и имеет собственный подробный
checklist: `numerical/LSODE2/MIRRORING_CHECKLIST.md`. Из него остаются следующие
реальные работы.

- [x] Закрыть accuracy/performance behavior Adams на non-stiff corpus, не
  ограничиваясь совпадением с аналитическим решением.
- [x] Привести раздел `Current action plan` checklist в соответствие с
  поставленными галочками: пункт про `MSBP/MXNCF` сейчас выглядит устаревшим,
  поскольку соответствующий parity test уже отмечен как locked.
- [ ] Усилить AOT infra для LSODE2: toolchain retry policy, file-lock cleanup,
  actionable spawn diagnostics и reproducibility checks.
- [ ] Окончательно разделить mandatory parity gates и advisory
  quality/performance stories.
- [x] Implement LSODE2 mid-run Adams/BDF re-evaluation for native solve.
  Locked by `lsode2_mixed_regime_ramp_native_switches_adams_to_bdf_acceptance`.
- [x] Close LSODE2 cold-rebuild method-switch handoff gap:
  native Adams/BDF switches now use a `JSTART=-1`-style handoff that preserves
  current `(t, y, h)`, step counters and available history instead of rebuilding
  the native step cycle as a fresh initial call.
- [ ] Continue full Fortran-grade method-switch trace audit:
  remaining work is side-by-side evidence for exact `METH/MUSED/MCUR/TSW/JSTART`
  ordering and `TSW` visibility on harder switch/retry windows.
- [ ] Сделать multi-run noise-robust summaries стандартом для LSODE2 story
  output, с ясно указанной методикой агрегации.

## P2: остальные IVP методы и общий API

Для Native BDF, Radau и Backward Euler уже добавлялся Numerical route через
общий `ODE_api2`, а также stiff correctness tests. Следующий этап здесь скорее
продуктовый, чем алгоритмический.

- [ ] Сверить public API и guides с фактически доступными backend/toolchain
  вариантами BDF/Radau/BE, особенно AOT Dense и numerical FD/user-Jacobian.
- [ ] Добавить story/performance coverage для тяжелых AOT routes этих stiff
  методов только там, где измерение отвечает практическому пользовательскому
  вопросу, а не дублирует BVP codegen diagnostics.
- [ ] Проверить, требуется ли аналогичный artifact lifecycle contract
  (`BuildIfMissing`/`RequirePrebuilt`) в IVP user-facing API и документации.

## P2: task documents и command interpreter

Инфраструктура dispatch IVP/BVP и executable shell уже появилась, однако
пользовательский формат документов является публичной поверхностью и требует
регрессионной дисциплины.

- [ ] Поддерживать end-to-end task-doc fixtures для IVP и BVP после изменений
  синтаксиса, substitutions и solver configuration.
- [ ] Проверить, что examples/task docs и `TASK_DOCS_GUIDE_EN.md` отражают
  актуальные ключи backend, AOT и parallel policy.
- [ ] При добавлении новых solver routes сначала расширять parser tests, затем
  примеры и README, чтобы executable не отставал от library API.

## P2: документация и presentation

- [ ] Связать README с новыми IVP/BVP user guides и task-doc guide, не
  дублируя в README длинные синтаксические детали.
- [ ] Помечать в story-ledgers старые экспериментальные выводы как
  superseded, если последующий correctness fix либо честная methodology их
  отменили.

## Постоянные правила для следующих изменений

- Correctness сначала, performance после: нельзя чинить красный тест снижением
  требований, пока не выяснено, не сломан ли алгоритм или generated callback.
- Cold, warm и prebuilt измерения являются разными экспериментами и должны
  называться явно в имени теста и заголовке таблицы.
- Forced chunking используется для доказательства корректности и изучения
  break-even; production рекомендация должна опираться на `Auto` и фактическую
  runtime-диагностику.
- Heavy release stories запускаются выборочно и фиксируются в ledger; обычные
  correctness tests должны оставаться пригодными для быстрого debug/CI цикла.

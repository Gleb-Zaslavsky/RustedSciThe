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
- `LSODE2` закрыт как основной production/mirroring фронт: ключевые
  Fortran-style switching/handoff сценарии, mandatory parity gates, story
  registry, clean pure numerical API, AOT lifecycle stories и пользовательские
  гайды уже приведены в рабочее состояние. Оставшиеся пункты по `LSODE2`
  являются hardening второго эшелона, а не блокерами нормального использования.

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

- [x] Добавить первый слой artifact lifecycle contract в registry/resolution/runtime
  path: `RegisteredAotArtifact` теперь хранит путь к generated manifest/header,
  проверяет соответствие `problem_key` manifest-derived key, сообщает наличие
  generated manifest/header и ожидаемых compiled outputs, а BVP generated
  handoff прокидывает эти поля в diagnostics. Resolver считает artifact
  `Compiled` только если manifest key согласован и compiled output существует;
  dynamic cdylib registration дополнительно отсекает manifest-key mismatch до
  попытки загрузки DLL. Это закрывает диагностический и safety слой, но не
  заменяет полноценную disk-persistent hash/ABI validation.
- [ ] Документировать место хранения артефактов, ключ проблемы и связь ключа с
  сеткой, frontend, matrix backend, toolchain и chunking policy.
- [x] Предусмотреть безопасную операцию очистки артефактов по ключу на уровне
  registry: `RegisteredAotArtifact::cleanup_generated_tree()` удаляет только
  manifest-marked generated tree, а `AotRegistry::cleanup_artifact_by_problem_key(...)`
  удаляет дерево и unregister-ит запись. Операция намеренно консервативна:
  marker manifest/header должен существовать внутри `crate_dir`, иначе cleanup
  отказывается выполнять recursive delete.
- [ ] Документировать user-facing artifact-directory workflow и решить, нужен
  ли solver-level wrapper поверх registry cleanup API.
- [ ] Определить поведение при stale artifact: проверка manifest/hash/ABI перед
  binding, а не загадочный runtime failure. Частично сделано на уровне
  registry manifest-key/existence contract; Rust/C/Zig dynamic registration
  now rejects manifest-key mismatch before loading a shared library. Остаются
  binary content hash, ABI-version marker и persistent artifact-cache validation.
- [ ] Добавить acceptance tests на missing, stale и intentionally reused
  artifact для `RequirePrebuilt`.

### AOT toolchain robustness

Внешние компиляторы и динамическая загрузка остаются источником
платформенно-зависимых сбоев.

- [ ] Добавить понятные diagnostics для отсутствующих `tcc`, `gcc` и `zig`,
  failed spawn, failed compile и failed dynamic load.
- [x] Для BVP generated handoff добавить ограниченный retry только для
  transient file-lock/spawn/build-runner failures; детерминированные compile
  errors не ретраятся. Одновременно BVP AOT lifecycle теперь сериализован в
  пределах процесса от initial selection до final runtime rebind, чтобы
  параллельные story tests не гонялись за один и тот же `problem_key` в
  глобальном linked-callback registry. 12 Core `after refactor` reruns для
  Auto chunking, combustion-3000 AtomView Banded stress и Frozen polynomial
  BuildIfMissing/RequirePrebuilt lifecycle не выявили практической просадки
  `jac_ms`, `fun_ms`, callback timings или `linear_ms`; ожидаемый overhead
  остается в cold-bootstrap/lifecycle части.
- [ ] Проверить очистку временных файлов и lock-поведение на Windows после
  неуспешной сборки/загрузки.
- [ ] Добавить determinism/reproducibility check для AOT manifest/artifact hash.

## P1: BVP Damped/Frozen - оставшиеся production gates

Status update 2026-06-01:

- [x] `BVP_DAMP_STORY_TESTS.md` теперь явно считает 12 Core release-прогоны
  source of truth, а старые 4 Core таблицы - comparison data для понимания
  влияния числа ядер, compiler/runtime шума и cold/warm lifecycle.
- [x] В начало BVP story ledger добавлена сводка production conclusions с
  привязкой к доказывающим story tests: AtomView, Banded route, `tcc`, cold vs
  warm/prebuilt lifecycle, chunking и Damped/Frozen artifact reuse.
- [x] `DampedSolverOptions` и `NRBVP` получили typed builder API для схемы
  производной: `with_scheme(BvpDerivativeScheme::...)`,
  `forward_derivative()`, `trapezoid_derivative()`, плюс legacy escape hatch
  `with_scheme_name(...)`.

### Финальная согласованность Sparse и Banded stories

После исправлений frontend/handoff важно один раз подтвердить, что старые
heavy comparisons по-прежнему рассказывают ту же историю, что новые lifecycle
tests.

- [x] Сопоставить актуальные 12 Core результаты Sparse/Banded race tables.
  `combustion_1000_lambdify_sparse_vs_banded_end_to_end_race`,
  `combustion_1000_aot_sparse_vs_banded_end_to_end_race` и
  `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`
  согласованы: correctness остается на уровне roundoff, а Banded выигрывает
  именно в linear-system и callback bind/assembly частях. Общий cold wall-clock
  может быть шумным из-за symbolic/bootstrap/toolchain стадий и не должен быть
  единственным основанием для вывода.
- [x] Связать обновленный вывод с
  `combustion_1000_end_to_end_banded_lapack_refine_statistics` после свежего
  12 Core multi-run release-прогона. Агрегированный 5-run блок добавлен в story
  ledger: все варианты `ok 5/5`, `C-tcc` конкурентен с Lambdify по total mean,
  compiled rows согласуются с Lambdify на уровне roundoff. Lapack-style Banded
  LU/refinement gate закрыт в той же методологии, что и остальные 12 Core
  Sparse/Banded выводы.
- [x] Проверить, что Banded advantage наблюдается в linear-system части для
  задач с узкой полосой и не объясняется случайным cold/warm смешением:
  актуальные 12 Core таблицы показывают меньший `linear_ms` для Banded в
  Lambdify, AOT и Auto-story маршрутах; warm/prebuilt сравнения учитываются
  отдельно от cold-build таблиц.
- [x] Пометить оставшиеся high-risk устаревшие либо дублирующие story tests как
  `superseded`/`historical`, чтобы будущий читатель не пытался интерпретировать
  старые аномальные таблицы. Низкорисковые пустые `Date/Conclusion` шаблоны
  можно чистить постепенно как документационный hygiene, но production выводы
  уже зафиксированы в Executive Summary и `Closed Production Findings`.

### Auto chunking как production gate

Forced `chunk4` нужен для диагностики, но production-default должен опираться
на `Auto` и native runtime diagnostics.

- [x] Добавить один heavy end-to-end test для `Auto` на большой задаче,
  проверяющий correctness и печатающий planned/actual jobs, fallback reason и
  work per job. Добавлен
  `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`; 12 Core
  release-результат вклеен и интерпретирован в story ledger.
- [x] Зафиксировать, что `Auto` вправе выбрать whole для малой нагрузки и
  chunked execution для достаточно тяжелой; тест не должен требовать
  конкретного выбора на любой машине без учета diagnostics. Закрыто 12 Core
  release-прогоном `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`:
  Auto выбрал Sequential/whole для combustion-1000 из-за `work_per_chunk_too_small`,
  actual jobs и fallback reasons согласованы с планом.
- [x] Обновить гайды конкретным примером чтения `aot.auto.*` и `aot.runtime.*`.
  EN/RU BVP guides теперь показывают `solver.get_statistics().diagnostics` и
  объясняют 12 Core результат `Sequential`/`work_per_chunk_too_small` как
  корректный Auto fallback, а не отказ chunking.

### Расширение Frozen coverage

Sparse и Banded combustion-1000 уже закрывают artifact lifecycle, а Banded
дополнительно закрывает тяжелую whole/chunk4 symbolic/AOT основу. Дальше
достаточно добавить не матрицу похожих тестов, а один качественно другой
сценарий.

- [x] Выбрать hard BVP, отличный от combustion, например задачу с boundary
  layer или сильной нелинейностью. Добавлен
  `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story`: это
  нелинейная polynomial-profile BVP `y = 1 + x^2` с нелинейным residual
  `z' = 2 + 0.1 * (y - (1 + x^2))^2`.
- [x] Проверить Frozen Lambdify против `AtomView + tcc AOT` на correctness и
  lifecycle reuse. 12 Core release-прогон
  `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story`
  подтвердил exact reported solution parity, `AotCompiled` selection,
  `BuildIfMissing` build row and strict `RequirePrebuilt` reuse without hidden
  compile/link.

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

Status update 2026-06-01:

- [x] Added LSODE2 story registry executive summary with source-of-truth
  conclusions for AtomView, `tcc`, warm/cold AOT, Banded/Sparse and chunking.
- [x] Added LSODE2 `BuildIfMissing -> RequirePrebuilt` story test and release
  result notes.
- [x] Added LSODE2 warm `Lambdify` vs strict `tcc RequirePrebuilt` story test
  and release result notes.
- [x] Added LSODE2 large synthetic chain chunking story harness
  (`lsode2_large_chain_tcc_chunking_sparse_banded_warm_story`). Release `n=96`
  data on the 12-core machine is green and shows no chunking win; the harness
  accepts `LSODE2_LARGE_CHUNK_DIMS` for optional larger sweeps (`192`, `384`).
- [x] Moved LSODE2 backend API syntax notes out of story tests and generic IVP
  guides into `LSODE2_USER_GUIDE_EN.md` / `LSODE2_USER_GUIDE_RU.md`.
- [x] Added LSODE2 non-stiff Adams corpus dashboard covering fixed Adams and
  automatic Adams/BDF routes on Sparse/Banded native paths.
- [x] Added LSODE2 symbolic-vs-pure-numerical closure dashboard covering
  Lambdify/AtomView, user analytical Jacobian closures, and FD Jacobian
  closures on Sparse/Banded native paths.
- [x] Added clean LSODE2 pure numerical constructors:
  `Lsode2ProblemConfig::new_numeric_fd_with_options(...)` and
  `Lsode2ProblemConfig::new_numeric_with_jacobian_options(...)`, plus short
  forms without explicit options. User-facing numerical LSODE2 examples no
  longer require symbolic `Vec<Expr>` placeholders or dummy Jacobian closures.
- [x] Added LSODE2 mixed-regime diagnostic story and native acceptance gate.
  `NativeSolve` now re-evaluates Adams/BDF during the full solve and observes
  real Adams -> BDF execution on the mixed-regime ramp case.
- [x] Updated `examples/lsode2_numerical_guide.rs` and LSODE2 EN/RU guides for
  the clean numerical constructors. The public numerical route no longer exposes
  symbolic placeholders or asks the user to pass a dummy Jacobian when finite
  differences are requested.

`LSODE2` уже сильно встроен в окружение и имеет собственный подробный
checklist: `numerical/LSODE2/MIRRORING_CHECKLIST.md`. Из него остаются следующие
реальные работы.

- [x] Закрыть accuracy/performance behavior Adams на non-stiff corpus, не
  ограничиваясь совпадением с аналитическим решением.
- [x] Привести раздел `Current action plan` checklist в соответствие с
  поставленными галочками: пункт про `MSBP/MXNCF` сейчас выглядит устаревшим,
  поскольку соответствующий parity test уже отмечен как locked.
- [ ] Hardening второго эшелона для LSODE2 AOT infra: toolchain retry policy,
  file-lock cleanup, actionable spawn diagnostics и reproducibility checks.
  Частично закрыто: `RequirePrebuilt`/missing-runtime diagnostics теперь
  печатают route, `problem_key`, build policy, codegen backend, compiler и
  output directory; transient retry classifier и сообщения после retry
  exhaustion / failed dynamic-load registration покрыты тестами. Manifest-derived
  `problem_key` reproducibility закрыта тестами на стабильность ключа и смену
  ключа при изменении backend identity/function layout. `RebuildAlways` теперь
  материализует IVP/LSODE2 AOT в изолированный подпуть и не пытается
  перезаписывать потенциально загруженный DLL/cdylib. Остаются optional disk
  cleanup для старых isolated rebuild directories и, если понадобится более
  строгий artifact-cache режим, binary artifact content hash / ABI validation.
- [x] Окончательно разделить mandatory parity gates и advisory
  quality/performance stories.
- [x] Implement LSODE2 mid-run Adams/BDF re-evaluation for native solve.
  Locked by `lsode2_mixed_regime_ramp_native_switches_adams_to_bdf_acceptance`.
- [x] Close LSODE2 cold-rebuild method-switch handoff gap:
  native Adams/BDF switches now use a `JSTART=-1`-style handoff that preserves
  current `(t, y, h)`, step counters and available history instead of rebuilding
  the native step cycle as a fresh initial call.
- [ ] Продолжить узкий Fortran-grade switch trace audit только для редких
  retry/error windows: базовая switching логика уже восстановлена, но для
  экстремальных окон еще полезны side-by-side traces с оригинальным LSODA.
- [ ] Накопить дополнительные release-измерения на новой 12-core машине и
  помечать старые 4-core выводы как comparison data, а не source of truth.
- [ ] Сделать multi-run noise-robust summaries стандартом для оставшегося
  LSODE2 story output, с ясно указанной методикой агрегации.

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

## P2: современный postprocessing API

Сейчас постпроцессинг работает, но вырос как набор отдельных функций:
`Utils::logger` пишет таблицы, `Utils::plots` рисует через plotters/gnuplot или
terminal, BVP и task parser вызывают эти функции напрямую, а LSODE2 task runner
поддерживает консервативный CSV-путь. Это нужно дополнить единым API, не ломая
старые публичные функции.

- [x] Добавить общий `PostprocessDataset`: axis/time mesh, solution matrix,
  имена переменных, имя оси и metadata/statistics.
- [x] Добавить `PostprocessAction` и `PostprocessPlan` для `txt`, `csv`,
  `plotters png`, `gnuplot png`, terminal plot и markdown/plain report.
- [x] Добавить `PostprocessReport`, который сообщает, какие файлы созданы,
  какие действия skipped/failed и почему.
- [x] Прокинуть новый фасад в BVP Damped/Frozen и LSODE2 как современный API,
  оставив legacy `save_to_file`, `save_to_csv`, `plot_result`,
  `gnuplot_result` и terminal plotting публичными для обратной совместимости.
- [x] Обновить task docs постепенно: сначала поддержать новый declarative plan
  как расширение существующего `postprocessing`, затем оставить старые ключи
  (`save_csv`, `csv_path`, `plot`) как compatibility aliases. IVP/BVP task
  runners теперь собирают `PostprocessPlan` из `save_csv/csv_path`,
  `save_txt/txt_path`, `write_report/report_path`, `plotters_png/plotters_dir`,
  `gnuplot_png/gnuplot_dir` и `terminal_plot`.
- [x] Добавить быстрые тесты на CSV/TXT/report через tempdir для facade,
  task-doc IVP/BVP routes и прямого LSODE2 solver API.
- [x] Добавить smoke tests для plotting без требования интерактивного GUI и
  без жесткой зависимости от установленного `gnuplot`. `PostprocessPlan`
  проверяет `plotters_png`, `gnuplot_png` с graceful `Skipped` при отсутствии
  бинарника и `terminal_plot`.
- [ ] Низкоприоритетно проверить legacy `Utils::plots::plots` на multi-series
  PNG-контракт. Новый фасадный smoke намеренно использует single-series dataset,
  чтобы защищать postprocessing API без привязки к старым особенностям
  именования/flush в plotters helper.

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

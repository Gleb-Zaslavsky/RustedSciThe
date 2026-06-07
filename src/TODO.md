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
- Исторический `src/numerical/BVP_Damp/BVP_traits2.rs` удален после проверки,
  что production code использует `BVP_traits.rs`.

Тело вида `self.clone()` внутри `Clone for Box<dyn Y>` выглядит как вызов
самого себя и может привести к переполнению стека при реально достигнутом пути.

- [x] Проверить, вызываются ли эти `Clone` реализации в Damped/Frozen/BVP_sci.
- [x] Заменить рекурсивную реализацию на объектно-безопасный `clone_box`/явное
  клонирование конкретного payload либо удалить недостижимую abstraction.
- [x] Добавить unit test, который действительно клонирует соответствующий
  trait object и проверяет содержимое результата.

### Закрыто: исторический `BVP_traits2` и nalgebra `Sparse_2`

- [x] Подтверждено, что production BVP Damped/Frozen runtime использует
  `BVP_traits.rs`, а `BVP_traits2.rs` был историческим экспериментом.
- [x] `BVP_traits2.rs` удален из дерева, после того как модуль был отключен из
  `BVP_Damp.rs` и targeted tests остались зелеными.
- [x] Удален активный `Sparse_2`/nalgebra `CsMatrix` runtime route из
  `YEnum`, `FunEnum`, `JacTypes`, `JacEnum` и `MatrixType`.
- [x] `GeneratedBackendConfig` больше не выбирает удаленный `Sparse_2`; старый
  provider-level `MatrixBackend::CsMatrix` безопасно переводится на production
  `Sparse`/faer route.
- [x] Док-комментарии обновлены так, чтобы `Sparse_2` не выглядел
  поддерживаемым solver backend. Standalone symbolic `CsMatrix` generator
  оставлен как legacy/reference helper, но solver selection до него не ведет.
- [x] Удален публичный `linear_sys_solvers_depot.rs`: единственный реально
  используемый helper был перенесен внутрь `BVP_traits.rs` как приватная
  dense nalgebra full/banded selection function. Это сохраняет Dense
  compatibility path, но больше не выставляет наружу ложное “depot” API.
- [x] Удален production debug-helper `jac_rowwise_printing()` и закомментированные
  вызовы из Damped/Frozen solver loops. При необходимости такую печать лучше
  вернуть как локальный test/debug helper, а не как публичную функцию traits.
- [x] `checkmem()` в BVP diagnostics больше не делает `System::new_all()` /
  `refresh_all()` из `get_statistics()`. Он оставлен как lightweight
  dense-equivalent matrix memory estimate; глобальная проверка free RAM остается
  в `task_check_mem()`, где она вызывается перед построением задачи.

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
- [x] Документировать место хранения артефактов, ключ проблемы и связь ключа с
  сеткой, frontend, matrix backend, toolchain и chunking policy. Закрыто в
  BVP Damped/Frozen user guides: добавлен workflow
  `BuildIfMissing -> RequirePrebuilt -> optional explicit cleanup`.
- [x] Предусмотреть безопасную операцию очистки артефактов по ключу на уровне
  registry: `RegisteredAotArtifact::cleanup_generated_tree()` удаляет только
  manifest-marked generated tree, а `AotRegistry::cleanup_artifact_by_problem_key(...)`
  удаляет дерево и unregister-ит запись. Операция намеренно консервативна:
  marker manifest/header должен существовать внутри `crate_dir`, иначе cleanup
  отказывается выполнять recursive delete.
- [x] Добавить user-facing solver-level wrapper поверх registry cleanup API:
  Damped/Frozen `NRBVP::cleanup_registered_aot_artifacts()` явно чистит
  registered generated trees из текущего resolver snapshot. `cleanup-on-drop`
  намеренно не добавлен: живые callbacks и загруженные DLL/shared libraries
  должны оставаться под явным контролем пользователя.
- [x] Документировать user-facing artifact-directory workflow: когда использовать
  BuildIfMissing, RequirePrebuilt, manual cleanup и почему cleanup не является
  automatic drop policy.
- [~] Определить поведение при stale artifact: проверка manifest/hash/ABI перед
  binding, а не загадочный runtime failure. Частично сделано на уровне
  registry manifest-key/existence contract; Rust/C/Zig dynamic registration
  rejects manifest-key mismatch before loading a shared library; handoff
  diagnostics print manifest key, expected outputs and contract issues. Остаются
  binary content hash, ABI-version marker и persistent artifact-cache validation.
- [x] Добавить acceptance tests на missing, stale/not-built и intentionally
  reused artifact для `RequirePrebuilt`. Закрыто без тяжелой компиляции:
  `generated_solver_handoff::require_prebuilt_errors_when_compiled_backend_is_missing`
  проверяет strict missing artifact (`AotMissing`),
  `generated_solver_handoff::require_prebuilt_errors_when_artifact_is_registered_but_not_built`
  проверяет registered-but-not-built/stale-not-ready route
  (`AotRegisteredButNotBuilt`), а
  `generated_solver_handoff::damped_solver_handoff_reuses_compiled_backend_when_only_param_values_change`
  плюс heavy lifecycle stories проверяют intentionally reused compiled backend
  без fallback в Lambdify.

### AOT toolchain robustness

Внешние компиляторы и динамическая загрузка остаются источником
платформенно-зависимых сбоев.

- [x] Добавить понятные diagnostics для отсутствующих `tcc`, `gcc` и `zig`,
  failed spawn, failed compile и failed dynamic load. BVP generated handoff now
  classifies missing/unreachable toolchain separately from transient file-lock
  failures, does not waste retry attempts on missing compiler executables, and
  prints PATH/compiler-override guidance in the build error.
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
- [x] `generated_solver_handoff` fake linked-runtime tests use a separate
  test-only registry mutex, so parallel `cargo test generated_solver_handoff`
  does not overwrite offset callbacks in the global linked-callback registry.
- [ ] Проверить очистку временных файлов и lock-поведение на Windows после
  неуспешной сборки/загрузки.
- [~] Добавить determinism/reproducibility check для AOT manifest/artifact
  hash. Manifest-derived identity уже покрыта codegen tests:
  `dense_problem_key_is_reproducible_and_tracks_backend_identity`,
  `dense_problem_key_changes_when_expressions_change_but_shape_stays_the_same`,
  `manifest_problem_key_changes_with_function_layout` и registry contract tests.
  Остается только более строгий future hardening слой: binary content hash и
  ABI-version marker для disk-persistent artifact cache.

## P1: BVP Damped/Frozen - оставшиеся production gates

Status update 2026-06-01:

- [x] `BVP_DAMP_STORY_TESTS.md` теперь явно считает 12 Core release-прогоны
  source of truth, а старые 4 Core таблицы - comparison data для понимания
  влияния числа ядер, compiler/runtime шума и cold/warm lifecycle.
- [x] В начало BVP story ledger добавлена сводка production conclusions с
  привязкой к доказывающим story tests: AtomView, Banded route, `tcc`, cold vs
  warm/prebuilt lifecycle, chunking и Damped/Frozen artifact reuse.
- [x] `DampedSolverOptions`, `FrozenSolverOptions` и оба `NRBVP` surface получили
  typed builder API для схемы производной:
  `with_scheme(BvpDerivativeScheme::...)`, `forward_derivative()`,
  `trapezoid_derivative()`, плюс legacy escape hatch `with_scheme_name(...)`.
  Frozen build request больше не hardcode-ит `"forward"`, а передает выбранную
  scheme в generated handoff.

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
- [x] Явно помечать каждый тест как `correctness gate`, `diagnostic`,
  `cold benchmark` или `warm/prebuilt benchmark`: в начало
  `BVP_CODEGEN_STORY_TESTS.md` добавлена taxonomy table для всех основных
  codegen story tests.
- [x] Не сравнивать `total_ms` между таблицами с различным lifecycle:
  `BVP_CODEGEN_STORY_TESTS.md` теперь явно предупреждает, что artifact-only,
  cold compiler, warm runtime и solver-level e2e таблицы отвечают на разные
  вопросы и не должны смешиваться как один wall-clock эксперимент.

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
- [x] Низкоприоритетно проверить legacy `Utils::plots::plots` на multi-series
  PNG-контракт. Закрыто тестом
  `legacy_plotters_plots_writes_one_png_per_series`: старый helper должен
  создавать отдельный непустой PNG для каждой колонки решения.

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
//========================================================================================
    BVP
Easy wins
1. [x] Закрыто: существование одновременно `BVP_traits.rs` и `BVP_traits2.rs`
было историческим артефактом. `BVP_traits2.rs` удален; production route
оставлен на очищенном `BVP_traits.rs`.

кроме того

Проблема: impl MatrixType for CsMatrix<f64> (nalgebra sparse) содержит mul() и solve_sys(), которые всегда паникуют через unreachable!(). Этот backend не используется — Sparse_2 вариант в YEnum/JacTypes присутствует, но ни один solver path его не активирует. Код занимает место, создаёт ложное впечатление поддержки nalgebra sparse.

Решение: [x] Удалено из активного runtime path: `CsMatrix<f64>` impl,
`Sparse_2` варианты `YEnum`/`JacTypes`/`FunEnum`/`JacEnum`, а также module
`BVP_traits2.rs`. Проверено targeted tests и `cargo check --lib`.

VP_traits.rs vs BVP_traits2.rs — две параллельные trait системы
Файлы: BVP_traits.rs:1-1420, BVP_traits2.rs:1-1178

Проблема: Два файла реализуют одни и те же концепции (VectorType, MatrixType, Fun, Jac, Y), но разными способами:

BVP_traits.rs — ручная диспетчеризация через match и downcast_ref с паникой при несовпадении типа.


Либо Выбрать один файл как основной, добавить в него BandedMatrixType, перевести JacEnum на #[enum_dispatch], удалить BVP_traits.rs. Или наоборот — удалить BVP_traits2.rs и оставить ручную диспетчеризацию, если enum_dispatch не даёт измеримого выигрыша. Тут надо обдумать


Проблема: YEnum и FunEnum используют #[enum_dispatch], но JacEnum — нет. В impl Jac for JacEnum диспетчеризация ручная через match. 
ПРИОРИТЕТ ЗДЕСЬ НЕ ГИБКОСТЬ ДЛЯ ДОБАВЛЕНИЯ НОВЫХ БЭКЕНДОВ - А ПЕРФОМАНС.

2. [x] linear_sys_solvers_depot.rs
Файл: linear_sys_solvers_depot.rs:1-51
Это наследие прежней архитектуры сейчас решение о том какой солвер линейной алгебры и какой тулчейн албебры Sparse, Banded, etc. принимается в другом месте. 
Решение: удален как публичный модуль; единственная использованная функция
перенесена в `BVP_traits.rs` как приватный helper для legacy Dense nalgebra
`solve_sys`.
3. [x] checkmem() — тяжёлый системный вызов в горячем пути
Файл: BVP_utils.rs

Проблема: checkmem() использует sysinfo::System — это полноценный системный вызов, который сканирует /proc (Linux) или использует Win32 API. Вызывается из get_statistics(), которая может дёргаться после каждой итерации Newton. На Windows это может быть особенно дорого.

Решение: `checkmem()` упрощен до lightweight dense-equivalent estimate без
обращения к системной памяти. `task_check_mem()` сохраняет системную проверку
free RAM как preflight warning перед генерацией задачи.

Риск: Низкий. Изменение затрагивает только diagnostic output.

4. [x] jac_rowwise_printing() — debug-функция в production коде
Файлы: BVP_traits.rs:1347-1369, BVP_traits2.rs:1105-1127

Проблема: Функция jac_rowwise_printing() — это debug-helper, который печатает матрицу построчно через println!. Находится в production trait-файлах, а не в тестовом модуле или debug-утилите.

Решение: удалена как неиспользуемая; оставались только закомментированные вызовы.
Если снова понадобится row-wise dump, его следует добавить локально в test/debug
модуль.

Риск: Очень низкий.


5. NRBVP struct дублируется в damped и frozen модулях
Файлы: NR_Damp_solver_damped.rs, NR_Damp_solver_frozen.rs

Проблема: Оба файла определяют struct NRBVP с ~40 полями, многие из которых идентичны (сетка, решение, параметры, таймеры, statistics). Это дублирование ведёт к расхождению: если в damped добавляется новое поле, frozen может его не получить.

Статус: частично закрыто phase-0. Добавлен `solver_common.rs` с общими
defaults, mesh constructors и legacy placeholder state. Поведение не менялось:
Damped сохраняет контракт `n_steps -> n_steps + 1` mesh points, Frozen сохраняет
контракт `n_steps -> n_steps` mesh points. Полная миграция общего `NRBVP`
state пока намеренно не сделана, потому что риск средний и требует отдельного
тестового прохода.

Решение: Выделить общую часть NRBVP в базовый struct (например, BvpSolverBase) в отдельном файле, а damped/frozen-specific поля добавить через композицию или generic.

Риск: Средний. Потребуется рефакторинг всех impl-блоков и конструкторов. Но это улучшит поддерживаемость.

10. DampedSolverOptions и FrozenSolverOptions — дублирование builder API
Файлы: NR_Damp_solver_damped.rs, NR_Damp_solver_frozen.rs

Проблема: Оба options-типа имеют ~30 методов builder API с одинаковой семантикой (сетка, толерантности, backend config, derivative scheme, postprocessing). Код дублируется.

Статус: частично закрыто. Derivative scheme больше не является raw-string gap:
Damped и Frozen имеют одинаковый typed builder surface, а tests проверяют, что
scheme попадает в solver/generated request. Оставшееся решение: выделить общий
BvpSolverOptions с общей частью, а специфичные для Damped/Frozen параметры
(damping factor, frozen strategy) добавить через extension traits или generic.
Phase-0 также вынес общие default values (`forward`, `Sparse`, `Dense`,
`max_iterations = 25`) в `solver_common.rs`, чтобы Damped/Frozen не расходились
по базовым presets.

Риск: Средний. Аналогично NRBVP.

11. [x] bound_step_Cantera(), bound_step_Cantera2(), bound_step() — три bound-step функции
Файл: BVP_utils_damped.rs:1-224

Проблема: Три функции с похожей логикой ограничения шага. bound_step_Cantera2() — это копия bound_step_Cantera() с небольшими отличиями. bound_step() — третий вариант. Различия не документированы.

Решение: [x] Реальный Damped production path проверен: solver использует
`bound_step_Cantera2(x, step, bounds)` с additive contract
`x_next = x + lambda * step`. Формулы не объединялись механически, потому что
legacy `bound_step(...)` использует другое соглашение о знаке step. Добавлены
characterization tests для no-clipping, upper/lower clipping, most restrictive
component, outward step from current boundary value, invariant "scaled step stays
inside bounds" и отдельный тест, фиксирующий отличие legacy twopnt contract.

Риск: Низкий. Production behavior не изменен; теперь есть regression guard на
самую критичную часть bounds/damping semantics.

12. [x] ОЧЕНЬ ВАЖНО! Parallel marking anti-pattern в adaptive_grid_basic.rs
Файл: adaptive_grid_basic.rs:159-208 (и все *_par функции)

Проблема: Параллельные версии grid refinement используют Mutex<HashMap<usize, ...>> для сбора маркеров от нескольких потоков. Это означает, что каждый поток блокируется при вставке, и накладные расходы на Mutex + HashMap могут превысить выигрыш от параллелизма для типичных размеров сетки (100-1000 интервалов).

Решение: [x] `easy_grid_refinement_par`, `pearson_grid_refinement_par` и
`grcar_smooke_grid_refinement_par` больше не используют `Mutex`/`par_bridge`.
Каждая строка строит локальный mark list через `into_par_iter()`, затем один
deterministic sequential merge воспроизводит legacy semantics. Добавлен быстрый
parity-test на coarse mesh (`10-20` точек как practically meaningful стартовая
сетка) и ignored benchmark-table для release-замеров.
Correctness coverage усилен отдельными invariant tests: refined grid должен
оставаться строго возрастающим, сохранять обе границы и все старые узлы,
`inserted` обязан совпадать с приростом длины сетки, а initial guess в новых
точках проверяется как линейная интерполяция между соседними старыми узлами.
Покрыты `refine_all`, `easy`, `pearson`, `grcar_smooke`, SciPy-style residual
thresholds и no-refinement identity case для плоского решения.

Риск: Низкий после parity test. Более глубокое объединение construction phase
остается в пункте 13.

Complex refactorings
13. [x] Massive code duplication в adaptive_grid_basic.rs — 5 алгоритмов x 2 (seq/par)
Файл: adaptive_grid_basic.rs:1-1254

Проблема: Пять алгоритмов адаптивного сгущения сетки (refine_all, easy, pearson, grcar_smooke, scipy), каждый в sequential и parallel версии. Это 10 функций с огромным дублированием кода. Алгоритмы различаются только в marking phase (какие интервалы сгущать), но construction phase (построение новой сетки с интерполяцией) идентична.

Решение: [x] Construction phase выделена в единый
`construct_refined_grid_from_marks(...)`: все алгоритмы теперь строят refined
grid и interpolated initial guess через один проверяемый путь. Это убрало
основную опасную копипасту: сохранение старых узлов, границ, порядок новых
точек, `inserted` count и линейную интерполяцию.

Marking phase пока намеренно оставлена в алгоритмах: критерии `easy`,
`pearson`, `grcar_smooke` и SciPy-style residual refinement различаются
математически, и дальнейшая унификация должна идти только после дополнительных
story/correctness tests. Parallelism применяется к marking phase без
`Mutex<HashMap>`; construction остается deterministic sequential assembly.

Риск: Низкий после targeted invariant coverage. Проверено `cargo test --lib
adaptive_grid -- --nocapture`, ignored benchmark-table и `cargo check --lib`.


14. Test file size explosion — BVP_Damp test story modules
Статус: частично закрыто. Тесты вынесены в `src/numerical/BVP_Damp/tests/`, подключены через `#[path = ...]` и `#[cfg(test)]`, чтобы не тянуть story-модули в production-сборку. Имена Rust-модулей временно сохранены для стабильности `cargo test` paths и существующего story ledger.

Файлы: `tests/aot_diagnostics.rs`, `tests/aot_race_stress.rs`, `tests/backend_compare.rs`, `tests/basic_correctness.rs`, `tests/classic_examples.rs`, `tests/common.rs`

Проблема: Два тестовых файла содержат ~11000 строк кода. BVP_Damp_tests3.rs — heavy diagnostic tests с IsolatedColdMetrics, RuntimeTuningSample, RuntimeTuningSummary, RuntimeTuningToolchain. BVP_Damp_tests4.rs — race/stress tests с RaceVariant, RaceRow, RaceSummaryRow, Aggregate. Это замедляет компиляцию тестов и усложняет навигацию.

Оставшееся решение: глубже разделить два самых крупных файла на тематические подмодули:

bvp_damp_diagnostic_tests.rs — diagnostic infrastructure
bvp_damp_race_tests.rs — race/stress tests
bvp_damp_toolchain_tests.rs — toolchain comparison tests
bvp_damp_aggregation_tests.rs — multi-run aggregation tests
Риск: Средний. Тесты слабо связаны между собой, но story helpers и isolated-child protocol нужно переносить осторожно, чтобы не потерять тесты и не изменить измеряемую методологию.
15. BVP_DAMP_STORY_TESTS.md — 4500 строк, монолитный ledger
Файл: BVP_DAMP_STORY_TESTS.md:1-4500

Проблема: Огромный markdown-файл, содержащий executive summary, 12 Core production conclusions, и ~50 per-test result blocks. Сложно навигировать, тяжело поддерживать.

Решение: Разделить на:

BVP_DAMP_STORY_SUMMARY.md — executive summary, 12 Core conclusions, backend vocabulary
BVP_DAMP_STORY_TESTS_SPARSE.md — sparse-related story tests
BVP_DAMP_STORY_TESTS_BANDED.md — banded-related story tests
BVP_DAMP_STORY_TESTS_FROZEN.md — frozen-related story tests
BVP_DAMP_STORY_TESTS_CHUNKING.md — chunking-related story tests
Риск: Низкий. Markdown-файлы, не влияют на компиляцию.

16. View subsystem integration — atom-based engine vs ExprLegacy
Файлы: View/lib.rs:1-46, symbolic_functions_BVP.rs

Проблема: В проекте существует два symbolic assembly backend: AtomView (новый, atom-based) и ExprLegacy (старый, expression-based). AtomView — это отдельный crate в src/symbolic/View/, который имеет свою копию codegen IR (CodegenIR_atom.rs), свой парсер, свой нормализатор. При этом ExprLegacy всё ещё используется по умолчанию в некоторых путях. Миграционный путь не документирован.

Решение:

Задокументировать миграционный план: какие модули переходят на AtomView, какие остаются на ExprLegacy.
Устранить дублирование codegen IR между CodegenIR.rs и CodegenIR_atom.rs.
Сделать AtomView default backend, ExprLegacy — compatibility path.
Риск: Высокий. Затрагивает symbolic engine, codegen, solver callbacks. Может потребовать перезаписи тестовых эталонов.

17. [mostly closed] ОЧЕНЬ ВАЖНО CodegenIR/AtomView CSE optimization
Файлы: CodegenIR.rs, CodegenIR_atom.rs, generated_solver_handoff.rs,
NR_Damp_solver_damped.rs, NR_Damp_solver_frozen.rs

Статус: AtomView CSE теперь имеет явный production/diagnostic переключатель:
`AtomOptimizationProfile::Full` сохраняет CSE-enabled поведение, а
`AtomOptimizationProfile::NoCse` реально отключает CSE при batch-lowering.
Профиль прокинут через `GeneratedBackendConfig`, Damped/Frozen builder API и
solver-level setters/getters. Добавлены correctness tests на сохранение значений
и на то, что `NoCse` действительно уменьшает Atom lowering work, но раздувает
generated source.

Full solver-level story `combustion_1000_banded_atomview_tcc_cse_profile_end_to_end_story`
закрывает главный production-вопрос: `Full` и `NoCse` оба корректны на реальном
Damped BVP solve, включая compile/link, rebind, Newton, callback timings и
solution diff. `NoCse` показал немного меньший mean wall-clock на
combustion-1000, но этот выигрыш частично объясняется compile-link выбросом у
`Full`; artifact/module/lowering/source stages не дают структурного выигрыша, а
hot residual callback у `NoCse` немного медленнее. Поэтому `Full` остается
production default, а `NoCse` - diagnostic/experimental profile. Дополнительная
работа здесь нужна только если захотим отдельный larger-grid или 5-10 run story.

Отдельный возможный долг: старый ExprLegacy/LinearBlockPlan CSE слой все еще
нужно рассматривать отдельно. `reuse_temps()` не равен полноценному удалению
общих подвыражений; если захотим оптимизировать ExprLegacy route дальше, нужен
отдельный CSE pass с parity tests и возможностью отключения.

Риск: Средний/высокий. CSE может изменить порядок вычислений и повлиять на
numerical accuracy из-за FP non-associativity. Поэтому любые изменения CSE
политики должны идти через explicit profile + correctness/story tests.

18. [x] Auto chunking maturity — production defaults use Auto, forced chunking is diagnostic
Файл: codegen_orchestrator.rs:324-372

Старый риск был в том, что `auto_parallel_recommendation()` и
`recommended_sparse_auto_parallel_plan()` существовали, но production presets
могли по умолчанию тащить forced `chunk4`. Актуальная проверка показывает, что
это уже не так: `GeneratedBackendConfig` по умолчанию использует
`AotExecutionPolicy::Auto`, а `AotChunkingPolicy::default()` не содержит
явного residual/sparse override. `DampedSolverOptions::{sparse_damped,
banded_damped}` и `FrozenSolverOptions::{sparse_frozen, banded_frozen}`
сохраняют этот контракт.

Решение: добавлены regression tests, фиксирующие, что production presets
остаются на `Auto` execution и пустой chunking policy. Forced `chunk4`,
`chunk8`, custom `ByTargetChunkCount` и explicit `Parallel(...)` остаются
доступными через builder/API, но используются как diagnostic/performance
experiments, а не как public default.

Оставшийся риск перенесен из "incorrect default" в normal tuning: Auto может
выбрать sequential для малых задач, что корректно; если новая машина меняет
break-even, это проверяется story tests и runtime diagnostics `aot.auto.*` /
`aot.runtime.*`, а не forced-default policy.

19. [x] AOT artifact cleanup — cleanup API существует, но нет user-facing workflow
Файл: generated_solver_handoff.rs

Проблема: RegisteredAotArtifact::cleanup_generated_tree() и AotRegistry::cleanup_artifact_by_problem_key() существуют. Нужно решить нужен ли  solver-level метод для пользователя типа метод NRBVP::cleanup_aot_artifacts() или DampedSolverOptions::with_cleanup_on_drop()? Либо такой метод только сделает систему уязвимей? 

Решение: добавлен явный solver-level метод
`NRBVP::cleanup_registered_aot_artifacts()` для Damped и Frozen. Он чистит
только artifacts, зарегистрированные в текущем `AotResolver`, и возвращает
количество удаленных записей. Автоматический cleanup-on-drop не добавлен:
это было бы опасно для одновременно живущих compiled callbacks и Windows DLL
lifecycle. Оставшееся действие перенесено в документацию artifact workflow.

20. Toolchain robustness — stale artifact detection и binary content hash validation
Файл: generated_solver_handoff.rs

Статус: частично закрыто. Retry policy для transient failures есть; missing
toolchain/failed spawn теперь классифицируется отдельно и не маскируется под
Windows file-lock transient failure. Manifest-key mismatch проверяется до
dynamic load для Rust/C/Zig runtime registration.

Остаются hardening-опции:

Stale artifact detection (артефакт существует, но собран с другой версией кода)
Binary content hash validation (проверка, что .dll/.so не повреждён)
ABI-version marker (проверка совместимости ABI при динамической загрузке)
Это лишние опции покрыавающие случаи которые едва ли могут появиться в реальной практике или нет?
Нужно ли добавить content hash в manifest, проверять при registration? Добавить ABI version string в exported symbols?

Риск: Средний. Изменение формата manifest может потребовать пересборки всех существующих артефактов.
//=================================================================================================================================

LSODE2

LSODE2: Архитектурное ревью — от быстрых побед до глубоких рефакторингов
Как читать
Каждый пункт имеет метку P1/P2/P3 (приоритет), оценку усилий (дни/недели) и выгоду. Сортировка — от быстрых побед с высокой отдачей к сложным рефакторингам, требующим взвешенного решения.

1.ОЧЕНЬ ВАЖНО finite_difference_jacobian_from_residual() строит полную плотную матрицу, затем конвертирует в sparse/banded
Файл: native_step_engine.rs

Проблема: Метод вычисляет FD-якобиан как DMatrix<f64>, затем вызывает .to_sparse() / .to_banded(). Для систем с N=1000 и разреженностью 1% это означает O(N²) памяти и времени на плотную матрицу, хотя нужны только O(nnz) элементов.

Оценка: P1, ~1 день

Выгода: Прямое ускорение FD-ветки для sparse/banded — до 10-100x по памяти, до 5-10x по времени на больших системах.

Предложение: Вычислять FD только для ненулевых позиций: для каждой ненулевой структуры (i,j) сделать одно возмущение j-й переменной и считать только i-й остаток. Для banded — возмущать полосу за раз (как оригинальный ODEPACK).

2. Lsode2NativeStatistics — ~80 полей с ручным merge
Файл: statistics.rs

Проблема: merge_native_statistics() в solver.rs вручную копирует ~80 полей через self.field = max(self.field, other.field). Добавление нового счётчика требует правки ~5 мест: определение поля, инициализация, increment, merge, table_report.

Оценка: P1, ~2 дня

Выгода: Устранение хрупкости при расширении статистики.

Предложение: Ввести HashMap<String, u64> или битовый массив счётчиков с дескрипторами. Либо макрос define_counters!, генерирующий struct + merge + report.

3. Lsode2ProblemConfig — ~20 полей с двунаправленной синхронизацией
Файл: config.rs

Проблема: sync_legacy_backend_from_policy() и sync_policy_fields_from_legacy_backend() — две функции, синхронизирующие одни и те же данные в противоположных направлениях. Вызов в неправильном порядке даёт inconsistent state.

Оценка: P1, ~1 день

Выгода: Устранение класса багов "тихая потеря конфигурации".

Предложение: Заменить двунаправленную синхронизацию на единый ResolvedPlan, который строится из policy-полей один раз в resolve() и дальше используется read-only. Убрать мутабельные sync-функции.

4. Lsode2ControllerConfig — дублирование инициализации в конструкторах
Файл: algorithm.rs

Проблема: adams_only(), bdf_only(), automatic_adams_bdf() — три конструктора, каждый вручную заполняет одни и те же поля с минимальными отличиями.

Оценка: P1, ~4 часа

Выгода: Упрощение поддержки, устранение copy-paste.

Предложение: Единый конструктор с enum Lsode2AlgorithmMode { AdamsOnly, BdfOnly, Automatic }, остальные поля — defaults с builder.

5. Lsode2BackendConfig — ~20 preset-методов с одинаковым паттерном
Файл: config.rs

Проблема: dense_symbolic_defaults(), native_sparse_faer(), native_banded_faithful_aot_c_tcc() и т.д. — каждый повторяет Self { jacobian_backend: ..., linear_solver_backend: ..., generated_backend: ... }.

Оценка: P1, ~4 часа

Выгода: Сокращение кода на ~300 строк, единая точка изменений.

Предложение: Таблица (name, jacobian, linear, generated) + метод from_preset(name). Или константные ассоциации.

6. Lsode2ProblemConfig builder — ~40 методов, многие тривиальные
Файл: config.rs

Проблема: with_backend(), with_linear_solver_policy(), with_native_*() — большинство просто устанавливают одно поле и возвращают self.

Оценка: P2, ~1 день

Выгода: Сокращение boilerplate, улучшение discoverability.

Предложение: Использовать #[derive(bon::Builder)] или derive_builder для автоматической генерации builder-методов. Либо сгруппировать в with_native_config(config: Lsode2NativeExecutionConfig).

7. Lsode2NativeStepEngine — enum-диспетчеризация с ручным match по всем вариантам
Файл: native_step_engine.rs

Проблема: Lsode2NativeStepEngine — enum с тремя вариантами (Dense/Sparse/Banded). Каждая операция требует match по всем вариантам. Добавление нового backend (например, GpuDense) требует правки ~10 match-сайтов.

Оценка: P2, ~2 дня

Выгода: Open-closed principle: новый backend = новый вариант без правки существующего кода.

Предложение: Выделить трейт Lsode2StepEngineBackend с методами factorize(), solve(), jacobian_storage_type(). Сделать Lsode2NativeStepEngine тонкой обёрткой Box<dyn Lsode2StepEngineBackend>. Либо сохранить enum, но генерировать match через макрос.

8. solve() — сложное ветвление с 4 режимами и 2 probe-потоками
Файл: solver.rs

Проблема: solve() диспетчеризует NativeSolve | BridgeSolve | ProbeBeforeBridge | Disabled. Внутри NativeSolve есть два probe-потока (LSODA-стиль и прямой). Вложенные замыкания с Rc<RefCell<...>> для передачи контекста.

Оценка: P2, ~3 дня

Выгода: Читаемость, тестируемость, снижение вероятности багов в flow.

Предложение: State machine с явными состояниями: Idle -> ProbingAdams -> ProbingBdf -> NativeAdams -> NativeBdf -> Bridge -> Finished. Каждый переход — отдельная функция. Убрать Rc<RefCell<...>> в пользу &mut self через enum state.

9. NativeStepResidualContext — Rc<RefCell<Option<...>>> для передачи контекста между замыканиями
Файл: native_step_engine.rs

Проблема: Для передачи residual/jacobian между замыканиями используется Rc<RefCell<Option<NativeStepResidualContext>>>. Это runtime-динамика с потенциальным panic!() при borrow().

Оценка: P2, ~1 день

Выгода: Безопасность, производительность (убираем RC-счётчики на каждом шаге).

Предложение: Если контекст известен на момент создания executor'а — передавать напрямую. Если нет — использовать OnceCell или Option с take().

10. with_lsode2_sparse_jacobian_artifact_suffix() — хрупкое именование AOT-артефактов
Файл: native_jacobian.rs

Проблема: Добавление суффикса _sj к имени артефакта, чтобы избежать коллизии с BVP AOT. Это соглашение, а не контракт — легко сломать при рефакторинге.

Оценка: P2, ~1 день

Выгода: Надёжность AOT-кэширования.

Предложение: Включить solver_kind: SolverKind в ProblemKey (BVP vs IVP vs LSODE2). Тогда суффикс генерируется автоматически из ключа, а не из хрупкой строки.

11. Lsode2SwitchTelemetryHints — потеря информации при max()-агрегации
Файл: solver.rs

Проблема: При сборе сигналов из Adams и BDF веток используется max() — если Adams дал ratio=50, а BDF ratio=200, то в telemetry попадёт 200, но будет неизвестно, какая семья его произвела.

Оценка: P2, ~1 день

Выгода: Точная диагностика при выборе метода.

Предложение: Хранить (value, source_family) для каждого сигнала. Или вести две копии: adams_hint и bdf_hint.

12. statistics() — слияние bridge и native статистики через max() может скрыть реальный путь
Файл: solver.rs

Проблема: Метод statistics() возвращает объединённую статистику, где n_steps = max(bridge.n_steps, native.n_steps). Если реально работал bridge, но native случайно имеет ненулевые поля — картина искажается.

Оценка: P2, ~2 дня

Выгода: Прозрачная диагностика.

Предложение: Хранить active_path: enum { Bridge, Native, Probe } и при слиянии брать поля только активного пути. Либо возвращать Either<BridgeStats, NativeStats>.

13. Lsode2SolveSummary — ~20 полей с глубоким optional-вложением
Файл: solver.rs

Проблема: native_integration_solve, native_integration_preview, native_step_probe — три Option-поля, из которых заполнено только одно. Пользователь должен знать, какое читать.

Оценка: P2, ~1 день

Выгода: Упрощение API для пользователя.

Предложение: Заменить на enum Lsode2SolveOutcome { NativeSolve(...), NativePreview(...), BridgeSolve(...), ProbeResult(...) }.

14. Тестовые файлы-монстры: story_tests2.rs (5500 строк), tests.rs (3822), parity_micro.rs (2224), step_cycle.rs (2186)
Файлы: story_tests2.rs, tests.rs, parity_micro.rs, step_cycle.rs

Проблема: Файлы >2000 строк тяжело навигировать, мерж-конфликты практически неразрешимы, cargo test не показывает, какой именно тест упал без фильтра.

Оценка: P2, ~2 дня

Выгода: Ускорение CI (лучший кэш инкрементальной компиляции), улучшение DX.

Предложение: Разбить по модулям: tests/adams_parity.rs, tests/bdf_parity.rs, tests/switch_parity.rs, tests/story_combustion.rs, tests/story_sublimation.rs. Использовать mod tests; в tests/ директории.

15. Bridge-путь (legacy BDF) всё ещё существует параллельно с Native-путём
Файлы: solver.rs, config.rs

Проблема: Lsode2Solver может делегировать BdfOdeSolver (bridge) или использовать полный LSODE2 native path. Bridge — это legacy-путь, который дублирует функциональность и требует отдельной поддержки.

Оценка: P3, ~1 неделя

Выгода: Устранение ~30% кода LSODE2, единый путь тестирования.

Предложение: После подтверждения, что native path покрывает все сценарии bridge (включая AOT), удалить bridge path. Interim: сделать bridge path deprecated с warning.

16. step_cycle.rs (2186 строк) — слишком много ответственности
Файл: step_cycle.rs

Проблема: Lsode2StepCycle владеет state, error_control, dstoda_state, iteration_mode, method, adams_pdest/pdlast. Методы: predict, finish_with_local_error, finish_with_correction, finish_after_converged_correction, apply_corrector_failure_policy, select_post_accept_order, select_and_apply_error_test_retry, select_dstoda_order_and_growth, coefficient_ratio_h_el1_for_method. Это ~10 ответственностей в одном классе.

Оценка: P3, ~3 дня

Выгода: Тестируемость, изоляция DSTODA-логики.

Предложение: Выделить DstodaOrderSelector, DstodaErrorTestHandler, DstodaCorrectorFailureHandler. Lsode2StepCycle становится оркестратором, делегирующим специализированным структурам.

17. nonlinear_driver.rs (972 строк) — коррекция, цикл, итерация, статистика
Файл: nonlinear_driver.rs

Проблема: Lsode2NonlinearStepDriver владеет step_cycle, correction_controller, statistics, iteration_mode. Методы: begin_step, submit_correction, compute_and_submit_correction_with_refresh_policy, compute_apply_and_submit_correction_with_refresh_policy, reject_after_nonlinear_failure, retry_after_stale_jacobian_nonlinear_failure.

Оценка: P3, ~2 дня

Выгода: Разделение correction loop и step orchestration.

Предложение: Выделить CorrectionLoop (отвечает только за итерации коррекции) и NonlinearStepOrchestrator (координирует correction + error test + order selection).

18. native_executor.rs — кэширование линеаризации через (t, c)-ключ
Файл: native_executor.rs

Проблема: has_current_linearization(t, c) проверяет, совпадает ли (t, c) с предыдущим. Если нет — пересобирает якобиан и факторизацию. Но c — это срез &[f64], сравнение идёт поэлементно. Для N=1000 это O(N) на каждом шаге.

Оценка: P3, ~2 дня

Выгода: Ускорение на больших системах, где линеаризация редко меняется.

Предложение: Использовать хэш состояния (например, blake3 от (t, y)) или индекс итерации Ньютона. Полное поэлементное сравнение — только если хэш совпал (защита от коллизий).

19. order_selection.rs — BDF-стиль RHUP/RHSM/RHDN без учёта Adams-стабильности
Файл: order_selection.rs

Проблема: select_bdf_like_order() использует ODEPACK-формулы RH, но Adams имеет stability-limited order (SM1/PDLAST), которые обрабатываются в step_cycle.rs отдельно. Логика размазана между двумя файлами.

Оценка: P3, ~2 дня

Выгода: Единая точка принятия решения о порядке.

Предложение: Объединить select_bdf_like_order и Adams stability-limiting в один OrderSelector, который знает текущий method family и применяет соответствующие ограничения.

20. method_switch.rs — probe gate с ICOUNT и step-advantage тест
Файл: method_switch.rs

Проблема: Логика переключения Adams↔BDF корректна, но preferred_family_and_reason_with_probe_gate_and_current() — функция с ~100 строками и 4 уровнями вложенных условий. Сложно верифицировать соответствие оригинальному LSODA.

Оценка: P3, ~2 дня

Выгода: Верифицируемость Fortran parity.

Предложение: Разбить на: probe_gate_decision(), step_advantage_test(), cost_based_preference(), stiffness_override(), convergence_override(). Каждая возвращает Option<MethodSwitchDecision>, главная функция комбинирует через chain или select_first.

21. parity_micro.rs (2224 строк) — label-by-label replay матрица
Файл: parity_micro.rs

Проблема: Ручная запись label-трасс (410, 430, 500, 620, 640, 670) с ODEPACK-стилем. Каждый тест воспроизводит конкретный сценарий из Fortran-кода. При изменении DSTODA-логики тесты могут требовать обновления, но их 2224 строки.

Оценка: P3, ~3 дня

Выгода: Упрощение поддержки parity-тестов.

Предложение: Ввести декларативный формат: label_trace!({410 => (jcur=1, ipup=0), 430 => ...}). Генерировать тесты из таблицы. Либо параметризовать через #[test_case].

22. MIRRORING_CHECKLIST.md — 192 строки ручного чеклиста
Файл: MIRRORING_CHECKLIST.md

Проблема: Чеклист Fortran-паритета ведётся вручную. Легко устаревает.

Оценка: P3, ~2 дня

Выгода: Автоматическая верификация паритета.

Предложение: Превратить в build.rs скрипт или cargo test --ignored, который запускает эталонные Fortran-трассы (если доступен бинарник) и сравнивает label-последовательности.

23. adams_engine.rs + dcfode.rs — два файла для коэффициентов
Файлы: adams_engine.rs, dcfode.rs

Проблема: Adams и BDF коэффициенты разнесены в разные файлы, хотя имеют идентичную структуру (ELCO/TESCO).

Оценка: P3, ~1 день

Выгода: Консистентность, единый трейт DcfodeTables.

Предложение: Ввести trait DcfodeTables { fn elco(order: usize) -> &[f64]; fn tesco(order: usize) -> (f64, f64, f64); }. Реализации: AdamsDcfodeTables и BdfDcfodeTables.

24. history.rs — backward_differences_to_nordsieck() только для orders 1..=5
Файл: history.rs

Проблема: Матрицы конвертации backward differences → Nordsieck хардкодом только для orders 1..=5. BDF может использовать order до 5, Adams до 12. Для order >5 конвертация не поддерживается.
Нужно посмотреть как в LSODA

 Добавить матрицы для orders 6..=12, либо генерировать через Pascal transform?

25. linear_backends.rs — три backend с разными crate-зависимостями
Файл: linear_backends.rs

Проблема: Dense (nalgebra), Sparse (faer), Banded (ручная LAPACK-стиль). Каждый backend тянет свою цепочку зависимостей. DirectSolverFactorization<S> — generic-обёртка, но не все операции одинаково эффективны для всех backend'ов.

Оценка: P3, ~3 дня

Выгода: Возможность добавлять новые linear solvers без изменения кода LSODE2.

Предложение: Выделить трейт LinearSolverBackend с методами factorize(&mut self, matrix: ...) -> Result<()> и solve(&self, rhs: &mut [f64]) -> Result<()>. Три существующих имплементации. Lsode2NativeStepEngine параметризуется impl LinearSolverBackend.
//===================================================================================================================================================================================


BVP_Sci
Архитектурное ревью: src/numerical/BVP_sci/
Общий вывод
BVP_sci — это работоспособный Rust-порт SciPy _bvp.py, но его архитектурный уровень значительно ниже BVP_Damp. Если BVP_Damp — production-grade фреймворк с полным покрытием Sparse/Banded/Dense, AOT с chunking/parallel, двумя symbolic бэкендами и pure numerical путём, то BVP_sci — well-engineered prototype, которому не хватает ~40% функциональности для паритета. Ключевые пробелы: отсутствие Banded-бэкенда, отсутствие Frozen Newton, отсутствие chunking/parallel в AOT, дублирование кода между faer/nalgebra версиями, и отсутствие интеграции с адаптивным сеточным фреймворком BVP_Damp.

P1: Быстрые победы (часы — 2 дня)
1. solve_newton() читает параметры из BVPResult::default() на каждом вызове
Файл: BVP_sci_faer.rs:847-880

Проблема: Каждый вызов solve_newton() создаёт новый BVPResult::default() и извлекает max_iter, max_njev, sigma, tau, n_trial из него. Это означает, что параметры стратегии зашиты в Default и не могут быть переопределены без изменения констант.


rust


// BVP_sci_faer.rs ~847
let default_result = BVPResult::default();
let max_iter = default_result.max_iter;
let max_njev = default_result.max_njev;
let sigma = default_result.sigma;
let tau = default_result.tau;
let n_trial = default_result.n_trial;
Решение: Сделать эти параметры аргументами solve_newton() с значениями по умолчанию.

Выгода: Устраняет технический долг, даёт caller-у контроль над стратегией. Оценка: 30 минут.

2. Дублирование кода между solve_bvp() и solve_bvp_sparse()
Файл: BVP_sci_faer.rs:1830

Проблема: Две функции (~200 строк каждая) почти идентичны. Различаются только комментарием в сигнатуре. solve_bvp() вызывает solve_bvp_sparse() внутри, но весь код подготовки продублирован.


rust


pub fn solve_bvp(ode, bc, x, y_init, tol, max_nodes, verbose, bc_tol, ...) -> BVPResult {
    // ~200 строк подготовки, вызова solve_newton, mesh refinement
    solve_bvp_sparse(ode, bc, x, y_init, tol, max_nodes, verbose, bc_tol, ...)
}

pub fn solve_bvp_sparse(ode, bc, x, y_init, tol, max_nodes, verbose, bc_tol, ...) -> BVPResult {
    // ТЕ ЖЕ САМЫЕ ~200 строк подготовки
}
Решение: Выделить общую логику в solve_bvp_impl(), оставить solve_bvp() и solve_bvp_sparse() как тонкие обёртки.

Выгода: Устраняет ~200 строк дублирования, снижает риск расхождения. Оценка: 1 час.

3. construct_global_jac() перестраивает структуру из triplets на каждой итерации Ньютона
Файл: BVP_sci_faer.rs

Проблема: Структура глобального Jacobian (sparsity pattern) не меняется между итерациями Ньютона, но construct_global_jac() каждый раз собирает Triplet-ы заново и строит SparseColMat из них. Это O(nnz) аллокаций на каждой итерации.
Решение: Кэшировать sparsity pattern после первого построения. Использовать SparseColMat::new_with_symbolic() или эквивалент faer для повторного использования структуры.
Выгода: Ускорение Newton-итераций на 10-30% для больших систем. Оценка: 2-4 часа.

4. estimate_fun_jac() возмущает каждую переменную индивидуально
Файл: BVP_sci_faer.rs
Проблема: Для каждой точки сетки и каждой переменной клонируется полная матрица y и вызывается fun(). Это O(n² × m) где n=число переменных, m=число точек сетки.
rust


// Псевдокод
for i in 0..n {
    let mut y_perturbed = y.clone();
    y_perturbed.row(i) += h;
    let f_plus = fun(x, &y_perturbed);
}
Решение: Использовать grouped perturbations (возмущать несколько переменных одновременно, если структура Jacobian позволяет) или перейти на автоматическое дифференцирование.

Выгода: Ускорение FD Jacobian в 2-4x для систем с n > 10. Оценка: 4-8 часов.

5. collocation_fun() вызывает fun() по одной точке за раз
Файл: BVP_sci_faer.rs:591-607

Проблема: Цикл по точкам сетки, каждая итерация вызывает fun() с одно-столбцовой матрицей. Сигнатура ODEFunction принимает полную матрицу, так что vectorization возможна.


rust


for (j, x_mid) in x_middle.iter().enumerate() {
    let y_mid = y_middle.col(j);
    f_mid[j] = fun(x_mid, &y_mid);
}
Решение: Передавать все точки одним вызовом fun(x_middle, &y_middle).

Выгода: Ускорение 2-5x для lambdify/AOT бэкендов за счёт уменьшения dispatch overhead. Оценка: 2-4 часа.

6. #[should_panic] тест — антипаттерн тестирования
Файл: BVP_sci_faer_tests.rs

Проблема: test_lane_emden_equation() помечен #[should_panic]. Это означает, что тест "проходит" только когда падает. Невозможно отличить ожидаемое падение от регрессии.

Решение: Рефакторить тест на Result<(), BVPError> и проверять конкретный вариант ошибки.

Выгода: Улучшает качество тестового покрытия. Оценка: 1 час.

7. stacked_matmul() — мёртвый код в nalgebra версии
Файл: BVP_sci_nalgebra.rs

Проблема: Функция stacked_matmul() определена, но не используется (закомментирована в faer версии). Это dead code.

Решение: Удалить или раскомментировать и использовать.

Выгода: Устраняет dead code. Оценка: 15 минут.

P2: Среднесрочные улучшения (2-5 дней)
8. Дублирование кода между BVP_sci_faer.rs и BVP_sci_nalgebra.rs
Файлы: BVP_sci_faer.rs:1830, BVP_sci_nalgebra.rs:1313

Проблема: Два файла разделяют ~70% идентичного алгоритмического кода. BVP_sci_nalgebra.rs — dense-прототип, который никогда не был интегрирован. Любое изменение алгоритма (например, mesh refinement) должно вноситься в оба файла.

Решение:

Выделить общий алгоритм в generic-функцию, параметризованную матричным типом (через trait).
Или удалить nalgebra-версию, если dense-бэкенд не нужен (SparseColMat faer уже покрывает dense как частный случай).
Выгода: Устраняет ~900 строк дублирования, делает код поддерживаемым. Оценка: 2-3 дня.

9. BvpSciSolverOptions — взрыв builder-методов
Файл: BVP_sci_symb.rs

Проблема: BvpSciSolverOptions имеет ~20 builder-методов, многие из которых дублируют функциональность BVPwrap методов. Это приводит к путанице: можно сконфигурировать solver через BvpSciSolverOptions::new().tol(1e-6).max_nodes(10000) или через BVPwrap::new().with_tol(1e-6).with_max_nodes(10000).

Решение: Уменьшить surface BvpSciSolverOptions до минимально необходимого, делегировать остальное в BVPwrap. Или наоборот — сделать BvpSciSolverOptions единственной точкой конфигурации.

Выгода: Упрощает API, снижает когнитивную нагрузку. Оценка: 1 день.

10. BVP_sci_symbolic_functions.rs дублирует BVP_Damp symbolic infrastructure
Файл: BVP_sci_symbolic_functions.rs:1253

Проблема: Jacobian_sci_faer перереализует многое из того, что Jacobian в symbolic_functions_BVP.rs уже предоставляет, но с faer-специфичными типами. Например, calc_jacobian_parallel_smart() — это копия calc_jacobian_parallel_smart() из BVP_Damp, но с SparseColMat вместо CsMat.

Решение: Рефакторить Jacobian_sci_faer чтобы он делегировал общую symbolic логику в Jacobian из symbolic_functions_BVP.rs, конвертируя только типы на выходе.

Выгода: Устраняет ~800 строк дублирования, обеспечивает единый symbolic pipeline. Оценка: 2-3 дня.

11. BVP_sci_aot.rs дублирует AOT lifecycle логику
Файл: BVP_sci_aot.rs:869

Проблема: ensure_sparse_generated_runtime() перереализует build/link/register логику, которая уже существует в generated_solver_handoff.rs для BVP_Damp. Отличается только типами (SparseColMat vs CsMat).

Решение: Обобщить generated_solver_handoff.rs чтобы он работал с обоими наборами типов, или сделать BVP_sci AOT тонкой обёрткой над BVP_Damp AOT.

Выгода: Устраняет ~500 строк дублирования, единый AOT lifecycle. Оценка: 2-3 дня.

12. BvpSciGeneratedBackendMode имеет 8 вариантов, но все только для Sparse
Файл: BVP_sci_aot.rs

Проблема: Enum имеет варианты AtomViewBuildIfMissingRelease{Rust,Gcc,Tcc,Zig} и AtomViewForRepeatedSolves, но все они работают только с SparseColMat. Нет вариантов для Banded или Dense.
Решение: Добавить Banded-варианты или явно указать в документации, что поддерживается только Sparse.
Выгода: Честная документация, понятные ограничения. Оценка: 4 часа.

13. Нет интеграции с adaptive grid framework BVP_Damp
Файл: BVP_sci_faer.rs

Проблема: SciPy-style mesh refinement (modify_mesh с insert_1/insert_2) встроен непосредственно в solve_bvp(). BVP_Damp имеет 5 алгоритмов адаптивной сетки (Pearson, Grcar-Smooke, TWOPNT, easy, SciPy) в adaptive_grid_basic.rs и adaptive_grid_twopoint.rs.

Решение: Заменить встроенную SciPy-реализацию на вызов scipy_grid_refinement() из adaptive_grid_basic.rs.
Выгода: Единый код сеточной адаптации, доступ к другим алгоритмам. Оценка: 1 день.

14. BVP_sci_generated_compare_tests.rs (1614 строк) — тест вместо документа
Файл: BVP_sci_generated_compare_tests.rs:1614

Проблема: Два #[ignore] теста с обширной inline-инфраструктурой. Это аналог story tests из BVP_Damp, но встроенный в тестовый файл, а не в отдельный документ. Невозможно прочитать результаты без запуска тестов.

Решение: Создать BVP_SCI_STORY_TESTS.md по аналогии с BVP_DAMP_STORY_TESTS.md и перенести туда документацию.

Выгода: Прозрачность результатов, возможность сравнения с BVP_Damp. Оценка: 4 часа.

P3: Сложные рефакторинги (недели)
15. Отсутствует Banded линейный бэкенд
Проблема: BVP_sci поддерживает только Sparse (faer SparseColMat). BVP_Damp имеет полную поддержку Dense/Sparse/Banded с LAPACK-стилем banded LU. Для BVP с banded Jacobian (типичный случай для 1D задач) banded solver на порядок быстрее sparse.

Требуемые изменения:

Реализовать construct_global_banded_jac() — сборка banded матрицы в LAPACK-формате.
Реализовать solve_banded_newton() — решение с banded LU.
Добавить Banded вариант в BvpSciWorkflow.
Добавить Banded AOT поддержку в BVP_sci_aot.rs.
Добавить Banded тесты.
Выгода: Паритет с BVP_Damp по линейным бэкендам. Для 1D BVP задач — ускорение 2-5x. Оценка: 1-2 недели.

16. Отсутствует Frozen Newton вариант
Проблема: BVP_sci реализует только Damped Newton. BVP_Damp имеет и Damped, и Frozen. Frozen Newton замораживает Jacobian на несколько итераций, что даёт значительное ускорение для задач, где Jacobian дорогой (особенно AOT с codegen).

Требуемые изменения:

Реализовать solve_frozen_newton() — вариант, где Jacobian вычисляется раз в K итераций.
Добавить Frozen вариант в BvpSciWorkflow.
Добавить параметр frozen_jacobian_update_period в BvpSciSolverOptions.
Добавить Frozen тесты.
Выгода: Паритет с BVP_Damp. Для AOT-бэкенда с дорогим codegen — ускорение 2-3x. Оценка: 1 неделя.

17. Отсутствует chunking/parallel execution в AOT
Проблема: BVP_sci AOT вызывает codegen-функции целиком (whole/sequential). BVP_Damp имеет chunk4/8x8 с параллельным Rayon execution и Auto policy, который выбирает оптимальную стратегию на основе бенчмарков.

Требуемые изменения:

Реализовать chunking для sparse residual и Jacobian (разбиение на блоки по строкам).
Реализовать ParallelResidualExecutor и ParallelSparseJacobianExecutor для BVP_sci типов.
Добавить Auto policy с бенчмаркингом overhead.
Интегрировать с codegen_orchestrator.rs.
Выгода: Паритет с BVP_Damp. Для больших систем (n > 50, m > 100) — ускорение 2-8x на многоядерных системах. Оценка: 2-3 недели.

18. Отсутствует Dense AOT путь
Проблема: AOT pipeline в BVP_sci поддерживает только Sparse. BVP_Damp имеет AOT для Dense, Sparse и Banded. Для малых систем (n < 10) dense AOT быстрее sparse AOT.

Требуемые изменения:

Реализовать dense Jacobian codegen в BVP_sci_aot.rs.
Реализовать dense Newton solver (или адаптировать BVP_sci_nalgebra.rs).
Добавить Dense вариант в BvpSciWorkflow.
Добавить Dense AOT тесты.
Выгода: Паритет с BVP_Damp. Для малых систем — ускорение 1.5-3x. Оценка: 1-2 недели.

19. BVP_sci_nalgebra.rs — dense прототип не интегрирован
Проблема: Файл существует как отдельный dense-прототип, но не связан с основным API. Нет способа выбрать dense бэкенд через BvpSciSolverOptions.

Решение: Интегрировать dense бэкенд в BvpSciWorkflow как DirectNumericFaerDense или удалить файл.

Выгода: Чистая архитектура. Оценка: 3-5 дней (если интегрировать) или 1 час (если удалить).

20. Проблема: BVP_sci и BVP_Damp имеют полностью несовместимые API. BVP_sci использует прямые функции (solve_bvp()), BVP_Damp использует trait-ы (BvpSparseSolverProvider, BvpSolverBundle). Невозможно переключаться между ними без переписывания кода.

Решение: Ввести общий BvpSolver trait (как BvpApi в BVP_api.rs), который оба модуля имплементируют.

Выгода: Возможность A/B сравнения, единый пользовательский API. Оценка: 2-3 недели.

## Phase 0: BVP_sci statistics + story-test infrastructure

- [x] **Phase 0.1** — `BvpSciStatistics` extended with `counters`, `timers`, `diagnostics` HashMaps; `BVPResult` gains `diagnostics` field; `solve_newton()` fixed to accept `strategy_params` instead of reading from `BVPResult::default()`
- [x] **Phase 0.2** — `BVP_SCI_STORY_TESTS.md` ledger documenting existing tests, how to run them, expected results, run history
- [x] **Phase 0.3** — `BVP_sci_story_tests.rs` with full story-test framework: `RaceVariant`, `RaceRow`, `RaceSummaryRow`, `Aggregate`, `run_race_variant()`, `run_race_samples()`, `summarize_samples()`, table printers, tests (combustion_200 lambdify/aot/correctness, combustion_1000 release matrix, combustion_200 ExprLegacy stability)
- [x] **Phase 0.4** — Process-isolated cold tests: `run_isolated_race_samples()`, encode/decode, `combustion_3000_sparse_isolated_stress_story()` (4 existing tests pass, new test `#[ignore]`d)
- [x] **Phase 1.1** — `strategy_params` now flow through the actual faer route:
  `BVPwrap::solve_bvp_wrap()` calls `solve_bvp_with_strategy_params()`, which
  passes explicit Newton/line-search parameters into `solve_newton()`.
  Regression test `solve_newton_uses_explicit_strategy_params` locks the
  `max_iter=0` behavior.
- [x] **Phase 1.2** — `solve_bvp_sparse()` no longer duplicates the full solve
  loop. It remains as a compatibility alias and delegates to the single
  production `solve_bvp_with_strategy_params()` implementation.
- [x] **Phase 1.5** — `collocation_fun()` evaluates midpoint RHS values through
  one batched `fun(&x_mid_all, &y_middle, p)` call instead of one call per mesh
  interval.
- [x] **Phase 1.6** — Lane-Emden faer test is no longer a broad
  `#[should_panic]`; it returns `Result<(), String>` and checks the shifted
  removable-singularity formulation explicitly.
- [x] **Phase 1.7** — removed the stale commented `stacked_matmul()` block from
  the faer implementation. No active caller existed.
- [x] **Phase 1.3 correctness gate** — `construct_global_jac_cached()` is now
  locked against `construct_global_jac()` by a full elementwise parity test with
  parameter columns and boundary-condition blocks. Performance measurements can
  still be extended later, but the cached Sparse/faer Jacobian route is no
  longer an unverified optimization.
- [x] **Phase 1.9 Banded foundation** — added `BVP_sci_banded.rs` as a safe
  adapter layer over the existing BVP_sci Sparse/faer global Jacobian:
  `infer_banded_profile()`, `sparse_global_jac_to_banded()`, and
  `solve_banded_lapack_faithful()`.  This reuses the shared
  `somelinalg::banded` storage and LAPACK-style banded LU without changing the
  production Newton loop yet.  Unit gate `BVP_sci_banded_tests` is green 4/4.
  The diagnostic gate is now green 5/5 and prints full-vs-collocation
  bandwidth.  For a combustion-shaped dense-block BVP, collocation rows are
  compact (`kl=5`, `ku=11`, storage amplification about `1.42`), while endpoint
  boundary-condition rows widen the full scalar band (`kl=1194`, amplification
  about `100.9` for n=6, m=200).  This forced the P3.15 route toward a
  boundary-aware/bordered-banded backend instead of naively factoring the full
  scalar band.
- [x] **Phase 1.9b Bordered/boundary-aware Banded route planner** — added
  `BVP_sci_bordered_banded.rs` with `BvpSciBandedRoutePolicy` and
  `profile_bordered_banded_global_jacobian()`.  The route planner classifies
  matrices as `FullScalarBanded`, `BorderedBanded`, or `SparseFallback` from
  measured scalar-band amplification, collocation-band amplification, shape, and
  non-finite diagnostics.  Unit gate `BVP_sci_bordered_banded_tests` is green
  4/4 and locks the key decision: typical endpoint-BC BVP_sci matrices should
  not use naive full scalar banded LU; they need a bordered/boundary-aware
  implementation or Sparse fallback.
- [x] **Phase 1.9c Safe AutoBanded production hook** — added
  `BvpSciLinearSolvePolicy::{Sparse, AutoBanded, RequireFullBanded}` and wired
  it through the faer solve path plus `BVPwrap`/`BvpSciSolverOptions`.
  Default remains Sparse for backward compatibility.  `AutoBanded` uses shared
  LAPACK-style banded LU only when the whole global Jacobian is compact; endpoint
  BC matrices that are bordered-banded candidates are diagnosed and solved by
  Sparse fallback.  `BVP_sci_faer_tests` is green 28/28, including compact
  full-banded and endpoint-BC fallback regression tests.
- [x] **Phase 1.9d Bordered solver structural extractor** — added
  `BVP_sci_bordered_solver.rs` as the first non-fallback brick for a real
  bordered-banded backend.  It extracts the sparse global Jacobian into interval
  diagonal/off-diagonal collocation blocks, optional collocation parameter
  blocks, endpoint boundary blocks, and optional boundary-parameter blocks, with
  reconstruction parity tests.  It also provides a correctness-only dense
  reference solve for the extracted layout and locks it against Sparse LU on
  parameter-free and parameterized synthetic endpoint-BC systems.  This still
  does not change production Newton behavior or make performance claims; it
  makes the future bordered solver work from an explicit tested block layout
  rather than ad-hoc sparse indexing.
- [x] **Phase 1.9e Native bordered structured solve correctness** — added
  `solve_bordered_banded_structured()`, which eliminates the collocation
  block-bidiagonal body interval-by-interval and solves the final `(n+k)` border
  system for `(y0, p)`.  It is tested against Sparse LU on synthetic endpoint-BC
  systems with and without unknown parameters.  This is the first native
  bordered algorithm, but it remains correctness-only and is not wired into the
  production Newton loop yet.
- [x] **Phase 1.9f Explicit experimental bordered Newton route** — added
  `BvpSciLinearSolvePolicy::ExperimentalBorderedBanded` and wired the structured
  bordered solver into `solve_newton` behind that explicit policy only.
  `Sparse` remains the default, and `AutoBanded` still uses full scalar banded
  only for compact whole matrices plus Sparse fallback for endpoint-BC bordered
  candidates.  Added solver-level regression
  `experimental_bordered_banded_linear_policy_matches_sparse_endpoint_problem`:
  it compares the experimental bordered route with Sparse on a real endpoint-BC
  problem, verifies solution parity, and asserts that the route does not silently
  fall back to Sparse.  The bordered solver tests now also cover wrong RHS
  length, malformed block layout, singular off-diagonal blocks, and singular
  border systems.  Current gates are green:
  `BVP_sci_bordered_solver_tests` 10/10 and `BVP_sci_faer_tests` 29/29.
- [x] **Phase 1.9g Global Jacobian memory telemetry + Rust AOT compare isolation** —
  `GlobalJacobianDiagnostics` now records dense-equivalent bytes/KiB, sparse CSC
  value/index/pointer bytes, total sparse storage, and dense/sparse storage
  ratio.  These counters are written into solver statistics and printed in the
  BVP_sci linear-policy story table, making the expected memory advantage of the
  sparse/global-Jacobian route visible instead of guessed.  The same pass also
  hardened BVP_sci generated backend story tables: Rust cold AOT builds now use
  process-unique table namespace + repeat-index output directories under a
  deliberately short `target/bsc/r2/...` root, while `Rust-warm` uses its own
  short `rw` prebuilt directory inside the same namespace.  This avoids Windows
  `os error 5` DLL replacement failures and MSVC `LNK1104` long-path failures
  around nested `target/release/deps/*.dll.lib` outputs.  Fast gates:
  `global_jacobian_diagnostics_detect_exact_degenerate_structure`,
  `compare_output_dirs_isolate_rust_cold_builds_from_loaded_dlls`,
  `compare_run_namespace_is_process_unique_for_stale_dll_hardening`, and
  `combustion_200_auto_banded_linear_policy_route_story` are green.
- [x] **Phase 1.9h Bordered route timing diagnostics** — the explicit
  `ExperimentalBorderedBanded` Newton route now records solver-facing timings
  for bordered block extraction and structured bordered solves:
  `bvp sci bordered extraction us`,
  `bvp sci bordered structured solve us`, plus extraction/solve/reuse/line-search
  call counters.  The BVP_sci linear-policy story table prints these columns
  together with memory counters.  This closes the diagnostic gap before
  optimizing P3.15: current `combustion_200_auto_banded_linear_policy_route_story`
  shows extraction is small compared with repeated structured solves, so the
  next production work should target reusable factorization/cache inside the
  bordered solver rather than route detection or block extraction.
- [x] **Phase 1.9i Reusable bordered factorization/cache** — the structured
  bordered solver now exposes `factor_bordered_banded_structured()` and
  `BvpSciBorderedStructuredFactorization`.  The explicit
  `ExperimentalBorderedBanded` Newton route factors the bordered block system
  once per Jacobian rebuild and reuses it for Newton reuse and line-search RHS
  solves.  A correctness gate compares cached multi-RHS solves against the
  one-shot structured solver and Sparse LU.  The debug
  `combustion_200_auto_banded_linear_policy_route_story` shows the intended
  effect: repeated structured solve time drops sharply while extraction and
  factorization are now visible as separate counters.
- [x] **Phase 1.9j Bordered route release-candidate story** — added ignored
  `combustion_linear_policy_release_candidate_story`, configurable through
  `BVP_SCI_LINEAR_POLICY_N_STEPS` and `BVP_SCI_LINEAR_POLICY_RUNS`.  It compares
  Sparse, safe `AutoBanded` fallback, and explicit
  `ExperimentalBorderedBanded` with multi-run correctness, wall-clock timings,
  route counters, memory diagnostics, and bordered extraction/factor/solve
  timings.  Debug smoke and 12 Core release `combustion-1000` are green.
  Release data confirms `ExperimentalBorderedBanded` as a credible
  production-candidate opt-in: correctness-equivalent, no Sparse fallback,
  slightly faster than safe `AutoBanded` fallback, and with small bordered
  extraction/factor/solve timings.  Automatic `AutoBanded` promotion remains
  intentionally blocked pending at least one larger mesh story and one
  non-combustion endpoint-BC story.
- [x] **Phase 1.9k Bordered route promotion gates** — added ignored
  `combustion_large_linear_policy_release_story` and
  `exponential_endpoint_linear_policy_release_story`.  The first scales the
  combustion endpoint-BC matrix to a larger mesh; the second checks a different
  endpoint-BC problem so the bordered route is not validated only on
  combustion.  Both tests compare Sparse, safe `AutoBanded` fallback, and
  explicit `ExperimentalBorderedBanded`; both assert route counters so the
  experimental route cannot silently fall back to Sparse.  Debug smoke and
  12 Core release runs are green for the default settings (`combustion n=3000,
  runs=3`; exponential endpoint `n=1000, runs=3`).  Release data confirms
  correctness and route behavior, but performance is mixed/parity rather than a
  decisive win: combustion-3000 safe `AutoBanded` fallback is slightly faster
  than explicit bordered in the recorded run, while the non-combustion endpoint
  case is roughly tied.  Therefore `ExperimentalBorderedBanded` stays a valid
  advanced opt-in, and automatic `AutoBanded` promotion remains blocked.
- [x] **Phase 1.10 Production-like stage visibility** — expanded
  `bvp_sci_production_like_end_to_end_compare_table` from a total-only table to
  a single production-facing table that also prints setup/solve split,
  speedup-vs-lambdify, residual/Jacobian/linear-solve totals, and core work
  counters.  This makes `best_total=Direct-num` interpretable without running
  the heavier diagnostic compare table separately.  Compile gate
  `cargo test --lib bvp_sci_production_like_end_to_end_compare_table --no-run`
  is green; release output has been refreshed, and the Rust AOT rows now use the
  same process-unique short artifact namespace hardening as PS.10.  Rust warmup
  failures are now reported as table rows instead of panicking the whole matrix.

Сводная таблица
#	Категория	Описание	Файл	Оценка
P1.1	[x] Tech debt	Параметры Newton из Default	BVP_sci_faer.rs:847	30 мин
P1.2	[x] Duplication	solve_bvp() / solve_bvp_sparse()	BVP_sci_faer.rs	1 час
P1.3	[x] Performance	Jacobian из triplets каждую итерацию	BVP_sci_faer.rs	2-4 часа
P1.4	Performance	FD Jacobian по одной переменной	BVP_sci_faer.rs	4-8 часов
P1.5	[x] Performance	collocation_fun() не vectorized	BVP_sci_faer.rs:591	2-4 часа
P1.6	[x] Testing	#[should_panic] антипаттерн	BVP_sci_faer_tests.rs	1 час
P1.7	[x] Dead code	stacked_matmul()	BVP_sci_faer.rs	15 мин
P1.8	[x] Hardening	Явная singular/rank диагностика вместо зависимости от `sp_lu()` error semantics	BVP_sci_faer.rs	2-4 часа

Закрыто: добавлены `GlobalJacobianDiagnostics` и `inspect_global_jacobian()`.
Newton path теперь записывает размеры/nnz/non-finite/пустые строки/колонки,
строго нулевые строки/колонки, а также dense-equivalent и sparse CSC memory
counters в `calc_statistics`; очевидно вырожденный глобальный Jacobian
останавливается до `sp_lu()`. Намеренно не добавлялись магические
pivot-threshold эвристики, чтобы не менять численное поведение хороших задач.
Добавлен unit-test `global_jacobian_diagnostics_detect_exact_degenerate_structure`;
актуальный `BVP_sci_faer_tests` зеленый: 29/29.
P2.8	Duplication	faer/nalgebra ~70% overlap	оба файла	2-3 дня
P2.9	API design	Builder method explosion	BVP_sci_symb.rs	1 день
P2.10	Duplication	Jacobian_sci_faer vs Jacobian	BVP_sci_symbolic_functions.rs	2-3 дня
P2.11	Duplication	AOT lifecycle	BVP_sci_aot.rs	2-3 дня

Частично снижено на уровне story-test infrastructure: Rust backend compare rows
больше не переиспользуют фиксированную output directory ни между cold
повторами, ни между разными запусками одного story-теста, а сам root
укорочен до `target/bsc/r2/...`, чтобы не ловить MSVC `LNK1104` на глубоком
`*.dll.lib` пути. `RustWarm` build failure больше не валит матрицу, а становится
строкой статуса. Это не заменяет общий artifact lifecycle contract; BVP_sci AOT
всё еще стоит сближать с hardened BVP_Damp/LSODE2 lifecycle API.
P2.12	Design	8 enum variants, все Sparse	BVP_sci_aot.rs	4 часа
P2.13	Integration	SciPy refinement vs adaptive_grid	BVP_sci_faer.rs	1 день
P2.14	Documentation	Story tests в test file	BVP_sci_generated_compare_tests.rs	4 часа
P3.15	Missing feature	Banded backend	—	1-2 недели
    Phase 1 foundation is done: sparse global-Jacobian profiling/conversion and
    shared banded LU solve parity are covered by `BVP_sci_banded_tests`.
    The bandwidth story shows that full scalar banded storage is too wide
    because of endpoint BC rows; production integration should be
    bordered/boundary-aware, not a direct full-matrix scalar banded LU.
    A route-planner foundation is now also covered by
    `BVP_sci_bordered_banded_tests`. A safe `AutoBanded` production hook is
    available for compact whole matrices and Sparse fallback, but the real
    performance backend still requires larger release story tests before it can
    be promoted from an explicit experimental route.  The structural extractor
    and reusable factorization/cache foundation is now covered by
    `BVP_sci_bordered_solver_tests`, including dense-reference solve parity,
    native structured solve parity, cached multi-RHS parity, and
    malformed/singular diagnostics.
    The structured route is now wired into Newton behind the explicit
    `ExperimentalBorderedBanded` policy and matches Sparse on a real endpoint-BC
    problem.  Sparse remains the default; AutoBanded remains safe fallback, not
    a forced bordered production backend.  Solver-facing timing diagnostics now
    show that repeated dense/nalgebra structured solves were the first obvious
    bottleneck; a first reusable bordered factor/cache is implemented, but
    Sparse remains the default and `AutoBanded` remains conservative.  The
    release-candidate story now justifies documenting `ExperimentalBorderedBanded`
    as an advanced opt-in.  Larger-mesh and non-combustion endpoint story gates
    are release-green for correctness/route behavior, but they do not yet prove
    a broad performance win.  Keep `AutoBanded` conservative until a stronger
    performance story or further bordered-solver optimization changes that
    evidence.
P3.16	Missing feature	Frozen Newton	—	1 неделя
P3.17	Missing feature	Chunking/parallel AOT	—	2-3 недели
P3.18	Missing feature	Dense AOT	—	1-2 недели
P3.19	Integration	nalgebra prototype	BVP_sci_nalgebra.rs	3-5 дней
P3.20	Architecture	Единый BVP trait	—	2-3 недели
Ключевые архитектурные решения, которые нужно принять
Унификация или разделение? Стоит ли объединять BVP_sci и BVP_Damp в один модуль с общими traits, или держать их раздельными как "SciPy-совместимый" и "production" варианты?

Banded приоритет. Banded backend — самый большой performance win для 1D BVP задач. Если BVP_sci позиционируется как "серьёзный backend", Banded обязателен.

Frozen Newton. Если целевая аудитория — задачи с дорогим Jacobian (AOT), Frozen Newton даёт 2-3x ускорение без изменения алгоритма.

Chunking/parallel. Если BVP_sci должен работать на многоядерных системах с большими сетками, chunking необходим. BVP_Damp уже имеет полную инфраструктуру для этого.

Судьба nalgebra-прототипа. Интегрировать как Dense backend или удалить? Текущее состояние (отдельный файл без связи с API) — наихудший вариант.


Couldn't compile the test.FAILED

failures:

failures:
    src\Utils\animation_2d.rs - Utils::animation_2d (line 15)
    src\Utils\animation_2d.rs - Utils::animation_2d::create_2d_animation (line 92)
    src\Utils\animation_3d.rs - Utils::animation_3d (line 15)
    src\Utils\animation_3d.rs - Utils::animation_3d::create_3d_animation (line 96)
    src\command_interpreter.rs - command_interpreter::task_parser (line 115)
    src\command_interpreter\task_parser.rs - command_interpreter::task_parser::ParseError (line 157)
    src\numerical\BDF.rs - numerical::BDF::BDF_api (line 107)
    src\numerical\BDF.rs - numerical::BDF::BDF_api (line 83)
    src\numerical\BDF\BDF_api.rs - numerical::BDF::BDF_api::ODEsolver::get_result (line 1199)
    src\numerical\BDF\BDF_api.rs - numerical::BDF::BDF_api::ODEsolver::new (line 419)
    src\numerical\BDF\BDF_api.rs - numerical::BDF::BDF_api::ODEsolver::set_stop_condition (line 732)
    src\numerical\BDF\BDF_api.rs - numerical::BDF::BDF_api::ODEsolver::solve (line 1167)
    src\numerical\BVP_sci.rs - numerical::BVP_sci::BVP_sci_faer (line 102)
    src\numerical\BVP_sci.rs - numerical::BVP_sci::BVP_sci_faer (line 81)
    src\numerical\BVP_sci\BVP_sci_symb.rs - numerical::BVP_sci::BVP_sci_symb (line 25)
    src\numerical\BVP_sci\BVP_sci_symbolic_functions.rs - numerical::BVP_sci::BVP_sci_symbolic_functions (line 16)
    src\numerical\ShootingBVP\Shooting_simple.rs - numerical::ShootingBVP::Shooting_simple (line 29)
    src\numerical\data_processing\LSQSplines.rs - numerical::data_processing::LSQSplines::BSpline::evaluate_batch (line 288)
    src\numerical\data_processing\LSQSplines.rs - numerical::data_processing::LSQSplines::deboor_batch (line 426)
    src\somelinalg\iterative_solvers_gpu.rs - somelinalg::iterative_solvers_gpu (line 53)
    src\symbolic\symbolic_engine.rs - symbolic::symbolic_engine::Expr (line 74)
    src\symbolic\symbolic_engine.rs - symbolic::symbolic_engine::Expr::Symbols (line 862)
    src\symbolic\symbolic_engine_derivatives.rs - symbolic::symbolic_engine_derivatives::Expr::all_arguments_are_variables (line 959)
    src\symbolic\symbolic_engine_derivatives.rs - symbolic::symbolic_engine_derivatives::Expr::diff (line 93)
    src\symbolic\symbolic_engine_derivatives.rs - symbolic::symbolic_engine_derivatives::Expr::diff1 (line 119)
    src\symbolic\symbolic_engine_derivatives.rs - symbolic::symbolic_engine_derivatives::Expr::parse_expression (line 907)
    src\symbolic\symbolic_engine_derivatives.rs - symbolic::symbolic_engine_derivatives::Expr::sym_to_str (line 603)
    src\symbolic\symbolic_functions_BVP.rs - symbolic::symbolic_functions_BVP::Jacobian::from_vectors (line 3084)
    src\symbolic\symbolic_functions_BVP.rs - symbolic::symbolic_functions_BVP::Jacobian::generate_BVP (line 5739)
    src\symbolic\symbolic_functions_BVP.rs - symbolic::symbolic_functions_BVP::Jacobian::lambdify_jacobian_SparseColMat_parallel2 (line 4932)
    src\symbolic\symbolic_functions_BVP.rs - symbolic::symbolic_functions_BVP::Jacobian::remove_numeric_suffix (line 5184)
    src\symbolic\symbolic_lambdify.rs - symbolic::symbolic_lambdify::Expr::lambdify1D (line 23)
    src\symbolic\symbolic_lambdify.rs - symbolic::symbolic_lambdify::Expr::lambdify_IVP (line 357)

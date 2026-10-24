# Руководство по подгонке данных и оптимизации

Модуль предоставляет API для символьной нелинейной подгонки параметров по
наблюдениям. Он предназначен для fitting-задач, а не заменяет ODE/BVP-солверы
или универсальный constrained optimizer.

## Выбор метода

| Метод | Когда выбирать | Отличительная особенность |
|---|---|---|
| `Method::LM` | Нужна универсальная символьная нелинейная least-squares подгонка. | Использует канонический LM-солвер least-squares из крейта. |
| `Method::GAVIN` | Нужен именно контроллер Gavin damping/Broyden. | Отдельный алгоритм с политиками Marquardt, Quadratic и Nielsen; поддерживает веса наблюдений и ограничения положительности параметров. |
| `Method::VARPRO` | Модель разделима на нелинейные параметры базиса и линейные коэффициенты. | Решает линейные коэффициенты внутри нелинейного поиска. |

Универсально самого быстрого метода нет. Сравнивайте методы на одних и тех же
данных, начальном приближении и критериях точности. VarPro специализирован и
не является заменой общего LM.

## Базовая подгонка через `UniversalFitting`

Builder собирает данные, модель, имена неизвестных и начальное приближение;
`build()` запускает выбранный маршрут. По умолчанию используются `ExprLegacy`
и `Auto` для параллельной обработки достаточно больших наборов наблюдений;
порог и выбор режима описаны ниже.

```rust,ignore
use RustedSciThe::numerical::optimization::{
    Method, UniversalFitting, UniversalFittingResult,
};

let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.4 * (-x / 1.7_f64).exp() + 0.35).collect();

let result = UniversalFitting::new()
    .with_method(Method::LM)
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_max_iterations(500)
    .build()?;

let UniversalFittingResult::LM(fit) = result else {
    unreachable!("выбран метод LM");
};
let parameters = fit.solution_map().expect("результат содержит параметры");
```

`with_equation_str` задаёт выражение модели. Имена в `with_unknowns` являются
подбираемыми параметрами; `with_arg` задаёт имя независимой переменной,
соответствующей каждому элементу `x`. Векторы `x` и `y` должны иметь одинаковую
длину, а начальное приближение должно соответствовать порядку имён параметров.

Builder принимает `&str` и итераторы строк; прежние вызовы с `String` и
`Vec<String>` остаются допустимыми.

## Параллельная обработка наблюдений и память

`FittingObservationPolicy` управляет работой между наблюдениями; он не
меняет `LambdifyExecutionPolicy`, который управляет вычислениями внутри одного
символьного вызова. Для классического LM по умолчанию действует `Auto`:
при 20 001 наблюдении и более, последовательных callback-ах и пуле Rayon с
несколькими потоками используются блоки по 1024 точки. Иначе вычисление
остаётся последовательным. Этот порог соответствует минимальному крупному
измеренному сценарию, а не гарантированному break-even для любого компьютера.

```rust,ignore
use RustedSciThe::numerical::optimization::sym_fitting::FittingObservationPolicy;

let fit = UniversalFitting::new()
    .with_observation_policy(FittingObservationPolicy::Sequential);
```

Параллельный маршрут использует временные sample-major буферы, затем
восстанавливает публичный equation-major порядок. Объём буфера Якобиана равен
`8 * samples * equations * fitted_parameters` байтам, дополнительно выделяется
буфер невязок. Выбирайте `Sequential`, если важнее ограничить пиковую память.
Измерения времени, аллокаций и peak heap приведены в
`FITTING_STORY_TESTS.md` для ExprLegacy и AtomViewNative.

При `Auto` явный параллельный callback policy оставляет обработку наблюдений
последовательной. Явное сочетание двух параллельных уровней возвращает typed
ошибку, в том числе при повторном solve после изменения конфигурации.
`Fitting::resolved_observation_policy()` показывает решение для текущего пула
Rayon. Universal LM принимает эту настройку; Gavin и VarPro сообщают о ней как
о неподдерживаемой опции. Native telemetry сохраняет прежнюю схему счётчиков и
таймингов; выбранный dispatch читается отдельно как конфигурация.

## Выбор символьного фронтенда

Для поддерживаемых Lambdify-маршрутов доступны:

- `SymbolicLambdifyFrontend::ExprLegacy` — прежний Expr-путь, используемый по
  умолчанию.
- `SymbolicLambdifyFrontend::AtomViewNative` — нативный путь на packed
  AtomView.

Оба фронтенда доступны для LM и Gavin. VarPro пока поддерживает только
`ExprLegacy` и последовательное выполнение. Если запросить для VarPro
неподдерживаемую комбинацию, builder вернёт `UniversalFittingError::UnsupportedOption`,
а не проигнорирует настройку.

```rust,ignore
use RustedSciThe::numerical::optimization::{Method, UniversalFitting};
use RustedSciThe::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend,
};

let x = vec![0.0, 0.5, 1.0];
let y = vec![1.0, 0.8, 0.6];
let result = UniversalFitting::new()
    .with_gavin()
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_gavin_positive_parameters(["tau"])
    .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
    .build()?;
```

Параллельное вычисление callback-ов включается явно. Порог `min_work` позволяет
не отправлять маленькую работу в Rayon; параллелизм не гарантирует ускорения
всего fit. Измеряйте полное решение, а не только preparation или один callback.

Описанные здесь fitting API используют Lambdify. AOT сейчас не является
fitting-бэкендом.

## Настройки Gavin

Gavin — отдельный LM-подобный контроллер, а не псевдоним канонического метода
LM. В `UniversalFitting` его выбирают через `Method::GAVIN` или `.with_gavin()`.
Доступны специализированные настройки:

```rust,ignore
use RustedSciThe::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;
use RustedSciThe::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;

use RustedSciThe::numerical::optimization::UniversalFitting;
let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.4 * (-x / 1.7_f64).exp() + 0.35).collect();
let weights = vec![1.0; x.len()];
let result = UniversalFitting::new()
    .with_gavin()
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_gavin_positive_parameters(["tau"])
    .with_gavin_update_policy(GavinLmUpdatePolicy::Marquardt)
    .with_gavin_solver_controls(1000, 1e-6, 1e-6, 1e-1)
    .with_gavin_weights(weights)
    .with_telemetry(LeastSquaresTelemetryMode::Detailed)
    .with_logging(false)
    .build()?;
```

Аргументы `with_gavin_solver_controls` идут в таком порядке: бюджет итераций,
допуск по градиенту, допуск по шагу параметров и порог gain ratio. Политика
обновления может быть Marquardt, Quadratic или Nielsen. Веса должны быть
неотрицательными и соответствовать наблюдениям. Положительность параметров —
явное решение пользователя; ограничение полезно в любой модели, где область
допустимых значений требует положительных параметров, а не только в химии.

`with_tolerance` задаёт для Gavin допуск по шагу параметров,
`with_g_tolerance` — допуск по градиенту. `f_tolerance` и `scale_diag` не имеют
эквивалента в Gavin и приводят к typed unsupported-option error. Для настройки
Gavin предпочтительны его специализированные методы, чтобы семантика
контроллера была явной.

Для прямого доступа к телеметрии символьной подготовки и provenance результата
используйте `GavinSymbolicFitting` из `optimization::gavin_fitting`; запускаемый
пример находится в `rus_gavin_fitting_guide`.

## Variable projection (VarPro)

VarPro подходит для разделимых моделей, например
`y(x) = a * exp(-x / tau) + b`: `tau` — нелинейный параметр, а `a` и `b` —
линейные коэффициенты. В универсальном builder нелинейные параметры задаются
через `with_parameters`, базис — через `with_basis_str`, фоновые/инвариантные
слагаемые — через `with_equations`.

```rust,ignore
use RustedSciThe::numerical::optimization::{UniversalFitting, UniversalFittingResult};

let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.5 * (-x / 1.7_f64).exp() + 0.8).collect();
let result = UniversalFitting::new()
    .with_varpro()
    .with_data(x, y)
    .with_arg("x")
    .with_parameters(["tau"])
    .with_initial_guess(vec![1.2])
    .with_basis_str("tau", "exp(-x/tau)")
    .with_equations_str(["1"])
    .build()?;
let UniversalFittingResult::VARPRO(fit) = result else {
    unreachable!("выбран VarPro");
};
let nonlinear_parameters = fit.nonlinear_parameters();
let linear_coefficients = fit
    .linear_coefficients()
    .expect("модель содержит линейные коэффициенты");
```

Если нужны низкоуровневое построение модели или настройки решателя VarPro,
используйте `SymbolicVarProBuilder` напрямую.

## Телеметрия, логирование и ошибки

Телеметрия и логирование выключены по умолчанию. Телеметрию можно включить
явно:

```rust,ignore
use RustedSciThe::numerical::optimization::UniversalFitting;
use RustedSciThe::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;

let configured = UniversalFitting::new()
    .with_telemetry(LeastSquaresTelemetryMode::Detailed);
```

Канонический least-squares solver в режиме `Off` не собирает runtime-счётчики
и не читает часы. Gavin пока предоставляет один полный telemetry snapshot,
поэтому любой включённый уровень запрашивает этот snapshot. Символьная
подготовка — отдельный lifecycle; у прямого Gavin builder для неё есть
`with_preparation_telemetry`.

Логирование включается через `.with_logging(true)` и использует установленный
приложением `log` logger; библиотека сама logger не устанавливает. Общие
настройки `UniversalFitting` применяются, когда их семантика совпадает.
Неподдерживаемая настройка возвращается как типизированная ошибка, а не
переинтерпретируется молча.

`UniversalFitting::build()` возвращает
`Result<UniversalFittingResult, UniversalFittingError>`. Прямой Gavin builder
возвращает `GavinLmError`, различающий ошибки конфигурации, символьной
подготовки, callback-ов, численного решения и статистической обработки.
Проверяйте вариант результата и termination/report солвера вместе со
значениями параметров.

## Примеры и результаты измерений

Запуск из корня репозитория:

```powershell
cargo run --no-default-features --example rus_universal_fitting_guide
cargo run --no-default-features --example rus_gavin_fitting_guide
```

Первый пример показывает classic LM и VarPro через общий facade, второй —
отдельный builder Gavin. Story-тесты и performance evidence, включая
сопоставление ExprLegacy/AtomView на одинаковых задачах, описаны в
[`GAVIN_STORY_TESTS.md`](GAVIN_STORY_TESTS.md). Измерения показывают, что
победитель зависит от количества наблюдений и политики контроллера; быстрое
symbolic preparation само по себе не доказывает более быстрый полный fit.

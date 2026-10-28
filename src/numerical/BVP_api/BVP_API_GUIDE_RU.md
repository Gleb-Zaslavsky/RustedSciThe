# Типизированный фасад BVP

`numerical::BVP_api` — тонкий фасад выбора между production BVP-солверами.
Он проверяет общие данные задачи, направляет запрос в native API и сохраняет
native-объект, результат и телеметрию выбранного маршрута. Фасад не заменяет
символьную подготовку, конфигурацию солвера, Newton/collocation или линейную
алгебру.

```rust
use nalgebra::DMatrix;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciSparseFactorization,
    BvpSolution, BvpSolver, BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = -y, with y(0) = 1.
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?;

    let config = BvpSolverConfig::Sci {
        options: BvpSciOptions::default()
            .with_assembly(BvpSciAssembly::ExprLegacy)
            .with_execution(BvpSciExecution::Lambdify)
            .with_telemetry(BvpSciTelemetryMode::Counters),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
        // State-major: one initial value per state at every mesh node.
        initial_guess: DMatrix::from_row_slice(1, 5, &[1.0, 0.78, 0.61, 0.47, 0.37]),
    };
    let mut solver = BvpSolver::new(problem, config)?;

    if let BvpSolution::Sci(result) = solver.solve()? {
        println!("y(1) = {:.8}", result.y[4]);
    }
    let dataset = solver.postprocess_dataset()?;
    println!("node-major dataset: {:?}", dataset.values.shape());

    if let BvpSolverTelemetrySnapshot::Sci(snapshot) = solver.telemetry_snapshot() {
        println!("native RHS calls: {}", snapshot.rhs_calls);
    }
    Ok(())
}
```

Полный запускаемый вариант находится в
[`examples/bvp_api_facade_sci.rs`](../../../../examples/bvp_api_facade_sci.rs).
Пример fixed-input rebinding для Damped находится в
[`examples/bvp_api_parameter_rebind.rs`](../../../../examples/bvp_api_parameter_rebind.rs).
Запускаемый пример цикла BVP_sci с фиксированными внешними параметрами:
[`examples/bvp_api_parameter_loop.rs`](../../../../examples/bvp_api_parameter_loop.rs).
Циклы фиксированных входных параметров через оба маршрута BVP_Damp, включая
различия checkpoint и отчётов об ошибках:
[`examples/bvp_api_damped_frozen_parameter_loop.rs`](../../../../examples/bvp_api_damped_frozen_parameter_loop.rs).

## Выбор маршрута и данные

`BvpProblem` хранит уравнения в порядке `state_names`, имя независимой
переменной, строго возрастающую сетку и скалярные граничные условия. Каждое
условие задаёт индекс состояния, один конец интервала и конечное значение.
Имена состояний должны быть уникальны и не совпадать с независимой переменной.
Всего требуется одно скалярное условие на каждое уравнение, но оба конца могут
ограничивать одну переменную, если этого требует математическая задача;
разрешимость системы проверяет сам solver.

Конфигурация solver-specific намеренно. Damped initial guess имеет форму
`state_count x interval_count`; Frozen/Naive и BVP_sci используют
`state_count x mesh_point_count`. Damped и Frozen принимают только равномерную
сетку, поскольку их native constructors задаются концами интервала и числом
шагов. Неравномерную сетку можно передать BVP_sci напрямую.

Адаптивное уточнение сетки в BVP_Damp выполняется самим solver во время
решения: пользователь не перезапускает его вручную при нарушении критериев
сетки. Явный facade-метод restart(mesh, guess) относится только к BVP_sci:
он позволяет начать с заданной сетки и профиля, сохраняя подготовленную
символьную модель.

Фасад не строит искусственную общую матрицу возможностей. Настройки
передаются native типами `DampedSolverOptions`, `FrozenSolverOptions` или
`BvpSciOptions`; BVP_sci также сохраняет sparse factorization и необязательную
AOT-конфигурацию. Для fixed inputs в Damped/Frozen перед первым solve вызовите
`bind_parameters(names, values)`, а затем `solve_with_parameter_values(values)`:
фасад переиспользует подготовленный маршрут и предыдущий Newton iterate.
`parameters` в BVP_sci имеют другую семантику: это свободные Newton-неизвестные,
а не fixed inputs. Фиксированные внешние inputs объявляются отдельно через
`BvpProblem::with_external_parameters(names, initial_values)` до подготовки
символьных callbacks. `solve_parameter_loop` меняет значения перед каждым
solve и использует последний успешный профиль как начальное приближение.
Фасад сохраняет нативную семантику циклов: Damped может передавать следующей
точке также новую уточнённую сетку, а Frozen/Naive сохраняет исходную сетку и
передаёт только успешный профиль. Ошибка отдельной точки записывается в отчёт и
не заменяет последний успешный checkpoint. Фасад не превращает уточнение сетки
в parameter continuation и не сводит разные native-отчёты к искусственному
общему типу.
Для неизвестных параметров и зависимых от них boundary callbacks используйте
отдельный free-parameter маршрут. Для численных RHS и Jacobian callbacks
обращайтесь напрямую к native BVP_Damp API.

Для неизвестных параметров фасад предоставляет отдельный вариант
`BvpSolverConfig::SciWithFreeParameters`. Обычные `n` скалярных граничных
условий остаются в `BvpProblem`, а `p` дополнительных уравнений, определяющих
параметры, задаются через `BvpSciFreeParameters`. Необязательный callback
задаёт аналитические строки Якобиана по значениям состояния слева/справа и
свободным параметрам; если его нет, используется нативное конечно-разностное
вычисление BVP_sci. Метод `parameter_guess()` читает текущее начальное
приближение Ньютона, `set_parameter_guess(...)` меняет его, а
`solve_with_parameter_guess(...)` меняет приближение и сразу запускает solve.
Это неизвестные Ньютона, а не fixed-input rebinding. После успешного solve
решённый вектор параметров становится текущим guess для следующего warm solve.
Полный пример для
`y' = p`, `y(0) = 0` и дополнительного уравнения, определяющего `p`, находится
в [`examples/bvp_api_sci_free_parameter.rs`](../../../examples/bvp_api_sci_free_parameter.rs).

### Границы возможностей фасада

Общая модель `BvpProblem` принимает только символьные уравнения. Damped и
Frozen/Naive используют нативную конфигурацию generated backend, но требуют
равномерную сетку: их конструкторы принимают границы интервала и число узлов
или интервалов. Нативные numerical RHS/Jacobian callbacks в этой модели не
представлены. По той же причине конфигурация BVP_sci с
`BvpSciExecution::Numerical` или `BvpSciAssembly::Numerical` отклоняется через
`UnsupportedRoute`; для callback-задач используйте native BVP_sci builder.
Важно: `BvpSciOptions::default()` выбирает Numerical execution, поэтому для
символьного маршрута фасада нужно явно задать `Lambdify` (assembly по
умолчанию становится ExprLegacy) или `Aot` с конфигурацией generated backend.

У семейств солверов намеренно разные defaults. Damped по умолчанию использует
forward scheme, Sparse, абсолютный tolerance `1e-6`, 25 нелинейных итераций,
AtomView и AOT-if-available с fallback на Lambdify; solver counters включены,
callback timings и logging выключены. Frozen использует стратегию
`Frozen_naive`, Sparse, tolerance `1e-6` и 25 итераций. Нативные BVP_sci
options по умолчанию выбирают Numerical/Dense, tolerance `1e-3`, telemetry
Off и последовательный dispatch callbacks. Поскольку facade принимает только
символьную задачу, BVP_sci default нужно переопределить, как описано выше.
Это defaults нативных конфигураций, а не общий preset фасада.

Фасад передаёт нативные options, но использует собственный ограниченный enum
`BvpSciSparseFactorization`: доступны только `StandardFaer` и `SuperLu`.
Экспериментальные reusable-faer варианты остаются только в прямом native
BVP_sci API: измеренного выигрыша они не показали. `SuperLu` выбирается явно и
сохраняет нативное поведение bridge/fallback. Dense-output sampling также
остаётся операцией native BVP_sci API; общий postprocessing экспортирует
сохранённые строки сетки.

После успешного решения solution_view() заимствует фактическую сетку и даёт
единый доступ по индексам узла и состояния через mesh(), state_count() и
value(node, state). Общая плотная матрица при этом не создаётся. Исходный
route-specific результат и его диагностика доступны через solution().
Метод execute_postprocessing(&plan) применяет общий план экспорта/графиков к
сохранённым строкам сетки и возвращает native report. Ошибки экспорта типизированы
и не удаляют результат решения. Плотная интерполяция BVP_sci не подразумевается:
для неё используйте native result API.

## Ошибки, телеметрия и логи

`BvpApiError` различает неверную задачу, неподдерживаемый маршрут, отсутствие
результата и ошибки native солверов. Native ошибки сохраняются источниками, а
не превращаются в строки. `solve()` очищает facade result перед новым запуском;
неудачный повторный запуск не выдаёт старый результат за новый. Для полного
solver-specific lifecycle, диагностики и изменения runtime-политик доступны
`native()` и `native_mut()`.

`telemetry_snapshot()` возвращает `BvpSolverTelemetrySnapshot` с отдельным
вариантом для каждого семейства. Native scopes и различие между измеренным
нулём и отсутствующим измерением сохраняются; фасад не суммирует и не
переименовывает метрики. Режим телеметрии задаётся native options. Логирование
тоже настраивается native options или native solver API; фасад не создаёт
второй logger и не возвращает текстовые логи в telemetry. Включённый logging
должен создать общий timestamped-файл до подготовки/решения BVP; ошибка
инициализации возвращается как типизированная native-ошибка, и расчёт не
продолжается с незаметно выключенным логом. Путь к файлу доступен через
`numerical::logging::solver_log_path()`.

Task-document parser — самостоятельная подсистема. Для чтения и исполнения
документов используйте `command_interpreter::task_parser_bvp`; он напрямую
строит native BVP-солверы и не зависит от этого фасада.

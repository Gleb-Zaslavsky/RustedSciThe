# Руководство по Python-интерфейсу RustedSciThe

Python-пакет является тонкой обёрткой над Rust-ядром. Численные алгоритмы и
символьные операции не переписаны на Python: `Expr` разбирается,
дифференцируется и упрощается в Rust, а solver владеет подготовкой,
коллбэками, workspace и телеметрией.

## Установка под Windows

```powershell
py -3.12 -m venv .venv-rustedscithe
.\.venv-rustedscithe\Scripts\python.exe -m pip install -U pip numpy maturin
& .\.venv-rustedscithe\Scripts\maturin.exe develop --manifest-path bindings/python/Cargo.toml
```

Пакет ориентирован на `abi3-py310` и NumPy 2.x. Расширение называется
`rustedscithe._native`, но пользовательский код должен импортировать обычный
`rustedscithe`.

## Независимые измерения маршрута

| Измерение | Значения | Смысл |
| --- | --- | --- |
| символьный frontend | `expr_legacy`, `atom_native` | подготовка и дифференцирование выражений |
| execution | `lambdify`, `aot` | closure в процессе или скомпилированный артефакт |
| layout | `dense`, `sparse`, `banded` | хранение и решение линейной системы |
| политика коллбэков | `sequential`, `parallel`, `auto` | планирование вычислений там, где это поддержано |

Проверяйте `capabilities("solver")`. Неподдержанный набор настроек даёт
`UnsupportedOptionError`, а не молча переключается на другой backend.

## IVP и continuation

`solve_ivp(...)` удобен для одиночного решения, а `create_ivp(...)` нужен для
явной подготовки, continuation и restart. Для серии параметров подготовьте
модель один раз:

```python
import numpy as np
from rustedscithe import BdfSolver, Expr

y = Expr.variable("y")
a = Expr.variable("a")
solver = BdfSolver(
    [-a * y], ["y"], "t", np.array([1.0]), 0.0, 0.5, 0.1,
    parameters=["a"], parameter_values=np.array([1.0]),
)
solver.prepare()
first = solver.solve()
second = solver.continue_with_parameters(np.array([2.0]), 1.0)
```

Continuation использует подготовленный символьный маршрут, но не начинает
новое решение с исходного `y0`. Для нового состояния и интервала используйте
`restart(...)`. Холодная AOT-компиляция может окупиться только на длинной
серии, поэтому сравнивайте отдельно preparation, callback и full solve.

В общем facade первый аргумент `create_ivp` выбирает семейство солвера, а
контроллер LSODE2 задаётся отдельно. Это не позволяет спутать выбор солвера с
режимом Adams/BDF:

```python
from rustedscithe import Expr, create_ivp

y = Expr.variable("y")
solver = create_ivp(
    "lsode2", [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
    controller="adams",  # также "bdf" или "auto"
)
result = solver.solve()
```

`"lsoda"` принимается как alias семейства LSODE2 для автоматического
контроллера Adams/BDF. Аргумент `controller=` допустим только для LSODE2.

## BVP

`solve_bvp("bvp_sci", ...)` и `solve_bvp("bvp_damp", ...)` имеют общий
внешний lifecycle, но намеренно разные граничные данные:

- BVP_sci принимает список символьных граничных невязок;
- BVP_Damp принимает mapping состояния к граничным условиям и требует bounds.

Нельзя сравнивать layout на разных уравнениях: Dense естественен для малых
плотных задач, а Sparse/Banded могут выигрывать на структурированных
диффузионных или collocation-системах. SuperLU в BVP_sci является opt-in;
доступность и fallback отражаются в телеметрии.

## NumPy и владение данными

Результаты имеют owned-методы `times_array()`, `states_array()`,
`mesh_array()` и `values_array()`. IVP states имеют форму `(time, state)`,
BVP_sci values — `(mesh, state)`, а BVP_Damp сохраняет native-ориентацию
`(state, mesh)`. Старые list-методы оставлены. Zero-copy views намеренно не
обещаются: подготовленные solvers переиспользуют изменяемые workspace.

Facade принимает NumPy-векторы и преобразует их перед входом в Rust.

## Ошибки, телеметрия и task parser

Используйте typed exceptions: `InvalidConfigurationError`,
`UnsupportedOptionError`, `PreparationError`, AOT-ошибки,
`CallbackError`, `SolverError` и `ExhaustionError`. Native-ошибки имеют поля
`stage`, `solver`, `route`; текст сообщения не следует разбирать программно.

`TaskDocument.parse(...)`, `task(...)` и `run_task_batch(...)` используют
общий Rust task parser. Batch продолжает выполнение после ошибки и может
вывести компактную таблицу через `render_table(...)`.

Примеры в `bindings/python/examples/` показывают Radau, BDF continuation,
BVP_sci, nonlinear/fitting и batch-задачи. Они используют уже проверенную
семантику основных Rust-гайдов, а не придумывают отдельные Python-алгоритмы.

## Отдельные руководства по солверам

Этот обзор намеренно краток. Подробные руководства сохраняют богатство
настроек соответствующих native-солверов, но показывают Python-синтаксис:

- [Radau5](RADAU_GUIDE_RU.md): жёсткие IVP, frontend/layout, output и телеметрия.
- [BDF](BDF_GUIDE_RU.md): плотный BDF, порядок, callbacks и continuation.
- [Backward Euler](BACKWARD_EULER_GUIDE_RU.md): фиксированный шаг и доступные
  dense-маршруты исполнения.
- [LSODE2](LSODE2_GUIDE_RU.md): контроллеры Adams/BDF, символьные маршруты,
  matrix layout и политика параллельных callbacks.
- [BVP_sci](BVP_SCI_GUIDE_RU.md): граничные невязки, Dense/Sparse/Banded,
  SuperLU и AOT.
- [BVP_Damp](BVP_DAMP_GUIDE_RU.md): наблюдения, bounds, уточнение сетки,
  layout и символьные/AOT-маршруты.
- [Нелинейные системы и fitting](NONLINEAR_FITTING_GUIDE_RU.md): root,
  least-squares и fitting API.

Подробно прокомментированные исполняемые Python-примеры собраны в
[`examples/README.md`](../examples/README.md).

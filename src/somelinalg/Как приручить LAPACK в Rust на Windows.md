Как приручить LAPACK в Rust на Windows
1. Сначала выбери стратегию

На Windows есть два основных ABI:

x86_64-pc-windows-msvc
x86_64-pc-windows-gnu

rustup отдельно подчёркивает, что Windows делится на MSVC ABI и GNU ABI. Для GNU ABI нужен GCC-based toolchain, а для MSVC — Visual Studio ecosystem.

Для связки Rust + OpenBLAS/LAPACK + MSYS2 самый прямой путь — это:

Rust toolchain: x86_64-pc-windows-gnu
Библиотеки: MSYS2 mingw64
Линковка: через openblas-src с feature system

Если хочешь остаться на msvc, то чаще всего придётся идти через vcpkg, а не через MSYS2. openblas-src прямо пишет, что на Windows OpenBLAS из исходников не собирается, и надо использовать system.

2. Установи GNU toolchain для Rust

В обычном Windows терминале:

rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
rustup show
Как понять, что всё хорошо

В выводе rustup show должен быть активный toolchain:

stable-x86_64-pc-windows-gnu (default)
Если этого нет

Ты всё ещё собираешь под msvc, и тогда MSYS2-библиотеки будут стыковаться заметно хуже. rustup book как раз объясняет различие между MSVC и GNU на Windows.

3. Установи MSYS2 и нужные пакеты

Открой MSYS2 MinGW x64 shell. Именно MinGW x64, не обычный MSYS shell.

Установи пакеты:

pacman -Syu
pacman -S --needed mingw-w64-x86_64-openblas
pacman -S --needed mingw-w64-x86_64-pkgconf
pacman -S --needed mingw-w64-x86_64-gcc

MSYS2-пакет mingw-w64-x86_64-openblas прямо описан как пакет, который даёт optimized blas, lapack, cblas для mingw-w64. mingw-w64-x86_64-pkgconf даёт pkg-config-совместимую утилиту, а mingw-w64-x86_64-gcc — GCC для MinGW-w64.

Как понять, что всё хорошо

Проверь:

which gcc
which pkg-config
pkg-config --libs openblas
pkg-config --cflags openblas

Ожидаемо увидишь что-то вроде:

/mingw64/bin/gcc
/mingw64/bin/pkg-config
-lopenblas
-IC:/msys64/mingw64/include/openblas -fopenmp
Что означает ошибка

Если pkg-config --libs openblas не работает, то openblas-src в system-режиме тоже не найдёт OpenBLAS.

Если which gcc пустой, то упадут crate’ы, использующие cc-rs, с ошибкой вида:

Failed to find tool. Is gcc.exe installed?
4. Сделай так, чтобы MINGW64 shell видел Rust

MSYS2 shell сам по себе обычно не знает про cargo и rustup, потому что они лежат в %USERPROFILE%\.cargo\bin. rustup именно туда ставит Rust binaries на Windows.

В MINGW64 shell добавь:

export PATH="$PATH:/c/Users/<ТВОЙ_ПОЛЬЗОВАТЕЛЬ>/.cargo/bin"

Проверь:

which cargo
which rustc
cargo --version
rustup show
Сделать это постоянным

Добавь ту же строку в ~/.bashrc.

Как понять, что всё хорошо

Должно быть примерно так:

/c/Users/Name/.cargo/bin/cargo
/c/Users/Name/.cargo/bin/rustc
Что означает ошибка

Если cargo не найден в MINGW64 shell, то сборка из этого shell не пойдёт, даже если gcc и pkg-config уже есть.

5. Настрой Cargo.toml

Минимально рабочая схема такая:

[dependencies]
lapack = { version = "0.20", optional = true }
openblas-src = { version = "0.10.15", default-features = false, features = ["system"], optional = true }

[features]
lapack = ["dep:lapack"]
openblas-system = ["lapack", "dep:openblas-src"]

openblas-src документирует feature system как режим, в котором bundled OpenBLAS не собирается, а используется системная библиотека. На Windows именно этот режим и нужен.

Что важно

src = "0.0.6" в Cargo.toml — это легитимная зависимость с crates.io, и это нормально. В Cargo зависимости с crates.io задаются просто именем и версией.

6. Явно подтяни linking crate в конечный crate

Это тот шаг, на котором у тебя всё наконец заработало.

В lib.rs:

#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;

Если тест — это integration test или отдельный test target, добавь то же самое прямо в тестовый модуль, а не только в lib.rs:

#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;
Почему это нужно

Иногда зависимость нужна не ради Rust API, а только ради линковки системной библиотеки. Без явного “подтягивания” provider crate итоговый binary/test crate может получить объявления lapack::*, но не получить саму библиотеку при линковке. Это известный острый угол у lapack-экосистемы.

Симптом, что этого не хватает

Ошибки линкера вида:

undefined reference to `dgbtrf_`
undefined reference to `dgbtrs_`
undefined reference to `dlamch_`

Это означает: Rust-код уже зовёт LAPACK symbols, но линкер не получил реальную библиотеку, которая эти symbols экспортирует.

7. Собирать лучше из MSYS2 MinGW x64 shell

Запускай сборку именно оттуда:

cargo clean
cargo test lapack_try --features openblas-system -vv
Почему именно так

MSYS2 shell обычно сам корректно настраивает окружение для pkg-config, gcc, mingw include/lib paths и связанных инструментов. Для этой связки это самый прямой путь.

8. Минимальный smoke test

Например, просто проверь, что зовётся хоть одна LAPACK-функция:

#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;

#[test]
fn lapack_try() {
    let eps = unsafe { lapack::dlamch(b'E') };
    assert!(eps > 0.0);
}

Если этот тест проходит, значит:

lapack crate подключён,
OpenBLAS/LAPACK найдены,
линковка работает,
runtime тоже в порядке.
Типовые ошибки и что они означают
9. Ошибка: Non-vcpkg builds are not supported on Windows. You must use the 'system' feature.
Что это значит

Ты пытался использовать openblas-src на Windows без feature system.

Что делать

Использовать:

openblas-src = { version = "...", default-features = false, features = ["system"] }

Это прямо соответствует документации openblas-src для Windows.

10. Ошибка: vcpkg failed to find OpenBLAS package
Что это значит

Ты собираешься под windows-msvc, а openblas-src пытается искать OpenBLAS через vcpkg, а не через MSYS2.

Что делать

Либо:

перейти на windows-gnu + MSYS2,
либо остаться на msvc и действительно использовать vcpkg.

Если твоя цель — именно MSYS2 OpenBLAS, то правильнее перейти на x86_64-pc-windows-gnu.

11. Ошибка: Failed to find tool. Is gcc.exe installed?
Что это значит

Какой-то crate использует cc-rs, а GCC недоступен в текущем shell.

Что делать

Установить и проверить:

pacman -S --needed mingw-w64-x86_64-gcc
which gcc
gcc --version

MSYS2-пакет mingw-w64-x86_64-gcc и есть нужный компилятор для GNU ABI на Windows.

12. Ошибка: pkg-config не находит openblas
Что это значит

openblas-src в system-режиме не может обнаружить установленную библиотеку.

Что делать

Проверить:

pacman -S --needed mingw-w64-x86_64-pkgconf
which pkg-config
pkg-config --libs openblas
pkg-config --cflags openblas

Если pkg-config --libs openblas не даёт -lopenblas, то Rust-линковка тоже не взлетит. mingw-w64-x86_64-pkgconf — официальный MSYS2 пакет для pkg-config-совместимой утилиты.

13. Ошибка: undefined reference to dgbtrf_ и похожие
Что это значит

LAPACK symbols объявлены, но реальная библиотека не дотянулась до конечного binary/test crate.

Что делать

Проверить, что feature активна:

cargo tree -e features | grep -i "openblas\|lapack"

Добавить:

#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;

в конечный crate или в сам test target.

Сделать:

cargo clean
cargo test --features openblas-system -vv
14. Ошибка: cargo tree показывает странности в зависимостях
Что это значит

Не обязательно ошибка. Например, запись вроде:

src = "0.0.6"

нормальна, если это реальный crate с crates.io. По Cargo Book dependency с crates.io задаётся просто именем и версией.

Подозрительно не это, а path dependency на каталог без корректного crate layout.

Рабочий чеклист
15. Краткая последовательность без объяснений

Установить GNU toolchain:

rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

В MSYS2 MinGW x64:

pacman -Syu
pacman -S --needed mingw-w64-x86_64-openblas
pacman -S --needed mingw-w64-x86_64-pkgconf
pacman -S --needed mingw-w64-x86_64-gcc

Добавить Rust в PATH внутри MINGW64 shell:

export PATH="$PATH:/c/Users/<USER>/.cargo/bin"

Проверить:

which cargo
which gcc
which pkg-config
pkg-config --libs openblas

В Cargo.toml:

[dependencies]
lapack = { version = "0.20", optional = true }
openblas-src = { version = "0.10.15", default-features = false, features = ["system"], optional = true }

[features]
lapack = ["dep:lapack"]
openblas-system = ["lapack", "dep:openblas-src"]

В lib.rs и при необходимости в test target:

#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;

Сборка:

cargo clean
cargo test --features openblas-system -vv

Smoke test:

let eps = unsafe { lapack::dlamch(b'E') };
assert!(eps > 0.0);
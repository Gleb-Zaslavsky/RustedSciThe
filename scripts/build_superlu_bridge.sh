#!/usr/bin/env bash
set -euo pipefail

readonly SUPERLU_COMMIT="d6a031ca56499eefa7c742d51fae5e8c644138cc"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/RustedSciThe"
SOURCE_DIR="${SUPERLU_SOURCE_DIR:-${CACHE_ROOT}/SuperLU-6.0.1}"
BUILD_DIR="${SUPERLU_BUILD_DIR:-${CACHE_ROOT}/SuperLU-6.0.1-build}"
OUTPUT_DIR="${SUPERLU_OUTPUT_DIR:-${ROOT}/test_reports/somelinalg_sparse_lu_reference/native}"
PARALLEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

for tool in git cmake ninja cc; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'Required tool not found: %s\n' "$tool" >&2
    exit 2
  }
done

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone --depth 1 --branch v6.0.1 https://github.com/xiaoyeli/superlu.git "$SOURCE_DIR"
fi
actual_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$SUPERLU_COMMIT" ]]; then
  printf 'Expected SuperLU commit %s, got %s at %s\n' "$SUPERLU_COMMIT" "$actual_commit" "$SOURCE_DIR" >&2
  exit 3
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -Denable_internal_blaslib=ON \
  -Denable_single=OFF \
  -Denable_double=ON \
  -Denable_complex=OFF \
  -Denable_complex16=OFF \
  -Denable_fortran=OFF \
  -Denable_tests=OFF \
  -Denable_examples=OFF \
  -Denable_doc=OFF
cmake --build "$BUILD_DIR" --parallel "$PARALLEL"

superlu_archive="${BUILD_DIR}/SRC/libsuperlu.a"
blas_archive="${BUILD_DIR}/CBLAS/libblas.a"
[[ -f "$superlu_archive" ]] || { printf 'Missing archive: %s\n' "$superlu_archive" >&2; exit 4; }
[[ -f "$blas_archive" ]] || { printf 'Missing archive: %s\n' "$blas_archive" >&2; exit 4; }
mkdir -p "$OUTPUT_DIR"
bridge="${OUTPUT_DIR}/libsuperlu_bridge.so"
cc -O3 -DNDEBUG -fPIC -shared -Wl,--no-undefined \
  -I"${SOURCE_DIR}/SRC" \
  "${ROOT}/src/somelinalg/sparse_lu_reference/superlu_bridge.c" \
  "$superlu_archive" "$blas_archive" -lm -lrt -o "$bridge"

printf 'SuperLU 6.0.1 commit=%s\n' "$actual_commit"
printf 'target=%s\n' "$(uname -sm)"
printf 'bridge=%s\n' "$bridge"
printf 'Set SUPERLU_LIBRARY=%s before selecting BvpSciSparseFactorization::SuperLu\n' "$bridge"

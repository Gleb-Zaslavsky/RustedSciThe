/* Version-pinned C ABI bridge shared by somelinalg's SuperLU API and controls. */
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "slu_ddefs.h"

#if defined(_WIN32)
#include <windows.h>
static uint64_t rs_now_ns(void) {
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency) ||
        frequency.QuadPart <= 0) {
        return 0;
    }
    const uint64_t ticks = (uint64_t)counter.QuadPart;
    const uint64_t hz = (uint64_t)frequency.QuadPart;
    /* Split quotient/remainder to avoid overflowing ticks * 1e9 on uptime. */
    return (ticks / hz) * 1000000000ULL + ((ticks % hz) * 1000000000ULL) / hz;
}
#define RS_EXPORT __declspec(dllexport)
#else
#include <time.h>
static uint64_t rs_now_ns(void) {
    struct timespec value = {0, 0};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}
#define RS_EXPORT __attribute__((visibility("default")))
#endif

/* Validate the public bridge boundary independently of the Rust wrapper. */
static int rs_valid_csc(int32_t n, int32_t nnz, const int32_t *col_ptr,
                        const int32_t *row_idx, const double *values) {
    if (n <= 0 || nnz < 0 || !col_ptr || (nnz > 0 && (!row_idx || !values))) return 0;
    if (col_ptr[0] != 0 || col_ptr[n] != nnz) return 0;
    for (int32_t column = 0; column < n; ++column) {
        const int32_t begin = col_ptr[column];
        const int32_t end = col_ptr[column + 1];
        if (begin < 0 || end < begin || end > nnz) return 0;
        int32_t previous_row = -1;
        for (int32_t index = begin; index < end; ++index) {
            if (row_idx[index] <= previous_row || row_idx[index] >= n ||
                !isfinite(values[index])) return 0;
            previous_row = row_idx[index];
        }
    }
    return 1;
}

RS_EXPORT int32_t rs_superlu_version(void) {
    return SUPERLU_MAJOR_VERSION * 10000 + SUPERLU_MINOR_VERSION * 100 + SUPERLU_PATCH_VERSION;
}

/*
 * One-shot CSC solve. Input CSC is borrowed and preserved; only the output
 * vector is written. The operation follows the same sequential COLAMD and
 * pivot defaults as SciPy's gstrf/gstrs route, with separate high-resolution
 * wall timings around factorization and triangular solve.
 */
RS_EXPORT int32_t rs_superlu_solve_csc(
    int32_t n,
    int32_t nnz,
    const int32_t *col_ptr,
    const int32_t *row_idx,
    const double *values,
    const double *rhs,
    double *solution,
    int32_t *factor_l_storage_nnz,
    int32_t *factor_u_storage_nnz,
    int32_t *factor_l_csc_nnz,
    int32_t *factor_u_csc_nnz,
    int32_t *solver_info,
    uint64_t *factor_ns,
    uint64_t *solve_ns
) {
    if (!rs_valid_csc(n, nnz, col_ptr, row_idx, values) || !rhs || !solution ||
        !factor_l_storage_nnz || !factor_u_storage_nnz ||
        !factor_l_csc_nnz || !factor_u_csc_nnz || !solver_info ||
        !factor_ns || !solve_ns) {
        return -1;
    }

    *factor_l_storage_nnz = 0;
    *factor_u_storage_nnz = 0;
    *factor_l_csc_nnz = 0;
    *factor_u_csc_nnz = 0;
    *solver_info = 0;
    *factor_ns = 0;
    *solve_ns = 0;

    int *perm_c = intMalloc(n);
    int *perm_r = intMalloc(n);
    if (!perm_c || !perm_r) {
        if (perm_c) SUPERLU_FREE(perm_c);
        if (perm_r) SUPERLU_FREE(perm_r);
        return -2;
    }

    SuperMatrix A = {0};
    SuperMatrix B = {0};
    SuperMatrix L = {0};
    SuperMatrix U = {0};
    SuperMatrix AC = {0};
    SuperLUStat_t stat;
    superlu_options_t options;
    GlobalLU_t glu = {0};
    StatInit(&stat);
    set_default_options(&options);
    options.ColPerm = COLAMD;
    options.DiagPivotThresh = 1.0;
    options.PrintStat = NO;

    memcpy(solution, rhs, (size_t)n * sizeof(double));
    dCreate_CompCol_Matrix(&A, n, n, nnz, (double *)values, (int *)row_idx,
                           (int *)col_ptr, SLU_NC, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&B, n, 1, solution, n, SLU_DN, SLU_D, SLU_GE);

    int *etree = int32Malloc(n);
    if (!etree) {
        if (A.Store) Destroy_SuperMatrix_Store(&A);
        if (B.Store) Destroy_SuperMatrix_Store(&B);
        StatFree(&stat);
        SUPERLU_FREE(perm_c);
        SUPERLU_FREE(perm_r);
        return -2;
    }

    const uint64_t factor_start = rs_now_ns();
    get_perm_c(options.ColPerm, &A, perm_c);
    sp_preorder(&options, &A, perm_c, etree, &AC);
    dgstrf(&options, &AC, sp_ienv(2), sp_ienv(1), etree, NULL, 0,
           perm_c, perm_r, &L, &U, &glu, &stat, solver_info);
    const uint64_t factor_end = rs_now_ns();
    *factor_ns = factor_end - factor_start;

    if (*solver_info == 0) {
        const uint64_t solve_start = rs_now_ns();
        int solve_info = 0;
        dgstrs(NOTRANS, &L, &U, perm_c, perm_r, &B, &stat, &solve_info);
        const uint64_t solve_end = rs_now_ns();
        *solve_ns = solve_end - solve_start;
        if (solve_info != 0) *solver_info = -3;
    }

    if (*solver_info == 0) {
        SCformat *l_store = (SCformat *)L.Store;
        NCformat *u_store = (NCformat *)U.Store;
        double *l_values = (double *)l_store->nzval;
        double *u_values = (double *)u_store->nzval;
        *factor_l_storage_nnz = (int32_t)l_store->nnz;
        *factor_u_storage_nnz = (int32_t)u_store->nnz;

        /* Match SciPy's LU_to_csc conversion, not the padded native buffers.
           SciPy moves the upper triangle stored in L's supernodes into U and
           inserts a unit diagonal into the materialized L. */
        for (int isup = 0; isup <= l_store->nsuper; ++isup) {
            const int icol_start = l_store->sup_to_col[isup];
            const int icol_end = l_store->sup_to_col[isup + 1];
            const int_t row_start = l_store->rowind_colptr[icol_start];
            const int_t row_end = l_store->rowind_colptr[icol_start + 1];

            for (int icol = icol_start; icol < icol_end; ++icol) {
                for (int_t iptr = u_store->colptr[icol];
                     iptr < u_store->colptr[icol + 1]; ++iptr) {
                    if (u_values[iptr] != 0.0) ++(*factor_u_csc_nnz);
                }

                double *src = l_values + l_store->nzval_colptr[icol];
                int_t iptr = row_start;
                for (; iptr < row_end; ++iptr) {
                    if (l_store->rowind[iptr] > icol) break;
                    if (*src != 0.0) ++(*factor_u_csc_nnz);
                    ++src;
                }

                ++(*factor_l_csc_nnz); /* SciPy explicitly stores unit L diagonal. */
                for (; iptr < row_end; ++iptr) {
                    if (*src != 0.0) ++(*factor_l_csc_nnz);
                    ++src;
                }
            }
        }
    }

    SUPERLU_FREE(etree);
    if (AC.Store) Destroy_CompCol_Permuted(&AC);
    if (L.Store) Destroy_SuperNode_Matrix(&L);
    if (U.Store) Destroy_CompCol_Matrix(&U);
    if (A.Store) Destroy_SuperMatrix_Store(&A);
    if (B.Store) Destroy_SuperMatrix_Store(&B);
    StatFree(&stat);
    SUPERLU_FREE(perm_c);
    SUPERLU_FREE(perm_r);
    return 0;
}

typedef struct {
    int32_t n;
    int *perm_c;
    int *perm_r;
    SuperMatrix L;
    SuperMatrix U;
    int32_t factor_l_storage_nnz;
    int32_t factor_u_storage_nnz;
    int32_t factor_l_csc_nnz;
    int32_t factor_u_csc_nnz;
} rs_superlu_factor;

static void rs_superlu_factor_destroy_contents(rs_superlu_factor *factor) {
    if (!factor) return;
    if (factor->L.Store) Destroy_SuperNode_Matrix(&factor->L);
    if (factor->U.Store) Destroy_CompCol_Matrix(&factor->U);
    if (factor->perm_c) SUPERLU_FREE(factor->perm_c);
    if (factor->perm_r) SUPERLU_FREE(factor->perm_r);
    free(factor);
}

/*
 * Persistent factor API for BVP Newton iterations. The CSC input is borrowed
 * only during factorization; the returned opaque handle owns L/U and pivot
 * permutations so subsequent RHS solves do not repeat symbolic or numeric LU.
 */
RS_EXPORT int32_t rs_superlu_factor_csc(
    int32_t n,
    int32_t nnz,
    const int32_t *col_ptr,
    const int32_t *row_idx,
    const double *values,
    void **factor_handle,
    int32_t *factor_l_storage_nnz,
    int32_t *factor_u_storage_nnz,
    int32_t *factor_l_csc_nnz,
    int32_t *factor_u_csc_nnz,
    int32_t *solver_info,
    uint64_t *factor_ns
) {
    if (!rs_valid_csc(n, nnz, col_ptr, row_idx, values) || !factor_handle ||
        !factor_l_storage_nnz || !factor_u_storage_nnz || !factor_l_csc_nnz ||
        !factor_u_csc_nnz || !solver_info || !factor_ns) {
        return -1;
    }
    *factor_handle = NULL;
    *factor_l_storage_nnz = 0;
    *factor_u_storage_nnz = 0;
    *factor_l_csc_nnz = 0;
    *factor_u_csc_nnz = 0;
    *solver_info = 0;
    *factor_ns = 0;

    rs_superlu_factor *factor = (rs_superlu_factor *)calloc(1, sizeof(*factor));
    if (!factor) return -2;
    factor->n = n;
    factor->perm_c = intMalloc(n);
    factor->perm_r = intMalloc(n);
    int *etree = int32Malloc(n);
    if (!factor->perm_c || !factor->perm_r || !etree) {
        if (etree) SUPERLU_FREE(etree);
        rs_superlu_factor_destroy_contents(factor);
        return -2;
    }

    SuperMatrix A = {0};
    SuperMatrix AC = {0};
    SuperLUStat_t stat;
    superlu_options_t options;
    GlobalLU_t glu = {0};
    StatInit(&stat);
    set_default_options(&options);
    options.ColPerm = COLAMD;
    options.DiagPivotThresh = 1.0;
    options.PrintStat = NO;
    dCreate_CompCol_Matrix(&A, n, n, nnz, (double *)values, (int *)row_idx,
                           (int *)col_ptr, SLU_NC, SLU_D, SLU_GE);

    const uint64_t factor_start = rs_now_ns();
    get_perm_c(options.ColPerm, &A, factor->perm_c);
    sp_preorder(&options, &A, factor->perm_c, etree, &AC);
    dgstrf(&options, &AC, sp_ienv(2), sp_ienv(1), etree, NULL, 0,
           factor->perm_c, factor->perm_r, &factor->L, &factor->U,
           &glu, &stat, solver_info);
    *factor_ns = rs_now_ns() - factor_start;

    if (*solver_info == 0) {
        SCformat *l_store = (SCformat *)factor->L.Store;
        NCformat *u_store = (NCformat *)factor->U.Store;
        double *l_values = (double *)l_store->nzval;
        double *u_values = (double *)u_store->nzval;
        factor->factor_l_storage_nnz = (int32_t)l_store->nnz;
        factor->factor_u_storage_nnz = (int32_t)u_store->nnz;
        for (int isup = 0; isup <= l_store->nsuper; ++isup) {
            const int icol_start = l_store->sup_to_col[isup];
            const int icol_end = l_store->sup_to_col[isup + 1];
            const int_t row_start = l_store->rowind_colptr[icol_start];
            const int_t row_end = l_store->rowind_colptr[icol_start + 1];
            for (int icol = icol_start; icol < icol_end; ++icol) {
                for (int_t iptr = u_store->colptr[icol];
                     iptr < u_store->colptr[icol + 1]; ++iptr) {
                    if (u_values[iptr] != 0.0) ++factor->factor_u_csc_nnz;
                }
                double *src = l_values + l_store->nzval_colptr[icol];
                int_t iptr = row_start;
                for (; iptr < row_end; ++iptr) {
                    if (l_store->rowind[iptr] > icol) break;
                    if (*src != 0.0) ++factor->factor_u_csc_nnz;
                    ++src;
                }
                ++factor->factor_l_csc_nnz;
                for (; iptr < row_end; ++iptr) {
                    if (*src != 0.0) ++factor->factor_l_csc_nnz;
                    ++src;
                }
            }
        }
        *factor_l_storage_nnz = factor->factor_l_storage_nnz;
        *factor_u_storage_nnz = factor->factor_u_storage_nnz;
        *factor_l_csc_nnz = factor->factor_l_csc_nnz;
        *factor_u_csc_nnz = factor->factor_u_csc_nnz;
        *factor_handle = factor;
    } else {
        rs_superlu_factor_destroy_contents(factor);
    }

    SUPERLU_FREE(etree);
    if (AC.Store) Destroy_CompCol_Permuted(&AC);
    if (A.Store) Destroy_SuperMatrix_Store(&A);
    StatFree(&stat);
    return 0;
}

RS_EXPORT int32_t rs_superlu_solve_factored(
    void *factor_handle,
    double *rhs_inout,
    uint64_t *solve_ns
) {
    if (!factor_handle || !rhs_inout || !solve_ns) return -1;
    rs_superlu_factor *factor = (rs_superlu_factor *)factor_handle;
    SuperMatrix B = {0};
    dCreate_Dense_Matrix(&B, factor->n, 1, rhs_inout, factor->n,
                         SLU_DN, SLU_D, SLU_GE);
    SuperLUStat_t stat;
    StatInit(&stat);
    int solve_info = 0;
    const uint64_t solve_start = rs_now_ns();
    dgstrs(NOTRANS, &factor->L, &factor->U, factor->perm_c, factor->perm_r,
           &B, &stat, &solve_info);
    *solve_ns = rs_now_ns() - solve_start;
    if (B.Store) Destroy_SuperMatrix_Store(&B);
    StatFree(&stat);
    return solve_info == 0 ? 0 : -3;
}

RS_EXPORT void rs_superlu_destroy_factor(void *factor_handle) {
    rs_superlu_factor_destroy_contents((rs_superlu_factor *)factor_handle);
}

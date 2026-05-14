# LSODE2 DSTODA Label Mapping (Fortran -> Rust)

This map documents how Fortran DSTODA label families are represented in Rust control flow.

## Terminal / near-terminal label families

| Fortran label family | Rust entry point | Expected retry action | Expected control-plane signature |
|---|---|---|---|
| `410` (stale-J refresh same step) | `Lsode2StepCycle::retry_after_stale_jacobian_nonlinear_failure` | `RetryWithJacobianRefresh` | `KFLAG=ConvergenceFailure`, `ICF=RefreshRequested`, `IREDO=CorrectorRefreshSameStep`, `IPUP=NeedsJacobianUpdate(FailurePath)` |
| `430` retry branch | `Lsode2StepCycle::retry_after_stale_jacobian_nonlinear_failure` (second pass) | `RetryWithJacobianRefresh` | `KFLAG=ConvergenceFailure`, `ICF=RefreshDidNotRecover`, `IREDO=CorrectorFailureRetry` |
| `430` terminal (MXNCF) | `Lsode2StepCycle::reject_after_nonlinear_failure` | `FailRepeatedConvergenceFailures` | `KFLAG=RepeatedConvergenceFailure`, `IPUP=UpToDate` |
| `500` (regular error-test reject) | `Lsode2StepCycle::finish_with_local_error` | `Retry` | `KFLAG=ErrorTestFailure`, `IRET=RetryAfterErrorTestFailure`, `IREDO=ErrorTestRetry` |
| `620` (error-test retry with clamp) | `Lsode2StepCycle::finish_with_local_error` after one reject | `Retry` | same flags as `500`; RH clamp path enforced by step-size assertion in test |
| `640` reset branch (nonterminal) | `Lsode2StepCycle::finish_with_local_error` after `KFLAG<=-3` precondition | `Retry` | `KFLAG=ErrorTestFailure`, `IRET=RestartWithDerivativeRefresh`, `IREDO=RepeatedErrorReset`, `IPUP=NeedsJacobianUpdate(FailurePath)` |
| `640` terminal repeated error | `Lsode2StepCycle::finish_with_local_error` with tight `max_error_test_failures` | `FailRepeatedErrorTestFailures` | `KFLAG=RepeatedErrorTestFailure`, `IRET=NormalFlow` |
| `670` (HMIN terminal) | `Lsode2StepCycle::reject_after_nonlinear_failure` with HMIN guard | `FailStepSizeUnderflow` | `KFLAG=RepeatedConvergenceFailure`, `IPUP=UpToDate` |

## Locked regression test

The matrix above is verified by:

- `numerical::LSODE2::parity_micro::dstoda_label_by_label_matrix_terminal_near_terminal_replay_matches_fortran_style`

## Note on `KFLAG` vs `ISTATE`

In ODEPACK, repeated error-test termination is surfaced to the caller as `ISTATE = -4`.
In LSODE2 Rust internals, `KFLAG` tracks only DSTODA classes (`0`, `-1`, `-2` groups), so repeated error-test terminal is represented as `KFLAG=RepeatedErrorTestFailure` (`code=-1`) plus retry action `FailRepeatedErrorTestFailures`.

use super::error::BandedError;

/// High-level policy for selecting a native linear solver backend.
///
/// For compact banded matrices, [`LinearSolverPolicy::Auto`] and
/// [`LinearSolverPolicy::ForceBanded`] use the faithful LAPACK-style banded LU
/// implementation. The dense general-pivot backend is kept as an explicit
/// diagnostic/legacy choice rather than a default path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolverPolicy {
    Auto,
    ForceBlockTridiagonal,
    ForceBlockTridiagonalConsistent,
    ForceBanded,
    ForceGeneralBandedPartialPivot,
    ForceFaerSparse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    Never,
    ToFaerSparse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearSolverConfig {
    /// Backend-selection policy.
    pub policy: LinearSolverPolicy,
    /// Optional fallback when the preferred native backend cannot factorize.
    pub fallback: FallbackPolicy,
    /// Guarded iterative-refinement attempts after the direct solve.
    ///
    /// The default is `0`: the faithful LAPACK-style banded solver already
    /// mirrors `DGBTRF`/`DGBTRS`, and current BVP workloads do not benefit from
    /// an unconditional extra refinement pass. Turn this on only for targeted
    /// experiments.
    pub iterative_refinement_steps: usize,
}

impl Default for LinearSolverConfig {
    fn default() -> Self {
        Self {
            policy: LinearSolverPolicy::Auto,
            fallback: FallbackPolicy::ToFaerSparse,
            iterative_refinement_steps: 0,
        }
    }
}

impl LinearSolverConfig {
    /// Default automatic policy.
    ///
    /// On a [`BandedAssembly`](crate::somelinalg::banded::BandedAssembly) this
    /// resolves to the faithful LAPACK-style banded LU backend with no
    /// refinement. On structured block-tridiagonal input it keeps using the
    /// structured native policy.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Forces the faithful LAPACK-style banded LU backend.
    ///
    /// This is the recommended default whenever the input is a general compact
    /// banded matrix rather than a known block-tridiagonal chain.
    pub fn faithful_banded() -> Self {
        Self {
            policy: LinearSolverPolicy::ForceBanded,
            fallback: FallbackPolicy::ToFaerSparse,
            iterative_refinement_steps: 0,
        }
    }

    /// Forces faithful banded LU and enables guarded iterative refinement.
    ///
    /// Prefer [`Self::faithful_banded`] for production defaults; this helper is
    /// mainly for diagnostics and solver-comparison tables.
    pub fn faithful_banded_with_refinement(iterative_refinement_steps: usize) -> Self {
        Self {
            iterative_refinement_steps,
            ..Self::faithful_banded()
        }
    }

    /// Returns a copy with a different guarded-refinement budget.
    pub fn with_iterative_refinement_steps(mut self, iterative_refinement_steps: usize) -> Self {
        self.iterative_refinement_steps = iterative_refinement_steps;
        self
    }
}

#[derive(Debug)]
pub enum LinearSolveError {
    Native(BandedError),
    Faer(String),
    InvalidPolicy(&'static str),
}

impl From<BandedError> for LinearSolveError {
    fn from(value: BandedError) -> Self {
        Self::Native(value)
    }
}

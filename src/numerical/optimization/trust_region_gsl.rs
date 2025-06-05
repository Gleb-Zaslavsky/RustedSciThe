/// gsl direct translation
use nalgebra::{DMatrix, DVector, Scalar};
use std::fmt;

// Error types
#[derive(Debug, Clone)]
pub enum GslError {
    NoMem,
    NoProgress,
    Sanity,
    Success,
    Failure,
}

impl fmt::Display for GslError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GslError::NoMem => write!(f, "failed to allocate memory"),
            GslError::NoProgress => write!(f, "no progress made"),
            GslError::Sanity => write!(f, "sanity check failed"),
            GslError::Success => write!(f, "success"),
            GslError::Failure => write!(f, "failure"),
        }
    }
}

pub type Result<T> = std::result::Result<T, GslError>;

// Forward declarations for external functions (not implemented here)
pub fn gsl_multifit_nlinear_eval_f(
    fdf: &mut MultifitNlinearFdf,
    x: &DVector<f64>,
    swts: Option<&DVector<f64>>,
    f: &mut DVector<f64>,
) -> Result<()> {
    // External function - not implemented
    unimplemented!("gsl_multifit_nlinear_eval_f")
}

pub fn gsl_multifit_nlinear_eval_df(
    x: &DVector<f64>,
    f: &DVector<f64>,
    swts: Option<&DVector<f64>>,
    h_df: f64,
    fdtype: FdType,
    fdf: &mut MultifitNlinearFdf,
    j: &mut DMatrix<f64>,
    workn: &mut DVector<f64>,
) -> Result<()> {
    // External function - not implemented
    unimplemented!("gsl_multifit_nlinear_eval_df")
}

pub fn nielsen_init(
    j: &DMatrix<f64>,
    diag: &DVector<f64>,
    mu: &mut f64,
    nu: &mut i64,
) -> Result<()> {
    // External function - not implemented
    unimplemented!("nielsen_init")
}

pub fn nielsen_accept(rho: f64, mu: &mut f64, nu: &mut i64) -> Result<()> {
    // External function - not implemented
    unimplemented!("nielsen_accept")
}

pub fn nielsen_reject(mu: &mut f64, nu: &mut i64) -> Result<()> {
    // External function - not implemented
    unimplemented!("nielsen_reject")
}

// Type definitions
#[derive(Debug, Clone)]
pub enum FdType {
    Forward,
    Central,
}
#[derive(Debug, Clone)]
pub struct MultifitNlinearFdf {
    // Function definition structure - not fully implemented
}

pub trait Scale {
    fn init(&self, j: &DMatrix<f64>, diag: &mut DVector<f64>);
    fn update(&self, j: &DMatrix<f64>, diag: &mut DVector<f64>);
}

pub trait Solver {
    fn alloc(&self, n: usize, p: usize) -> Box<dyn SolverState>;
    fn free(&self, state: Box<dyn SolverState>);
    fn rcond(&self, rcond: &mut f64, state: &dyn SolverState) -> Result<()>;
}


pub trait SolverState {}

pub trait Trs {
    fn alloc(&self, params: &MultifitNlinearParameters, n: usize, p: usize) -> Box<dyn TrsState>;
    fn free(&self, state: Box<dyn TrsState>);
    fn init(&self, trust_state: &MultifitNlinearTrustState, state: &mut dyn TrsState)
    -> Result<()>;
    fn preloop(
        &self,
        trust_state: &MultifitNlinearTrustState,
        state: &mut dyn TrsState,
    ) -> Result<()>;
    fn step(
        &self,
        trust_state: &MultifitNlinearTrustState,
        delta: f64,
        dx: &mut DVector<f64>,
        state: &mut dyn TrsState,
    ) -> Result<()>;
    fn preduction(
        &self,
        trust_state: &MultifitNlinearTrustState,
        dx: &DVector<f64>,
        pred_reduction: &mut f64,
        state: &mut dyn TrsState,
    ) -> Result<()>;
}

pub trait TrsState {}

pub struct MultifitNlinearParameters {
    pub trs: Box<dyn Trs>,
    pub solver: Box<dyn Solver>,
    pub scale: Box<dyn Scale>,
    pub h_df: f64,
    pub fdtype: FdType,
    pub factor_up: f64,
    pub factor_down: f64,
    pub avmax: f64,
}

pub struct MultifitNlinearTrustState<'a> {
    pub x: Option<&'a DVector<f64>>,
    pub f: &'a DVector<f64>,
    pub g: &'a DVector<f64>,
    pub j: &'a DMatrix<f64>,
    pub diag: &'a DVector<f64>,
    pub sqrt_wts: Option<&'a DVector<f64>>,
    pub mu: &'a mut f64,
    pub params: &'a MultifitNlinearParameters,
    pub solver_state: &'a mut dyn SolverState,
    pub fdf: Option<&'a mut MultifitNlinearFdf>,
    pub avratio: &'a mut f64,
}

pub struct TrustState {
    n: usize,                           // number of observations
    p: usize,                           // number of parameters
    delta: f64,                         // trust region radius
    mu: f64,                            // LM parameter
    nu: i64,                            // for updating LM parameter
    diag: DVector<f64>,                 // D = diag(J^T J)
    x_trial: DVector<f64>,              // trial parameter vector
    f_trial: DVector<f64>,              // trial function vector
    workp: DVector<f64>,                // workspace, length p
    workn: DVector<f64>,                // workspace, length n
    trs_state: Box<dyn TrsState>,       // workspace for trust region subproblem
    solver_state: Box<dyn SolverState>, // workspace for linear least squares solver
    avratio: f64,                       // current |a| / |v|
    params: MultifitNlinearParameters,  // tunable parameters
}

impl TrustState {
    pub fn alloc(params: &MultifitNlinearParameters, n: usize, p: usize) -> Result<Self> {
        let diag = DVector::zeros(p);
        let workp = DVector::zeros(p);
        let workn = DVector::zeros(n);
        let x_trial = DVector::zeros(p);
        let f_trial = DVector::zeros(n);

        let trs_state = params.trs.alloc(params, n, p);
        let solver_state = params.solver.alloc(n, p);

        Ok(TrustState {
            n,
            p,
            delta: 0.0,
            mu: 0.0,
            nu: 0,
            diag,
            x_trial,
            f_trial,
            workp,
            workn,
            trs_state,
            solver_state,
            avratio: 0.0,
            params: *params,
        })
    }

    pub fn init(
        &mut self,
        swts: Option<&DVector<f64>>,
        fdf: &mut MultifitNlinearFdf,
        x: &DVector<f64>,
        f: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
        g: &mut DVector<f64>,
    ) -> Result<()> {
        // evaluate function and Jacobian at x and apply weight transform
        gsl_multifit_nlinear_eval_f(fdf, x, swts, f)?;

        gsl_multifit_nlinear_eval_df(
            x,
            f,
            swts,
            self.params.h_df,
            self.params.fdtype.clone(),
            fdf,
            j,
            &mut self.workn,
        )?;

        // compute g = J^T f
        g.gemv(1.0, &j.transpose(), f, 0.0);

        // initialize diagonal scaling matrix D
        self.params.scale.init(j, &mut self.diag);

        // compute initial trust region radius
        let dx = trust_scaled_norm(&self.diag, x);
        self.delta = 0.3 * f64::max(1.0, dx);

        // initialize LM parameter
        nielsen_init(j, &self.diag, &mut self.mu, &mut self.nu)?;

        // initialize trust region method solver
        let mut trust_state = MultifitNlinearTrustState {
            x: Some(x),
            f,
            g,
            j,
            diag: &self.diag,
            sqrt_wts: swts,
            mu: &mut self.mu,
            params: &self.params,
            solver_state: self.solver_state.as_mut(),
            fdf: Some(fdf),
            avratio: &mut self.avratio,
        };

        self.params
            .trs
            .init(&trust_state, self.trs_state.as_mut())?;

        // set default parameters
        self.avratio = 0.0;

        Ok(())
    }
/*
trust_iterate()
  This function performs 1 iteration of the trust region algorithm.
It calls a user-specified method for computing the next step
(LM or dogleg), then tests if the computed step is acceptable.

Args: vstate - trust workspace
      swts   - data weights (NULL if unweighted)
      fdf    - function and Jacobian pointers
      x      - on input, current parameter vector
               on output, new parameter vector x + dx
      f      - on input, f(x)
               on output, f(x + dx)
      J      - on input, J(x)
               on output, J(x + dx)
      g      - on input, g(x) = J(x)' f(x)
               on output, g(x + dx) = J(x + dx)' f(x + dx)
      dx     - (output only) parameter step vector

Return:
1) GSL_SUCCESS if we found a step which reduces the cost
function

2) GSL_ENOPROG if 15 successive attempts were to made to
find a good step without success

3) If a scaling matrix D is used, inputs and outputs are
set to the unscaled quantities (ie: J and g)
*/
    pub fn iterate(
        &mut self,
        swts: Option<&DVector<f64>>,
        fdf: &mut MultifitNlinearFdf,
        x: &mut DVector<f64>,
        f: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
        g: &mut DVector<f64>,
        dx: &mut DVector<f64>,
    ) -> Result<()> {
        // ratio actual_reduction/predicted_reduction
        let mut rho: f64 = 1.0;
        // found step dx 
        let mut foundstep = false;
         // consecutive rejected steps 
        let mut bad_steps = 0;
        
        // initialize trust region subproblem with this Jacobian
        // We need to create the trust_state temporarily for preloop
        {
            let trust_state = MultifitNlinearTrustState {
                x: Some(x),
                f,
                g,
                j,
                diag: &self.diag,
                sqrt_wts: swts,
                mu: &mut self.mu,
                params: &self.params,
                solver_state: self.solver_state.as_mut(),
                fdf: Some(fdf),
                avratio: &mut self.avratio,
            };
            
            self.params.trs.preloop(&trust_state, self.trs_state.as_mut())?;
        } // trust_state is dropped here, releasing all borrows

        // loop until we find an acceptable step dx
        while !foundstep {
            // calculate new step - create trust_state temporarily
            let step_result = {
                let trust_state = MultifitNlinearTrustState {
                    x: Some(x),
                    f,
                    g,
                    j,
                    diag: &self.diag,
                    sqrt_wts: swts,
                    mu: &mut self.mu,
                    params: &self.params,
                    solver_state: self.solver_state.as_mut(),
                    fdf: Some(fdf),
                    avratio: &mut self.avratio,
                };
                
                self.params.trs.step(&trust_state, self.delta, dx, self.trs_state.as_mut())
            }; // trust_state is dropped here
        //occasionally the iterative methods (ie: CG Steihaug) can fail to find a step,
        // so in this case skip rho calculation and count it as a rejected step 
            if step_result.is_ok() {
                // compute x_trial = x + dx
                trust_trial_step(x, dx, &mut self.x_trial);

                // compute f_trial = f(x + dx)
                gsl_multifit_nlinear_eval_f(fdf, &self.x_trial, swts, &mut self.f_trial)?;

                // check if step should be accepted or rejected
                if self
                    .trust_eval_step(f, &self.f_trial.clone(), g, j, dx, &mut rho)
                    .is_ok()
                {
                    foundstep = true;
                }
            } else {
                // an iterative TRS method failed to find a step vector
                rho = -1.0;
            }

            // update trust region radius
            if rho > 0.75 {
                self.delta *= self.params.factor_up;
            } else if rho < 0.25 {
                self.delta /= self.params.factor_down;
            }

            if foundstep {
                // step was accepted

                // compute J <- J(x + dx)
                gsl_multifit_nlinear_eval_df(
                    &self.x_trial,
                    &self.f_trial,
                    swts,
                    self.params.h_df,
                    self.params.fdtype.clone(),
                    fdf,
                    j,
                    &mut self.workn,
                )?;

                // update x <- x + dx
                x.copy_from(&self.x_trial);

                // update f <- f(x + dx)
                f.copy_from(&self.f_trial);

                // compute new g = J^T f
                g.gemv(1.0, &j.transpose(), f, 0.0);

                // update scaling matrix D
                self.params.scale.update(j, &mut self.diag);

                // step accepted, decrease LM parameter
                nielsen_accept(rho, &mut self.mu, &mut self.nu)?;

                bad_steps = 0;
            } else {
                // step rejected, increase LM parameter
                nielsen_reject(&mut self.mu, &mut self.nu)?;

                bad_steps += 1;
                if bad_steps > 15 {
                    // if more than 15 consecutive rejected steps, report no progress
                    return Err(GslError::NoProgress);
                }
            }
        }

        Ok(())
    }

    pub fn rcond(&self, rcond: &mut f64) -> Result<()> {
        self.params.solver.rcond(rcond, self.solver_state.as_ref())
    }

    pub fn avratio(&self) -> f64 {
        self.avratio
    }
/*
trust_eval_step()
  Evaluate proposed step to determine if it should be
accepted or rejected
*/
    fn trust_eval_step(
        &mut self,
        // 
        f: &DVector<f64>,
        f_trial: &DVector<f64>,
        g: &DVector<f64>,
        j: &DMatrix<f64>,
        dx: &DVector<f64>,
        rho: &mut f64,
    ) -> Result<()> {
        // reject step if acceleration is too large compared to velocity
        if self.avratio > self.params.avmax {
            return Err(GslError::Failure);
        }

        // compute rho
        *rho = self.trust_calc_rho(f, f_trial, g, j, dx);
        if *rho <= 0.0 {
            return Err(GslError::Failure);
        }

        Ok(())
    }
    /*
    trust_calc_rho()
      Calculate ratio of actual reduction to predicted
    reduction.

    rho = actual_reduction / predicted_reduction

    actual_reduction = 1 - ( ||f+|| / ||f|| )^2
    predicted_reduction = -2 g^T dx / ||f||^2 - ( ||J*dx|| / ||f|| )^2
                        = -2 fhat . beta - ||beta||^2

    where: beta = J*dx / ||f||

    Inputs: f        - f(x)
            f_trial  - f(x + dx)
            g        - gradient J^T f
            J        - Jacobian
            dx       - proposed step, size p
            state    - workspace

    Return: rho = actual_reduction / predicted_reduction
    If actual_reduction is < 0, return rho = -1
    */
    fn trust_calc_rho(
        &mut self,
        // f(x)
        f: &DVector<f64>,
        // f(x+dx)
        f_trial: &DVector<f64>,
        g: &DVector<f64>,
        // jacobian
        j: &DMatrix<f64>,
        // proposed step
        dx: &DVector<f64>,
    ) -> f64 {
        let normf = f.norm();
        let normf_trial = f_trial.norm();

        // if ||f(x+dx)|| > ||f(x)|| reject step immediately
        if normf_trial >= normf {
            return -1.0;
        }
        // create trust state instance
        let trust_state = MultifitNlinearTrustState {
            x: None,
            f,
            g,
            j,
            diag: &self.diag,
            sqrt_wts: None,
            mu: &mut self.mu.clone(),
            params: &self.params,
            solver_state: self.solver_state.as_mut(),
            fdf: None,
            avratio: &mut self.avratio.clone(),
        };

        // compute numerator of rho (actual reduction)
        let u = normf_trial / normf;
        let actual_reduction = 1.0 - u * u;

        // compute denominator of rho (predicted reduction)
        let mut pred_reduction = 0.0;
        let status = self.params.trs.preduction(
            &trust_state,
            dx,
            &mut pred_reduction,
            self.trs_state.as_mut(),
        );

        if status.is_err() {
            return -1.0;
        }

        if pred_reduction > 0.0 {
            actual_reduction / pred_reduction
        } else {
            -1.0
        }
    }
}

// Standalone functions (outside impl block)

/// Compute x_trial = x + dx
fn trust_trial_step(x: &DVector<f64>, dx: &DVector<f64>, x_trial: &mut DVector<f64>) {
    let n = x.len();

    for i in 0..n {
        let dxi = dx[i];
        let xi = x[i];
        x_trial[i] = xi + dxi;
    }
}

/// Compute || diag(D) * a ||
fn trust_scaled_norm(d: &DVector<f64>, a: &DVector<f64>) -> f64 {
    let n = a.len();
    let mut e2 = 0.0;

    for i in 0..n {
        let di = d[i];
        let ai = a[i];
        let u = di * ai;
        e2 += u * u;
    }

    e2.sqrt()
}

// Trust region type definition
pub struct TrustType;

impl TrustType {
    pub const fn new() -> Self {
        TrustType
    }

    pub fn name(&self) -> &'static str {
        "trust-region"
    }

    pub fn alloc(
        &self,
        params: &MultifitNlinearParameters,
        n: usize,
        p: usize,
    ) -> Result<TrustState> {
        TrustState::alloc(params, n, p)
    }

    pub fn init(
        &self,
        state: &mut TrustState,
        swts: Option<&DVector<f64>>,
        fdf: &mut MultifitNlinearFdf,
        x: &DVector<f64>,
        f: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
        g: &mut DVector<f64>,
    ) -> Result<()> {
        state.init(swts, fdf, x, f, j, g)
    }

    pub fn iterate(
        &self,
        state: &mut TrustState,
        swts: Option<&DVector<f64>>,
        fdf: &mut MultifitNlinearFdf,
        x: &mut DVector<f64>,
        f: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
        g: &mut DVector<f64>,
        dx: &mut DVector<f64>,
    ) -> Result<()> {
        state.iterate(swts, fdf, x, f, j, g, dx)
    }

    pub fn rcond(&self, state: &TrustState, rcond: &mut f64) -> Result<()> {
        state.rcond(rcond)
    }

    pub fn avratio(&self, state: &TrustState) -> f64 {
        state.avratio()
    }
}

// Global constant
pub const GSL_MULTIFIT_NLINEAR_TRUST: TrustType = TrustType::new();

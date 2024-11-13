#![allow(warnings)]
// Domain1D, OneDim, MultiJac, writelog, and debuglog. 
// 

use std::cmp::max;
use std::cmp::min;
use std::f64::EPSILON;
use std::time::Instant;
use crate::numerical::BVP_NR::Domain1D;
struct Indx {
    m_nv: usize,
    m_np: usize,
}

impl Indx {
    fn new(nv: usize, np: usize) -> Self {
        Indx { m_nv: nv, m_np: np }
    }

    fn index(&self, m: usize, j: usize) -> usize {
        j * self.m_nv + m
    }
}

/*
checking if the solution is within the bounds of the defined domains. It iterates over each component and grid point, calculates the new value
 after applying the step, and checks if the new value violates the bounds. If a violation is detected, it logs an error message
and calculates the factor by which the step should be damped to remain within the bounds.
*/
/// This function checks if the solution is within the bounds of the defined domains.
/// It iterates over each component and grid point, calculates the new value after applying the step,
/// and checks if the new value violates the bounds. If a violation is detected, it logs an error message
/// and calculates the factor by which the step should be damped to remain within the bounds.
///
/// # Parameters
///
/// * `x`: A slice of f64 representing the current solution.
/// * `step`: A slice of f64 representing the step to be applied to the solution.
/// * `r`: A reference to a `Domain1D` struct representing the domain.
/// * `loglevel`: An integer representing the log level.
///
/// # Returns
///
/// A f64 representing the factor by which the step should be damped to remain within the bounds.
fn bound_step(x: &[f64], step: &[f64], r: &Domain1D, loglevel: i32) -> f64 {

    let np = r.n_points();
    let nv = r.n_components();
    let index = Indx::new(nv, np);
    let mut fbound = 1.0;
    let mut wrote_title = false;
    let separator = format!("\n     {:=>69}", ""); // equals sign separator

    for m in 0..nv {
        let above = r.upper_bound(m);
        let below = r.lower_bound(m);

        for j in 0..np {
            let val = x[index.index(m, j)];
            if loglevel > 0 && (val > above + 1.0e-12 || val < below - 1.0e-12) {
                writelog("\nERROR: solution out of bounds.\n");
                writelog(&format!(
                    "domain {}: {:>20}({}) = {:10.3e} ({:10.3e}, {:10.3e})\n",
                    r.domain_index(),
                    r.component_name(m),
                    j,
                    val,
                    below,
                    above
                ));
            }

            let newval = val + step[index.index(m, j)];

            if newval > above {
                fbound = max(0.0, min(fbound, (above - val) / (newval - val)));
            } else if newval < below {
                fbound = min(fbound, (val - below) / (val - newval));
            }

            if loglevel > 1 && (newval > above || newval < below) {
                if !wrote_title {
                    let header = format!(
                        "     {:=>10}Undamped Newton step takes solution out of bounds{:=>10}",
                        "", ""
                    );
                    writelog(&format!("\n{}", header));
                    writelog(&separator);
                    writelog(&format!(
                        "\n     {:<7}   {:23}   {:<9}   {:<9}   {:<9}",
                        "Domain/", "", "Value", "Min", "Max"
                    ));
                    writelog(&format!(
                        "\n     {:8}  {:<9}     {:<9}   {:6}      {:5}       {:5}",
                        "Grid Loc", "Component", "Value", "Change", "Bound", "Bound"
                    ));
                    writelog(&separator);
                    wrote_title = true;
                }
                let domain_length = r.domain_index().to_string().len();
                let grid_length = j.to_string().len();
                let padding = 9;
                let format_string = format!(
                    "{{:<{}d}} / {{:<{}d}}{:>{}}",
                    domain_length, grid_length, "", padding - 3 - domain_length - grid_length
                );
                writelog(&format!("\n     {}", format_string), r.domain_index(), j);
                writelog(&format!(
                    " {:<12} {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}",
                    r.component_name(m),
                    val,
                    step[index.index(m, j)],
                    below,
                    above
                ));
            }
        }
    }

    if loglevel > 1 && wrote_title {
        writelog(&separator);
    }
    fbound
}


fn norm_square(x: &[f64], step: &[f64], r: &Domain1D) -> f64 {
    //Initialize two variables, sum and f2max, to 0.0. These variables will store the sum of squares of the norm and the maximum square of 
    //the norm, respectively.
    let mut sum = 0.0;
    let mut f2max = 0.0;
    // Retrieve the number of components (nv) and the number of points (np) from the domain r.
    let nv = r.n_components();
    let np = r.n_points();
    // terate over each component n from 0 to nv - 1. For each component, calculate the sum of absolute values of the solution vector x at 
    //each point j and store it in the variable esum.
    for n in 0..nv {
        let mut esum = 0.0;
        for j in 0..np {
            esum += x[nv * j + n].abs();
        }
        // Calculate the error weight (ewt) for each component using the relative tolerance (rtol) and absolute tolerance (atol) provided
        // by the domain r.
        let ewt = r.rtol(n) * esum / np as f64 + r.atol(n);
        // Iterate over each point j from 0 to np - 1. For each point, calculate the scaled step value (f) by dividing the step value at
        // that point by the error weight. Update the sum variable by adding the square of f to it. Update the f2max variable by taking the
        // maximum of its current value and the square of f.
        for j in 0..np {
            let f = step[nv * j + n] / ewt;
            sum += f * f;
            f2max = f2max.max(f * f);
        }
    }
    sum
}

struct MultiNewton {
    m_n: usize,
    m_x: Vec<f64>,
    m_stp: Vec<f64>,
    m_stp1: Vec<f64>,
    m_max_damp_iter: usize,
    m_damp_factor: f64,
    m_max_age: usize,
    m_elapsed: f64,
}

impl MultiNewton {
    fn new(sz: usize) -> Self {
        MultiNewton {
            m_n: sz,
            m_x: vec![0.0; sz],
            m_stp: vec![0.0; sz],
            m_stp1: vec![0.0; sz],
            m_max_damp_iter: 10,
            m_damp_factor: 2.0,
            m_max_age: 10,
            m_elapsed: 0.0,
        }
    }

    fn resize(&mut self, sz: usize) {
        self.m_n = sz;
        self.m_x.resize(self.m_n, 0.0);
        self.m_stp.resize(self.m_n, 0.0);
        self.m_stp1.resize(self.m_n, 0.0);
    }

    fn norm2(&self, x: &[f64], step: &[f64], r: &OneDim) -> f64 {
        // The method starts by initializing two variables, sum and f, to 0.0. sum will store the sum of squares of the norm, and f
        // will temporarily store the result of the norm_square function for each domain.
        let mut sum = 0.0;
        let nd = r.n_domains();
        // For each domain, the norm_square function is called with the corresponding slices of x and step, and the domain object. 
        //The result is stored in f.
        for n in 0..nd {
            let f = norm_square(&x[r.start(n)..], &step[r.start(n)..], &r.domain(n));
            sum += f; // The sum of squares of the norm for each domain is accumulated by adding f to sum.
        } // After the loop, the sum is divided by the total number of points in all domains (r.size()) to get the average sum of squares
        // of the norm.
        sum /= r.size() as f64;
        sum.sqrt()
    }
// . This method calculates the Newton step for the given solution vector x and domain r.
    fn step(&self, x: &mut [f64], step: &mut [f64], r: &OneDim, jac: &mut MultiJac, loglevel: i32) {
        // calling the eval method of the OneDim struct r with None as the first argument, the solution vector x as the second argument, and
        // the step vector as the third argument. This evaluates the residual function at the current solution vector x and stores the result 
        //in the step vector.
        r.eval(None, x, step);
        // iterates over each element in the step vector and negates it. This is done to calculate the Newton step, which is the negative of
        // the residual function divided by the Jacobian matrix.
        for n in 0..r.size() {
            step[n] = -step[n];
        }
        // calls the solve method of the MultiJac struct jac with the step vector as both the input and output vectors. This solves the linear 
        //system of equations represented by the Jacobian matrix and the residual function.
        if let Err(_) = jac.solve(step, step) { // If the Jacobian solve is successful (i.e., the return value of the solve method is Ok(())), 
            if jac.info() > 0 { //  the method continues to the next part. If the Jacobian solve fails (i.e., the return value is Err(_)), 
                //the method checks if the Jacobian matrix is singular.
                let row = (jac.info() - 1) as usize;
                for n in 0..r.n_domains() {
                    let dom = r.domain(n);
                    let n_comp = dom.n_components();
                    if row >= dom.loc() && row < dom.loc() + n_comp * dom.n_points() {
                        let offset = row - dom.loc();
                        let pt = offset / n_comp;
                        let comp = offset - pt * n_comp;
                        panic!(
                            "Jacobian is singular for domain {}, component {} at point {}\n(Matrix row {})",
                            dom.id(),
                            dom.component_name(comp),
                            pt,
                            row
                        );
                    }
                }
            }
            panic!("Jacobian solve failed");
        }
    }

    fn bound_step(&self, x0: &[f64], step0: &[f64], r: &OneDim, loglevel: i32) -> f64 {
        let mut fbound = 1.0;
        for i in 0..r.n_domains() {
            fbound = fbound.min(bound_step(&x0[r.start(i)..], &step0[r.start(i)..], &r.domain(i), loglevel));
        }
        fbound
    }

    fn damp_step(
        &self,
        x0: &[f64],
        step0: &[f64],
        x1: &mut [f64],
        step1: &mut [f64],
        s1: &mut f64,
        r: &OneDim,
        jac: &mut MultiJac,
        loglevel: i32,
        writetitle: bool,
    ) -> i32 {
        if loglevel > 0 && writetitle {
            writelog(&format!("\n\n  {:-^70}", " Damped Newton iteration "));
            writelog(&format!(
                "\n  {:<4}  {:<10}   {:<10}  {:<7}  {:<7}  {:<7}  {:<5}  {:<3}\n",
                "Iter", "F_damp", "F_bound", "log(ss)", "log(s0)", "log(s1)", "N_jac", "Age"
            ));
            writelog(&format!("  {:->70}", ""));
        }

        let s0 = self.norm2(x0, step0, r);
        let fbound = self.bound_step(x0, step0, r, loglevel - 1);

        if fbound < 1.0e-10 {
            debuglog("\n  No damped step can be taken without violating solution component bounds.", loglevel);
            return -3;
        }

        let mut alpha = fbound * 1.0;
        let mut m = 0;
        while m < self.m_max_damp_iter {
            for j in 0..self.m_n {
                x1[j] = x0[j] + alpha * step0[j];
            }

            self.step(x1, step1, r, jac, loglevel - 1);
            *s1 = self.norm2(x1, step1, r);

            if loglevel > 0 {
                let ss = r.ssnorm(x1, step1);
                writelog(&format!(
                    "\n  {:<4}  {:<9.3e}   {:<9.3e}   {:>6.3}   {:>6.3}   {:>6.3}    {:<5}  {}/{}",
                    m, alpha, fbound, (ss + EPSILON).log10(), (s0 + EPSILON).log10(), (s1 + EPSILON).log10(),
                    jac.n_evals(), jac.age(), self.m_max_age
                ));
            }

            if *s1 < 1.0 || *s1 < s0 {
                break;
            }
            alpha /= self.m_damp_factor;
            m += 1;
        }

        if m < self.m_max_damp_iter {
            if *s1 > 1.0 {
                debuglog("\n  Damping coefficient found (solution has not converged yet)", loglevel);
                0
            } else {
                debuglog("\n  Damping coefficient found (solution has converged)", loglevel);
                1
            }
        } else {
            debuglog("\n  No damping coefficient found (max damping iterations reached)", loglevel);
            -2
        }
    }

    fn solve(
        &mut self,
        x0: &mut [f64],
        x1: &mut [f64],
        r: &OneDim,
        jac: &mut MultiJac,
        loglevel: i32,
    ) -> i32 {
        let t0 = Instant::now();
        let mut status = 0;
        let mut force_new_jac = false;
        let mut write_header = true;
        let mut s1 = 1.0e30;

        self.m_x.copy_from_slice(x0);

        let rdt = r.rdt();
        let mut n_jac_reeval = 0;
        loop {
            if jac.age() > self.m_max_age {
                if loglevel > 1 {
                    writelog(&format!("\n  Maximum Jacobian age reached ({}), updating it.", self.m_max_age));
                }
                force_new_jac = true;
            }

            if force_new_jac {
                r.eval(None, &mut self.m_x, &mut self.m_stp, 0.0, 0);
                jac.eval(&self.m_x, &self.m_stp, 0.0);
                jac.update_transient(rdt, r.transient_mask().as_slice());
                force_new_jac = false;
            }

            self.step(&mut self.m_x, &mut self.m_stp, r, jac, loglevel - 1);
            jac.increment_age();

            status = self.damp_step(&self.m_x, &self.m_stp, x1, &mut self.m_stp1, &mut s1, r, jac, loglevel - 1, write_header);
            write_header = false;

            if status == 0 {
                self.m_x.copy_from_slice(x1);
            } else if status == 1 {
                if rdt == 0.0 {
                    jac.set_age(0);
                }
                break;
            } else if status < 0 {
                if jac.age() > 1 {
                    force_new_jac = true;
                    if n_jac_reeval > 3 {
                        break;
                    }
                    n_jac_reeval += 1;
                    if loglevel > 1 {
                        writelog("\n  Re-evaluating Jacobian (damping coefficient not found with this Jacobian)");
                    }
                } else {
                    break;
                }
            }
        }

        if loglevel > 1 {
            writelog(&format!("\n  {:->70}", ""));
        }

        if status < 0 {
            self.m_x.copy_from_slice(x1);
        }
        self.m_elapsed += t0.elapsed().as_secs_f64();
        status
    }
}

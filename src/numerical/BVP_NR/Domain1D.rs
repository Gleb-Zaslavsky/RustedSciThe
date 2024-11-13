#![allow(warnings)]
/**
 * @file domain1d.rs
 */

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

use std::collections::HashSet;
use std::sync::Arc;
use std::cell::RefCell;

use crate::one_d::{MultiJac, Refiner};
use crate::base::{AnyMap, Solution, SolutionArray};
use crate::thermo::ThermoPhase;

pub struct Domain1D {
    nv: usize,
    points: usize,
    time: f64,
    name: Vec<String>,
    max: Vec<f64>,
    min: Vec<f64>,
    rtol_ss: Vec<f64>,
    atol_ss: Vec<f64>,
    rtol_ts: Vec<f64>,
    atol_ts: Vec<f64>,
    z: Vec<f64>,
    slast: Vec<f64>,
    refiner: Option<RefCell<Refiner>>,
    solution: Option<Arc<Solution>>,
    container: Option<Arc<Container>>,
    state: Option<Arc<State>>,
    left: Option<Arc<Domain1D>>,
    right: Option<Arc<Domain1D>>,
    jstart: usize,
    iloc: usize,
}

impl Domain1D {
    pub fn new(nv: usize, points: usize, time: f64) -> Self {
        let mut domain = Domain1D {
            nv,
            points,
            time,
            name: vec![String::new(); nv],
            max: vec![0.0; nv],
            min: vec![0.0; nv],
            rtol_ss: vec![1.0e-4; nv],
            atol_ss: vec![1.0e-9; nv],
            rtol_ts: vec![1.0e-4; nv],
            atol_ts: vec![1.0e-11; nv],
            z: vec![0.0; points],
            slast: vec![0.0; nv * points],
            refiner: None,
            solution: None,
            container: None,
            state: None,
            left: None,
            right: None,
            jstart: 0,
            iloc: 0,
        };
        domain.resize(nv, points);
        domain
    }

    pub fn set_solution(&mut self, sol: Arc<Solution>) {
        if sol.thermo().is_none() {
            panic!("Domain1D::setSolution: Missing or incomplete Solution object.");
        }
        if let Some(ref solution) = self.solution {
            solution.thermo().unwrap().remove_species_lock();
        }
        self.solution = Some(sol.clone());
        sol.thermo().unwrap().add_species_lock();
    }

    pub fn resize(&mut self, nv: usize, np: usize) {
        if nv != self.nv || self.refiner.is_none() {
            self.nv = nv;
            self.refiner = Some(RefCell::new(Refiner::new(self)));
        }
        self.nv = nv;
        self.name.resize(nv, String::new());
        self.max.resize(nv, 0.0);
        self.min.resize(nv, 0.0);
        self.rtol_ss.resize(nv, 1.0e-4);
        self.atol_ss.resize(nv, 1.0e-9);
        self.rtol_ts.resize(nv, 1.0e-4);
        self.atol_ts.resize(nv, 1.0e-11);
        self.points = np;
        self.z.resize(np, 0.0);
        self.slast.resize(nv * np, 0.0);
        self.locate();
    }

    pub fn component_name(&self, n: usize) -> String {
        if !self.name[n].is_empty() {
            self.name[n].clone()
        } else {
            format!("component {}", n)
        }
    }

    pub fn component_index(&self, name: &str) -> usize {
        for n in 0..self.nv {
            if name == self.component_name(n) {
                return n;
            }
        }
        panic!("Domain1D::componentIndex: no component named {}", name);
    }

    pub fn set_transient_tolerances(&mut self, rtol: f64, atol: f64, n: Option<usize>) {
        if let Some(n) = n {
            self.rtol_ts[n] = rtol;
            self.atol_ts[n] = atol;
        } else {
            for n in 0..self.nv {
                self.rtol_ts[n] = rtol;
                self.atol_ts[n] = atol;
            }
        }
    }

    pub fn set_steady_tolerances(&mut self, rtol: f64, atol: f64, n: Option<usize>) {
        if let Some(n) = n {
            self.rtol_ss[n] = rtol;
            self.atol_ss[n] = atol;
        } else {
            for n in 0..self.nv {
                self.rtol_ss[n] = rtol;
                self.atol_ss[n] = atol;
            }
        }
    }

    pub fn need_jac_update(&mut self) {
        if let Some(ref container) = self.container {
            container.jacobian().set_age(10000);
            container.save_stats();
        }
    }

    pub fn get_meta(&self) -> AnyMap {
        let wrap_tols = |tols: &Vec<f64>| -> AnyValue {
            let unique_tols: HashSet<_> = tols.iter().cloned().collect();
            if unique_tols.len() == 1 {
                AnyValue::Scalar(tols[0])
            } else {
                let mut out = AnyMap::new();
                for (i, &tol) in tols.iter().enumerate() {
                    out.insert(self.component_name(i), AnyValue::Scalar(tol));
                }
                AnyValue::Map(out)
            }
        };

        let mut state = AnyMap::new();
        state.insert("type".to_string(), AnyValue::Scalar(self.type()));
        state.insert("points".to_string(), AnyValue::Scalar(self.points as f64));
        if self.nv > 0 && self.points > 0 {
            let mut tolerances = AnyMap::new();
            tolerances.insert("transient-abstol".to_string(), wrap_tols(&self.atol_ts));
            tolerances.insert("steady-abstol".to_string(), wrap_tols(&self.atol_ss));
            tolerances.insert("transient-reltol".to_string(), wrap_tols(&self.rtol_ts));
            tolerances.insert("steady-reltol".to_string(), wrap_tols(&self.rtol_ss));
            state.insert("tolerances".to_string(), AnyValue::Map(tolerances));
        }
        state
    }

    pub fn to_array(&self, normalize: bool) -> Arc<SolutionArray> {
        if self.state.is_none() {
            panic!("Domain1D::toArray: Domain needs to be installed in a container before calling asArray.");
        }
        let ret = self.as_array(self.state.as_ref().unwrap().data() + self.iloc);
        if normalize {
            ret.normalize();
        }
        ret
    }

    pub fn from_array(&mut self, arr: Arc<SolutionArray>) {
        if self.state.is_none() {
            panic!("Domain1D::fromArray: Domain needs to be installed in a container before calling fromArray.");
        }
        self.resize(self.nv, arr.size());
        self.container.as_ref().unwrap().resize();
        self.from_array_impl(&arr, self.state.as_ref().unwrap().data() + self.iloc);
        self.finalize(self.state.as_ref().unwrap().data() + self.iloc);
    }

    pub fn set_meta(&mut self, meta: &AnyMap) {
        let set_tols = |tols: &AnyValue, which: &str, out: &mut Vec<f64>| {
            if !tols.has_key(which) {
                return;
            }
            let tol = &tols[which];
            if tol.is_scalar() {
                out.fill(tol.as_f64());
            } else {
                for i in 0..self.nv {
                    let name = self.component_name(i);
                    if tol.has_key(&name) {
                        out[i] = tol[&name].as_f64();
                    } else {
                        warn!("Domain1D::setMeta: No {} found for component '{}'", which, name);
                    }
                }
            }
        };

        if meta.has_key("tolerances") {
            let tols = &meta["tolerances"];
            set_tols(tols, "transient-abstol", &mut self.atol_ts);
            set_tols(tols, "transient-reltol", &mut self.rtol_ts);
            set_tols(tols, "steady-abstol", &mut self.atol_ss);
            set_tols(tols, "steady-reltol", &mut self.rtol_ss);
        }
    }

    pub fn locate(&mut self) {
        if let Some(ref left) = self.left {
            self.jstart = left.last_point() + 1;
            self.iloc = left.loc() + left.size();
        } else {
            self.jstart = 0;
            self.iloc = 0;
        }
        if let Some(ref right) = self.right {
            right.locate();
        }
    }

    pub fn setup_grid(&mut self, n: usize, z: &[f64]) {
        if n > 1 {
            self.resize(self.nv, n);
            for j in 0..self.points {
                self.z[j] = z[j];
            }
        }
    }

    pub fn show(&self, x: &[f64]) {
        let nn = self.nv / 5;
        for i in 0..nn {
            writeline('-', 79, false, true);
            writelog("\n          z ");
            for n in 0..5 {
                writelog(" {:>10} ", self.component_name(i * 5 + n));
            }
            writeline('-', 79, false, true);
            for j in 0..self.points {
                writelog("\n {:10.4} ", self.z[j]);
                for n in 0..5 {
                    let v = self.value(x, i * 5 + n, j);
                    writelog(" {:10.4} ", v);
                }
            }
            writelog("\n");
        }
        let nrem = self.nv - 5 * nn;
        writeline('-', 79, false, true);
        writelog("\n          z ");
        for n in 0..nrem {
            writelog(" {:>10} ", self.component_name(nn * 5 + n));
        }
        writeline('-', 79, false, true);
        for j in 0..self.points {
            writelog("\n {:10.4} ", self.z[j]);
            for n in 0..nrem {
                let v = self.value(x, nn * 5 + n, j);
                writelog(" {:10.4} ", v);
            }
        }
        writelog("\n");
    }

    pub fn set_profile(&mut self, name: &str, values: &[f64], soln: &mut [f64]) {
        for n in 0..self.nv {
            if name == self.component_name(n) {
                for j in 0..self.points {
                    soln[self.index(n, j) + self.iloc] = values[j];
                }
                return;
            }
        }
        panic!("Domain1D::setProfile: unknown component: {}", name);
    }

    pub fn get_initial_soln(&self, x: &mut [f64]) {
        for j in 0..self.points {
            for n in 0..self.nv {
                x[self.index(n, j)] = self.initial_value(n, j);
            }
        }
    }

    pub fn initial_value(&self, n: usize, j: usize) -> f64 {
        unimplemented!("Domain1D::initialValue");
    }
}

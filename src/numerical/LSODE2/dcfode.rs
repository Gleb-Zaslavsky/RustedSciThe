//! Faithful ODEPACK-style coefficient generation for LSODE2.
//!
//! This module mirrors the BDF branch of `DCFODE` from `odepack.f`.
//! The goal is not micro-optimization, but to have one canonical place for
//! `ELCO`/`TESCO` data that the rest of `LSODE2` can reuse as more of the
//! original `DSTODA` choreography is ported.

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2DcfodeError {
    InvalidOrder { order: usize, max_order: usize },
}

impl std::fmt::Display for Lsode2DcfodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder { order, max_order } => write!(
                f,
                "LSODE2 DCFODE supports BDF orders 1..={max_order}, got {order}"
            ),
        }
    }
}

impl std::error::Error for Lsode2DcfodeError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2BdfDcfodeTables {
    // Mirror Fortran's 1-based indexing to keep the translation faithful.
    // Only orders 1..=5 are exposed, but rows 1..=6 are needed.
    elco: [[f64; 6]; 7],
    tesco: [[f64; 6]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2BdfOrderCoefficients {
    pub order: usize,
    pub el: [f64; 6],
    pub tesco1: f64,
    pub tesco2: f64,
    pub tesco3: f64,
}

impl Lsode2BdfDcfodeTables {
    pub const MAX_ORDER: usize = 5;

    pub fn generate() -> Self {
        // Mirror the BDF branch of DCFODE:
        //   PC(1) = 1
        //   RQ1FAC = 1
        //   DO NQ = 1,5 ...
        let mut elco = [[0.0; 6]; 7];
        let mut tesco = [[0.0; 6]; 4];
        let mut pc = [0.0; 7];
        pc[1] = 1.0;
        let mut rq1fac = 1.0;

        for nq in 1..=Self::MAX_ORDER {
            let fnq = nq as f64;
            let nqp1 = nq + 1;

            pc[nqp1] = 0.0;
            for ib in 1..=nq {
                let i = nq + 2 - ib;
                pc[i] = pc[i - 1] + fnq * pc[i];
            }
            pc[1] = fnq * pc[1];

            for i in 1..=nqp1 {
                elco[i][nq] = pc[i] / pc[2];
            }
            elco[2][nq] = 1.0;
            tesco[1][nq] = rq1fac;
            tesco[2][nq] = nqp1 as f64 / elco[1][nq];
            tesco[3][nq] = (nq as f64 + 2.0) / elco[1][nq];
            rq1fac /= fnq;
        }

        Self { elco, tesco }
    }

    pub fn order(&self, order: usize) -> Result<Lsode2BdfOrderCoefficients, Lsode2DcfodeError> {
        if !(1..=Self::MAX_ORDER).contains(&order) {
            return Err(Lsode2DcfodeError::InvalidOrder {
                order,
                max_order: Self::MAX_ORDER,
            });
        }

        let mut el = [0.0; 6];
        for i in 1..=order + 1 {
            el[i - 1] = self.elco[i][order];
        }

        Ok(Lsode2BdfOrderCoefficients {
            order,
            el,
            tesco1: self.tesco[1][order],
            tesco2: self.tesco[2][order],
            tesco3: self.tesco[3][order],
        })
    }

    pub fn tesco2(&self, order: usize) -> Result<f64, Lsode2DcfodeError> {
        Ok(self.order(order)?.tesco2)
    }

    pub fn tesco1(&self, order: usize) -> Result<f64, Lsode2DcfodeError> {
        Ok(self.order(order)?.tesco1)
    }

    pub fn tesco3(&self, order: usize) -> Result<f64, Lsode2DcfodeError> {
        Ok(self.order(order)?.tesco3)
    }
}

impl Default for Lsode2BdfDcfodeTables {
    fn default() -> Self {
        Self::generate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf_dcfode_reproduces_known_tesco_constants() {
        let table = Lsode2BdfDcfodeTables::generate();

        assert_eq!(table.tesco2(1).unwrap(), 2.0);
        assert_eq!(table.tesco2(2).unwrap(), 4.5);
        assert!((table.tesco2(3).unwrap() - 22.0 / 3.0).abs() < 1e-12);
        assert!((table.tesco2(4).unwrap() - 125.0 / 12.0).abs() < 1e-12);
        assert!((table.tesco2(5).unwrap() - 13.7).abs() < 1e-12);
    }

    #[test]
    fn bdf_dcfode_reproduces_first_bdf_method_coefficients() {
        let table = Lsode2BdfDcfodeTables::generate();

        let q1 = table.order(1).unwrap();
        assert_eq!(q1.el[0], 1.0);
        assert_eq!(q1.el[1], 1.0);

        let q2 = table.order(2).unwrap();
        assert!((q2.el[0] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(q2.el[1], 1.0);
        assert!((q2.el[2] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn bdf_dcfode_rejects_invalid_order() {
        let table = Lsode2BdfDcfodeTables::generate();
        let err = table.order(6).unwrap_err();
        assert!(matches!(
            err,
            Lsode2DcfodeError::InvalidOrder {
                order: 6,
                max_order: 5
            }
        ));
    }
}

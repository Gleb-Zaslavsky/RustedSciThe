//! Faithful ODEPACK-style Adams coefficient bootstrap for LSODE2.
//!
//! This mirrors the `METH = 1` branch of `DCFODE` from `odepack.f` and keeps
//! the same 1-based indexing semantics for `ELCO`/`TESCO` tables.

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2AdamsDcfodeError {
    InvalidOrder { order: usize, max_order: usize },
}

impl std::fmt::Display for Lsode2AdamsDcfodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder { order, max_order } => write!(
                f,
                "LSODE2 Adams DCFODE supports orders 1..={max_order}, got {order}"
            ),
        }
    }
}

impl std::error::Error for Lsode2AdamsDcfodeError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2AdamsDcfodeTables {
    // 1-based mirror for ELCO(13,12).
    elco: [[f64; 13]; 14],
    // 1-based mirror for TESCO(3,12).
    tesco: [[f64; 13]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2AdamsOrderCoefficients {
    pub order: usize,
    pub el: [f64; 14],
    pub tesco1: f64,
    pub tesco2: f64,
    pub tesco3: f64,
}

impl Lsode2AdamsDcfodeTables {
    pub const MAX_ORDER: usize = 12;

    pub fn generate() -> Self {
        // Faithful port of DCFODE METH=1 branch.
        let mut elco = [[0.0; 13]; 14];
        let mut tesco = [[0.0; 13]; 4];
        let mut pc = [0.0; 13];

        elco[1][1] = 1.0;
        elco[2][1] = 1.0;
        tesco[1][1] = 0.0;
        tesco[2][1] = 2.0;
        tesco[1][2] = 1.0;
        tesco[3][12] = 0.0;
        pc[1] = 1.0;
        let mut rqfac = 1.0;

        for nq in 2..=Self::MAX_ORDER {
            let rq1fac = rqfac;
            rqfac /= nq as f64;
            let nqm1 = nq - 1;
            let fnqm1 = nqm1 as f64;
            let nqp1 = nq + 1;

            // Form coefficients of p(x) * (x + nq - 1).
            pc[nq] = 0.0;
            for ib in 1..=nqm1 {
                let i = nqp1 - ib;
                pc[i] = pc[i - 1] + fnqm1 * pc[i];
            }
            pc[1] = fnqm1 * pc[1];

            // Compute integral on [-1, 0] of p(x) and x*p(x).
            let mut pint = pc[1];
            let mut xpin = pc[1] / 2.0;
            let mut tsign = 1.0;
            for (i, coeff) in pc.iter().enumerate().take(nq + 1).skip(2) {
                tsign = -tsign;
                let fi = i as f64;
                pint += tsign * *coeff / fi;
                xpin += tsign * *coeff / (fi + 1.0);
            }

            elco[1][nq] = pint * rq1fac;
            elco[2][nq] = 1.0;
            for (i, coeff) in pc.iter().enumerate().take(nq + 1).skip(2) {
                let fi = i as f64;
                elco[i + 1][nq] = rq1fac * *coeff / fi;
            }
            let agamq = rqfac * xpin;
            let ragq = 1.0 / agamq;
            tesco[2][nq] = ragq;
            if nq < Self::MAX_ORDER {
                tesco[1][nqp1] = ragq * rqfac / nqp1 as f64;
            }
            tesco[3][nqm1] = ragq;
        }

        Self { elco, tesco }
    }

    pub fn order(
        &self,
        order: usize,
    ) -> Result<Lsode2AdamsOrderCoefficients, Lsode2AdamsDcfodeError> {
        if !(1..=Self::MAX_ORDER).contains(&order) {
            return Err(Lsode2AdamsDcfodeError::InvalidOrder {
                order,
                max_order: Self::MAX_ORDER,
            });
        }

        let mut el = [0.0; 14];
        for (i, value) in el.iter_mut().enumerate().take(order + 2).skip(1) {
            *value = self.elco[i][order];
        }

        Ok(Lsode2AdamsOrderCoefficients {
            order,
            el,
            tesco1: self.tesco[1][order],
            tesco2: self.tesco[2][order],
            tesco3: self.tesco[3][order],
        })
    }
}

impl Default for Lsode2AdamsDcfodeTables {
    fn default() -> Self {
        Self::generate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adams_dcfode_keeps_expected_low_order_constants() {
        let t = Lsode2AdamsDcfodeTables::generate();
        let q1 = t.order(1).unwrap();
        let q2 = t.order(2).unwrap();
        let q12 = t.order(12).unwrap();

        assert_eq!(q1.el[1], 1.0);
        assert_eq!(q1.el[2], 1.0);
        assert_eq!(q1.tesco1, 0.0);
        assert_eq!(q1.tesco2, 2.0);
        assert!(q1.tesco3.is_finite() && q1.tesco3 > 0.0);

        assert_eq!(q2.tesco1, 1.0);
        assert!(q2.tesco2.is_finite() && q2.tesco2 > 0.0);
        assert!(q2.tesco3.is_finite() && q2.tesco3 > 0.0);
        assert_eq!(q12.tesco3, 0.0);
    }

    #[test]
    fn adams_dcfode_rejects_invalid_order() {
        let t = Lsode2AdamsDcfodeTables::generate();
        let err = t.order(13).unwrap_err();
        assert!(matches!(
            err,
            Lsode2AdamsDcfodeError::InvalidOrder {
                order: 13,
                max_order: 12
            }
        ));
    }

    #[test]
    fn adams_dcfode_tesco2_is_positive_and_finite_for_supported_orders() {
        let t = Lsode2AdamsDcfodeTables::generate();
        for q in 1..=Lsode2AdamsDcfodeTables::MAX_ORDER {
            let cur = t.order(q).unwrap().tesco2;
            assert!(
                cur.is_finite() && cur > 0.0,
                "TESCO(2,q) should stay finite and positive in Adams DCFODE, q={q}, value={cur:e}"
            );
        }
    }
}

//! Immutable Radau IIA coefficient tables and derived transforms.

/// Three-stage Radau IIA collocation coefficients (order five).
#[derive(Debug, Clone, Copy)]
/// Radau IIA order-5 collocation and eigentransformation constants.
pub(crate) struct RadauIia5 {
    pub(crate) c: [f64; 3],
    pub(crate) a: [[f64; 3]; 3],
    pub(crate) b: [f64; 3],
    pub(crate) e: [f64; 3],
    /// Dense-output polynomial coefficients: `Q = Z^T * p`.
    pub(crate) p: [[f64; 3]; 3],
    pub(crate) t: [[f64; 3]; 3],
    pub(crate) ti: [[f64; 3]; 3],
    pub(crate) mu_real: f64,
    pub(crate) mu_complex: (f64, f64),
}

impl RadauIia5 {
    /// Return the immutable coefficient set used by every Radau5 step.
    pub(crate) fn new() -> Self {
        let sqrt6 = 6.0_f64.sqrt();
        let cube_root_3 = 3.0_f64.powf(1.0 / 3.0);
        let cube_root_3_sq = 3.0_f64.powf(2.0 / 3.0);
        Self {
            c: [(4.0 - sqrt6) / 10.0, (4.0 + sqrt6) / 10.0, 1.0],
            a: [
                [
                    (88.0 - 7.0 * sqrt6) / 360.0,
                    (296.0 - 169.0 * sqrt6) / 1800.0,
                    (-2.0 + 3.0 * sqrt6) / 225.0,
                ],
                [
                    (296.0 + 169.0 * sqrt6) / 1800.0,
                    (88.0 + 7.0 * sqrt6) / 360.0,
                    (-2.0 - 3.0 * sqrt6) / 225.0,
                ],
                [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
            ],
            b: [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
            e: [
                (-13.0 - 7.0 * sqrt6) / 3.0,
                (-13.0 + 7.0 * sqrt6) / 3.0,
                -1.0 / 3.0,
            ],
            p: [
                [
                    13.0 / 3.0 + 7.0 * sqrt6 / 3.0,
                    -23.0 / 3.0 - 22.0 * sqrt6 / 3.0,
                    10.0 / 3.0 + 5.0 * sqrt6,
                ],
                [
                    13.0 / 3.0 - 7.0 * sqrt6 / 3.0,
                    -23.0 / 3.0 + 22.0 * sqrt6 / 3.0,
                    10.0 / 3.0 - 5.0 * sqrt6,
                ],
                [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0],
            ],
            t: [
                [
                    0.09443876248897524,
                    -0.14125529502095421,
                    0.03002919410514742,
                ],
                [
                    0.25021312296533332,
                    0.20412935229379994,
                    -0.38294211275726192,
                ],
                [1.0, 1.0, 0.0],
            ],
            ti: [
                [
                    4.17871859155190428,
                    0.32768282076106237,
                    0.52337644549944951,
                ],
                [
                    -4.17871859155190428,
                    -0.32768282076106237,
                    0.47662355450055044,
                ],
                [
                    0.50287263494578682,
                    -2.57192694985560522,
                    0.59603920482822492,
                ],
            ],
            mu_real: 3.0 + cube_root_3_sq - cube_root_3,
            mu_complex: (
                3.0 + 0.5 * (cube_root_3 - cube_root_3_sq),
                -0.5 * (3.0_f64.powf(5.0 / 6.0) + 3.0_f64.powf(7.0 / 6.0)),
            ),
        }
    }
}

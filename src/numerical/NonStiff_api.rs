use nalgebra::DVector;

pub struct RK45 {
    f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y0: DVector<f64>,
    t0: f64,
    t_end: f64,
    pub t: f64,
    pub y: DVector<f64>,
    h: f64,
}
impl RK45 {
    pub fn new() -> RK45 {
        RK45 {
            f: Box::new(|_t, y| {
                let mut dydt = DVector::zeros(y.len());
                dydt[0] = y[1];
                dydt[1] = -y[0];
                dydt
            }),
            y0: DVector::from_vec(vec![1.0, 0.0]),
            t0: 0.0,
            t: 0.0,
            y: DVector::from_vec(vec![1.0, 0.0]),
            t_end: 10.0,
            h: 0.1,
        }
    } //new
    pub fn set_initial(
        &mut self,
        f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        h: f64,
    ) {
        self.f = f;
        self.y0 = y0.clone();
        self.t0 = t0;
        self.h = h;
        self.y = y0;
        self.t = t0;
    }

    pub fn _step_impl(&mut self) -> bool {
        // Butcher tableau coefficients for RK45
        let a: [[f64; 6]; 6] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
            [
                1932.0 / 2197.0,
                -7200.0 / 2197.0,
                7296.0 / 2197.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                439.0 / 216.0,
                -8.0,
                3680.0 / 513.0,
                -845.0 / 4104.0,
                0.0,
                0.0,
            ],
            [
                -8.0 / 27.0,
                2.0,
                -3544.0 / 2565.0,
                1859.0 / 4104.0,
                -11.0 / 40.0,
                0.0,
            ],
        ];
        let c = [0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0];
        let b = [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ];

        let mut t = self.t;
        let y = &self.y;
        let f = &self.f;
        let h = self.h;

        let mut k = vec![DVector::zeros(y.len()); 6];

        k[0] = h * f(t, &y);
        for i in 1..6 {
            let mut y_temp = y.clone();
            for j in 0..i {
                y_temp += a[i - 1][j] * &k[j];
            }
            k[i] = h * f(t + c[i], &y_temp);
        }

        let mut y_next = y.clone();
        for i in 0..6 {
            y_next += b[i] * &k[i];
        }

        t += h;
        self.t = t;
        self.y = y_next.clone();
        return true;
    }
}

pub struct DormandPrince {
    pub f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub y0: DVector<f64>,
    pub t0: f64,
    pub t: f64,
    pub y: DVector<f64>,
    h: f64,
}

impl DormandPrince {
    pub fn new() -> DormandPrince {
        DormandPrince {
            f: Box::new(|_t, y| {
                let mut dydt = DVector::zeros(y.len());
                dydt[0] = y[1];
                dydt[1] = -y[0];
                dydt
            }),
            y0: DVector::from_vec(vec![1.0, 0.0]),
            t0: 0.0,
            t: 0.0,
            y: DVector::from_vec(vec![1.0, 0.0]),

            h: 0.1,
        }
    }

    pub fn set_initial(
        &mut self,
        f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        h: f64,
    ) {
        self.f = f;
        self.y0 = y0.clone();
        self.t0 = t0;
        self.h = h;
        self.y = y0;
        self.t = t0;
    }

    pub fn _step_impl(&mut self) -> bool {
        // Butcher tableau coefficients for Dormand-Prince
        let a: [[f64; 6]; 6] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 45.0, 32.0 / 45.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 6561.0,
                64448.0 / 6561.0,
                -212.0 / 6561.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -3556.0 / 3168.0,
                46732.0 / 3168.0,
                -4275.0 / 3168.0,
                2187.0 / 6561.0,
                0.0,
            ],
        ];
        let c = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
        let b = [
            19.0 / 216.0,
            0.0,
            16.0 / 216.0,
            512.0 / 1083.0,
            256.0 / 1083.0,
            -212.0 / 6561.0,
            1.0 / 8.0,
        ];

        let mut t = self.t;
        let y = &self.y;
        let f = &self.f;
        let h = self.h;

        let mut k = vec![DVector::zeros(y.len()); 6];

        k[0] = h * f(t, &y);
        for i in 1..6 {
            let mut y_temp = y.clone();
            for j in 0..i {
                y_temp += a[i - 1][j] * &k[j];
            }
            k[i] = h * f(t + c[i] * h, &y_temp);
        }

        let mut y_next = y.clone();
        for i in 0..6 {
            y_next += b[i] * &k[i];
        }

        t += h;
        self.t = t;
        self.y = y_next.clone();
        return true;
    }
}

/*
fn main() {
    // Example usage
    let f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> = Box::new(|t, y| {
        let mut dydt = DVector::zeros(y.len());
        dydt[0] = y[1];
        dydt[1] = -y[0];
        dydt
    });

    let y0 = DVector::from_vec(vec![1.0, 0.0]);
    let t0 = 0.0;
    let t_end = 10.0;
    let h = 0.1;

    let result = rk45(f, y0, t0, t_end, h);

    for (t, y) in result {
        println!("t: {}, y: {:?}", t, y);
    }
}



use nalgebra::DVector;

fn rk45(
    f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y0: DVector<f64>,
    t0: f64,
    t_end: f64,
    h: f64,
) -> Vec<(f64, DVector<f64>)> {
    let mut t = t0;
    let mut y = y0;
    let mut result = vec![(t, y.clone())];

    while t < t_end {
        let k1 = h * f(t, &y);
        let k2 = h * f(t + h / 4.0, &(y + 0.25 * &k1));
        let k3 = h * f(t + 3.0 * h / 8.0, &(y + 3.0 / 32.0 * &k1 + 9.0 / 32.0 * &k2));
        let k4 = h * f(t + 12.0 * h / 13.0, &(y + 1932.0 / 2197.0 * &k1 - 7200.0 / 2197.0 * &k2 + 7296.0 / 2197.0 * &k3));
        let k5 = h * f(t + h, &(y + 439.0 / 216.0 * &k1 - 8.0 * &k2 + 3680.0 / 513.0 * &k3 - 845.0 / 4104.0 * &k4));
        let k6 = h * f(t + h / 2.0, &(y - 8.0 / 27.0 * &k1 + 2.0 * &k2 - 3544.0 / 2565.0 * &k3 + 1859.0 / 4104.0 * &k4 - 11.0 / 40.0 * &k5));

        let y_next = y + 16.0 / 135.0 * &k1 + 6656.0 / 12825.0 * &k3 + 28561.0 / 56430.0 * &k4 - 9.0 / 50.0 * &k5 + 2.0 / 55.0 * &k6;

        t += h;
        y = y_next.clone();
        result.push((t, y));
    }

    result
}

fn main() {
    // Example usage
    let f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> = Box::new(|t, y| {
        let mut dydt = DVector::zeros(y.len());
        dydt[0] = y[1];
        dydt[1] = -y[0];
        dydt
    });

    let y0 = DVector::from_vec(vec![1.0, 0.0]);
    let t0 = 0.0;
    let t_end = 10.0;
    let h = 0.1;

    let result = rk45(f, y0, t0, t_end, h);

    for (t, y) in result {
        println!("t: {}, y: {:?}", t, y);
    }
}

*/

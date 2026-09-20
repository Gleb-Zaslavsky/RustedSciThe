//! Correctness gate for the NASA TP-1907 CHON + graphite reproducer.
//!
//! The case is a fixed-temperature, fixed-pressure chemical-equilibrium
//! system in log-mole variables. It is deliberately kept as a prepared
//! symbolic residual/Jacobian problem. The compact equations in this file are
//! a diagnostic rederivation. The exact KiThe-exported 18-equation payload is
//! stored beside this module and round-tripped through the public string API.
//! The fixed graphite active-set reduction leaves one structural null
//! direction, so this is a bounded least-squares trace gate rather than a
//! unique-root acceptance test.
//!
//! Debug run:
//! ```text
//! cargo test --lib numerical::Nonlinear_systems::tp1907_chon_graphite_tests -- --nocapture --test-threads=1
//! ```
//!
//! Release run:
//! ```text
//! cargo test --release --lib numerical::Nonlinear_systems::tp1907_chon_graphite_tests -- --nocapture --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use crate::numerical::Nonlinear_systems::engine::{DiagnosticsOptions, SolveOptions};
    use crate::numerical::Nonlinear_systems::engine::{LinearSolverKind, solve_linear_system};
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use crate::numerical::Nonlinear_systems::prelude::{
        BacktrackingLevenbergMarquardtMethod, Bounds, JacobianProvider, LevenbergMarquardtMethod,
        LevenbergMarquardtMinpack, NielsenLevenbergMarquardtMethod, NonlinearProblem,
        NonlinearSolverMethod, SolveResult, SymbolicNonlinearProblem, SymbolicProblemOptions,
        TrustRegionLMMethod, TrustRegionMethod,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::DVector;

    const TEMPERATURE: f64 = 700.0;
    const G0_OVER_T: f64 = 0.12027235504272604;
    const CANONICAL_TP1907_FIXTURE: &str = include_str!("fixtures/tp1907_rst_parity_fixture.txt");
    const CANONICAL_TP1907_EQUATION_HASH: u64 = 0x6dc5_1479_2152_d82c;
    // Published physical moles are retained to check the external physical
    // publication contract independently of the internal solver coordinates.
    const ORACLE_MOLES: [f64; 18] = [
        5.364815999999999e6,
        7.740862308907562e6,
        3.7322365529513024e6,
        8.812251532548694e7,
        3.1395937772166847e-6,
        3.375886706085238e-26,
        2.4074906124597773e7,
        6.038820479607477e7,
        3.564057592071103e-24,
        3.6776307673835574e4,
        2.447993369668292e-13,
        3.8177643238841245e-28,
        4.472920518461631e8,
        3.0531829258636216e-17,
        1.5156204451699374e-22,
        6.346227228849742e-10,
        2.788514732859965e-22,
        5.871218126541678e5,
    ];

    // Full-precision accepted internal coordinates exported by the KiThe
    // normalized-discovery route. Do not reconstruct these through `ln` of
    // published moles: that loses enough precision to obscure graph parity.
    const ORACLE_LOG_MOLES: [f64; 18] = [
        1.54953726370934710e1,
        1.58620236487799229e1,
        1.51325182239758078e1,
        1.82942386238111325e1,
        -1.26714171367262978e1,
        -5.86505544002986241e1,
        1.69966806163510533e1,
        1.79163043590012769e1,
        -5.39911425629876405e1,
        1.05126091036093179e1,
        -2.90383375527450127e1,
        -6.31327176080713812e1,
        1.99187222990660473e1,
        -3.80277619521825017e1,
        -5.02410471559785137e1,
        -2.11779904307943170e1,
        -4.96313629455999532e1,
        1.32829875945798417e1,
    ];

    fn fixture_vec_payload<'a>(fixture: &'a str, marker: &str) -> &'a str {
        let marker_offset = fixture
            .find(marker)
            .unwrap_or_else(|| panic!("TP-1907 fixture is missing marker '{marker}'"));
        let bytes = fixture.as_bytes();
        let start = marker_offset
            + fixture[marker_offset..]
                .find('[')
                .expect("TP-1907 fixture marker must start a vector");
        let mut depth = 0_usize;
        for (offset, byte) in bytes[start..].iter().copied().enumerate() {
            match byte {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return &fixture[start + 1..start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("TP-1907 fixture vector for '{marker}' is not balanced");
    }

    fn fixture_string_vector(marker: &str) -> Vec<String> {
        let payload = fixture_vec_payload(CANONICAL_TP1907_FIXTURE, marker);
        let bytes = payload.as_bytes();
        let mut values = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != b'"' {
                index += 1;
                continue;
            }
            index += 1;
            let mut value = String::new();
            while index < bytes.len() {
                match bytes[index] {
                    b'\\' => {
                        index += 1;
                        let escaped = *bytes
                            .get(index)
                            .expect("TP-1907 fixture has a trailing escape");
                        value.push(escaped as char);
                    }
                    b'"' => break,
                    byte => value.push(byte as char),
                }
                index += 1;
            }
            assert!(
                index < bytes.len(),
                "TP-1907 fixture has an unterminated string"
            );
            values.push(value);
            index += 1;
        }
        values
    }

    fn fixture_numeric_vector(marker: &str) -> DVector<f64> {
        DVector::from_vec(fixture_numeric_values(fixture_vec_payload(
            CANONICAL_TP1907_FIXTURE,
            marker,
        )))
    }

    fn fixture_scalar(marker: &str) -> f64 {
        let start = CANONICAL_TP1907_FIXTURE
            .find(marker)
            .expect("TP-1907 fixture scalar marker must exist")
            + marker.len();
        let value = CANONICAL_TP1907_FIXTURE[start..]
            .split(';')
            .next()
            .expect("TP-1907 fixture scalar must terminate with a semicolon")
            .trim();
        value
            .parse::<f64>()
            .expect("TP-1907 fixture scalar must be a finite number")
    }

    fn fixture_bounds() -> Bounds {
        let payload = fixture_vec_payload(
            CANONICAL_TP1907_FIXTURE,
            "let log_mole_bounds: Option<Vec<(f64, f64)>> = Some(vec![",
        );
        let limits = payload
            .split('(')
            .skip(1)
            .filter_map(|tuple| tuple.split_once(')'))
            .map(|(tuple, _)| {
                let values = fixture_numeric_values(tuple);
                assert_eq!(values.len(), 2, "TP-1907 bound tuple must have two values");
                (values[0], values[1])
            })
            .collect::<Vec<_>>();
        Bounds::new(limits).expect("exact TP-1907 fixture bounds must be valid")
    }

    fn fixture_numeric_values(payload: &str) -> Vec<f64> {
        payload
            .split(',')
            .filter_map(|token| {
                let token = token.trim();
                (!token.is_empty()).then_some(token)
            })
            .map(|token| {
                token.parse::<f64>().unwrap_or_else(|error| {
                    panic!("TP-1907 fixture numeric value '{token}' is invalid: {error}")
                })
            })
            .collect()
    }

    fn fixture_numeric_matrix(marker: &str) -> nalgebra::DMatrix<f64> {
        let mut remaining = fixture_vec_payload(CANONICAL_TP1907_FIXTURE, marker);
        let mut rows = Vec::new();
        while let Some(row_start) = remaining.find("vec![") {
            remaining = &remaining[row_start..];
            let row_payload = fixture_vec_payload(remaining, "vec![");
            rows.push(fixture_numeric_values(row_payload));
            remaining = &remaining["vec![".len() + row_payload.len() + 1..];
        }
        let column_count = rows
            .first()
            .map(Vec::len)
            .expect("TP-1907 fixture matrix must contain rows");
        assert!(
            rows.iter().all(|row| row.len() == column_count),
            "TP-1907 fixture matrix rows have inconsistent widths"
        );
        let values = rows.into_iter().flatten().collect::<Vec<_>>();
        nalgebra::DMatrix::from_row_slice(values.len() / column_count, column_count, &values)
    }

    fn stable_equation_hash(equations: &[String]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for equation in equations {
            for byte in (equation.len() as u64).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3_u64);
            }
            for byte in equation.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3_u64);
            }
        }
        hash
    }

    fn canonical_fixture_equations() -> Vec<String> {
        let equations = fixture_string_vector("let equations: Vec<String> = vec![");
        assert_eq!(
            equations.len(),
            18,
            "TP-1907 fixture must contain 18 equations"
        );
        assert_eq!(
            stable_equation_hash(&equations),
            CANONICAL_TP1907_EQUATION_HASH,
            "TP-1907 equation fixture changed; refresh all parity expectations together"
        );
        equations
    }

    fn canonical_fixture_problem() -> SymbolicNonlinearProblem {
        let variables = fixture_string_vector("let variables: Vec<String> = vec![");
        let parameters = fixture_string_vector("let equation_parameters: Vec<String> = vec![");
        let parameter_values =
            fixture_numeric_vector("let equation_parameter_values: Vec<f64> = vec![");
        SymbolicNonlinearProblem::from_strings_with_options(
            canonical_fixture_equations(),
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(parameters)
                .with_equation_parameter_values(parameter_values),
        )
        .expect("exact KiThe TP-1907 fixture must parse through the public string API")
    }

    // The first thirteen equations are generated from the same coefficient
    // rows as the paste-ready TP-1907 system.  Keeping the common g0 and
    // log-sum terms factored makes accidental transcription changes visible.
    const LINEAR_ROWS: [[f64; 18]; 13] = [
        [
            0.0,
            -0.6509337356549053,
            0.21697791188496898,
            0.14542568793360044,
            0.149197702983754,
            0.006093255081016968,
            0.29839540596750774,
            0.22684318201613918,
            -0.007485999099534877,
            0.44010710985172685,
            -0.07903822305090336,
            -0.1505904470022719,
            -0.014971998199069754,
            -0.08652422215043824,
            -0.07155222395136852,
            0.07764547903238539,
            -0.14310444790273705,
            0.2885301358363375,
        ],
        [
            0.0,
            3.3993498887762963e-16,
            -0.8164965809277257,
            0.40824829046386346,
            -2.5495124165822223e-17,
            4.567876413043148e-17,
            -1.6996749443881481e-17,
            -3.399349888776296e-17,
            -5.550500990267546e-17,
            -3.399349888776296e-17,
            -5.948862305358518e-17,
            -5.099024833164445e-17,
            -1.1101001980535093e-16,
            -8.498374721940741e-17,
            -1.4872155763396295e-17,
            0.0,
            -2.974431152679259e-17,
            0.40824829046386324,
        ],
        [
            0.0,
            3.983165516740518e-16,
            5.869928129933394e-16,
            -0.6619809476665833,
            -0.01818074009459659,
            0.1684748582099276,
            -0.03636148018919303,
            0.05696631896306899,
            -0.03434139795646007,
            -0.08888361824024948,
            0.05898640119580196,
            0.15231420034806403,
            -0.06868279591292013,
            0.02464500323934202,
            0.09332779915226201,
            0.07514705905766554,
            0.18665559830452402,
            0.6619809476665849,
        ],
        [
            0.0,
            2.9228675765054765e-16,
            -1.4757521661985515e-17,
            -2.4977893266521802e-17,
            -0.9723049259813404,
            0.02793258179868931,
            0.11235802566214353,
            0.09823481014595227,
            -0.008160080076021647,
            0.16037695841719368,
            -0.02228329559221288,
            -0.036406511108404135,
            -0.016320160152043295,
            -0.030443375668234532,
            -0.01412321551619124,
            0.04205579731488054,
            -0.02824643103238248,
            -6.601300363295047e-17,
        ],
        [
            0.0,
            1.0319307203355628e-16,
            1.6400232926072495e-16,
            2.9406885646838763e-16,
            8.191458257893971e-18,
            -0.847088965779729,
            0.06782808632874131,
            0.2175833555826476,
            -0.07087653964688698,
            0.030865589846225085,
            0.07887872960701937,
            0.22863399886092578,
            -0.14175307929377395,
            0.008002189960132477,
            0.1497552692539064,
            0.18366931241827705,
            0.2995105385078128,
            1.6165240014470602e-16,
        ],
        [
            0.0,
            6.322262047063343e-16,
            3.501789681355293e-17,
            -2.374354010270983e-19,
            1.5937806663230025e-16,
            1.0318264141777957e-16,
            -0.8742325212005133,
            0.24801652455482517,
            -0.024698741034505388,
            0.37974314340552084,
            -0.04631013943969762,
            -0.06792153784488988,
            -0.049397482069010776,
            -0.07100888047420299,
            -0.021611398405192255,
            0.11320256307481652,
            -0.04322279681038451,
            3.183776648058203e-17,
        ],
        [
            0.0,
            7.695862291059086e-16,
            1.4507524738862408e-16,
            1.7988079617071384e-16,
            2.065663669566315e-16,
            3.7675000992021353e-16,
            3.794401387868872e-16,
            -0.8046363591053719,
            -0.07973214831135052,
            0.4491333859006356,
            0.005851900793493631,
            0.0914359498983378,
            -0.15946429662270103,
            -0.07388024751785681,
            0.08558404910484413,
            0.26187256050883956,
            0.17116809820968826,
            4.546698463291556e-17,
        ],
        [
            0.0,
            -9.524470816514522e-17,
            -4.950038152651186e-17,
            -8.479592866151912e-17,
            -2.406041474959957e-17,
            -1.3002777165899649e-16,
            -2.83621657486951e-17,
            -8.817253600630607e-17,
            -0.9443612559916791,
            0.025028942564509365,
            0.069310917870949,
            0.024066290927412866,
            0.22911108962897034,
            0.1838664626854341,
            -0.04524462694353614,
            -0.07508682769352808,
            -0.09048925388707228,
            -3.827656624622236e-17,
        ],
        [
            0.0,
            4.294567531736067e-15,
            1.7219448436288174e-16,
            2.48706210572226e-17,
            1.1796483996431353e-15,
            1.0622879037557194e-15,
            2.2737995338547752e-15,
            2.6326813942312193e-15,
            1.17865117184229e-16,
            -0.2943574006319992,
            -0.031167254184565032,
            -0.15237324268009603,
            0.1800774686219318,
            0.05887148012640071,
            -0.12120598849553098,
            0.8830722018960114,
            -0.24241197699106196,
            3.003337751533305e-16,
        ],
        [
            0.0,
            -2.7658190704679374e-16,
            2.9775687551448806e-17,
            9.383443109772607e-17,
            7.686111675464084e-17,
            1.1592598713705144e-17,
            -1.2773731412837291e-16,
            -1.1308251952802873e-16,
            8.530402827066008e-17,
            -1.6360875165757368e-16,
            -0.9330531792422753,
            0.20174122794427587,
            0.15130592095820697,
            0.21435005469079313,
            0.06304413373258617,
            -3.718380719490311e-18,
            0.12608826746517235,
            -3.611203863457941e-17,
        ],
        [
            0.0,
            -1.0988750225478982e-15,
            1.7018177846448862e-16,
            4.326672196343078e-16,
            -3.040709338661008e-16,
            1.1260535283343713e-16,
            -6.096011163650322e-16,
            -4.985766162171617e-16,
            3.05226172367092e-17,
            -1.0146642160282925e-15,
            1.7997399804912185e-16,
            -0.7710996009560592,
            0.07009996372327822,
            0.3154498367547521,
            0.2453498730314737,
            -1.0895723125136027e-16,
            0.4906997460629474,
            -7.034794450771415e-17,
        ],
        [
            0.0,
            6.664097774159498e-16,
            -7.42293346162675e-17,
            -1.6280140578501274e-16,
            1.957827250604248e-16,
            -6.923737947178703e-17,
            4.158920969622883e-16,
            3.901619897261511e-16,
            3.4197651617483987e-16,
            9.79615355499473e-16,
            2.657218347295609e-16,
            1.9554881499464167e-16,
            -0.6741998624632414,
            0.6741998624632427,
            -0.1348399724926486,
            1.1134119131273858e-16,
            -0.2696799449852972,
            -5.543668559058618e-17,
        ],
        [
            0.0,
            -1.0255969220602796e-15,
            1.7168581989460165e-16,
            3.960417766352291e-16,
            -2.8878854329571795e-16,
            9.348292670821763e-17,
            -5.791286704372155e-16,
            -4.78275720461545e-16,
            1.1520510054913132e-16,
            -1.0969697789661388e-15,
            7.137285690585904e-17,
            3.1186835300168864e-16,
            -1.7067422303575003e-16,
            0.0,
            -0.8944271909999157,
            -1.40030442081604e-16,
            0.44721359549995837,
            -5.159016287216987e-17,
        ],
    ];

    const LOG_Y17: [f64; 13] = [
        -0.2885301358363375,
        -0.40824829046386324,
        -0.6619809476665849,
        6.601300363295047e-17,
        -1.6165240014470602e-16,
        -3.183776648058203e-17,
        -4.546698463291556e-17,
        3.827656624622236e-17,
        -3.003337751533305e-16,
        3.611203863457941e-17,
        7.034794450771415e-17,
        5.543668559058618e-17,
        5.159016287216987e-17,
    ];

    const LOG_SUM: [f64; 13] = [
        -0.3564844377398704,
        0.4082482904638624,
        0.09191374158934537,
        0.687329821787971,
        -0.2050084856452984,
        0.45744126624323334,
        0.05266710714143924,
        0.6237982608385405,
        -0.2805052876611025,
        0.17652357445124137,
        -0.3504998186163895,
        0.40451991747794125,
        0.44721359549996015,
    ];

    fn linear_expression(prefix: &str, row: &[f64; 18]) -> String {
        row.iter()
            .enumerate()
            .filter(|(_, coefficient)| coefficient.abs() > 0.0)
            .enumerate()
            .map(|(term, (index, coefficient))| {
                if term == 0 {
                    format!("{coefficient:.17e} * {prefix}{index}")
                } else {
                    format!("{coefficient:+.17e} * {prefix}{index}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn log_sum_expression() -> String {
        (0..18)
            .map(|index| format!("exp(y{index})"))
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn tp1907_equations() -> Vec<String> {
        let log_sum = log_sum_expression();
        let mut equations = Vec::with_capacity(18);
        for row in 0..13 {
            let y_terms = linear_expression("y", &LINEAR_ROWS[row]);
            let g_terms = linear_expression("g0_", &LINEAR_ROWS[row]);
            equations.push(format!(
                "{y_terms} + {G0_OVER_T:.17e} * ({g_terms}) / T + ({LOG_Y17:.17e}) * ln(exp(y17)) + ({LOG_SUM:.17e}) * ln({log_sum})",
                G0_OVER_T = G0_OVER_T,
                LOG_Y17 = LOG_Y17[row],
                LOG_SUM = LOG_SUM[row],
            ));
        }
        equations.extend([
            "exp(y0) - 5364815.999999999".to_string(),
            "exp(y1) + exp(y17) + exp(y2) + exp(y3) - 100182736".to_string(),
            "2 * exp(y6) + 2 * exp(y7) + 3 * exp(y9) + 4 * exp(y1) + exp(y15) + exp(y4) + exp(y5) - 200000000".to_string(),
            "2 * exp(y12) + 2 * exp(y13) + exp(y10) + exp(y11) + exp(y8) + exp(y9) - 894620880.0000001".to_string(),
            "2 * exp(y11) + 2 * exp(y16) + 2 * exp(y3) + 2 * exp(y5) + exp(y10) + exp(y13) + exp(y14) + exp(y15) + exp(y2) + exp(y7) - 240365472".to_string(),
        ]);
        equations
    }

    fn tp1907_expression_equations() -> Vec<Expr> {
        let variable = |index| Expr::Var(format!("y{index}"));
        let parameter = |index| Expr::Var(format!("g0_{index}"));
        let temperature = Expr::Var("T".to_string());
        let log_sum = (0..18)
            .map(|index| Expr::Exp(variable(index).boxed()))
            .reduce(|sum, term| sum + term)
            .expect("TP-1907 has species");
        let log_sum = Expr::Ln(log_sum.boxed());
        let log_y17 = Expr::Ln(Expr::Exp(variable(17).boxed()).boxed());

        let mut equations = Vec::with_capacity(18);
        for row in 0..13 {
            let y_terms = LINEAR_ROWS[row]
                .iter()
                .enumerate()
                .map(|(index, coefficient)| Expr::Const(*coefficient) * variable(index))
                .reduce(|sum, term| sum + term)
                .expect("reaction row has coefficients");
            let g_terms = LINEAR_ROWS[row]
                .iter()
                .enumerate()
                .map(|(index, coefficient)| Expr::Const(*coefficient) * parameter(index))
                .reduce(|sum, term| sum + term)
                .expect("reaction row has coefficients");
            equations.push(
                y_terms
                    + Expr::Const(G0_OVER_T) * g_terms / temperature.clone()
                    + Expr::Const(LOG_Y17[row]) * log_y17.clone()
                    + Expr::Const(LOG_SUM[row]) * log_sum.clone(),
            );
        }
        equations.extend([
            Expr::Exp(variable(0).boxed()) - Expr::Const(5_364_815.999_999_999),
            Expr::Exp(variable(1).boxed())
                + Expr::Exp(variable(17).boxed())
                + Expr::Exp(variable(2).boxed())
                + Expr::Exp(variable(3).boxed())
                - Expr::Const(100_182_736.0),
            Expr::Const(2.0) * Expr::Exp(variable(6).boxed())
                + Expr::Const(2.0) * Expr::Exp(variable(7).boxed())
                + Expr::Const(3.0) * Expr::Exp(variable(9).boxed())
                + Expr::Const(4.0) * Expr::Exp(variable(1).boxed())
                + Expr::Exp(variable(15).boxed())
                + Expr::Exp(variable(4).boxed())
                + Expr::Exp(variable(5).boxed())
                - Expr::Const(200_000_000.0),
            Expr::Const(2.0) * Expr::Exp(variable(12).boxed())
                + Expr::Const(2.0) * Expr::Exp(variable(13).boxed())
                + Expr::Exp(variable(10).boxed())
                + Expr::Exp(variable(11).boxed())
                + Expr::Exp(variable(8).boxed())
                + Expr::Exp(variable(9).boxed())
                - Expr::Const(894_620_880.000_000_1),
            Expr::Const(2.0) * Expr::Exp(variable(11).boxed())
                + Expr::Const(2.0) * Expr::Exp(variable(16).boxed())
                + Expr::Const(2.0) * Expr::Exp(variable(3).boxed())
                + Expr::Const(2.0) * Expr::Exp(variable(5).boxed())
                + Expr::Exp(variable(10).boxed())
                + Expr::Exp(variable(13).boxed())
                + Expr::Exp(variable(14).boxed())
                + Expr::Exp(variable(15).boxed())
                + Expr::Exp(variable(2).boxed())
                + Expr::Exp(variable(7).boxed())
                - Expr::Const(240_365_472.0),
        ]);
        equations
    }

    fn tp1907_problem() -> (SymbolicNonlinearProblem, DVector<f64>, Bounds) {
        let variables = (0..18).map(|index| format!("y{index}")).collect();
        let parameters = std::iter::once("T".to_string())
            .chain((0..18).map(|index| format!("g0_{index}")))
            .collect();
        let parameter_values = DVector::from_vec(vec![
            TEMPERATURE,
            -1.12446021115026888e5,
            -2.13170551189015736e5,
            -2.54631726236580202e5,
            -5.51213378891847096e5,
            1.33615869738429552e5,
            -1.55139954474190337e5,
            -9.71743984271630470e4,
            -3.80706770892712770e5,
            3.61263767705759616e5,
            -1.88510850753486360e5,
            -6.21446958310785703e4,
            -1.41891230313522334e5,
            -1.39857084358844266e5,
            -8.07814187503829162e4,
            1.32213985757463466e5,
            -9.50548687019174831e4,
            -1.49514500249614095e5,
            -6.36346609712339068e3,
        ]);
        let problem = SymbolicNonlinearProblem::from_strings_with_options(
            tp1907_equations(),
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(parameters)
                .with_equation_parameter_values(parameter_values),
        )
        .expect("TP-1907 symbolic problem should parse");
        let initial = DVector::from_vec(vec![
            1.54953726370934710e1,
            1.77275335633924200e1,
            -4.60517018598809145e1,
            1.77311816211308155e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            1.99187634081709426e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            -4.60517018598809145e1,
            1.80640058000136321e1,
            -4.60517018598809145e1,
        ]);
        let lower = -7.08396418532264079e2;
        let upper = [
            1.54953726370944711e1,
            1.77275335633934183e1,
            1.84225064363622977e1,
            1.84225064363622977e1,
            1.91138279245133090e1,
            1.86045239424631390e1,
            1.84206807439533655e1,
            1.84206807439533655e1,
            2.06119105887318881e1,
            1.80152156358451982e1,
            1.92976711230230862e1,
            1.86045239424631390e1,
            1.99187634081719409e1,
            1.92976711230230862e1,
            1.92976711230230862e1,
            1.91138279245133090e1,
            1.86045239424631390e1,
            1.84225064363622977e1,
        ];
        let bounds = Bounds::new(upper.into_iter().map(|value| (lower, value)).collect())
            .expect("TP-1907 bounds should be valid");
        (problem, initial, bounds)
    }

    fn tp1907_problem_from_expressions() -> SymbolicNonlinearProblem {
        let variables = (0..18).map(|index| format!("y{index}")).collect();
        let parameters = std::iter::once("T".to_string())
            .chain((0..18).map(|index| format!("g0_{index}")))
            .collect();
        let parameter_values = DVector::from_vec(vec![
            TEMPERATURE,
            -1.12446021115026888e5,
            -2.13170551189015736e5,
            -2.54631726236580202e5,
            -5.51213378891847096e5,
            1.33615869738429552e5,
            -1.55139954474190337e5,
            -9.71743984271630470e4,
            -3.80706770892712770e5,
            3.61263767705759616e5,
            -1.88510850753486360e5,
            -6.21446958310785703e4,
            -1.41891230313522334e5,
            -1.39857084358844266e5,
            -8.07814187503829162e4,
            1.32213985757463466e5,
            -9.50548687019174831e4,
            -1.49514500249614095e5,
            -6.36346609712339068e3,
        ]);
        SymbolicNonlinearProblem::from_expressions_with_options(
            tp1907_expression_equations(),
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(parameters)
                .with_equation_parameter_values(parameter_values),
        )
        .expect("programmatic TP-1907 symbolic problem should prepare")
    }

    fn oracle_log_moles() -> DVector<f64> {
        DVector::from_vec(ORACLE_LOG_MOLES.to_vec())
    }

    fn options(bounds: Bounds) -> SolveOptions {
        SolveOptions {
            tolerance: 1.0e-8,
            max_iterations: 500,
            bounds: Some(bounds),
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    /// Row scaling used by chemical-equilibrium solvers when inventories are
    /// supplied as absolute mole counts. It preserves the zero set while
    /// putting thermodynamic and inventory equations on comparable scales.
    ///
    /// The fixed graphite active-set reduction is structurally rank deficient.
    /// Scaling therefore improves conditioning but does not turn this bounded
    /// least-squares experiment into a unique-root problem.
    struct InventoryScaledProblem<'a> {
        base: &'a SymbolicNonlinearProblem,
        scales: DVector<f64>,
        coordinate_shift: f64,
    }

    impl<'a> InventoryScaledProblem<'a> {
        fn new(base: &'a SymbolicNonlinearProblem) -> Self {
            Self {
                base,
                scales: DVector::from_vec(vec![
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0 / 5.364815999999999e6,
                    1.0 / 1.00182736e8,
                    1.0 / 2.0e8,
                    1.0 / 8.9462088e8,
                    1.0 / 2.40365472e8,
                ]),
                coordinate_shift: 0.0,
            }
        }

        fn from_denominators(
            base: &'a SymbolicNonlinearProblem,
            denominators: &DVector<f64>,
        ) -> Self {
            Self::from_denominators_and_log_shift(base, denominators, 0.0)
        }

        fn from_denominators_and_log_shift(
            base: &'a SymbolicNonlinearProblem,
            denominators: &DVector<f64>,
            coordinate_shift: f64,
        ) -> Self {
            assert!(
                denominators
                    .iter()
                    .all(|value| value.is_finite() && *value != 0.0)
            );
            assert!(coordinate_shift.is_finite());
            Self {
                base,
                scales: denominators.map(|value| 1.0 / value),
                coordinate_shift,
            }
        }

        fn physical_coordinates(&self, x: &DVector<f64>) -> DVector<f64> {
            x.map(|value| value + self.coordinate_shift)
        }

        fn scale_residual(&self, residual: &mut DVector<f64>) {
            residual.component_mul_assign(&self.scales);
        }

        fn scale_jacobian(&self, jacobian: &mut nalgebra::DMatrix<f64>) {
            for row in 0..jacobian.nrows() {
                for column in 0..jacobian.ncols() {
                    jacobian[(row, column)] *= self.scales[row];
                }
            }
        }
    }

    impl NonlinearProblem for InventoryScaledProblem<'_> {
        fn dimension(&self) -> usize {
            self.base.dimension()
        }

        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
            let physical_x = self.physical_coordinates(x);
            let mut residual = self.base.residual(&physical_x)?;
            self.scale_residual(&mut residual);
            Ok(residual)
        }
    }

    impl JacobianProvider for InventoryScaledProblem<'_> {
        fn jacobian(
            &self,
            x: &DVector<f64>,
        ) -> Result<nalgebra::DMatrix<f64>, crate::numerical::Nonlinear_systems::error::SolveError>
        {
            let physical_x = self.physical_coordinates(x);
            let mut jacobian = self.base.jacobian(&physical_x)?;
            self.scale_jacobian(&mut jacobian);
            Ok(jacobian)
        }
    }

    #[derive(Debug)]
    struct KitheStyleLmTrace {
        x: DVector<f64>,
        residual_norm: f64,
        iterations: usize,
        accepted_steps: usize,
        rejected_steps: usize,
        final_lambda: f64,
        last_alpha: f64,
    }

    #[derive(Debug)]
    struct HistoricalMinpackTrace {
        x: DVector<f64>,
        residual_norm: f64,
        iterations: usize,
        accepted_steps: usize,
        rejected_steps: usize,
        termination: TerminationReason,
    }

    /// Replays the pre-refactor 0.4.15 Minpack policy on the current engine
    /// primitives. This is deliberately test-only: it is an A/B diagnostic,
    /// not a second production implementation. The replay preserves the old
    /// approximate gradient test, old trust-region branch ordering, zero-based
    /// `nfev`, and the old generic engine's one-step-per-outer-iteration
    /// behavior. Keeping this local makes the historical comparison explicit
    /// without weakening the current Fortran-grade implementation.
    fn solve_historical_minpack_policy<P: JacobianProvider>(
        problem: &P,
        initial: DVector<f64>,
        bounds: &Bounds,
        method: &LevenbergMarquardtMinpack,
        options: &SolveOptions,
    ) -> Result<HistoricalMinpackTrace, crate::numerical::Nonlinear_systems::error::SolveError>
    {
        use crate::numerical::Nonlinear_systems::trust_region_LM::solve_trust_region_subproblem;

        let mut x = initial;
        let mut residual = problem.residual(&x)?;
        let mut diag = crate::numerical::Nonlinear_systems::engine::scaling_vector(
            &problem.jacobian(&x)?,
            true,
        );
        let xnorm = diag.component_mul(&x).norm();
        let mut delta = method.factor * if xnorm > 0.0 { xnorm } else { 1.0 };
        let mut par = 0.0;
        let mut nfev = 0_usize;
        let mut accepted_steps = 0_usize;
        let mut rejected_steps = 0_usize;

        for iteration in 0..options.max_iterations {
            let jacobian = problem.jacobian(&x)?;
            let fnorm = residual.norm();
            let jtj = jacobian.transpose() * &jacobian;
            let gradient = jacobian.transpose() * &residual;

            // This is the old 0.4.15 approximation, intentionally retained
            // only to explain historical behavior. It is not MINPACK's
            // actual scaled-gradient calculation.
            let mut gnorm = 0.0_f64;
            if fnorm > 0.0 {
                for column_index in 0..jacobian.ncols() {
                    let column_norm = jacobian.column(column_index).norm();
                    if column_norm == 0.0 {
                        continue;
                    }
                    let mut sum = 0.0;
                    for row_index in 0..jacobian.ncols() {
                        sum += jtj[(row_index, column_index)] * (gradient[row_index] / fnorm);
                    }
                    let denominator = if method.mode == 2 {
                        diag[column_index]
                    } else {
                        column_norm
                    };
                    if denominator != 0.0 {
                        gnorm = gnorm.max(sum.abs() / denominator);
                    }
                }
            }
            if gnorm <= method.gtol {
                return Ok(HistoricalMinpackTrace {
                    x,
                    residual_norm: fnorm,
                    iterations: iteration,
                    accepted_steps,
                    rejected_steps,
                    termination: TerminationReason::Converged,
                });
            }

            if method.mode != 2 {
                for column_index in 0..jacobian.ncols() {
                    let column_norm = jacobian.column(column_index).norm();
                    diag[column_index] = diag[column_index].max(column_norm).max(1.0);
                }
            }

            let subproblem = solve_trust_region_subproblem(&jacobian, &residual, &diag, delta, par)
                .map_err(|message| {
                    crate::numerical::Nonlinear_systems::error::SolveError::LinearSolveFailure(
                        message.to_string(),
                    )
                })?;
            let pvec = subproblem.step;
            par = subproblem.lambda;
            let pnorm = crate::numerical::Nonlinear_systems::engine::scaled_norm(&diag, &pvec);
            let current_xnorm = diag.component_mul(&x).norm();
            if iteration == 0 {
                delta = delta.min(pnorm);
            }

            let mut trial_x = &x - &pvec;
            trial_x = bounds.project(&trial_x);
            let trial_residual = problem.residual(&trial_x)?;
            nfev += 1;
            let fnorm1 = trial_residual.norm();
            let actred = if 0.1 * fnorm1 < fnorm {
                1.0 - (fnorm1 / fnorm).powi(2)
            } else {
                -1.0
            };
            let j_p = &jacobian * &pvec;
            let temp1 = j_p.norm() / fnorm.max(1e-300);
            let temp2 = (par.sqrt() * pnorm) / fnorm.max(1e-300);
            let prered = temp1 * temp1 + (temp2 * temp2) / 0.5;
            let dirder = -(temp1 * temp1 + temp2 * temp2);
            let ratio = if prered != 0.0 { actred / prered } else { 0.0 };

            // Exact ordering used by the old implementation. In particular,
            // the old branch updated delta in the middle ratio interval.
            if ratio > 0.25 {
                if par == 0.0 || ratio < 0.75 {
                    delta = pnorm / 0.5;
                    par = 0.5 * par;
                } else {
                    delta = pnorm / 0.5;
                }
            } else {
                let mut temp = 0.5;
                if actred < 0.0 {
                    temp = 0.5 * dirder / (dirder + 0.5 * actred);
                }
                if 0.1 * fnorm1 >= fnorm || temp < 0.1 {
                    temp = 0.1;
                }
                delta = temp * delta.min(pnorm / 0.1);
                if par != 0.0 {
                    par /= temp;
                }
            }

            if ratio > 1.0e-4 {
                x = trial_x;
                residual = trial_residual;
                accepted_steps += 1;
                if residual.norm() < options.tolerance {
                    return Ok(HistoricalMinpackTrace {
                        x,
                        residual_norm: residual.norm(),
                        iterations: iteration + 1,
                        accepted_steps,
                        rejected_steps,
                        termination: TerminationReason::Converged,
                    });
                }
            } else {
                rejected_steps += 1;
            }

            if nfev >= method.maxfev {
                return Ok(HistoricalMinpackTrace {
                    x,
                    residual_norm: residual.norm(),
                    iterations: iteration + 1,
                    accepted_steps,
                    rejected_steps,
                    termination: TerminationReason::MaxIterations,
                });
            }
            if actred.abs() <= f64::EPSILON && prered <= f64::EPSILON && 0.5 * ratio <= 1.0 {
                return Ok(HistoricalMinpackTrace {
                    x,
                    residual_norm: residual.norm(),
                    iterations: iteration + 1,
                    accepted_steps,
                    rejected_steps,
                    termination: TerminationReason::Stagnation,
                });
            }
            if delta <= f64::EPSILON * current_xnorm {
                return Ok(HistoricalMinpackTrace {
                    x,
                    residual_norm: residual.norm(),
                    iterations: iteration + 1,
                    accepted_steps,
                    rejected_steps,
                    termination: TerminationReason::Stagnation,
                });
            }
        }

        Ok(HistoricalMinpackTrace {
            x,
            residual_norm: residual.norm(),
            iterations: options.max_iterations,
            accepted_steps,
            rejected_steps,
            termination: TerminationReason::MaxIterations,
        })
    }

    /// A diagnostic-only reference for the policy used by the KiThe facade:
    /// `(J^T J + lambda I) step = -J^T F`, strict feasibility, and a short
    /// backtracking search before increasing `lambda`. It intentionally lives
    /// in this test gate until an A/B trace establishes which policy helps the
    /// chemistry corpus; it is not a second production implementation.
    fn solve_kithe_style_lm_trace<P: JacobianProvider, F: Fn(&DVector<f64>) -> bool>(
        problem: &P,
        initial: DVector<f64>,
        feasible: F,
    ) -> Result<KitheStyleLmTrace, crate::numerical::Nonlinear_systems::error::SolveError> {
        let mut x = initial;
        let mut residual = problem.residual(&x)?;
        let mut residual_norm = residual.norm();
        let mut lambda = 1.0e-3;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;
        let mut last_alpha = 0.0;
        let mut iterations = 0;

        for iteration in 0..500 {
            iterations = iteration + 1;
            let jacobian = problem.jacobian(&x)?;
            let mut normal = jacobian.transpose() * &jacobian;
            for diagonal in 0..normal.nrows() {
                normal[(diagonal, diagonal)] += lambda;
            }
            let rhs = -(jacobian.transpose() * &residual);
            let step = solve_linear_system(LinearSolverKind::Lu, &normal, &rhs)?;

            if !step.iter().all(|value| value.is_finite()) || step.norm() <= 1.0e-12 {
                break;
            }

            let mut accepted = None;
            let mut alpha = 1.0;
            while alpha >= 1.0e-6 {
                let candidate = &x + alpha * &step;
                if feasible(&candidate) {
                    let candidate_residual = problem.residual(&candidate)?;
                    let candidate_norm = candidate_residual.norm();
                    if candidate_norm.is_finite() && candidate_norm < residual_norm {
                        accepted = Some((candidate, candidate_residual, candidate_norm, alpha));
                        break;
                    }
                }
                alpha *= 0.5;
            }

            if let Some((candidate, candidate_residual, candidate_norm, accepted_alpha)) = accepted
            {
                x = candidate;
                residual = candidate_residual;
                residual_norm = candidate_norm;
                lambda = (lambda * 0.3).max(1.0e-15);
                accepted_steps += 1;
                last_alpha = accepted_alpha;
            } else {
                lambda = (lambda * 10.0).min(1.0e15);
                rejected_steps += 1;
                last_alpha = 0.0;
            }
        }

        Ok(KitheStyleLmTrace {
            x,
            residual_norm,
            iterations,
            accepted_steps,
            rejected_steps,
            final_lambda: lambda,
            last_alpha,
        })
    }

    fn raw_inventory_max_abs(problem: &SymbolicNonlinearProblem, x: &DVector<f64>) -> f64 {
        problem
            .residual(x)
            .expect("TP-1907 raw residual")
            .iter()
            .skip(13)
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    }

    fn assert_finite_bounded_decrease(
        name: &str,
        x: &DVector<f64>,
        residual_norm: f64,
        initial_norm: f64,
        bounds: &Bounds,
    ) {
        assert!(
            x.iter().all(|value| value.is_finite()),
            "{name} produced non-finite x"
        );
        assert!(
            x.iter()
                .zip(bounds.as_slice())
                .all(|(value, (lower, upper))| *value >= *lower && *value <= *upper),
            "{name} left the declared bounds"
        );
        assert!(
            residual_norm.is_finite() && residual_norm < initial_norm,
            "{name} did not decrease the scaled residual"
        );
    }

    fn robust_methods() -> Vec<(&'static str, NonlinearSolverMethod)> {
        vec![
            (
                "minpack",
                NonlinearSolverMethod::LevenbergMarquardtMinpack(
                    LevenbergMarquardtMinpack::default(),
                ),
            ),
            (
                "nielsen",
                NonlinearSolverMethod::NielsenLevenbergMarquardt(
                    NielsenLevenbergMarquardtMethod::default(),
                ),
            ),
            (
                "trust-region-lm",
                NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default()),
            ),
        ]
    }

    fn print_result(
        name: &str,
        result: &Result<SolveResult, crate::numerical::Nonlinear_systems::error::SolveError>,
    ) {
        match result {
            Ok(result) => println!(
                "[TP-1907 gate] method={name} termination={:?} converged={} residual_l2={:.6e} iterations={} residual_calls={} jacobian_calls={} linear_solves={}",
                result.termination,
                result.termination == TerminationReason::Converged,
                result.residual_norm,
                result.iterations,
                result.statistics.residual_evaluations,
                result.statistics.jacobian_evaluations,
                result.statistics.linear_solves,
            ),
            Err(error) => println!("[TP-1907 gate] method={name} error={error}"),
        }
    }

    fn print_exact_replay_result(
        method: &str,
        scaling: &str,
        result: &Result<SolveResult, crate::numerical::Nonlinear_systems::error::SolveError>,
        base: &SymbolicNonlinearProblem,
        coordinate_shift: f64,
    ) {
        match result {
            Ok(result) => println!(
                "[TP-1907 exact replay] method={method} scaling={scaling} termination={:?} residual_l2={:.6e} raw_inventory_max={:.6e} iterations={} accepted={} rejected={} R={} J={} L={}",
                result.termination,
                result.residual_norm,
                raw_inventory_max_abs(base, &result.x.map(|value| value + coordinate_shift)),
                result.iterations,
                result.statistics.accepted_steps,
                result.statistics.rejected_steps,
                result.statistics.residual_evaluations,
                result.statistics.jacobian_evaluations,
                result.statistics.linear_solves,
            ),
            Err(error) => {
                println!("[TP-1907 exact replay] method={method} scaling={scaling} error={error}")
            }
        }
    }

    fn print_historical_replay_result(
        scaling: &str,
        result: &Result<
            HistoricalMinpackTrace,
            crate::numerical::Nonlinear_systems::error::SolveError,
        >,
        base: &SymbolicNonlinearProblem,
        coordinate_shift: f64,
    ) {
        match result {
            Ok(result) => println!(
                "[TP-1907 exact replay] method=LM-Minpack-0.4.15-policy scaling={scaling} termination={:?} residual_l2={:.6e} raw_inventory_max={:.6e} iterations={} accepted={} rejected={}",
                result.termination,
                result.residual_norm,
                raw_inventory_max_abs(base, &result.x.map(|value| value + coordinate_shift)),
                result.iterations,
                result.accepted_steps,
                result.rejected_steps,
            ),
            Err(error) => println!(
                "[TP-1907 exact replay] method=LM-Minpack-0.4.15-policy scaling={scaling} error={error}"
            ),
        }
    }

    #[test]
    fn tp1907_chon_graphite_reproducer_has_finite_expected_initial_state() {
        let (problem, initial, bounds) = tp1907_problem();
        assert_eq!(problem.dimension(), 18);
        assert!(initial.iter().all(|value| value.is_finite()));
        for (value, (lower, upper)) in initial.iter().zip(bounds.as_slice()) {
            assert!(*value >= *lower && *value <= *upper);
        }
        let residual = problem.residual(&initial).expect("initial residual");
        let jacobian = problem.jacobian(&initial).expect("initial Jacobian");
        assert_eq!(residual.len(), 18);
        assert_eq!((jacobian.nrows(), jacobian.ncols()), (18, 18));
        assert!(residual.iter().all(|value| value.is_finite()));
        assert!(jacobian.iter().all(|value| value.is_finite()));
        assert!((residual.norm() - 124.8643832783359).abs() < 1.0e-8);
    }

    #[test]
    fn tp1907_chon_graphite_reproducer_is_rank_deficient_without_phase_control() {
        let (problem, initial, _) = tp1907_problem();
        let singular_values = problem
            .jacobian(&initial)
            .expect("initial Jacobian")
            .svd(false, false)
            .singular_values;
        let rank = singular_values
            .iter()
            .filter(|value| **value > 1.0e-12)
            .count();
        assert_eq!(rank, 17, "TP-1907 active-set reduction changed rank");
        assert!(singular_values[17] <= 1.0e-12);
    }

    #[test]
    fn tp1907_rederived_graph_is_finite_at_the_exported_kithe_oracle() {
        let (problem, _, bounds) = tp1907_problem();
        let programmatic_problem = tp1907_problem_from_expressions();
        let scaled_problem = InventoryScaledProblem::new(&problem);
        let oracle = oracle_log_moles();
        bounds
            .validate(&oracle)
            .expect("KiThe oracle must satisfy RST bounds");

        let raw_residual = problem.residual(&oracle).expect("oracle raw residual");
        let raw_l2 = raw_residual.norm();
        let scaled_l2 = scaled_problem
            .residual(&oracle)
            .expect("oracle scaled residual")
            .norm();
        let raw_inventory_max = raw_inventory_max_abs(&problem, &oracle);
        let programmatic_l2 = programmatic_problem
            .residual(&oracle)
            .expect("programmatic oracle residual")
            .norm();
        println!(
            "[TP-1907 oracle parity] string_raw_l2={raw_l2:.6e}; programmatic_raw_l2={programmatic_l2:.6e}; scaled_l2={scaled_l2:.6e}; raw_inventory_max={raw_inventory_max:.6e}"
        );
        println!(
            "[TP-1907 oracle parity] reaction_rows={:?}",
            raw_residual.rows(0, 13).as_slice()
        );

        for (index, (expected, actual)) in ORACLE_MOLES
            .iter()
            .zip(oracle.iter().map(|value| value.exp()))
            .enumerate()
        {
            let relative_error = (actual - expected).abs() / expected.abs().max(1.0);
            assert!(
                relative_error < 1.0e-12,
                "full-precision log-mole fixture no longer reproduces published mole {index}"
            );
        }

        assert!(
            raw_l2 < 2.0e-3,
            "rederived TP-1907 graph exceeded the established transcription-gap envelope"
        );
        assert!(
            scaled_l2 < 2.0e-3,
            "scaled rederived TP-1907 graph exceeded the established transcription-gap envelope"
        );
        assert!(
            raw_inventory_max < 2.0e-6,
            "oracle inventory balance changed"
        );
        assert!(
            (programmatic_l2 - raw_l2).abs() < 1.0e-10,
            "string and programmatic TP-1907 construction disagree"
        );
    }

    #[test]
    fn tp1907_canonical_export_round_trips_through_the_public_string_api() {
        let problem = canonical_fixture_problem();
        let y_final = fixture_numeric_vector("let y_final: Vec<f64> = vec![");
        let expected_raw = fixture_numeric_vector("let raw_residual: Vec<f64> = vec![");
        let row_denominators = fixture_numeric_vector("let residual_row_scale_w: Vec<f64> = vec![");
        let expected_jacobian = fixture_numeric_matrix("let raw_jacobian: Vec<Vec<f64>> = vec![");
        let expected_jt_f = fixture_numeric_vector("let jt_f_raw: Vec<f64> = vec![");
        let actual_raw = problem
            .residual(&y_final)
            .expect("canonical TP-1907 residual must evaluate");
        let actual_jacobian = problem
            .jacobian(&y_final)
            .expect("canonical TP-1907 Jacobian must evaluate");
        let actual_scaled = actual_raw.component_div(&row_denominators);
        let expected_scaled = expected_raw.component_div(&row_denominators);
        let raw_drift = (&actual_raw - &expected_raw).amax();
        let scaled_drift = (&actual_scaled - &expected_scaled).amax();
        let jacobian_drift = (&actual_jacobian - &expected_jacobian).amax();
        let actual_jt_f = actual_jacobian.transpose() * &actual_raw;
        let jt_f_drift = (&actual_jt_f - &expected_jt_f).amax();

        println!(
            "[TP-1907 canonical string round-trip] raw_l2={:.16e}; scaled_l2={:.16e}; raw_drift={raw_drift:.3e}; scaled_drift={scaled_drift:.3e}; jacobian_drift={jacobian_drift:.3e}; jt_f_drift={jt_f_drift:.3e}",
            actual_raw.norm(),
            actual_scaled.norm(),
        );
        assert!(
            raw_drift <= 1.0e-12,
            "canonical TP-1907 residual changed after Expr::to_string()/parse round-trip"
        );
        assert!(
            scaled_drift <= 1.0e-12,
            "canonical scaled TP-1907 residual changed after Expr::to_string()/parse round-trip"
        );
        assert!(
            jacobian_drift <= 1.0e-12,
            "canonical TP-1907 Jacobian changed after Expr::to_string()/parse round-trip"
        );
        assert!(
            jt_f_drift <= 1.0e-8,
            "canonical TP-1907 J^T F changed after Expr::to_string()/parse round-trip"
        );
    }

    #[test]
    fn tp1907_chon_graphite_methods_are_finite_and_do_not_claim_a_missing_root() {
        let (problem, initial, bounds) = tp1907_problem();
        let scaled_problem = InventoryScaledProblem::new(&problem);
        let initial_residual_norm = scaled_problem
            .residual(&initial)
            .expect("initial residual")
            .norm();
        let mut failures = Vec::new();
        for (name, method) in robust_methods() {
            let result = method.solve(&scaled_problem, initial.clone(), options(bounds.clone()));
            print_result(name, &result);
            match result {
                Ok(result) => {
                    if result.termination == TerminationReason::Converged
                        || !result.residual_norm.is_finite()
                        || result.residual_norm > 1.0e-2 * initial_residual_norm
                        || !result.x.iter().all(|value| value.is_finite())
                        || result
                            .x
                            .iter()
                            .zip(bounds.as_slice())
                            .any(|(value, (lower, upper))| *value < *lower || *value > *upper)
                    {
                        failures.push(format!(
                            "{name}: termination={:?}, residual={:.6e}",
                            result.termination, result.residual_norm
                        ));
                    }
                }
                Err(error) => failures.push(format!("{name}: error={error}")),
            }
        }
        assert!(
            failures.is_empty(),
            "TP-1907 robust gate failures: {failures:?}"
        );
    }

    #[test]
    fn tp1907_backtracking_lm_reports_finite_residual_and_balance() {
        let (problem, initial, bounds) = tp1907_problem();
        let scaled_problem = InventoryScaledProblem::new(&problem);
        let initial_norm = scaled_problem
            .residual(&initial)
            .expect("TP-1907 initial scaled residual")
            .norm();
        let reference_initial = initial.clone();
        let result = NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
            BacktrackingLevenbergMarquardtMethod::default(),
        )
        .solve(&scaled_problem, initial, options(bounds.clone()))
        .expect("TP-1907 backtracking LM should return a typed result");
        let raw_inventory_max = raw_inventory_max_abs(&problem, &result.x);
        let reference =
            solve_kithe_style_lm_trace(&scaled_problem, reference_initial, |candidate| {
                bounds.validate(candidate).is_ok()
            })
            .expect("TP-1907 reference backtracking trace should remain finite");

        println!(
            "[TP-1907 backtracking LM] termination={:?} residual_l2={:.6e} initial_l2={:.6e} raw_inventory_max={:.6e} iterations={} accepted={} rejected={} reference_residual={:.6e} reference_raw_inventory_max={:.6e}",
            result.termination,
            result.residual_norm,
            initial_norm,
            raw_inventory_max,
            result.iterations,
            result.statistics.accepted_steps,
            result.statistics.rejected_steps,
            reference.residual_norm,
            raw_inventory_max_abs(&problem, &reference.x),
        );
        assert!(result.x.iter().all(|value| value.is_finite()));
        assert!(result.residual_norm.is_finite() && result.residual_norm < initial_norm);
        assert!(raw_inventory_max.is_finite());
        assert!((result.residual_norm - reference.residual_norm).abs() <= 1.0e-12);
        assert!(
            (raw_inventory_max - raw_inventory_max_abs(&problem, &reference.x)).abs() <= 1.0e-6
        );
        assert!(
            result
                .x
                .iter()
                .zip(reference.x.iter())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max)
                <= 1.0e-10,
            "public backtracking LM diverged from its reference policy"
        );
        assert!(
            result.termination != TerminationReason::Converged,
            "incomplete TP-1907 fixture must not be reported as a physical root"
        );
        bounds
            .validate(&result.x)
            .expect("backtracking LM must preserve TP-1907 bounds");
    }

    #[test]
    fn tp1907_lm_policy_trace_separates_damping_from_feasible_backtracking() {
        let (problem, initial, bounds) = tp1907_problem();
        let scaled_problem = InventoryScaledProblem::new(&problem);
        let initial_norm = scaled_problem
            .residual(&initial)
            .expect("initial scaled residual")
            .norm();

        let diagonal = NonlinearSolverMethod::LevenbergMarquardt(
            LevenbergMarquardtMethod::default(),
        )
        .solve(&scaled_problem, initial.clone(), options(bounds.clone()));
        let identity = NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod {
            diag_scaling: false,
            increase_factor: 10.0,
            decrease_factor: 10.0,
            ..LevenbergMarquardtMethod::default()
        })
        .solve(&scaled_problem, initial.clone(), options(bounds.clone()))
        .expect("identity-damped LM trace should remain finite");
        let kithe_style =
            solve_kithe_style_lm_trace(&scaled_problem, initial.clone(), |candidate| {
                bounds.validate(candidate).is_ok()
            })
            .expect("KiThe-style LM trace should remain finite");
        let mut unbounded_options = options(bounds.clone());
        unbounded_options.bounds = None;
        let identity_unbounded =
            NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod {
                diag_scaling: false,
                increase_factor: 10.0,
                decrease_factor: 10.0,
                ..LevenbergMarquardtMethod::default()
            })
            .solve(&scaled_problem, initial.clone(), unbounded_options);
        let mut trust_region_options = options(bounds.clone());
        trust_region_options.bounds = None;
        let trust_region_unbounded = NonlinearSolverMethod::TrustRegion(
            TrustRegionMethod::default(),
        )
        .solve(&scaled_problem, initial.clone(), trust_region_options);
        let kithe_style_log_feasible =
            solve_kithe_style_lm_trace(&scaled_problem, initial, |candidate| {
                candidate.iter().all(|value| {
                    let moles = value.exp();
                    value.is_finite() && moles.is_finite() && moles > 0.0
                })
            })
            .expect("log-feasible KiThe-style LM trace should remain finite");

        println!(
            "[TP-1907 LM policy trace] same scaled symbolic residual/Jacobian, initial point, and bounds"
        );
        println!(
            "policy                         | scaled_l2 | raw_inventory_max | term/iterations | accepted/rejected | final_lambda | last_alpha"
        );
        println!(
            "--------------------------------------------------------------------------------------------------------------------------------"
        );
        match &diagonal {
            Ok(result) => println!(
                "RST diagonal single-trial       | {:9.3e} | {:17.3e} | {:?}/{:10} | {:8}/{:<8} |            - |          -",
                result.residual_norm,
                raw_inventory_max_abs(&problem, &result.x),
                result.termination,
                result.iterations,
                result.statistics.accepted_steps,
                result.statistics.rejected_steps,
            ),
            Err(error) => println!(
                "RST diagonal single-trial       |         - |                 - | error/{:<15} |        -/       - |            - |          -",
                error,
            ),
        }
        println!(
            "RST identity single-trial       | {:9.3e} | {:17.3e} | {:?}/{:10} | {:8}/{:<8} |            - |          -",
            identity.residual_norm,
            raw_inventory_max_abs(&problem, &identity.x),
            identity.termination,
            identity.iterations,
            identity.statistics.accepted_steps,
            identity.statistics.rejected_steps,
        );
        println!(
            "KiThe-style identity/backtrack  | {:9.3e} | {:17.3e} | trace/{:13} | {:8}/{:<8} | {:11.3e} | {:10.3e}",
            kithe_style.residual_norm,
            raw_inventory_max_abs(&problem, &kithe_style.x),
            kithe_style.iterations,
            kithe_style.accepted_steps,
            kithe_style.rejected_steps,
            kithe_style.final_lambda,
            kithe_style.last_alpha,
        );
        match &identity_unbounded {
            Ok(result) => println!(
                "RST identity unbounded         | {:9.3e} | {:17.3e} | {:?}/{:10} | {:8}/{:<8} |            - |          -",
                result.residual_norm,
                raw_inventory_max_abs(&problem, &result.x),
                result.termination,
                result.iterations,
                result.statistics.accepted_steps,
                result.statistics.rejected_steps,
            ),
            Err(error) => println!(
                "RST identity unbounded         |         - |                 - | error/{:<15} |        -/       - |            - |          -",
                error,
            ),
        }
        match &trust_region_unbounded {
            Ok(result) => println!(
                "RST Trust Region unbounded     | {:9.3e} | {:17.3e} | {:?}/{:10} | {:8}/{:<8} |            - |          -",
                result.residual_norm,
                raw_inventory_max_abs(&problem, &result.x),
                result.termination,
                result.iterations,
                result.statistics.accepted_steps,
                result.statistics.rejected_steps,
            ),
            Err(error) => println!(
                "RST Trust Region unbounded     |         - |                 - | error/{:<15} |        -/       - |            - |          -",
                error,
            ),
        }
        println!(
            "KiThe-style log-feasible       | {:9.3e} | {:17.3e} | trace/{:13} | {:8}/{:<8} | {:11.3e} | {:10.3e}",
            kithe_style_log_feasible.residual_norm,
            raw_inventory_max_abs(&problem, &kithe_style_log_feasible.x),
            kithe_style_log_feasible.iterations,
            kithe_style_log_feasible.accepted_steps,
            kithe_style_log_feasible.rejected_steps,
            kithe_style_log_feasible.final_lambda,
            kithe_style_log_feasible.last_alpha,
        );

        if let Ok(result) = &diagonal {
            assert_finite_bounded_decrease(
                "RST diagonal",
                &result.x,
                result.residual_norm,
                initial_norm,
                &bounds,
            );
        }
        assert_finite_bounded_decrease(
            "RST identity",
            &identity.x,
            identity.residual_norm,
            initial_norm,
            &bounds,
        );
        if let Ok(result) = &identity_unbounded {
            assert!(
                result.x.iter().all(|value| value.is_finite())
                    && result.residual_norm.is_finite()
                    && result.residual_norm < initial_norm,
                "RST unbounded identity did not produce a finite residual decrease"
            );
        }
        if let Ok(result) = &trust_region_unbounded {
            assert!(
                result.x.iter().all(|value| value.is_finite())
                    && result.residual_norm.is_finite()
                    && result.residual_norm < initial_norm,
                "RST unbounded Trust Region did not produce a finite residual decrease"
            );
        }
        assert!(
            kithe_style_log_feasible
                .x
                .iter()
                .all(|value| value.is_finite())
                && kithe_style_log_feasible.residual_norm.is_finite()
                && kithe_style_log_feasible.residual_norm < initial_norm,
            "log-feasible KiThe-style trace did not produce a finite residual decrease"
        );
        assert_finite_bounded_decrease(
            "KiThe-style",
            &kithe_style.x,
            kithe_style.residual_norm,
            initial_norm,
            &bounds,
        );
    }

    /// Diagnostic replay for the 0.4.15 -> current nonlinear-method change.
    ///
    /// This deliberately does not assert that every method must converge on
    /// the ill-conditioned physical system. It keeps the exact KiThe graph,
    /// seed, and bounds fixed, then separates the two likely explanations for
    /// the historical regression: raw inventory scaling versus the current
    /// MINPACK/LM acceptance policy. The final row is the historical
    /// KiThe-style identity-damped, feasible-backtracking policy used as an
    /// A/B reference, not a second production solver.
    #[test]
    #[ignore = "release diagnostic: exact TP-1907 raw/scaled policy replay"]
    fn tp1907_exact_replay_traces_current_and_historical_lm_policies() {
        let problem = canonical_fixture_problem();
        let initial = fixture_numeric_vector("let initial_log_moles: Vec<f64> = vec![");
        let bounds = fixture_bounds();
        let denominators = fixture_numeric_vector("let residual_row_scale_w: Vec<f64> = vec![");
        let scaled_problem = InventoryScaledProblem::from_denominators(&problem, &denominators);
        // KiThe's extensive normalization uses the physical input inventory
        // scale, not the external inventory multiplier used to construct it.
        let normalization_scale = fixture_scalar("let normalization_scale: f64 = ");
        let log_scale = normalization_scale.ln();
        let normalized_initial = initial.map(|value| value - log_scale);
        let normalized_bounds = Bounds::new(
            bounds
                .as_slice()
                .iter()
                .map(|(lower, upper)| (lower - log_scale, upper - log_scale))
                .collect(),
        )
        .expect("normalized TP-1907 bounds must be valid");
        let mut normalized_denominators = denominators.clone();
        // This is an intentionally limited diagnostic surrogate for the
        // extensive-recovery route: it shifts log-mole coordinates and scales
        // the balance rows by S. KiThe also rebuilds the normalized seed,
        // composition, and inventory request, so this is not claimed to be an
        // exact reproduction of its normalized problem.
        for denominator in normalized_denominators.iter_mut().skip(13) {
            *denominator *= normalization_scale;
        }
        let normalized_problem = InventoryScaledProblem::from_denominators_and_log_shift(
            &problem,
            &normalized_denominators,
            log_scale,
        );

        assert_eq!(problem.dimension(), 18);
        assert_eq!(initial.len(), problem.dimension());
        assert_eq!(bounds.len(), problem.dimension());
        assert_eq!(denominators.len(), problem.dimension());
        bounds
            .validate(&initial)
            .expect("exact TP-1907 seed must satisfy exact fixture bounds");

        let raw_initial = problem
            .residual(&initial)
            .expect("exact TP-1907 raw seed residual");
        let scaled_initial = scaled_problem
            .residual(&initial)
            .expect("exact TP-1907 scaled seed residual");
        println!(
            "[TP-1907 exact replay] seed raw_l2={:.6e} scaled_l2={:.6e}; same exact fixture for every route",
            raw_initial.norm(),
            scaled_initial.norm(),
        );
        println!(
            "method/policy       | scaling | residual_l2 | raw_inventory_max | termination/iterations | accepted/rejected | R/J/L"
        );
        println!(
            "---------------------------------------------------------------------------------------------------------------"
        );

        let routes = [
            (
                "LM-current",
                NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod::default()),
            ),
            (
                "LM-current-identity",
                NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod {
                    diag_scaling: false,
                    ..LevenbergMarquardtMethod::default()
                }),
            ),
            (
                "LM-Minpack-current",
                NonlinearSolverMethod::LevenbergMarquardtMinpack(
                    LevenbergMarquardtMinpack::default(),
                ),
            ),
            (
                "LM-Minpack-no-progress-stop",
                NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack {
                    // This is a diagnostic only: it suppresses MINPACK's
                    // method-specific progress exits so the generic engine
                    // can reveal whether the current policy stops too early.
                    ftol: 0.0,
                    xtol: 0.0,
                    gtol: 0.0,
                    ..LevenbergMarquardtMinpack::default()
                }),
            ),
            (
                "TrustRegionLM-current",
                NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default()),
            ),
            (
                "BacktrackingLM-current",
                NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
                    BacktrackingLevenbergMarquardtMethod::default(),
                ),
            ),
        ];
        for (name, method) in routes {
            let raw = method.clone().solve(&problem, initial.clone(), {
                let mut options = options(bounds.clone());
                options.tolerance = 1.0e-12;
                options
            });
            print_exact_replay_result(name, "raw", &raw, &problem, 0.0);
            let scaled = method.clone().solve(&scaled_problem, initial.clone(), {
                let mut options = options(bounds.clone());
                options.tolerance = 1.0e-12;
                options
            });
            print_exact_replay_result(name, "row-scaled", &scaled, &problem, 0.0);
            let normalized = method.solve(&normalized_problem, normalized_initial.clone(), {
                let mut options = options(normalized_bounds.clone());
                options.tolerance = 1.0e-12;
                options
            });
            print_exact_replay_result(
                name,
                "row-scaled+log-normalized",
                &normalized,
                &problem,
                log_scale,
            );
        }

        let historical_method = LevenbergMarquardtMinpack::default();
        let historical_raw = solve_historical_minpack_policy(
            &problem,
            initial.clone(),
            &bounds,
            &historical_method,
            &options(bounds.clone()),
        );
        print_historical_replay_result("raw", &historical_raw, &problem, 0.0);
        let historical_scaled = solve_historical_minpack_policy(
            &scaled_problem,
            initial.clone(),
            &bounds,
            &historical_method,
            &options(bounds.clone()),
        );
        print_historical_replay_result("row-scaled", &historical_scaled, &problem, 0.0);
        let historical_normalized = solve_historical_minpack_policy(
            &normalized_problem,
            normalized_initial.clone(),
            &normalized_bounds,
            &historical_method,
            &options(normalized_bounds.clone()),
        );
        print_historical_replay_result(
            "row-scaled+log-normalized",
            &historical_normalized,
            &problem,
            log_scale,
        );
        let historical_normalized = historical_normalized
            .as_ref()
            .expect("historical normalized replay should remain finite");
        assert_eq!(
            historical_normalized.termination,
            TerminationReason::Converged,
            "the old policy replay must expose its historical false-success path"
        );
        assert!(
            historical_normalized.residual_norm > 1.0e-12,
            "the historical Converged result must fail the current root tolerance"
        );
        assert!(
            raw_inventory_max_abs(
                &problem,
                &historical_normalized.x.map(|value| value + log_scale)
            ) > 1.0e-3,
            "the historical Converged result must fail the physical balance check"
        );

        let raw_trace = solve_kithe_style_lm_trace(&problem, initial.clone(), |candidate| {
            bounds.validate(candidate).is_ok()
        })
        .expect("historical TP-1907 raw policy replay should remain finite");
        let scaled_trace = solve_kithe_style_lm_trace(&scaled_problem, initial, |candidate| {
            bounds.validate(candidate).is_ok()
        })
        .expect("historical TP-1907 scaled policy replay should remain finite");
        let normalized_trace =
            solve_kithe_style_lm_trace(&normalized_problem, normalized_initial, |candidate| {
                normalized_bounds.validate(candidate).is_ok()
            })
            .expect("historical TP-1907 normalized policy replay should remain finite");
        for (name, trace, coordinate_shift) in [
            ("KiThe-style-current-coordinates", raw_trace, 0.0),
            ("KiThe-style-row-scaled", scaled_trace, 0.0),
            (
                "KiThe-style-row-scaled+log-normalized",
                normalized_trace,
                log_scale,
            ),
        ] {
            println!(
                "[TP-1907 exact replay] method={name} residual_l2={:.6e} raw_inventory_max={:.6e} trace/iterations={} accepted/rejected={}/{} final_lambda={:.6e} last_alpha={:.6e}",
                trace.residual_norm,
                raw_inventory_max_abs(&problem, &trace.x.map(|value| value + coordinate_shift)),
                trace.iterations,
                trace.accepted_steps,
                trace.rejected_steps,
                trace.final_lambda,
                trace.last_alpha,
            );
        }
    }
}

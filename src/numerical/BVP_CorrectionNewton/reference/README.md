# Frozen independent combustion references

- Generator: `scripts/bvp_correction_newton_stiff_combustion_reference.py`.
- Regenerate from the repository root with `python scripts/bvp_correction_newton_stiff_combustion_reference.py`.
- Python: 2.5.4 NumPy / 1.18.1 SciPy.
- Fine solve tolerance: `2e-09`; maximum adaptive nodes: `30000`.
- Runtime Rust tests consume only the checked-in CSV files and do not require Python, NumPy, or SciPy.
- These are verification models, not a validated detailed chemical mechanism.

## COMB-01

- `solve_bvp` SciPy tolerance: `2e-09`; independent coarse solve tolerance: `1e-6`.
- Adaptive mesh nodes (fine/coarse): `938` / `340`.
- Maximum difference on 2049 common points: `1.008882088982e-09`.
- Maximum SciPy interval RMS residual (fine): `1.990744900140e-09`.
- Maximum boundary-condition residual: `6.220613273358e-19`.
- Maximum independent first-integral error: `4.440892098501e-16`.
- Profile summary: theta_max=1.463600254680e+00, fuel_out=2.681998726598e-01, rate_max=5.312993399488e+00 at x=0.61914062.
- SciPy emitted `RuntimeWarning: overflow encountered in dot` during a trial collocation step; the final reference solve returned success and passed the residual, BC, invariant, and mesh checks.
- Frozen profile: `comb01_scipy.csv` (2049 uniform samples).

## COMB-02

- `solve_bvp` SciPy tolerance: `2e-09`; independent coarse solve tolerance: `1e-6`.
- Adaptive mesh nodes (fine/coarse): `3852` / `1793`.
- Maximum difference on 2049 common points: `3.322661257243e-09`.
- Maximum SciPy interval RMS residual (fine): `1.994805024430e-09`.
- Maximum boundary-condition residual: `1.092576342212e-20`.
- Maximum independent first-integral error: `5.329070518201e-15`.
- Profile summary: theta_max=2.911398565784e+00, B_max=5.516938168889e-01 at x=0.15039062, rate1_max=7.910079373904e+01 at x=0.10791016, rate2_max=1.047250548437e+01 at x=0.16210938.
- Frozen profile: `comb02_scipy.csv` (2049 uniform samples).

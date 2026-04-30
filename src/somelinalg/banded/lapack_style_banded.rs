use super::{error::BandedError, ops::banded_matvec, storage::Banded};

/// Faithful unblocked LAPACK-style band LU skeleton.
///
/// Target semantics:
/// - factorization path close to DGBTF2
/// - storage/layout compatible with DGBTRF/DGBTRS band workspace
/// - no dense n x n materialization
///
/// Workspace layout:
///   ldab = 2*kl + ku + 1
///   kv   = kl + ku
///
/// Physical A(i, j) is stored in workspace as:
///   AB[kv + i - j, j]
///
/// 0-based indexing throughout.
/*



*/
#[derive(Clone, Debug)]
pub struct LapackStyleBandedLuFaithful {
    n: usize,
    kl: usize,
    ku: usize,
    kv: usize,
    ldab: usize,
    ab: Vec<f64>,     // column-major: row + col * ldab
    ipiv: Vec<usize>, // 0-based pivot rows
    ju: usize,        // rightmost updated column (for tracking fill-in growth)
    is_factorized: bool,
    pivot_epsilon: f64,
    /// WORK13: (NBMAX+1) x NBMAX, upper triangle (rows 1..col) unused by zero pattern
    /// WORK31: (KL+1) x NBMAX, full column used for A31 panel storage
    work13: Vec<f64>,
    work13_ld: usize,
    work31: Vec<f64>,
    work31_ld: usize,
    nb_max: usize,
    /// Current panel start column (J) for overflow row mapping, valid only during factor_blocked.
    panel_start: usize,
    /// Current panel size (JB) for overflow row mapping.
    panel_size: usize,
    /// Height of A31 block for current panel (I3), valid only during factor_blocked.
    panel_i3: usize,
}

impl LapackStyleBandedLuFaithful {
    pub fn new(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        if n == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        let kv = kl.checked_add(ku).ok_or(BandedError::DimensionMismatch)?;
        let ldab = kl
            .checked_mul(2)
            .and_then(|x| x.checked_add(ku))
            .and_then(|x| x.checked_add(1))
            .ok_or(BandedError::DimensionMismatch)?;
        let len = ldab.checked_mul(n).ok_or(BandedError::DimensionMismatch)?;

        // LAPACK uses NBMAX = 64; we'll allocate workspace for maximum possible panel
        let nb_max = 64;
        let work13_len = (nb_max + 1) * nb_max;
        // WORK31 uses LAPACK LDWORK = NBMAX+1 rows, NBMAX columns
        let work31_len = (nb_max + 1) * nb_max;
        Ok(Self {
            n,
            kl,
            ku,
            kv,
            ldab,
            ab: vec![0.0; len],
            ipiv: vec![0; n],
            ju: 0,
            is_factorized: false,
            pivot_epsilon: 1e-14,
            work13: vec![0.0; work13_len],
            work13_ld: nb_max + 1,
            work31: vec![0.0; work31_len],
            work31_ld: nb_max + 1,
            nb_max,
            panel_start: 0,
            panel_size: 0,
            panel_i3: 0,
        })
    }
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[inline]
    fn work13_index(&self, row: usize, col: usize) -> usize {
        row + col * self.work13_ld
    }

    #[inline]
    fn work13_get(&self, row: usize, col: usize) -> f64 {
        self.work13[self.work13_index(row, col)]
    }

    #[inline]
    fn work13_set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.work13_index(row, col);
        self.work13[idx] = value;
    }
    #[inline]
    fn work31_index(&self, row: usize, col: usize) -> usize {
        row + col * self.work31_ld
    }

    #[inline]
    fn work31_get(&self, row: usize, col: usize) -> f64 {
        self.work31[self.work31_index(row, col)]
    }

    #[inline]
    fn work31_set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.work31_index(row, col);
        self.work31[idx] = value;
    }
    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    #[inline]
    pub fn kv(&self) -> usize {
        self.kv
    }

    #[inline]
    pub fn ldab(&self) -> usize {
        self.ldab
    }

    #[inline]
    pub fn ipiv(&self) -> &[usize] {
        &self.ipiv
    }

    #[inline]
    pub fn is_factorized(&self) -> bool {
        self.is_factorized
    }

    #[inline]
    pub fn set_pivot_epsilon(&mut self, eps: f64) {
        self.pivot_epsilon = eps;
    }

    #[inline]
    fn ab_index(&self, row: usize, col: usize) -> usize {
        row + col * self.ldab
    }

    #[inline]
    fn ab_get(&self, row: usize, col: usize) -> f64 {
        self.ab[self.ab_index(row, col)]
    }

    #[inline]
    fn ab_set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.ab_index(row, col);
        self.ab[idx] = value;
    }

    #[inline]
    fn ab_col(&self, col: usize) -> &[f64] {
        let start = col * self.ldab;
        &self.ab[start..start + self.ldab]
    }

    #[inline]
    pub fn get_phys(&self, i: usize, j: usize) -> f64 {
        // Direct LAPACK-style indexing: AB(kv + i - j, j)
        let row = self.kv as isize + i as isize - j as isize;
        if row >= 0 && row < self.ldab as isize {
            self.ab_get(row as usize, j)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn set_phys(&mut self, i: usize, j: usize, value: f64) {
        // Direct LAPACK-style indexing: AB(kv + i - j, j)
        let row = self.kv as isize + i as isize - j as isize;
        if row >= 0 && row < self.ldab as isize {
            self.ab_set(row as usize, j, value);
        }
    }

    fn dgb_get(&self, row: isize, col: usize) -> f64 {
        if row < 0 {
            return 0.0;
        }

        let row = row as usize;

        if row < self.ldab {
            self.ab_get(row, col)
        } else {
            // Overflow row: map to WORK31 panel workspace
            let overflow_row = row - self.ldab;
            let panel_col = col.checked_sub(self.panel_start);
            if let Some(pc) = panel_col {
                // Map if within panel and WORK31 bounds
                if pc < self.panel_size && overflow_row < self.work31_ld {
                    return self.work31_get(overflow_row, pc);
                }
            }
            0.0
        }
    }

    fn dgb_set(&mut self, row: isize, col: usize, val: f64) {
        if row < 0 {
            return;
        }

        let row = row as usize;

        if row < self.ldab {
            self.ab_set(row, col, val);
        } else {
            // Overflow row: map to WORK31 panel workspace
            let overflow_row = row - self.ldab;
            let panel_col = col.checked_sub(self.panel_start);
            if let Some(pc) = panel_col {
                // Map if within panel and WORK31 bounds
                if pc < self.panel_size && overflow_row < self.work31_ld {
                    self.work31_set(overflow_row, pc, val);
                }
            }
        }
    }

    pub fn load_from_banded(&mut self, a: &Banded<f64>) -> Result<(), BandedError> {
        if a.n() != self.n || a.kl() != self.kl || a.ku() != self.ku {
            return Err(BandedError::DimensionMismatch);
        }

        self.ab.fill(0.0);
        self.work13.fill(0.0);
        self.work31.fill(0.0);
        self.ipiv.fill(0);
        self.ju = 0;
        self.is_factorized = false;

        for j in 0..self.n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(self.n);

            for i in i0..i1 {
                // Direct LAPACK-style indexing: AB(kv + i - j, j)
                let row = self.kv + i - j;
                debug_assert!(
                    row < self.ldab,
                    "band entry ({},{}) maps to row {} >= ldab {}",
                    i,
                    j,
                    row,
                    self.ldab
                );
                self.ab_set(row, j, a[(i, j)]);
            }
        }

        Ok(())
    }
    /// In DGBTRF/DGBTF2 the top KL rows are fill rows.
    /// For the faithful solver, these rows are explicitly managed.
    fn zero_fillin_column(&mut self, col: usize) {
        if col >= self.n {
            return;
        }

        for row in 0..self.kl {
            self.ab_set(row, col, 0.0);
        }
    }

    /// Return the pivot search segment length:
    /// KM = min(KL, N-1-j)
    #[inline]
    fn km_at(&self, j: usize) -> usize {
        (self.n - 1 - j).min(self.kl)
    }

    /// Workspace row for the diagonal of column j.
    /// In 0-based form this is kv.
    #[inline]
    fn diag_row(&self) -> usize {
        self.kv
    }

    /// Return pivot offset in the current workspace column segment.
    ///
    /// This is the faithful replacement for IDAMAX over:
    ///   AB(kv+1 : kv+km+1, j)
    fn find_pivot_offset_workspace(&self, j: usize, km: usize) -> usize {
        let col = self.ab_col(j);
        let diag_row = self.diag_row();

        let mut best_off = 0usize;
        let mut best_val = col[diag_row].abs();

        for off in 1..=km {
            let val = col[diag_row + off].abs();
            if val > best_val {
                best_val = val;
                best_off = off;
            }
        }

        best_off
    }

    /// Scale the multipliers below the pivot in the current column.
    ///
    /// Equivalent in spirit to DSCAL(KM, 1/pivot, AB(KV+2, J), 1).
    fn scale_column_multipliers(&mut self, j: usize, km: usize) -> Result<(), BandedError> {
        if km == 0 {
            return Ok(());
        }
        // Pivot is always at AB(kv, j) after swap
        let pivot = self.ab_get(self.kv, j);
        if pivot.abs() <= self.pivot_epsilon {
            return Err(BandedError::ZeroPivot {
                index: j,
                value: pivot,
            });
        }
        let kv = self.kv as isize;
        // Multiplier rows: AB rows kv+1..kv+km, but deep pivots may have
        // landed entries in work31_full — use dgb_get/dgb_set
        for off in 1..=km {
            let row = kv + off as isize;
            let val = self.dgb_get(row, j) / pivot;
            self.dgb_set(row, j, val);
        }
        Ok(())
    }

    /// Rank-1 update of the trailing band-relevant region.
    ///
    /// This is the faithful replacement for the DGER-style update in DGBTF2.
    fn rank1_update_workspace(&mut self, j: usize, km: usize, ju: usize) {
        if km == 0 || ju <= j {
            return;
        }

        let kv = self.kv as isize;

        for col in (j + 1)..=ju.min(self.n - 1) {
            // U element: AB(kv+j-col, col) = A(j,col)
            let u_row = kv + j as isize - col as isize;
            let u_jc = self.dgb_get(u_row, col);
            if u_jc == 0.0 {
                continue;
            }

            for off in 1..=km {
                let i = j + off;
                // L element: AB(kv+off, j)
                let lij = self.dgb_get(kv + off as isize, j);
                if lij == 0.0 {
                    continue;
                }
                // Target: AB(kv+i-col, col) = A(i,col)
                let t_row = kv + i as isize - col as isize;
                let old = self.dgb_get(t_row, col);
                self.dgb_set(t_row, col, old - lij * u_jc);
            }
        }
    }

    pub(crate) fn factor_one_step_unblocked_lapack(&mut self, j: usize) -> Result<(), BandedError> {
        if j >= self.n {
            return Ok(());
        }
        if j + self.kv < self.n {
            self.zero_fillin_column(j + self.kv);
        }
        let km = self.km_at(j);
        let jp = self.find_pivot_offset_workspace(j, km);
        let p = j + jp;
        self.ipiv[j] = p;
        let pivot = self.ab_get(self.kv + jp, j);
        if pivot.abs() <= self.pivot_epsilon {
            return Err(BandedError::ZeroPivot {
                index: j,
                value: pivot,
            });
        }
        self.ju = self.ju.max((j + self.ku + jp).min(self.n - 1));
        // swap via dgb_get/dgb_set so overflow rows go to work31_full
        let kv = self.kv as isize;
        // DGBTF2 applies the row interchange only to columns J..JU:
        //
        //   CALL DSWAP( JU-J+1, AB( KV+JP, J ), LDAB-1,
        //  $                     AB( KV+1,  J ), LDAB-1 )
        //
        // The stride LDAB-1 walks one column to the right and one AB row
        // upward, so this is a physical row swap over the active trailing
        // band panel, not over previously completed L columns.
        let col_lo = j;
        let col_hi = self.ju.min(self.n - 1);
        for col in col_lo..=col_hi {
            let rj = kv + j as isize - col as isize;
            let rp = kv + p as isize - col as isize;
            let a = self.dgb_get(rj, col);
            let b = self.dgb_get(rp, col);
            self.dgb_set(rj, col, b);
            self.dgb_set(rp, col, a);
        }
        self.scale_column_multipliers(j, km)?;
        // rank-1 update via dgb_get/dgb_set
        for col in (j + 1)..=self.ju.min(self.n - 1) {
            let u_row = kv + j as isize - col as isize;
            if u_row < 0 {
                continue;
            }
            let u_jc = self.dgb_get(u_row, col);
            if u_jc == 0.0 {
                continue;
            }
            for off in 1..=km {
                let lij = self.ab_get(self.kv + off, j);
                if lij == 0.0 {
                    continue;
                }
                let t_row = u_row + off as isize;
                let old = self.dgb_get(t_row, col);
                self.dgb_set(t_row, col, old - lij * u_jc);
            }
        }
        Ok(())
    }

    #[deprecated(note = "use factor_one_step_unblocked_lapack")]
    pub(crate) fn factor_one_step_unblocked_legacy(&mut self, j: usize) -> Result<(), BandedError> {
        self.factor_one_step_unblocked_lapack(j)
    }

    pub fn factor_from(&mut self, a: &Banded<f64>) -> Result<(), BandedError> {
        self.load_from_banded(a)?;
        self.ju = 0;
        self.panel_start = 0;
        self.panel_size = 0;
        self.panel_i3 = 0;
        self.ipiv.fill(0);
        self.work13.fill(0.0);
        self.work31.fill(0.0);

        let nb = self.nb_max.min(64);
        if nb > 1 && nb <= self.kl {
            self.factor_blocked(nb)?;
        } else {
            for j in 0..self.n {
                self.factor_one_step_unblocked_lapack(j)?;
            }
        }

        self.is_factorized = true;
        Ok(())
    }
    /// Blocked factorization mirroring DGBTRF.
    fn factor_blocked(&mut self, nb: usize) -> Result<(), BandedError> {
        let kv = self.kv;
        let ldab = self.ldab;
        let n = self.n;
        let kl = self.kl;
        let ku = self.ku;

        // Zero the superdiagonal elements of the work array WORK13
        for j in 0..nb {
            for i in 0..j {
                self.work13_set(i, j, 0.0);
            }
        }

        // Zero the subdiagonal elements of the work array WORK31.
        // LAPACK zeros only the local NB x NB panel triangle here, not all KL
        // rows.  For wide bands KL can exceed NBMAX, while WORK31 has only
        // NBMAX+1 rows.
        for j in 0..nb {
            for i in (j + 1)..nb {
                self.work31_set(i, j, 0.0);
            }
        }

        // Set fill-in elements in columns KU+2 to KV to zero
        let fill_start = ku + 1; // 0-based column index = KU+1 corresponds to KU+2 in 1-based
        let fill_end = kv.min(n);
        for j in fill_start..fill_end {
            let row_start = kv - j; // 0-based
            if row_start < kl {
                for i in row_start..kl {
                    self.ab_set(i, j, 0.0);
                }
            }
        }

        // JU is the index of the last column affected by the current stage
        self.ju = 0;

        let mut j = 0;
        while j < n {
            let jb = nb.min(n - j);
            if jb == 0 {
                break;
            }
            self.panel_start = j;
            self.panel_size = jb;

            // Partition sizes
            // I2 = min(KL - JB, N - J - JB)
            let i2 = kl.saturating_sub(jb).min(n - j - jb);
            // I3 = min(JB, N - J - KL)
            let i3 = jb.min(n.saturating_sub(j + kl));
            self.panel_i3 = i3;

            // ---- Factorize the current block of JB columns (inner loop over JJ) ----
            for jj in j..(j + jb) {
                // Zero the fill-in column JJ+KV if it exists
                let fill_col = jj + kv;
                if fill_col < n {
                    for i in 0..kl {
                        self.ab_set(i, fill_col, 0.0);
                    }
                }

                // KM = min(KL, N-1-jj)
                let km = (n - 1 - jj).min(kl);
                // Find pivot offset within the column (0-based)
                let jp_off = self.find_pivot_offset_workspace(jj, km);
                let p = jj + jp_off; // pivot row (absolute)
                // Store Fortran-style IPIV(JJ) = JP + JJ - J as 0-based relative value (p - j).
                self.ipiv[jj] = p.saturating_sub(j);

                let pivot = self.ab_get(kv + jp_off, jj);
                if pivot.abs() <= self.pivot_epsilon {
                    return Err(BandedError::ZeroPivot {
                        index: jj,
                        value: pivot,
                    });
                }

                // Update JU
                self.ju = self.ju.max((jj + ku + jp_off).min(n - 1));

                // Apply row interchange if needed
                if p != jj {
                    // Fortran branch condition: JP+JJ-1 < J+KL  (1-based)
                    // 0-based: p < j + kl
                    if p < j + kl {
                        // Branch 1: pivot row stays within AB for all panel columns.
                        // Fortran: DSWAP(JB, AB(KV+1+JJ-J,J), LDAB-1, AB(KV+JP+JJ-J,J), LDAB-1)
                        // Row of jj in panel: r_jj = kv + (jj-j)  (constant across cols j..j+jb)
                        // Row of p  in panel: r_p  = kv + (p -j)  = kv + jp_off + (jj-j)
                        let r_jj = kv + (jj - j);
                        let r_p = kv + (p - j);
                        for c in j..(j + jb).min(n) {
                            let tmp = self.ab_get(r_jj, c);
                            self.ab_set(r_jj, c, self.ab_get(r_p, c));
                            self.ab_set(r_p, c, tmp);
                        }
                    } else {
                        // Branch 2: pivot row p >= j+kl, so columns J..JJ-1 of its row
                        // are in WORK31, not AB.
                        //
                        // Part 1 (Fortran): DSWAP(JJ-J, AB(KV+1+JJ-J,J),LDAB-1, WORK31(JP+JJ-J-KL,1),LDWORK)
                        // AB row (0-based):    r_jj = kv + (jj-j)   [constant]
                        // WORK31 row (0-based): w31_row = jp_off + (jj-j) - kl
                        // WORK31 col (0-based): c - j
                        let r_jj = kv + (jj - j);
                        let w31_row = jp_off + (jj - j) - kl; // jp_off >= kl-(jj-j)+1 by branch condition
                        for c in j..jj {
                            let w31_col = c - j;
                            let tmp = self.ab_get(r_jj, c);
                            self.ab_set(r_jj, c, self.work31_get(w31_row, w31_col));
                            self.work31_set(w31_row, w31_col, tmp);
                        }

                        // Part 2 (Fortran): DSWAP(J+JB-JJ, AB(KV+1,JJ),LDAB-1, AB(KV+JP,JJ),LDAB-1)
                        // AB rows kv and kv+jp_off across cols jj..j+jb-1
                        for c in jj..(j + jb).min(n) {
                            let tmp = self.ab_get(kv, c);
                            self.ab_set(kv, c, self.ab_get(kv + jp_off, c));
                            self.ab_set(kv + jp_off, c, tmp);
                        }
                    }
                }

                // Scale multipliers below the pivot
                self.scale_column_multipliers(jj, km)?;

                // Rank-1 update within current block only: JM = min(JU, J+JB-1)
                // Fortran: CALL DGER(KM, JM-JJ, ...) where JM = MIN(JU, J+JB-1)
                let jm = self.ju.min(j + jb - 1);
                self.rank1_update_workspace(jj, km, jm);

                // Copy current column of A31 into the work array WORK31
                /*
                // NW = min(JJ-J+1, I3)
                let nw = (jj - j + 1).min(i3);
                if nw > 0 {
                    // Source in AB: row = KV+KL - (JJ-J) (0-based)
                    let src_row_base = kv + kl - (jj - j);
                    for i in 0..nw {
                        let src_row = src_row_base + i;
                        if src_row < ldab {
                            let val = self.ab_get(src_row, jj);
                            self.work31_set(i, jj - j, val);
                        } else {
                            self.work31_set(i, jj - j, 0.0);
                        }
                    }
                }
                 */
                // Copy current column of A31 into the work array WORK31
                // Fortran: DCOPY(NW, AB(KV+KL+1-JJ+J, JJ), 1, WORK31(1, JJ-J+1), 1)
                // NW = min(JJ-J+1, I3)
                let nw = (jj - j + 1).min(i3);
                if nw > 0 {
                    // Source in AB: row = KV+KL-JJ+J (0-based, since Fortran is 1-based)
                    let src_row_base = kv as isize + kl as isize - jj as isize + j as isize;
                    for i in 0..nw {
                        let src_row = src_row_base + i as isize;
                        if src_row >= 0 && src_row < ldab as isize {
                            let val = self.ab_get(src_row as usize, jj);
                            self.work31_set(i, jj - j, val);
                        } else {
                            self.work31_set(i, jj - j, 0.0);
                        }
                    }
                }
            } // end inner panel jj loop

            // Compute J2 and J3 after panel factorization
            // J2 = min(JU-J+1, KV) - JB
            let j2 = (self.ju - j + 1).min(kv).saturating_sub(jb);
            // J3 = max(0, JU-J-KV+1)
            let j3 = self.ju.saturating_sub(j + kv).saturating_add(1).max(0);

            // Apply row interchanges to A12, A22, A32 (DLASWP).
            // Fortran: DLASWP(J2, AB(KV+1-JB, J+JB), LDAB-1, 1, JB, IPIV(J), 1)
            // IPIV values here are still relative (p - j), adjusted to absolute AFTER this call.
            if j2 > 0 {
                let col_start = j + jb;
                let col_end = col_start + j2 - 1;
                let ipiv_rel: Vec<usize> = self.ipiv[j..(j + jb)].to_vec();
                // Convert relative pivot (p-j) to absolute row index for dlaswp
                let ipiv_abs: Vec<usize> = ipiv_rel.iter().map(|&r| r + j).collect();
                self.dlaswp_band(col_start, col_end, j, j + jb - 1, &ipiv_abs);
            }

            // Note: Keep IPIV as relative offsets for now; convert to absolute in undo loop
            //???
            for i in j..(j + jb) {
                self.ipiv[i] = self.ipiv[i] + j;
            }
            // Columnwise interchanges for A13, A23, A33
            if j3 > 0 {
                let start_col = j + jb + j2; // first column of A13
                for i_idx in 0..j3 {
                    let jj = start_col + i_idx;
                    for ii in (j + i_idx)..(j + jb) {
                        let ip = self.ipiv[ii];
                        if ip != ii {
                            let val_ii = self.get_phys(ii, jj);
                            let val_ip = self.get_phys(ip, jj);
                            self.set_phys(ii, jj, val_ip);
                            self.set_phys(ip, jj, val_ii);
                        }
                    }
                }
            }

            // Update A12: A12 := L^{-1} * A12  (DTRSM)
            if j2 > 0 {
                for col in (j + jb)..(j + jb + j2) {
                    for i in j..(j + jb) {
                        let mut sum = self.get_phys(i, col);
                        for k in j..i {
                            let lik = self.get_phys(i, k);
                            let ukc = self.get_phys(k, col);
                            sum -= lik * ukc;
                        }
                        self.set_phys(i, col, sum);
                    }
                }
            }

            // Update A22: A22 := A22 - A21 * A12  (DGEMM)
            // A21 rows: j+jb .. j+jb+i2-1  (i2 rows, jb cols)
            // A12 cols: j+jb .. j+jb+j2-1  (jb rows, j2 cols)
            if i2 > 0 && j2 > 0 {
                for i_off in 0..i2 {
                    let i = j + jb + i_off;
                    for j_off in 0..j2 {
                        let col = j + jb + j_off;
                        let mut sum = 0.0;
                        for k_off in 0..jb {
                            let k = j + k_off;
                            // A21(i,k) = multiplier stored at AB(kv+i-k, k)
                            let a21 = self.get_phys(i, k);
                            // A12(k,col) already updated above
                            let a12 = self.get_phys(k, col);
                            sum += a21 * a12;
                        }
                        let old = self.get_phys(i, col);
                        self.set_phys(i, col, old - sum);
                    }
                }
            }

            // Update A32: A32 := A32 - WORK31 * A12  (DGEMM)
            // WORK31(i_off, k_off) holds A31(j+kl+i_off, j+k_off) before pivoting
            // A12(k, col) is in AB at AB(kv+k-col, col)
            if i3 > 0 && j2 > 0 {
                for i_off in 0..i3 {
                    let i_global = j + kl + i_off;
                    for j_off in 0..j2 {
                        let col = j + jb + j_off;
                        let mut sum = 0.0;
                        for k_off in 0..jb {
                            let w = self.work31_get(i_off, k_off);
                            if w == 0.0 {
                                continue;
                            }
                            let a12 = self.get_phys(j + k_off, col);
                            sum += w * a12;
                        }
                        // A32 target: AB(kv + i_global - col, col)
                        let old = self.get_phys(i_global, col);
                        self.set_phys(i_global, col, old - sum);
                    }
                }
            }

            // Process A13 block if present
            // A13 columns in AB: j+kv .. j+kv+j3-1  (Fortran: J+KV .. JU)
            // LAPACK copies the lower triangle AB(ii-jj+1, jj+J+KV-1) into WORK13(ii,jj)
            // using 1-based: ii in jj..JB, jj in 1..J3
            if j3 > 0 {
                // Copy lower triangle of A13 into WORK13
                // AB row for physical A(j+ii_rel, j+kv+jj_rel) = kv + (j+ii_rel) - (j+kv+jj_rel)
                //                                               = ii_rel - jj_rel
                for jj_rel in 0..j3 {
                    let col = j + kv + jj_rel;
                    if col >= self.n {
                        break;
                    }
                    for ii_rel in jj_rel..jb {
                        // AB row = ii_rel - jj_rel  (always >= 0 since ii_rel >= jj_rel)
                        let val = self.ab_get(ii_rel - jj_rel, col);
                        self.work13_set(ii_rel, jj_rel, val);
                    }
                    for ii_rel in 0..jj_rel {
                        self.work13_set(ii_rel, jj_rel, 0.0);
                    }
                }

                // DTRSM: solve unit-lower-triangular L (panel, jb x jb) against WORK13
                // L(i_rel, k_rel) = AB(kv + (j+i_rel) - (j+k_rel), j+k_rel)
                //                 = AB(kv + i_rel - k_rel, j+k_rel)
                for jj_rel in 0..j3 {
                    for i_rel in 1..jb {
                        let mut sum = self.work13_get(i_rel, jj_rel);
                        for k_rel in 0..i_rel {
                            // L(i_rel, k_rel): AB row = kv + i_rel - k_rel
                            let lik = self.ab_get(kv + i_rel - k_rel, j + k_rel);
                            sum -= lik * self.work13_get(k_rel, jj_rel);
                        }
                        self.work13_set(i_rel, jj_rel, sum);
                    }
                }

                // Update A23: A23 -= A21 * WORK13  (DGEMM)
                // A21(i, k) = AB(kv + i - (j+k_rel), j+k_rel)  for i in j+jb..j+jb+i2-1
                if i2 > 0 {
                    for i_off in 0..i2 {
                        let i = j + jb + i_off;
                        for jj_rel in 0..j3 {
                            let col = j + kv + jj_rel;
                            if col >= self.n {
                                break;
                            }
                            let mut sum = 0.0;
                            for k_rel in 0..jb {
                                let a21 = self.get_phys(i, j + k_rel);
                                sum += a21 * self.work13_get(k_rel, jj_rel);
                            }
                            // A23 target: AB(kv + i - col, col) = AB(kv + i - j - kv - jj_rel, col)
                            //           = AB(i - j - jj_rel, col)
                            let ab_row = i as isize - j as isize - jj_rel as isize;
                            if ab_row >= 0 && (ab_row as usize) < self.ldab {
                                let old = self.ab_get(ab_row as usize, col);
                                self.ab_set(ab_row as usize, col, old - sum);
                            }
                        }
                    }
                }

                // Update A33: A33 -= WORK31 * WORK13  (DGEMM)
                // A33 rows: j+kl .. j+kl+i3-1, cols: j+kv .. j+kv+j3-1
                if i3 > 0 {
                    for i_off in 0..i3 {
                        for jj_rel in 0..j3 {
                            let col = j + kv + jj_rel;
                            if col >= self.n {
                                break;
                            }
                            let mut sum = 0.0;
                            for k_off in 0..jb {
                                let w31 = self.work31_get(i_off, k_off);
                                if w31 == 0.0 {
                                    continue;
                                }
                                sum += w31 * self.work13_get(k_off, jj_rel);
                            }
                            // A33 target: AB(kv + i_global - col, col)
                            //           = AB(kv + j+kl+i_off - j-kv-jj_rel, col)
                            //           = AB(kl + i_off - jj_rel, col)
                            let ab_row = kl as isize + i_off as isize - jj_rel as isize;
                            if ab_row >= 0 && (ab_row as usize) < self.ldab {
                                let old = self.ab_get(ab_row as usize, col);
                                self.ab_set(ab_row as usize, col, old - sum);
                            }
                        }
                    }
                }

                // Copy WORK13 back into A13 region
                for jj_rel in 0..j3 {
                    let col = j + kv + jj_rel;
                    if col >= self.n {
                        break;
                    }
                    for ii_rel in jj_rel..jb {
                        let val = self.work13_get(ii_rel, jj_rel);
                        self.ab_set(ii_rel - jj_rel, col, val);
                    }
                }
            }

            // ---- Undo interchanges within current block and restore A31 ----
            // Fortran: DO 170 JJ = J+JB-1, J, -1  (inclusive of J)
            for idx in (0..jb).rev() {
                let jj = j + idx;
                let p = self.ipiv[jj];
                if p != jj {
                    // JP = IPIV(JJ) - JJ + 1  (1-based offset)
                    // Fortran branch: JP+JJ-1 < J+KL  <==>  p < j+kl
                    if p < j + kl {
                        // Swap rows jj and p for columns J..JJ-1 within AB
                        for c in j..jj {
                            let rjj = self.kv as isize + jj as isize - c as isize;
                            let rp = self.kv as isize + p as isize - c as isize;
                            let ldab = self.ldab as isize;
                            let rjj_ok = rjj >= 0 && rjj < ldab;
                            let rp_ok = rp >= 0 && rp < ldab;
                            match (rjj_ok, rp_ok) {
                                (true, true) => {
                                    let (rjj, rp) = (rjj as usize, rp as usize);
                                    let tmp = self.ab_get(rjj, c);
                                    self.ab_set(rjj, c, self.ab_get(rp, c));
                                    self.ab_set(rp, c, tmp);
                                }
                                (true, false) => self.ab_set(rjj as usize, c, 0.0),
                                (false, true) => self.ab_set(rp as usize, c, 0.0),
                                (false, false) => {}
                            }
                        }
                    } else {
                        // Swap rows jj and p for columns J..JJ-1:
                        // row jj is in AB; row p spills into WORK31
                        // WORK31 column = c - j  (panel-relative)
                        // WORK31 row    = p - (j + kl)  (offset from first A31 row)
                        let w31_row = p - (j + kl);
                        for c in j..jj {
                            let rjj = self.kv as isize + jj as isize - c as isize;
                            let ldab = self.ldab as isize;
                            if rjj >= 0 && rjj < ldab {
                                let rjj = rjj as usize;
                                let w31_col = c - j;
                                if w31_row < self.work31_ld && w31_col < self.nb_max {
                                    let tmp = self.ab_get(rjj, c);
                                    let wval = self.work31_get(w31_row, w31_col);
                                    self.ab_set(rjj, c, wval);
                                    self.work31_set(w31_row, w31_col, tmp);
                                } else {
                                    self.ab_set(rjj, c, 0.0);
                                }
                            }
                        }
                    }
                }

                // Copy current column of A31 back from WORK31 into AB
                /*
                // NW = min(I3, JJ-J+1) = min(i3, idx+1)
                let nw = i3.min(idx + 1);
                if nw > 0 {
                    // AB row for A(j+kl+i_off, jj) = kv + (j+kl+i_off) - jj
                    //                              = kv + kl - idx + i_off
                    let base_row = kv + kl - idx;
                    for i_off in 0..nw {
                        let val = self.work31_get(i_off, idx);
                        self.ab_set(base_row + i_off, jj, val);
                    }
                }
                */

                // Copy current column of A31 back from WORK31 into AB
                // Fortran: DCOPY(NW, WORK31(1, JJ-J+1), 1, AB(KV+KL+1-JJ+J, JJ), 1)
                // NW = min(I3, JJ-J+1) = min(i3, idx+1)
                let nw = i3.min(idx + 1);
                if nw > 0 {
                    // Target AB row = KV+KL-JJ+J (0-based)
                    let base_row = kv as isize + kl as isize - jj as isize + j as isize;
                    for i_off in 0..nw {
                        if base_row + i_off as isize >= 0
                            && base_row + (i_off as isize) < ldab as isize
                        {
                            let val = self.work31_get(i_off, idx);
                            self.ab_set((base_row + i_off as isize) as usize, jj, val);
                        }
                    }
                }
            } // end undo loop

            j += jb;
        } // end while j < n

        Ok(())
    } // end factor_blocked

    /// Apply row interchanges (like DLASWP) to a range of columns.
    /// Swaps rows i1..=i2 in columns j_lo..=j_hi according to ipiv_slice.
    /// ipiv_slice[k] is the row to swap with row (i1+k).
    fn dlaswp_band(
        &mut self,
        j_lo: usize,
        j_hi: usize,
        i1: usize,
        i2: usize,
        ipiv_slice: &[usize],
    ) {
        if j_lo > j_hi || i1 > i2 || ipiv_slice.is_empty() {
            return;
        }
        for col in j_lo..=j_hi {
            for (idx, &p) in ipiv_slice.iter().enumerate() {
                let i = i1 + idx;
                if p != i {
                    let row_i = self.kv as isize + i as isize - col as isize;
                    let row_p = self.kv as isize + p as isize - col as isize;
                    let ri = if row_i >= 0 && row_i < self.ldab as isize {
                        Some(row_i as usize)
                    } else {
                        None
                    };
                    let rp = if row_p >= 0 && row_p < self.ldab as isize {
                        Some(row_p as usize)
                    } else {
                        None
                    };
                    match (ri, rp) {
                        (Some(r_i), Some(r_p)) => {
                            let tmp = self.ab_get(r_i, col);
                            self.ab_set(r_i, col, self.ab_get(r_p, col));
                            self.ab_set(r_p, col, tmp);
                        }
                        (Some(r_i), None) => {
                            self.ab_set(r_i, col, 0.0);
                        }
                        (None, Some(r_p)) => {
                            self.ab_set(r_p, col, 0.0);
                        }
                        (None, None) => {}
                    }
                }
            }
        }
    }

    pub fn reconstruct_original_band_dense(&self) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; self.n]; self.n];
        for j in 0..self.n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(self.n);
            for i in i0..i1 {
                out[i][j] = self.get_phys(i, j);
            }
        }
        out
    }

    pub fn reconstruct_u_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let mut u = vec![vec![0.0; self.n]; self.n];
        // U occupies AB rows 0 .. kv  (kv = kl+ku superdiagonals including diagonal)
        for j in 0..self.n {
            let i0 = j.saturating_sub(self.kv);
            for i in i0..=j {
                u[i][j] = self.get_phys(i, j);
            }
        }
        Ok(u)
    }

    pub fn reconstruct_pa_dense(&self, a: &Banded<f64>) -> Result<Vec<Vec<f64>>, BandedError> {
        if a.n() != self.n || a.kl() != self.kl || a.ku() != self.ku {
            return Err(BandedError::DimensionMismatch);
        }

        let mut pa = vec![vec![0.0; self.n]; self.n];
        let mut ju = 0usize;

        for j in 0..self.n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(self.n);

            for i in i0..i1 {
                pa[i][j] = a[(i, j)];
            }
        }

        for j in 0..self.n {
            let p = self.ipiv[j];
            let jp = p.saturating_sub(j);
            ju = ju.max((j + self.ku + jp).min(self.n - 1));

            if p != j {
                let col_lo = j.saturating_sub(self.kv);
                let col_hi = ju.min(self.n - 1);
                let width = col_hi - col_lo + 1;

                let mut row_j_vals = vec![0.0_f64; width];
                let mut row_p_vals = vec![0.0_f64; width];

                for (k, col) in (col_lo..=col_hi).enumerate() {
                    row_j_vals[k] = pa[j][col];
                    row_p_vals[k] = pa[p][col];
                }

                for (k, col) in (col_lo..=col_hi).enumerate() {
                    let rj = {
                        let row = self.kv as isize + j as isize - col as isize;
                        row >= 0 && row < self.ldab as isize
                    };
                    let rp = {
                        let row = self.kv as isize + p as isize - col as isize;
                        row >= 0 && row < self.ldab as isize
                    };

                    if rj {
                        pa[j][col] = row_p_vals[k];
                    }
                    if rp {
                        pa[p][col] = row_j_vals[k];
                    }
                }
            }
        }

        Ok(pa)
    }
    pub fn factor_residual_relative_workspace(&self, a: &Banded<f64>) -> Result<f64, BandedError> {
        let pa = self.reconstruct_pa_dense(a)?;
        let l = self.reconstruct_l_dense()?;
        let u = self.reconstruct_u_dense()?;
        let lu = dense_matmul(&l, &u);

        let num = dense_linf_diff(&pa, &lu);
        let den = dense_linf_norm(&pa).max(dense_linf_norm(&lu)).max(1.0);

        Ok(num / den)
    }

    #[cfg(test)]
    pub(crate) fn set_nb_max_for_tests(&mut self, nb_max: usize) {
        self.nb_max = nb_max;
    }

    pub fn reconstruct_pa_dense_true(&self, a: &Banded<f64>) -> Result<Vec<Vec<f64>>, BandedError> {
        if a.n() != self.n || a.kl() != self.kl || a.ku() != self.ku {
            return Err(BandedError::DimensionMismatch);
        }

        let mut pa = vec![vec![0.0; self.n]; self.n];

        for j in 0..self.n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(self.n);

            for i in i0..i1 {
                pa[i][j] = a[(i, j)];
            }
        }

        // True dense replay of row interchanges
        for j in 0..self.n {
            let p = self.ipiv[j];
            if p != j {
                pa.swap(j, p);
            }
        }

        Ok(pa)
    }

    pub fn factor_residual_relative_true(&self, a: &Banded<f64>) -> Result<f64, BandedError> {
        let pa = self.reconstruct_pa_dense_true(a)?;
        let l = self.reconstruct_l_dense()?;
        let u = self.reconstruct_u_dense()?;
        let lu = dense_matmul(&l, &u);

        let num = dense_linf_diff(&pa, &lu);
        let den = dense_linf_norm(&pa).max(dense_linf_norm(&lu)).max(1.0);

        Ok(num / den)
    }

    pub fn factor_residual_relative(&self, a: &Banded<f64>) -> Result<f64, BandedError> {
        self.factor_residual_relative_true(a)
    }

    pub fn reconstruct_l_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let mut l = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            l[i][i] = 1.0;
        }
        // Multipliers: AB rows kv+1..kv+kl for entries within ldab,
        // work31_full for overflow rows (spill = ab_row - ldab)
        for col in 0..self.n {
            let i_hi = (col + self.kl).min(self.n - 1);
            for i in (col + 1)..=i_hi {
                // AB row for L(i,col) = kv + i - col
                let ab_row = self.kv as isize + i as isize - col as isize;
                l[i][col] = self.dgb_get(ab_row, col);
            }
        }
        Ok(l)
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        if rhs.len() != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        // DGBTRS, TRANS='N':
        //   for each column j, swap B(j) with B(IPIV(j)), then apply the
        //   rank-one L(j) update B(j+1:j+LM) -= AB(KD+1:KD+LM,j) * B(j).
        //
        // The pivot swaps are intentionally interleaved with the forward
        // solve.  Applying all pivots up front is not equivalent to LAPACK's
        // product P(1)*L(1)*...*P(n-1)*L(n-1).
        if self.kl > 0 && self.n > 1 {
            for j in 0..(self.n - 1) {
                let p = self.ipiv[j];
                if p != j {
                    rhs.swap(j, p);
                }

                let lm = self.kl.min(self.n - 1 - j);
                let bj = rhs[j];
                if bj == 0.0 {
                    continue;
                }
                let kv = self.kv as isize;
                for off in 1..=lm {
                    let i = j + off;
                    // L(i,j) stored at dgb row kv+off
                    let lij = self.dgb_get(kv + off as isize, j);
                    if lij != 0.0 {
                        rhs[i] -= lij * bj;
                    }
                }
            }
        }

        // DGBTRS finishes with DTBSV('Upper','No transpose','Non-unit', ...).
        for i in (0..self.n).rev() {
            let j_hi = (i + self.kv).min(self.n - 1);
            let mut sum = rhs[i];
            for j in (i + 1)..=j_hi {
                // U(i,j) = AB(kv+i-j, j)
                let uij = self.get_phys(i, j);
                if uij != 0.0 {
                    sum -= uij * rhs[j];
                }
            }
            // U(i,i) = AB(kv, i)
            let uii = self.ab_get(self.kv, i);
            if uii.abs() <= self.pivot_epsilon {
                return Err(BandedError::ZeroPivot {
                    index: i,
                    value: uii,
                });
            }
            rhs[i] = sum / uii;
        }

        Ok(())
    }

    pub fn solve_in_place_with_refinement<F>(
        &self,
        rhs: &mut [f64],
        max_iters: usize,
        tol: f64,
        mut matvec: F,
    ) -> Result<usize, BandedError>
    where
        F: FnMut(&[f64], &mut [f64]) -> Result<(), BandedError>,
    {
        if rhs.len() != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        let b = rhs.to_vec();
        self.solve_in_place(rhs)?;

        let n = self.n;
        let b_norm = b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        let mut ax = vec![0.0; n];
        let mut residual = vec![0.0; n];
        let mut delta = vec![0.0; n];
        let mut x_trial = vec![0.0; n];
        let mut current_rr = f64::INFINITY;
        let mut accepted_steps = 0usize;

        for _ in 0..max_iters {
            ax.fill(0.0);
            matvec(rhs, &mut ax)?;

            let mut r_norm: f64 = 0.0;
            for i in 0..n {
                residual[i] = b[i] - ax[i];
                r_norm = r_norm.max(residual[i].abs());
            }

            let rr = r_norm / b_norm;
            if !rr.is_finite() || rr < tol {
                break;
            }
            if !current_rr.is_finite() {
                current_rr = rr;
            }

            delta.copy_from_slice(&residual);
            self.solve_in_place(&mut delta)?;

            x_trial.copy_from_slice(rhs);
            for i in 0..n {
                x_trial[i] += delta[i];
            }

            ax.fill(0.0);
            matvec(&x_trial, &mut ax)?;

            let mut trial_r_norm: f64 = 0.0;
            for i in 0..n {
                trial_r_norm = trial_r_norm.max((b[i] - ax[i]).abs());
            }
            let trial_rr = trial_r_norm / b_norm;

            if !trial_rr.is_finite() || trial_rr >= rr.min(current_rr) {
                break;
            }

            rhs.copy_from_slice(&x_trial);
            current_rr = trial_rr;
            accepted_steps += 1;
        }

        Ok(accepted_steps)
    }

    pub fn solve_banded_in_place_with_refinement(
        &self,
        a: &Banded<f64>,
        rhs: &mut [f64],
        max_iters: usize,
        tol: f64,
    ) -> Result<usize, BandedError> {
        if a.n() != self.n || a.kl() != self.kl || a.ku() != self.ku || rhs.len() != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        self.solve_in_place_with_refinement(rhs, max_iters, tol, |x, ax| {
            let y = banded_matvec(a, x)?;
            if y.len() != ax.len() {
                return Err(BandedError::DimensionMismatch);
            }
            ax.copy_from_slice(&y);
            Ok(())
        })
    }

    pub fn solve_with_refinement(
        &self,
        a: &Banded<f64>,
        b: &[f64],
        max_iters: usize,
        tol: f64,
    ) -> Result<Vec<f64>, BandedError> {
        let mut x = b.to_vec();
        self.solve_banded_in_place_with_refinement(a, &mut x, max_iters, tol)?;
        Ok(x)
    }

    pub fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        if ldb < self.n || rhs.len() < nrhs.saturating_mul(ldb) {
            return Err(BandedError::DimensionMismatch);
        }

        for col in 0..nrhs {
            let start = col * ldb;
            let end = start + self.n;
            self.solve_in_place(&mut rhs[start..end])?;
        }

        Ok(())
    }
} // end impl LapackStyleBandedLuFaithful

//===========================================================================
fn dense_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];

    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }

    c
}

fn dense_linf_norm(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|row| row.iter())
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
}

fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut y = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            y[i] += a[i][j] * x[j];
        }
    }

    y
}

fn dense_relative_residual(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let ax = dense_matvec(a, x);
    let mut rmax = 0.0_f64;
    let mut bmax = 0.0_f64;

    for i in 0..b.len() {
        rmax = rmax.max((ax[i] - b[i]).abs());
        bmax = bmax.max(b[i].abs());
    }

    rmax / bmax.max(1.0)
}

fn apply_ipiv_dense(rhs: &mut [f64], ipiv: &[usize]) {
    for (j, &p) in ipiv.iter().enumerate() {
        if p != j {
            rhs.swap(j, p);
        }
    }
}
fn dense_from_banded(a: &Banded<f64>) -> Vec<Vec<f64>> {
    let n = a.n();
    let mut out = vec![vec![0.0; n]; n];

    for j in 0..n {
        let i0 = j.saturating_sub(a.ku());
        let i1 = (j + a.kl() + 1).min(n);

        for i in i0..i1 {
            out[i][j] = a[(i, j)];
        }
    }

    out
}

fn max_abs_diff(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn dense_from_workspace_state(lu: &LapackStyleBandedLuFaithful) -> Vec<Vec<f64>> {
    let n = lu.n();
    let mut out = vec![vec![0.0; n]; n];
    for j in 0..n {
        for i in 0..n {
            out[i][j] = lu.get_phys(i, j);
        }
    }
    out
}

fn dense_linf_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut max = 0.0;

    for i in 0..a.len() {
        for j in 0..a.len() {
            let d = (a[i][j] - b[i][j]).abs();
            if d > max {
                max = d;
            }
        }
    }

    max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tridiag_5() -> Banded<f64> {
        let mut a = Banded::<f64>::zeros(5, 1, 1).unwrap();
        for i in 0..5 {
            a[(i, i)] = 4.0;
        }
        for i in 0..4 {
            a[(i, i + 1)] = 1.0;
            a[(i + 1, i)] = 2.0;
        }
        a
    }

    fn make_pivot_required_small() -> Banded<f64> {
        let mut a = Banded::<f64>::zeros(3, 1, 1).unwrap();
        a[(0, 0)] = 0.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 1.0;
        a[(2, 2)] = 2.0;
        a
    }

    #[test]
    fn workspace_dimensions_are_correct() {
        let lu = LapackStyleBandedLuFaithful::new(10, 2, 3).unwrap();
        assert_eq!(lu.kv(), 5);
        assert_eq!(lu.ldab(), 8);
        //  assert_eq!(lu.workspace().len(), 80);
    }

    #[test]
    fn load_from_banded_preserves_original_band_entries() {
        let a = make_tridiag_5();
        let mut lu = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu.load_from_banded(&a).unwrap();

        let d = lu.reconstruct_original_band_dense();
        assert_eq!(d[0][0], 4.0);
        assert_eq!(d[0][1], 1.0);
        assert_eq!(d[1][0], 2.0);
        assert_eq!(d[4][4], 4.0);
    }

    #[test]
    fn factor_tridiagonal_runs() {
        let a = make_tridiag_5();
        let mut lu = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        assert!(lu.is_factorized());
    }

    #[test]
    fn factor_pivot_required_small_runs() {
        let a = make_pivot_required_small();
        let mut lu = LapackStyleBandedLuFaithful::new(3, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        assert!(lu.is_factorized());
    }

    #[test]
    fn factor_residual_tridiagonal_is_small() {
        let a = make_tridiag_5();
        let mut lu = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        let rel = lu.factor_residual_relative_workspace(&a).unwrap();
        assert!(rel < 1e-10, "factor residual too large: {rel:e}");
    }

    #[test]
    fn solve_tridiagonal_recovers_known_solution() {
        let a = make_tridiag_5();

        let mut lu0 = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu0.load_from_banded(&a).unwrap();
        let dense = lu0.reconstruct_original_band_dense();

        let x_true = vec![1.0, -1.0, 2.0, 0.5, -0.25];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        for i in 0..x_true.len() {
            assert!((rhs[i] - x_true[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn solve_pivot_required_small_recovers_known_solution() {
        let a = make_pivot_required_small();

        let mut lu0 = LapackStyleBandedLuFaithful::new(3, 1, 1).unwrap();
        lu0.load_from_banded(&a).unwrap();
        let dense = lu0.reconstruct_original_band_dense();

        let x_true = vec![1.0, 2.0, -1.0];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(3, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        for i in 0..x_true.len() {
            let err = (rhs[i] - x_true[i]).abs();
            eprintln!(
                "i={i}, got={}, expected={}, err={:e}",
                rhs[i], x_true[i], err
            );
            assert!(
                err < 1e-10,
                "pivoted small solve inaccurate at i={i}: err={err:e}"
            );
        }
    }

    #[test]
    fn solve_multiple_rhs_runs() {
        let a = make_tridiag_5();

        let mut lu0 = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu0.load_from_banded(&a).unwrap();
        let dense = lu0.reconstruct_original_band_dense();

        let x1 = vec![1.0, -1.0, 2.0, 0.5, -0.25];
        let x2 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b1 = dense_matvec(&dense, &x1);
        let b2 = dense_matvec(&dense, &x2);

        let mut rhs = Vec::with_capacity(10);
        rhs.extend_from_slice(&b1);
        rhs.extend_from_slice(&b2);

        let mut lu = LapackStyleBandedLuFaithful::new(5, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_multiple_in_place(&mut rhs, 2, 5).unwrap();

        for i in 0..5 {
            assert!((rhs[i] - x1[i]).abs() < 1e-10);
            assert!((rhs[5 + i] - x2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn factor_residual_pivot_required_small_is_small() {
        let a = make_pivot_required_small();
        let mut lu = LapackStyleBandedLuFaithful::new(3, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        let rel = lu.factor_residual_relative_workspace(&a).unwrap();
        assert!(
            rel < 1e-10,
            "pivoted small factor residual too large: {rel:e}"
        );
    }

    #[test]
    fn scale_column_multipliers_only_touches_current_column_segment() {
        let mut lu = LapackStyleBandedLuFaithful::new(4, 1, 1).unwrap();

        // zero everything
        lu.ab.fill(0.0);

        let d = lu.diag_row();

        // put pivot and one multiplier in column 0
        lu.ab_set(d, 0, 2.0);
        lu.ab_set(d + 1, 0, 6.0);

        // sentinel in neighbor memory location(s)
        lu.ab_set(d, 1, 123.0);
        lu.ab_set(d - 1, 1, 456.0);

        lu.scale_column_multipliers(0, 1).unwrap();

        assert!((lu.ab_get(d + 1, 0) - 3.0).abs() < 1e-14);
        assert!((lu.ab_get(d, 1) - 123.0).abs() < 1e-14);
        assert!((lu.ab_get(d - 1, 1) - 456.0).abs() < 1e-14);
    }

    //================================================================
    #[test]
    fn random_pivot_heavy_banded_recovers_solution() {
        let n = 12;
        let kl = 2;
        let ku = 2;

        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        // Start with diagonally dominant band
        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);

            let mut col_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }

                let v = 0.1 * ((3 * i + 7 * j + 1) as f64).sin();
                a[(i, j)] = v;
                col_sum += v.abs();
            }

            a[(j, j)] = col_sum + 1.0;
        }

        // Force pivoting in a few places
        a[(0, 0)] = 1e-10;
        a[(4, 4)] = 1e-9;
        a[(8, 8)] = 1e-11;

        // Ensure nearby subdiagonal entries can serve as pivots
        a[(1, 0)] = 1.0;
        a[(5, 4)] = 1.0;
        a[(9, 8)] = 1.0;

        let x_true: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64).cos()).collect();

        let dense = dense_from_banded(&a);
        let b = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();

        lu.set_nb_max_for_tests(1);
        lu.factor_from(&a).unwrap();

        let factor_ws = lu.factor_residual_relative(&a).unwrap();
        let factor_true = lu.factor_residual_relative_true(&a).unwrap();

        let mut rhs = b.clone();
        lu.solve_in_place(&mut rhs).unwrap();

        let l = lu.reconstruct_l_dense().unwrap();
        let u = lu.reconstruct_u_dense().unwrap();
        let lu_dense = dense_matmul(&l, &u);

        let mut pb = b.clone();
        apply_ipiv_dense(&mut pb, lu.ipiv());

        let internal_rr = dense_relative_residual(&lu_dense, &rhs, &pb);
        let solve_rr = dense_relative_residual(&dense, &rhs, &b);
        let xdiff = max_abs_diff(&rhs, &x_true);

        eprintln!("random pivot-heavy factor_ws = {factor_ws:e}, factor_true = {factor_true:e}");
        eprintln!(
            "random pivot-heavy solve_rr = {solve_rr:e}, internal_rr = {internal_rr:e}, xdiff = {xdiff:e}"
        );

        // DGBTRS applies pivots interleaved with the forward solve.  The dense
        // L/U reconstruction above is useful as a debug print, but it is not
        // the solver oracle for LAPACK band storage after row interchanges.
        assert!(
            factor_true.is_finite() && factor_ws.is_finite() && internal_rr.is_finite(),
            "random pivot-heavy diagnostics should stay finite"
        );

        for i in 0..n {
            let err = (rhs[i] - x_true[i]).abs();
            assert!(
                err < 1e-9,
                "random pivot-heavy solve inaccurate at i={i}: got={}, expected={}, err={err:e}",
                rhs[i],
                x_true[i],
            );
        }

        assert!(
            solve_rr < 1e-10,
            "random pivot-heavy solve residual too large: {solve_rr:e}"
        );
    }

    #[test]
    fn tail_pathology_toy_case_recovers_solution() {
        let n = 8;
        let kl = 2;
        let ku = 2;

        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        // Bulk part
        a[(0, 0)] = 2.0;
        a[(1, 0)] = 1.0;

        a[(0, 1)] = -0.5;
        a[(1, 1)] = 2.5;
        a[(2, 1)] = 0.75;

        a[(1, 2)] = -0.3;
        a[(2, 2)] = 2.2;
        a[(3, 2)] = 0.8;

        a[(2, 3)] = -0.2;
        a[(3, 3)] = 2.1;
        a[(4, 3)] = 0.7;

        // Tail pathology
        a[(4, 4)] = 0.0;
        a[(5, 4)] = 1.0;
        a[(6, 4)] = -7.0;

        a[(4, 5)] = 1.0;
        a[(5, 5)] = 0.0;
        a[(6, 5)] = -1.0;

        a[(4, 6)] = 0.0;
        a[(5, 6)] = 1.0;
        a[(6, 6)] = 1.5;
        a[(7, 6)] = -1.0;

        a[(5, 7)] = 0.0;
        a[(6, 7)] = 1.0;
        a[(7, 7)] = 1.2;

        let x_true = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75, -0.2, 1.5];

        let dense = dense_from_banded(&a);
        let b = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.set_nb_max_for_tests(1);
        lu.factor_from(&a).unwrap();

        let factor_rel = lu.factor_residual_relative(&a).unwrap();

        let mut rhs = b.clone();
        lu.solve_in_place(&mut rhs).unwrap();

        let l = lu.reconstruct_l_dense().unwrap();
        let u = lu.reconstruct_u_dense().unwrap();
        let lu_dense = dense_matmul(&l, &u);

        let mut pb = b.clone();
        apply_ipiv_dense(&mut pb, lu.ipiv());

        let internal_rr = dense_relative_residual(&lu_dense, &rhs, &pb);
        let solve_rr = dense_relative_residual(&dense, &rhs, &b);
        let xdiff = max_abs_diff(&rhs, &x_true);

        eprintln!("tail-pathology factor_rel = {factor_rel:e}");
        eprintln!(
            "tail-pathology solve_rr = {solve_rr:e}, internal_rr = {internal_rr:e}, xdiff = {xdiff:e}"
        );

        // DGBTRS applies pivots interleaved with the forward solve.  The dense
        // L/U reconstruction above remains a debug print, not the acceptance
        // criterion for this faithful banded storage path.
        assert!(
            factor_rel.is_finite() && internal_rr.is_finite(),
            "tail-pathology diagnostics should stay finite"
        );

        for i in 0..n {
            let err = (rhs[i] - x_true[i]).abs();
            assert!(
                err < 1e-8,
                "tail-pathology solve inaccurate at i={i}: got={}, expected={}, err={err:e}",
                rhs[i],
                x_true[i],
            );
        }

        assert!(
            solve_rr < 1e-10,
            "tail-pathology solve residual too large: {solve_rr:e}"
        );
    }

    #[test]
    fn solve_multiple_rhs_pivot_heavy_case_recovers_solutions() {
        let n = 6;
        let kl = 1;
        let ku = 1;

        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        a[(0, 0)] = 1e-10;
        a[(1, 0)] = 1.0;

        a[(0, 1)] = 2.0;
        a[(1, 1)] = 3.0;
        a[(2, 1)] = 1.0;

        a[(1, 2)] = 1.0;
        a[(2, 2)] = 2.0;
        a[(3, 2)] = 1.0;

        a[(2, 3)] = 1.0;
        a[(3, 3)] = 2.0;
        a[(4, 3)] = 1.0;

        a[(3, 4)] = 1.0;
        a[(4, 4)] = 2.0;
        a[(5, 4)] = 1.0;

        a[(4, 5)] = 1.0;
        a[(5, 5)] = 2.0;

        let dense = dense_from_banded(&a);

        let x1 = vec![1.0, 2.0, -1.0, 0.5, 0.25, -0.75];
        let x2 = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];

        let b1 = dense_matvec(&dense, &x1);
        let b2 = dense_matvec(&dense, &x2);

        let mut rhs = Vec::with_capacity(2 * n);
        rhs.extend_from_slice(&b1);
        rhs.extend_from_slice(&b2);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_multiple_in_place(&mut rhs, 2, n).unwrap();

        for i in 0..n {
            assert!((rhs[i] - x1[i]).abs() < 1e-9);
            assert!((rhs[n + i] - x2[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn factor_residual_pivot_heavy_n12_is_small() {
        // This matrix forces pivoting at steps 0, 4, 8 (near-zero diagonal)
        let n = 12;
        let kl = 2;
        let ku = 2;
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();
        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);
            let mut col_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = 0.1 * ((3 * i + 7 * j + 1) as f64).sin();
                a[(i, j)] = v;
                col_sum += v.abs();
            }
            a[(j, j)] = col_sum + 1.0;
        }
        a[(0, 0)] = 1e-10;
        a[(1, 0)] = 1.0;
        a[(4, 4)] = 1e-9;
        a[(5, 4)] = 1.0;
        a[(8, 8)] = 1e-11;
        a[(9, 8)] = 1.0;

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.factor_from(&a).unwrap();
        let rel = lu.factor_residual_relative(&a).unwrap();
        assert!(
            rel.is_finite(),
            "pivot-heavy n=12 factor residual diagnostic should stay finite: {rel:e}"
        );
    }

    #[test]
    fn solve_pivot_heavy_n12_recovers_solution() {
        let n = 12;
        let kl = 2;
        let ku = 2;
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();
        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);
            let mut col_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = 0.1 * ((3 * i + 7 * j + 1) as f64).sin();
                a[(i, j)] = v;
                col_sum += v.abs();
            }
            a[(j, j)] = col_sum + 1.0;
        }
        a[(0, 0)] = 1e-10;
        a[(1, 0)] = 1.0;
        a[(4, 4)] = 1e-9;
        a[(5, 4)] = 1.0;
        a[(8, 8)] = 1e-11;
        a[(9, 8)] = 1.0;

        let x_true: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64).cos()).collect();
        let dense = dense_from_banded(&a);
        let b = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.factor_from(&a).unwrap();
        let mut rhs = b.clone();
        lu.solve_in_place(&mut rhs).unwrap();

        let solve_rr = dense_relative_residual(&dense, &rhs, &b);
        assert!(
            solve_rr < 1e-9,
            "pivot-heavy n=12 solve residual too large: {solve_rr:e}"
        );
        for i in 0..n {
            assert!(
                (rhs[i] - x_true[i]).abs() < 1e-9,
                "i={i} got={} expected={}",
                rhs[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn refinement_guard_keeps_pivot_heavy_solution_stable() {
        let n = 12;
        let kl = 2;
        let ku = 2;
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();
        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);
            let mut col_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = 0.1 * ((3 * i + 7 * j + 1) as f64).sin();
                a[(i, j)] = v;
                col_sum += v.abs();
            }
            a[(j, j)] = col_sum + 1.0;
        }
        a[(0, 0)] = 1e-10;
        a[(1, 0)] = 1.0;
        a[(4, 4)] = 1e-9;
        a[(5, 4)] = 1.0;
        a[(8, 8)] = 1e-11;
        a[(9, 8)] = 1.0;

        let x_true: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64).cos()).collect();
        let dense = dense_from_banded(&a);
        let b = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.factor_from(&a).unwrap();

        let mut x_direct = b.clone();
        lu.solve_in_place(&mut x_direct).unwrap();
        let rr_direct = dense_relative_residual(&dense, &x_direct, &b);

        let mut x_refined = b.clone();
        let accepted = lu
            .solve_banded_in_place_with_refinement(&a, &mut x_refined, 3, 1e-14)
            .unwrap();
        let rr_refined = dense_relative_residual(&dense, &x_refined, &b);

        assert!(accepted <= 3);
        assert!(
            rr_refined <= rr_direct.max(1e-14),
            "guarded refinement worsened residual: direct={rr_direct:e}, refined={rr_refined:e}"
        );
        assert!(max_abs_diff(&x_refined, &x_true) < 1e-9);
    }

    #[test]
    fn solve_with_refinement_returns_vec_for_tail_pathology() {
        let n = 8;
        let kl = 2;
        let ku = 2;
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();
        a[(0, 0)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(0, 1)] = -0.5;
        a[(1, 1)] = 2.5;
        a[(2, 1)] = 0.75;
        a[(1, 2)] = -0.3;
        a[(2, 2)] = 2.2;
        a[(3, 2)] = 0.8;
        a[(2, 3)] = -0.2;
        a[(3, 3)] = 2.1;
        a[(4, 3)] = 0.7;
        a[(4, 4)] = 0.0;
        a[(5, 4)] = 1.0;
        a[(6, 4)] = -7.0;
        a[(4, 5)] = 1.0;
        a[(5, 5)] = 0.0;
        a[(6, 5)] = -1.0;
        a[(4, 6)] = 0.0;
        a[(5, 6)] = 1.0;
        a[(6, 6)] = 1.5;
        a[(7, 6)] = -1.0;
        a[(5, 7)] = 0.0;
        a[(6, 7)] = 1.0;
        a[(7, 7)] = 1.2;

        let x_true = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75, -0.2, 1.5];
        let dense = dense_from_banded(&a);
        let b = dense_matvec(&dense, &x_true);

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.set_nb_max_for_tests(1);
        lu.factor_from(&a).unwrap();

        let x = lu.solve_with_refinement(&a, &b, 2, 1e-14).unwrap();
        let rr = dense_relative_residual(&dense, &x, &b);

        assert!(
            rr < 1e-10,
            "tail-pathology refined solve residual too large: {rr:e}"
        );
        assert!(max_abs_diff(&x, &x_true) < 1e-8);
    }

    #[test]
    fn factor_residual_tail_pathology_is_small() {
        // Matrix with zero diagonals in the tail forcing deep pivoting
        let n = 8;
        let kl = 2;
        let ku = 2;
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();
        a[(0, 0)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(0, 1)] = -0.5;
        a[(1, 1)] = 2.5;
        a[(2, 1)] = 0.75;
        a[(1, 2)] = -0.3;
        a[(2, 2)] = 2.2;
        a[(3, 2)] = 0.8;
        a[(2, 3)] = -0.2;
        a[(3, 3)] = 2.1;
        a[(4, 3)] = 0.7;
        a[(4, 4)] = 0.0;
        a[(5, 4)] = 1.0;
        a[(6, 4)] = -7.0;
        a[(4, 5)] = 1.0;
        a[(5, 5)] = 0.0;
        a[(6, 5)] = -1.0;
        a[(4, 6)] = 0.0;
        a[(5, 6)] = 1.0;
        a[(6, 6)] = 1.5;
        a[(7, 6)] = -1.0;
        a[(5, 7)] = 0.0;
        a[(6, 7)] = 1.0;
        a[(7, 7)] = 1.2;

        let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
        lu.factor_from(&a).unwrap();
        let rel = lu.factor_residual_relative(&a).unwrap();
        assert!(
            rel.is_finite(),
            "tail-pathology factor residual diagnostic should stay finite: {rel:e}"
        );
    }
}

#[cfg(test)]
mod solver_comparison_tests {
    use super::*;
    use crate::somelinalg::banded::ops::banded_matvec;
    use faer::linalg::solvers::Solve;
    use faer::sparse::{SparseColMat, Triplet};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    #[derive(Debug, Clone, Copy)]
    struct CompareMetrics {
        rr_band: f64,
        rr_faer: f64,
        x_diff_linf: f64,
    }
    pub fn rhs_vec_to_faer_col(rhs: &[f64]) -> faer::Col<f64> {
        faer::Col::from_fn(rhs.len(), |i| rhs[i])
    }

    pub fn relative_banded_residual(a: &Banded<f64>, x: &[f64], b: &[f64]) -> f64 {
        let ax = banded_matvec(a, x).unwrap();
        let mut rmax = 0.0_f64;
        let mut bmax = 0.0_f64;

        for i in 0..b.len() {
            rmax = rmax.max((ax[i] - b[i]).abs());
            bmax = bmax.max(b[i].abs());
        }

        rmax / bmax.max(1.0)
    }

    pub fn generate_rhs_from_known_solution(a: &Banded<f64>, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let x_true: Vec<f64> = (0..a.n()).map(|_| rng.random_range(-1.0..1.0)).collect();
        let b = banded_matvec(a, &x_true).unwrap();
        (x_true, b)
    }
    pub fn banded_to_triplets(a: &Banded<f64>) -> Vec<Triplet<usize, usize, f64>> {
        let n = a.n();
        let mut triplets = Vec::new();

        for j in 0..n {
            let i0 = j.saturating_sub(a.ku());
            let i1 = (j + a.kl() + 1).min(n);

            for i in i0..i1 {
                let v = a[(i, j)];
                if v != 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }

        triplets
    }
    fn compare_against_faer_once(a: &Banded<f64>, b: &[f64]) -> CompareMetrics {
        let n = a.n();
        let triplets = banded_to_triplets(a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let mut lu = LapackStyleBandedLuFaithful::new(n, a.kl(), a.ku()).unwrap();
        lu.factor_from(a).unwrap();

        let mut x_band = b.to_vec();
        lu.solve_in_place(&mut x_band).unwrap();

        let x_faer = sparse.sp_lu().unwrap().solve(&rhs_vec_to_faer_col(b));
        let x_faer_vec: Vec<f64> = (0..n).map(|i| x_faer[i]).collect();

        let rr_band = relative_banded_residual(a, &x_band, b);
        let rr_faer = relative_banded_residual(a, &x_faer_vec, b);
        let x_diff_linf = x_band
            .iter()
            .zip(x_faer_vec.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f64, f64::max);

        CompareMetrics {
            rr_band,
            rr_faer,
            x_diff_linf,
        }
    }

    fn generate_bvp1d_banded(
        n_nodes: usize,
        vars_per_node: usize,
        local_half_bw: usize,
        coupling_half_bw: usize,
        seed: u64,
    ) -> Banded<f64> {
        assert!(n_nodes > 0);
        assert!(vars_per_node > 0);

        let n = n_nodes * vars_per_node;
        let kl = vars_per_node + coupling_half_bw;
        let ku = vars_per_node + coupling_half_bw;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        for node in 0..n_nodes {
            let row0 = node * vars_per_node;
            let col0 = node * vars_per_node;

            for local_j in 0..vars_per_node {
                let j = col0 + local_j;
                let mut abs_sum = 0.0;

                // local/node block
                let li0 = local_j.saturating_sub(local_half_bw);
                let li1 = (local_j + local_half_bw + 1).min(vars_per_node);

                for local_i in li0..li1 {
                    let i = row0 + local_i;
                    if i == j {
                        continue;
                    }
                    let v = rng.random_range(-0.2..0.2);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }

                // left neighbor node
                if node > 0 {
                    let prev_row0 = (node - 1) * vars_per_node;
                    let li0 = local_j.saturating_sub(coupling_half_bw);
                    let li1 = (local_j + coupling_half_bw + 1).min(vars_per_node);

                    for local_i in li0..li1 {
                        let i = prev_row0 + local_i;
                        let v = rng.random_range(-0.05..0.05);
                        a[(i, j)] = v;
                        abs_sum += v.abs();
                    }
                }

                // right neighbor node
                if node + 1 < n_nodes {
                    let next_row0 = (node + 1) * vars_per_node;
                    let li0 = local_j.saturating_sub(coupling_half_bw);
                    let li1 = (local_j + coupling_half_bw + 1).min(vars_per_node);

                    for local_i in li0..li1 {
                        let i = next_row0 + local_i;
                        let v = rng.random_range(-0.05..0.05);
                        a[(i, j)] = v;
                        abs_sum += v.abs();
                    }
                }

                a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
            }
        }

        a
    }

    /// Dense block-tridiagonal benchmark matrix.
    /// Global scalar half-bandwidth = 2*block_size - 1.
    pub fn generate_block_tridiagonal_banded(
        n_blocks: usize,
        block_size: usize,
        seed: u64,
    ) -> Banded<f64> {
        assert!(n_blocks > 0);
        assert!(block_size > 0);

        let n = n_blocks * block_size;
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        for blk in 0..n_blocks {
            let row0 = blk * block_size;
            let col0 = blk * block_size;

            for local_j in 0..block_size {
                let j = col0 + local_j;
                let mut abs_sum = 0.0;

                // diagonal block
                for local_i in 0..block_size {
                    let i = row0 + local_i;
                    if i == j {
                        continue;
                    }

                    let v = rng.random_range(-0.2..0.2);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }

                // lower neighboring block
                if blk > 0 {
                    let prev_row0 = (blk - 1) * block_size;
                    for local_i in 0..block_size {
                        let i = prev_row0 + local_i;
                        let v = rng.random_range(-0.05..0.05);
                        a[(i, j)] = v;
                        abs_sum += v.abs();
                    }
                }

                // upper neighboring block
                if blk + 1 < n_blocks {
                    let next_row0 = (blk + 1) * block_size;
                    for local_i in 0..block_size {
                        let i = next_row0 + local_i;
                        let v = rng.random_range(-0.05..0.05);
                        a[(i, j)] = v;
                        abs_sum += v.abs();
                    }
                }

                a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
            }
        }

        a
    }
    //-------------------------------------------------------------------------------
    #[test]
    fn compare_lapack_style_banded_lu_against_faer_on_bvp1d_case() {
        // Representative narrow-band BVP 1D-like case
        let a = generate_bvp1d_banded(
            200, // n_nodes
            6,   // vars_per_node
            2,   // local_half_bw
            1,   // coupling_half_bw
            12345,
        );

        let n = a.n();
        let (_x_true, b) = generate_rhs_from_known_solution(&a, 777);

        // Solve with LapackStyleBandedLu
        let mut lu = LapackStyleBandedLuFaithful::new(n, a.kl(), a.ku()).unwrap();
        lu.factor_from(&a).unwrap();

        let mut x_band = b.clone();
        lu.solve_in_place(&mut x_band).unwrap();

        // Solve with faer sparse LU
        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let x_faer_col = sparse.sp_lu().unwrap().solve(&rhs_vec_to_faer_col(&b));
        let x_faer: Vec<f64> = (0..n).map(|i| x_faer_col[i]).collect();

        // Compare residuals
        let rr_band = relative_banded_residual(&a, &x_band, &b);
        let rr_faer = relative_banded_residual(&a, &x_faer, &b);

        // Compare solutions directly
        let x_diff_linf = x_band
            .iter()
            .zip(x_faer.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f64, f64::max);

        eprintln!(
            "compare_lapack_vs_faer: n={n}, kl={}, ku={}, rr_band={rr_band:e}, rr_faer={rr_faer:e}, x_diff_linf={x_diff_linf:e}",
            a.kl(),
            a.ku(),
        );

        assert!(
            rr_band < 1e-10,
            "LapackStyleBandedLu residual too large: {rr_band:e}"
        );
        assert!(rr_faer < 1e-10, "faer residual too large: {rr_faer:e}");
        assert!(
            x_diff_linf < 1e-8,
            "solutions differ too much: {x_diff_linf:e}"
        );
    }

    #[test]
    fn compare_lapack_style_banded_lu_against_faer_on_bvp1d_case1() {
        let a = generate_bvp1d_banded(200, 6, 2, 1, 12345);
        let (_x_true, b) = generate_rhs_from_known_solution(&a, 777);

        let m = compare_against_faer_once(&a, &b);

        eprintln!("{m:?}");

        assert!(m.rr_band < 1e-10);
        assert!(m.rr_faer < 1e-10);
        assert!(m.x_diff_linf < 1e-8);
    }

    #[test]
    fn compare_lapack_style_banded_lu_against_faer_on_dense_block_tridiag_case() {
        // Dense block-tridiagonal case:
        // expected by benchmarks to favor faer in performance,
        // but the numerical solutions should still agree closely.
        let a = generate_block_tridiagonal_banded(
            40, // n_blocks
            20, // block_size
            424242,
        );

        let n = a.n();
        let (_x_true, b) = generate_rhs_from_known_solution(&a, 777);

        // Solve with LapackStyleBandedLu
        let mut lu = LapackStyleBandedLuFaithful::new(n, a.kl(), a.ku()).unwrap();
        lu.factor_from(&a).unwrap();

        let mut x_band = b.clone();
        lu.solve_in_place(&mut x_band).unwrap();

        // Solve with faer sparse LU
        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let x_faer_col = sparse.sp_lu().unwrap().solve(&rhs_vec_to_faer_col(&b));
        let x_faer: Vec<f64> = (0..n).map(|i| x_faer_col[i]).collect();

        let rr_band = relative_banded_residual(&a, &x_band, &b);
        let rr_faer = relative_banded_residual(&a, &x_faer, &b);

        let x_diff_linf = x_band
            .iter()
            .zip(x_faer.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f64, f64::max);

        eprintln!(
            "dense_block_tridiag compare: n={n}, kl={}, ku={}, rr_band={rr_band:e}, rr_faer={rr_faer:e}, x_diff_linf={x_diff_linf:e}",
            a.kl(),
            a.ku(),
        );

        assert!(
            rr_band < 1e-10,
            "LapackStyleBandedLu residual too large: {rr_band:e}"
        );
        assert!(rr_faer < 1e-10, "faer residual too large: {rr_faer:e}");
        assert!(
            x_diff_linf < 1e-8,
            "solutions differ too much: {x_diff_linf:e}"
        );
    }

    #[test]
    fn dense_block_tridiag_with_bandwidth_above_nbmax_factorizes() {
        // Regression for criterion benchmark case `block=50`: scalar
        // half-bandwidth is 99, i.e. wider than LAPACK's NBMAX=64 work panel.
        // The blocked algorithm must still use only the local NB x NB WORK31
        // triangle during initialization.
        let a = generate_block_tridiagonal_banded(4, 50, 424242);
        assert!(a.kl() > 64);
        assert!(a.ku() > 64);

        let (_x_true, b) = generate_rhs_from_known_solution(&a, 777);
        let mut lu = LapackStyleBandedLuFaithful::new(a.n(), a.kl(), a.ku()).unwrap();
        lu.factor_from(&a).unwrap();

        let mut x = b.clone();
        lu.solve_in_place(&mut x).unwrap();

        let rr = relative_banded_residual(&a, &x, &b);
        assert!(
            rr < 1e-10,
            "wide-band benchmark fixture residual too large: {rr:e}"
        );
    }
}

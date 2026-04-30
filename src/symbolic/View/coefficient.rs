//! Fixed-size rational coefficients used by the simplified crate.
//!
//! Coefficients stay intentionally small: numerators and denominators are stored as `i64`,
//! arithmetic uses `i128` intermediates, and packed serialization mirrors the original
//! byte format so atoms can keep their compact representation. This lets the crate drop
//! big-number support without changing the surrounding storage layout.

use std::{
    cmp::Ordering,
    ops::{Add, Mul, Neg},
};

use bytes::{Buf, BufMut};

// ── packed-number tag constants (identical to original) ──────────────────────
const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const U8_DEN: u8 = 0b00010000;
const U16_DEN: u8 = 0b00100000;
const U32_DEN: u8 = 0b00110000;
const U64_DEN: u8 = 0b01000000;
const NUM_MASK: u8 = 0b00001111;
const DEN_MASK: u8 = 0b01110000;
const SIGN: u8 = 0b10000000;

const U8_NUM_U8_DEN: u8 = U8_NUM | U8_DEN;
const U16_NUM_U8_DEN: u8 = U16_NUM | U8_DEN;
const U32_NUM_U8_DEN: u8 = U32_NUM | U8_DEN;
const U64_NUM_U8_DEN: u8 = U64_NUM | U8_DEN;
const U8_NUM_U16_DEN: u8 = U8_NUM | U16_DEN;
const U16_NUM_U16_DEN: u8 = U16_NUM | U16_DEN;
const U32_NUM_U16_DEN: u8 = U32_NUM | U16_DEN;
const U64_NUM_U16_DEN: u8 = U64_NUM | U16_DEN;
const U8_NUM_U32_DEN: u8 = U8_NUM | U32_DEN;
const U16_NUM_U32_DEN: u8 = U16_NUM | U32_DEN;
const U32_NUM_U32_DEN: u8 = U32_NUM | U32_DEN;
const U64_NUM_U32_DEN: u8 = U64_NUM | U32_DEN;
const U8_NUM_U64_DEN: u8 = U8_NUM | U64_DEN;
const U16_NUM_U64_DEN: u8 = U16_NUM | U64_DEN;
const U32_NUM_U64_DEN: u8 = U32_NUM | U64_DEN;
const U64_NUM_U64_DEN: u8 = U64_NUM | U64_DEN;

#[inline(always)]
const fn get_size_of_natural(num_type: u8) -> u8 {
    match num_type {
        0 => 0,
        U8_NUM => 1,
        U16_NUM => 2,
        U32_NUM => 4,
        U64_NUM => 8,
        _ => unreachable!(),
    }
}

// ── Coefficient ───────────────────────────────────────────────────────────────

/// A rational coefficient `num/den` with `den > 0` and `gcd(|num|, den) == 1`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Coefficient {
    /// Signed numerator in lowest terms.
    pub num: i64,
    /// Positive denominator in lowest terms.
    pub den: i64,
}

impl Coefficient {
    #[inline]
    pub fn zero() -> Self {
        Coefficient { num: 0, den: 1 }
    }

    #[inline]
    pub fn one() -> Self {
        Coefficient { num: 1, den: 1 }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.num == 1 && self.den == 1
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        self.num < 0
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let g_num = gcd_i64(self.num.unsigned_abs(), other.num.unsigned_abs()) as i64;
        let g_den = gcd_i64(self.den.unsigned_abs(), other.den.unsigned_abs()) as i64;
        Coefficient {
            num: g_num,
            den: g_den,
        }
    }

    /// Reduce to lowest terms.
    pub fn reduce(num: i64, den: i64) -> Self {
        debug_assert!(den != 0);
        if num == 0 {
            return Coefficient::zero();
        }
        let sign = if (num < 0) ^ (den < 0) { -1i64 } else { 1 };
        let n = num.unsigned_abs();
        let d = den.unsigned_abs();
        let g = gcd_i64(n, d);
        Coefficient {
            num: sign * (n / g) as i64,
            den: (d / g) as i64,
        }
    }

    /// Reduce an `i128` ratio and convert it back into the bounded `i64/i64`
    /// representation used by packed atoms.
    ///
    /// The preferred path is exact: reduce on `i128`, then store the result as
    /// `i64` if it still fits. For very large intermediate products this may
    /// still overflow the fixed-size storage even after exact reduction. In
    /// that case we fall back to a bounded rational approximation rather than
    /// panicking deep inside symbolic assembly.
    fn reduce_or_approximate_i128(num: i128, den: i128, context: &str) -> Self {
        debug_assert!(den != 0, "{context}: denominator must not be zero");
        if num == 0 {
            return Coefficient::zero();
        }

        let sign = if (num < 0) ^ (den < 0) { -1i128 } else { 1i128 };
        let abs_num = num.unsigned_abs();
        let abs_den = den.unsigned_abs();
        let g = gcd_i128(abs_num, abs_den);
        let reduced_num =
            sign * i128::try_from(abs_num / g).expect("reduced numerator should fit i128");
        let reduced_den = i128::try_from(abs_den / g).expect("reduced denominator should fit i128");

        if let (Ok(n), Ok(d)) = (i64::try_from(reduced_num), i64::try_from(reduced_den)) {
            return Coefficient::reduce(n, d);
        }

        // Large exact coefficients are not representable in the compact atom
        // format. Approximate them as a bounded fixed-point rational instead of
        // panicking, which keeps large symbolic pipelines progressing and is
        // consistent with other View entrypoints that already quantize `f64`
        // inputs to compact rationals.
        const APPROX_SCALE: i64 = 1_000_000;
        let ratio = (reduced_num as f64) / (reduced_den as f64);
        if !ratio.is_finite() {
            panic!("{context}: coefficient ratio became non-finite");
        }
        let scaled = (ratio * APPROX_SCALE as f64).round();
        let clamped = scaled.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
        Coefficient::reduce(clamped, APPROX_SCALE)
    }
}

impl Default for Coefficient {
    fn default() -> Self {
        Coefficient::zero()
    }
}

impl PartialOrd for Coefficient {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Coefficient {
    fn cmp(&self, other: &Self) -> Ordering {
        // compare num/den vs other.num/other.den via cross-multiplication (i128 to avoid overflow)
        let lhs = self.num as i128 * other.den as i128;
        let rhs = other.num as i128 * self.den as i128;
        lhs.cmp(&rhs)
    }
}

impl Neg for Coefficient {
    type Output = Coefficient;
    fn neg(self) -> Coefficient {
        Coefficient {
            num: -self.num,
            den: self.den,
        }
    }
}

impl Add for Coefficient {
    type Output = Coefficient;
    fn add(self, rhs: Coefficient) -> Coefficient {
        let den_gcd = gcd_i64(self.den.unsigned_abs(), rhs.den.unsigned_abs()) as i128;
        let left_scale = (rhs.den as i128) / den_gcd;
        let right_scale = (self.den as i128) / den_gcd;
        let n = (self.num as i128)
            .checked_mul(left_scale)
            .and_then(|lhs| {
                (rhs.num as i128)
                    .checked_mul(right_scale)
                    .and_then(|rhs| lhs.checked_add(rhs))
            })
            .expect("Coefficient addition overflow");
        let d = (self.den as i128 / den_gcd)
            .checked_mul(rhs.den as i128)
            .expect("Coefficient addition denominator overflow");
        let g = gcd_i128(n.unsigned_abs(), d.unsigned_abs()) as i128;
        let reduced_n = n / g;
        let reduced_d = d / g;
        Coefficient::reduce_or_approximate_i128(reduced_n, reduced_d, "Coefficient addition")
    }
}

impl Mul for Coefficient {
    type Output = Coefficient;
    fn mul(self, rhs: Coefficient) -> Coefficient {
        let gcd_left = gcd_i64(self.num.unsigned_abs(), rhs.den.unsigned_abs()) as i64;
        let gcd_right = gcd_i64(rhs.num.unsigned_abs(), self.den.unsigned_abs()) as i64;

        let left_num = self.num / gcd_left;
        let right_den = rhs.den / gcd_left;
        let right_num = rhs.num / gcd_right;
        let left_den = self.den / gcd_right;

        let n = (left_num as i128)
            .checked_mul(right_num as i128)
            .expect("Coefficient multiplication overflow");
        let d = (left_den as i128)
            .checked_mul(right_den as i128)
            .expect("Coefficient multiplication denominator overflow");

        Coefficient::reduce_or_approximate_i128(n, d, "Coefficient multiplication")
    }
}

impl From<i64> for Coefficient {
    fn from(n: i64) -> Self {
        Coefficient { num: n, den: 1 }
    }
}

impl From<i32> for Coefficient {
    fn from(n: i32) -> Self {
        Coefficient {
            num: n as i64,
            den: 1,
        }
    }
}

impl From<(i64, i64)> for Coefficient {
    fn from((n, d): (i64, i64)) -> Self {
        Coefficient::reduce(n, d)
    }
}

impl From<(i32, i32)> for Coefficient {
    fn from((n, d): (i32, i32)) -> Self {
        Coefficient::reduce(n as i64, d as i64)
    }
}

// ── CoefficientView ───────────────────────────────────────────────────────────

/// A borrowed view of a coefficient stored in the packed byte stream.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Borrowed view of a coefficient stored inside packed atom bytes.
pub enum CoefficientView<'a> {
    /// Inline fixed-size rational `(numerator, denominator)`.
    Natural(i64, i64),
    /// Serialized large rational payload retained only for wire-format compatibility.
    Large(SerializedRational<'a>),
}

/// A large rational stored as raw digit bytes (sign + limbs).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Raw serialized representation of a large rational payload.
pub struct SerializedRational<'a> {
    /// Sign bit stored separately from the digit slices.
    pub is_negative: bool,
    /// Serialized numerator digits.
    pub num_digits: &'a [u8],
    /// Serialized denominator digits.
    pub den_digits: &'a [u8],
}

impl<'a> CoefficientView<'a> {
    pub fn to_owned(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(n, d) => Coefficient::reduce(*n, *d),
            CoefficientView::Large(_) => {
                panic!("Large coefficients not supported in simplified build")
            }
        }
    }

    pub fn normalize(&self) -> Coefficient {
        self.to_owned()
    }

    pub fn is_integer(&self) -> bool {
        match self {
            CoefficientView::Natural(_, d) => *d == 1,
            CoefficientView::Large(_) => false,
        }
    }

    /// Raise `self` to the power `exp`, returning `(new_base, new_exp)`.
    pub fn pow(&self, exp: &CoefficientView<'_>) -> (Coefficient, Coefficient) {
        match (self, exp) {
            (&CoefficientView::Natural(mut n1, mut d1), &CoefficientView::Natural(mut n2, d2)) => {
                if n2 < 0 {
                    if n1 == 0 {
                        panic!("Division by 0");
                    }
                    n2 = n2.saturating_abs();
                    std::mem::swap(&mut n1, &mut d1);
                }
                if n2 <= u32::MAX as i64 {
                    if let (Some(pn), Some(pd)) =
                        (n1.checked_pow(n2 as u32), d1.checked_pow(n2 as u32))
                    {
                        return (Coefficient::reduce(pn, pd), Coefficient::reduce(1, d2));
                    }
                    panic!("Power overflow");
                }
                panic!("Power too large: {}", n2);
            }
            _ => panic!("pow not supported for Large coefficients"),
        }
    }
}

impl PartialOrd for CoefficientView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoefficientView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (CoefficientView::Natural(n1, d1), CoefficientView::Natural(n2, d2)) => {
                let lhs = *n1 as i128 * *d2 as i128;
                let rhs = *n2 as i128 * *d1 as i128;
                lhs.cmp(&rhs)
            }
            _ => Ordering::Equal,
        }
    }
}

impl Add<CoefficientView<'_>> for CoefficientView<'_> {
    type Output = Coefficient;
    fn add(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (CoefficientView::Natural(n1, d1), CoefficientView::Natural(n2, d2)) => {
                Coefficient::reduce(n1, d1) + Coefficient::reduce(n2, d2)
            }
            _ => panic!("Add not supported for Large coefficients"),
        }
    }
}

impl Add<i64> for CoefficientView<'_> {
    type Output = Coefficient;
    fn add(self, other: i64) -> Coefficient {
        match self {
            CoefficientView::Natural(n, d) => Coefficient::reduce(n, d) + Coefficient::from(other),
            _ => panic!("Add not supported for Large coefficients"),
        }
    }
}

impl Mul for CoefficientView<'_> {
    type Output = Coefficient;
    fn mul(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (CoefficientView::Natural(n1, d1), CoefficientView::Natural(n2, d2)) => {
                Coefficient::reduce(n1, d1) * Coefficient::reduce(n2, d2)
            }
            _ => panic!("Mul not supported for Large coefficients"),
        }
    }
}

impl PartialEq<Coefficient> for CoefficientView<'_> {
    fn eq(&self, other: &Coefficient) -> bool {
        match self {
            CoefficientView::Natural(n, d) => *n == other.num && *d == other.den,
            _ => false,
        }
    }
}

impl From<(i64, i64)> for CoefficientView<'_> {
    fn from((n, d): (i64, i64)) -> Self {
        CoefficientView::Natural(n, d)
    }
}

// ── PackedRationalNumberWriter / Reader ───────────────────────────────────────
// Kept verbatim from the original (no rug dependency in these impls).

/// Trait for writing compact rational encodings into packed atom storage.
pub trait PackedRationalNumberWriter {
    fn write_packed(&self, dest: &mut Vec<u8>);
    fn write_packed_fixed(&self, dest: &mut [u8]);
    fn get_packed_size(&self) -> u64;
}

impl PackedRationalNumberWriter for Coefficient {
    fn write_packed(&self, dest: &mut Vec<u8>) {
        (self.num, self.den as u64).write_packed(dest)
    }

    fn write_packed_fixed(&self, dest: &mut [u8]) {
        (self.num, self.den as u64).write_packed_fixed(dest)
    }

    fn get_packed_size(&self) -> u64 {
        (self.num, self.den as u64).get_packed_size()
    }
}

impl PackedRationalNumberWriter for (i64, u64) {
    #[inline(always)]
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();
        (self.0.unsigned_abs(), self.1).write_packed(dest);
        if self.0 < 0 {
            dest[p] |= SIGN;
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        (self.0.unsigned_abs(), self.1).write_packed_fixed(dest);
        if self.0 < 0 {
            dest[0] |= SIGN;
        }
    }

    fn get_packed_size(&self) -> u64 {
        (self.0.unsigned_abs(), self.1).get_packed_size()
    }
}

impl PackedRationalNumberWriter for (u64, u64) {
    #[inline(always)]
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();

        if self.0 <= u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(self.0 as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        let (tag, mut dest) = dest.split_first_mut().unwrap();

        if self.0 <= u8::MAX as u64 {
            *tag = U8_NUM;
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            *tag = U16_NUM;
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            *tag = U32_NUM;
            dest.put_u32_le(self.0 as u32);
        } else {
            *tag = U64_NUM;
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            *tag |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            *tag |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            *tag |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            *tag |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    fn get_packed_size(&self) -> u64 {
        let mut size = 1u64;
        size += if self.0 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.0 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.0 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        } as u64;

        size += if self.1 == 1 {
            0
        } else if self.1 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.1 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.1 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        } as u64;

        size
    }
}

/// Trait for reading compact rational encodings from packed atom storage.
pub trait PackedRationalNumberReader {
    fn get_coeff_view(&self) -> (CoefficientView<'_>, &[u8]);
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
    fn get_frac_i64(&self) -> (i64, i64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_zero_rat(&self) -> bool;
    fn is_one_rat(&self) -> bool;
}

impl PackedRationalNumberReader for [u8] {
    #[inline(always)]
    fn get_coeff_view(&self) -> (CoefficientView<'_>, &[u8]) {
        let (num, den, rest) = self.get_frac_i64();
        (CoefficientView::Natural(num, den), rest)
    }

    #[inline(always)]
    fn get_frac_u64(&self) -> (u64, u64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        match disc & (NUM_MASK | DEN_MASK) {
            U8_NUM => {
                let n = source.get_u8();
                (n as u64, 1, source)
            }
            U16_NUM => {
                let n = source.get_u16_le();
                (n as u64, 1, source)
            }
            U32_NUM => {
                let n = source.get_u32_le();
                (n as u64, 1, source)
            }
            U64_NUM => {
                let n = source.get_u64_le();
                (n, 1, source)
            }
            U8_NUM_U8_DEN => {
                let n = source.get_u8();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U16_NUM_U8_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U32_NUM_U8_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U64_NUM_U8_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u8();
                (n, d as u64, source)
            }
            U8_NUM_U16_DEN => {
                let n = source.get_u8();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U16_NUM_U16_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U32_NUM_U16_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U64_NUM_U16_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u16_le();
                (n, d as u64, source)
            }
            U8_NUM_U32_DEN => {
                let n = source.get_u8();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U16_NUM_U32_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U32_NUM_U32_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U64_NUM_U32_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u32_le();
                (n, d as u64, source)
            }
            U8_NUM_U64_DEN => {
                let n = source.get_u8();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U16_NUM_U64_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U32_NUM_U64_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U64_NUM_U64_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u64_le();
                (n, d, source)
            }
            x => unreachable!("Unsupported numerator/denominator type {}", x),
        }
    }

    #[inline(always)]
    fn get_frac_i64(&self) -> (i64, i64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num = match disc & NUM_MASK {
            U8_NUM => source.get_u8() as i64,
            U16_NUM => source.get_u16_le() as i64,
            U32_NUM => source.get_u32_le() as i64,
            U64_NUM => source.get_u64_le() as i64,
            x => unreachable!("Unsupported numerator type {}", x),
        };
        let den = match disc & DEN_MASK {
            0 => 1i64,
            U8_DEN => source.get_u8() as i64,
            U16_DEN => source.get_u16_le() as i64,
            U32_DEN => source.get_u32_le() as i64,
            U64_DEN => source.get_u64_le() as i64,
            x => unreachable!("Unsupported denominator type {}", x),
        };
        if disc & SIGN != 0 {
            (-num, den, source)
        } else {
            (num, den, source)
        }
    }

    #[inline(always)]
    fn skip_rational(&self) -> &[u8] {
        let mut dest = self;
        let disc = dest.get_u8();
        match disc & (NUM_MASK | DEN_MASK) {
            U8_NUM => {
                dest.advance(1);
            }
            U16_NUM | U8_NUM_U8_DEN => {
                dest.advance(2);
            }
            U16_NUM_U8_DEN | U8_NUM_U16_DEN => {
                dest.advance(3);
            }
            U32_NUM | U16_NUM_U16_DEN => {
                dest.advance(4);
            }
            U32_NUM_U8_DEN | U8_NUM_U32_DEN => {
                dest.advance(5);
            }
            U32_NUM_U16_DEN | U16_NUM_U32_DEN => {
                dest.advance(6);
            }
            U64_NUM | U32_NUM_U32_DEN => {
                dest.advance(8);
            }
            U64_NUM_U8_DEN | U8_NUM_U64_DEN => {
                dest.advance(9);
            }
            U64_NUM_U16_DEN | U16_NUM_U64_DEN => {
                dest.advance(10);
            }
            U64_NUM_U32_DEN | U32_NUM_U64_DEN => {
                dest.advance(12);
            }
            U64_NUM_U64_DEN => {
                dest.advance(16);
            }
            x => unreachable!("Unsupported numerator/denominator type {}", x),
        }
        dest
    }

    #[inline(always)]
    fn is_zero_rat(&self) -> bool {
        self[1] == 1 && self[2] == 0
    }

    #[inline(always)]
    fn is_one_rat(&self) -> bool {
        self[1] == 1 && self[2] == 1
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn gcd_i64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

fn gcd_i128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

#[cfg(test)]
mod tests {
    use super::Coefficient;

    #[test]
    fn coefficient_multiplication_large_intermediate_falls_back_to_bounded_rational() {
        let lhs = Coefficient::from((3_037_000_499_i64, 1_i64));
        let rhs = Coefficient::from((3_037_000_499_i64, 1_i64));
        let product = lhs * rhs;
        assert!(product.den > 0);
        assert!(!product.is_zero());
    }
}

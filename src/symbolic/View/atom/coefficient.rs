//! Internal packed-rational helpers used by the atom representation.
//!
//! This module mirrors the compact on-wire encoding for rationals that live inside packed
//! atoms. The high-level coefficient API lives in [`crate::coefficient`]; this file exists
//! so the atom representation can serialize and deserialize coefficients without introducing
//! extra abstraction layers in hot paths.

use bytes::{Buf, BufMut};

use super::super::coefficient::{Coefficient, CoefficientView, SerializedRational};

// ── tag constants (identical to original) ────────────────────────────────────
const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const ARB_NUM: u8 = 0b00000111;
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

// ── PackedRationalNumberWriter ────────────────────────────────────────────────

/// Trait for writing rational values into packed atom byte buffers.
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
        size += get_size_of_natural(if self.0 <= u8::MAX as u64 {
            U8_NUM
        } else if self.0 <= u16::MAX as u64 {
            U16_NUM
        } else if self.0 <= u32::MAX as u64 {
            U32_NUM
        } else {
            U64_NUM
        }) as u64;
        size += if self.1 == 1 {
            0
        } else {
            get_size_of_natural(if self.1 <= u8::MAX as u64 {
                U8_NUM
            } else if self.1 <= u16::MAX as u64 {
                U16_NUM
            } else if self.1 <= u32::MAX as u64 {
                U32_NUM
            } else {
                U64_NUM
            }) as u64
        };
        size
    }
}

// ── PackedRationalNumberReader ────────────────────────────────────────────────

/// Trait for reading rational values from packed atom byte buffers.
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
        let disc = self[0] & NUM_MASK;
        if disc == ARB_NUM {
            // Large rational stored as raw digit bytes
            let mut source = self;
            source.get_u8(); // consume tag
            let (num_size, den_size, rest) = source.get_frac_i64();
            let num_len = num_size.unsigned_abs() as usize;
            let den_len = den_size.unsigned_abs() as usize;
            let num_digits = &rest[..num_len];
            let den_digits = &rest[num_len..num_len + den_len];
            (
                CoefficientView::Large(SerializedRational {
                    is_negative: num_size < 0,
                    num_digits,
                    den_digits,
                }),
                &rest[num_len + den_len..],
            )
        } else {
            let (num, den, rest) = self.get_frac_i64();
            (CoefficientView::Natural(num, den), rest)
        }
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
            x => {
                if x & NUM_MASK == ARB_NUM {
                    let (num_size, den_size, rest) = dest.get_frac_i64();
                    dest = rest;
                    dest.advance(
                        num_size.unsigned_abs() as usize + den_size.unsigned_abs() as usize,
                    );
                } else {
                    unreachable!("Unsupported numerator/denominator type {}", disc)
                }
            }
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

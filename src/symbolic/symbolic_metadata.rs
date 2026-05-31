//! Lightweight traversal metadata for symbolic expressions.
//!
//! This module provides a first-level cached traversal that computes several
//! useful properties of an expression in one pass:
//! - whether it contains any variables,
//! - the sorted list of variables,
//! - a structural signature suitable for fast equality heuristics.
//!
//! The cache is external to `Expr` on purpose. This keeps Phase 1 changes low
//! risk while still reducing repeated recursive walks in differentiation,
//! lambdification, and utility methods.

use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashMap;

/// Cached metadata for a single expression node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExprMetadata {
    /// Lightweight structural signature for the expression subtree.
    pub(crate) signature: u64,
    /// True when the subtree contains at least one variable.
    pub(crate) contains_any_variable: bool,
    /// Sorted list of unique variable names found in the subtree.
    pub(crate) variables: Vec<String>,
}

/// External cache for one traversal over an expression tree.
#[derive(Default)]
pub(crate) struct TraversalCache {
    metadata_by_ptr: HashMap<usize, ExprMetadata>,
}

/// Cached structural signatures for consumers that do not need variable sets.
///
/// Code generation CSE is one such consumer: constructing and merging
/// `Vec<String>` metadata for every node makes large residual lowering pay for
/// information it never reads.
#[derive(Default)]
pub(crate) struct SignatureCache {
    signature_by_ptr: HashMap<usize, u64>,
}

impl SignatureCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn signature(&mut self, expr: &Expr) -> u64 {
        let key = expr as *const Expr as usize;
        if let Some(existing) = self.signature_by_ptr.get(&key) {
            return *existing;
        }

        let signature = match expr {
            Expr::Var(name) => combine_signature(1, &[hash_string(name)]),
            Expr::Const(value) => combine_signature(2, &[value.to_bits()]),
            Expr::Add(lhs, rhs) => self.binary_signature(3, lhs, rhs),
            Expr::Sub(lhs, rhs) => self.binary_signature(4, lhs, rhs),
            Expr::Mul(lhs, rhs) => self.binary_signature(5, lhs, rhs),
            Expr::Div(lhs, rhs) => self.binary_signature(6, lhs, rhs),
            Expr::Pow(base, exp) => self.binary_signature(7, base, exp),
            Expr::Exp(inner) => self.unary_signature(8, inner),
            Expr::Ln(inner) => self.unary_signature(9, inner),
            Expr::sin(inner) => self.unary_signature(10, inner),
            Expr::cos(inner) => self.unary_signature(11, inner),
            Expr::tg(inner) => self.unary_signature(12, inner),
            Expr::ctg(inner) => self.unary_signature(13, inner),
            Expr::arcsin(inner) => self.unary_signature(14, inner),
            Expr::arccos(inner) => self.unary_signature(15, inner),
            Expr::arctg(inner) => self.unary_signature(16, inner),
            Expr::arcctg(inner) => self.unary_signature(17, inner),
        };
        self.signature_by_ptr.insert(key, signature);
        signature
    }

    fn unary_signature(&mut self, tag: u64, inner: &Expr) -> u64 {
        let inner_signature = self.signature(inner);
        combine_signature(tag, &[inner_signature])
    }

    fn binary_signature(&mut self, tag: u64, lhs: &Expr, rhs: &Expr) -> u64 {
        let lhs_signature = self.signature(lhs);
        let rhs_signature = self.signature(rhs);
        combine_signature(tag, &[lhs_signature, rhs_signature])
    }
}

impl TraversalCache {
    /// Create a new empty traversal cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Return cached metadata for `expr`, computing it if needed.
    pub(crate) fn metadata(&mut self, expr: &Expr) -> ExprMetadata {
        let key = expr as *const Expr as usize;
        if let Some(existing) = self.metadata_by_ptr.get(&key) {
            return existing.clone();
        }

        let computed = self.compute_metadata(expr);
        self.metadata_by_ptr.insert(key, computed.clone());
        computed
    }

    /// Check if an expression contains a specific variable using cached metadata.
    pub(crate) fn contains_variable(&mut self, expr: &Expr, var_name: &str) -> bool {
        let metadata = self.metadata(expr);
        metadata.contains_any_variable
            && metadata
                .variables
                .binary_search_by(|candidate| candidate.as_str().cmp(var_name))
                .is_ok()
    }

    /// Return the sorted list of unique variables found in an expression.
    pub(crate) fn variables(&mut self, expr: &Expr) -> Vec<String> {
        self.metadata(expr).variables
    }

    /// Return the structural signature of an expression.
    pub(crate) fn signature(&mut self, expr: &Expr) -> u64 {
        self.metadata(expr).signature
    }

    fn compute_metadata(&mut self, expr: &Expr) -> ExprMetadata {
        match expr {
            Expr::Var(name) => ExprMetadata {
                signature: combine_signature(1, &[hash_string(name)]),
                contains_any_variable: true,
                variables: vec![name.clone()],
            },
            Expr::Const(value) => ExprMetadata {
                signature: combine_signature(2, &[value.to_bits()]),
                contains_any_variable: false,
                variables: Vec::new(),
            },
            Expr::Add(lhs, rhs) => self.binary_metadata(3, lhs, rhs),
            Expr::Sub(lhs, rhs) => self.binary_metadata(4, lhs, rhs),
            Expr::Mul(lhs, rhs) => self.binary_metadata(5, lhs, rhs),
            Expr::Div(lhs, rhs) => self.binary_metadata(6, lhs, rhs),
            Expr::Pow(base, exp) => self.binary_metadata(7, base, exp),
            Expr::Exp(inner) => self.unary_metadata(8, inner),
            Expr::Ln(inner) => self.unary_metadata(9, inner),
            Expr::sin(inner) => self.unary_metadata(10, inner),
            Expr::cos(inner) => self.unary_metadata(11, inner),
            Expr::tg(inner) => self.unary_metadata(12, inner),
            Expr::ctg(inner) => self.unary_metadata(13, inner),
            Expr::arcsin(inner) => self.unary_metadata(14, inner),
            Expr::arccos(inner) => self.unary_metadata(15, inner),
            Expr::arctg(inner) => self.unary_metadata(16, inner),
            Expr::arcctg(inner) => self.unary_metadata(17, inner),
        }
    }

    fn unary_metadata(&mut self, tag: u64, inner: &Expr) -> ExprMetadata {
        let child = self.metadata(inner);
        ExprMetadata {
            signature: combine_signature(tag, &[child.signature]),
            contains_any_variable: child.contains_any_variable,
            variables: child.variables,
        }
    }

    fn binary_metadata(&mut self, tag: u64, lhs: &Expr, rhs: &Expr) -> ExprMetadata {
        let left = self.metadata(lhs);
        let right = self.metadata(rhs);
        ExprMetadata {
            signature: combine_signature(tag, &[left.signature, right.signature]),
            contains_any_variable: left.contains_any_variable || right.contains_any_variable,
            variables: merge_sorted_unique(&left.variables, &right.variables),
        }
    }
}

/// Compute a structural signature with a fresh traversal cache.
pub(crate) fn structural_signature(expr: &Expr) -> u64 {
    let mut cache = SignatureCache::new();
    cache.signature(expr)
}

#[cfg(test)]
mod tests {
    use super::{SignatureCache, TraversalCache};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn signature_only_cache_matches_full_traversal_signature() {
        let expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("x_0".to_string())),
                Box::new(Expr::Const(3.5)),
            )),
            Box::new(Expr::Exp(Box::new(Expr::Sub(
                Box::new(Expr::Var("temperature".to_string())),
                Box::new(Expr::Const(298.15)),
            )))),
        );

        let mut full = TraversalCache::new();
        let mut signatures = SignatureCache::new();
        assert_eq!(signatures.signature(&expr), full.signature(&expr));
    }
}

fn merge_sorted_unique(lhs: &[String], rhs: &[String]) -> Vec<String> {
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => {
                merged.push(lhs[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(rhs[j].clone());
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(lhs[i].clone());
                i += 1;
                j += 1;
            }
        }
    }

    while i < lhs.len() {
        merged.push(lhs[i].clone());
        i += 1;
    }

    while j < rhs.len() {
        merged.push(rhs[j].clone());
        j += 1;
    }

    merged
}

fn combine_signature(tag: u64, parts: &[u64]) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15u64 ^ tag;
    for part in parts {
        acc ^= part
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64)
            .wrapping_add(acc << 6)
            .wrapping_add(acc >> 2);
    }
    acc
}

fn hash_string(value: &str) -> u64 {
    let mut acc = 0xCBF2_9CE4_8422_2325u64;
    for byte in value.as_bytes() {
        acc ^= u64::from(*byte);
        acc = acc.wrapping_mul(0x1000_0000_01B3);
    }
    acc
}

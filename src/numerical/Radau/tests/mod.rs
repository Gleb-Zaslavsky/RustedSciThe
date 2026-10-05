//! Thematic story-test layout for the second-generation Radau solver.
//!
//! The archived monolithic tests remain enabled through `Radau_test_old` as a
//! reference. New tests will be split by contract so fast correctness tests
//! stay separate from lifecycle, backend, and performance evidence.

mod aot;
mod api;
mod backends;
mod continuation_stories;
mod correctness;
mod error_contracts;
mod execution_policy;
mod large_performance;
mod lifecycle;
mod native_callbacks;
mod performance;
mod policy_stories;
mod process_isolated;
mod story_support;
mod telemetry_stories;
mod workloads;

/// Versioned outer job metadata and generic result-acceptance checks for external agents.
pub mod agent_job;
/// Typed catalog and comparisons for reported native numerical metrics.
pub mod agent_metrics;
/// Stable model/run identities used by the optional agent worker.
pub mod agent_provenance;
pub mod agent_worker;
/// Bounded worker-owned AOT artifact cache lifecycle.
pub mod agent_worker_cache;
pub mod agent_worker_control;
pub mod bvp_dialogue;
pub mod docx_parser;
pub mod ivp_dialogue;
/// Machine-readable task capabilities, result schema and non-executing JSON preflight.
pub mod task_contract;
/// Strict JSON adapter for the native task-document parsers.
pub mod task_json;
/// parse document with structure like " title1 key1: value1, value2 key2: value2 title2 key3:value3, value4" into HashMap
pub mod task_parser;
pub mod task_parser_bvp;
/// Solver-settings-only adapter for hybrid BVP clients.
pub mod task_parser_bvp_hybrid;
pub mod task_parser_common;
pub mod task_parser_ivp;
/// Hybrid IVP adapter that extracts stiff-solver policy without parsing the
/// caller-owned physical model.
pub mod task_parser_ivp_hybrid;
/// Typed task-document adapter for symbolic nonlinear root and least-squares problems.
pub mod task_parser_nonlinear;
/// Typed task-document adapter for LM, Gavin, and VarPro fitting jobs.
pub mod task_parser_optimization;
///
mod task_parser_tests;
/// Fixed named task configurations for constrained agent workflows.
pub mod task_presets;
pub mod task_runner;
pub mod task_wizard;

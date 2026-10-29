pub mod bvp_dialogue;
pub mod docx_parser;
pub mod ivp_dialogue;
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
///
mod task_parser_tests;
pub mod task_runner;
pub mod task_wizard;

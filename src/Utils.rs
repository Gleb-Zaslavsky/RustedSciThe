//! different utility modules used throughout the project
/// tiny module to save solution into file
pub mod logger;
/// tiny module to plot result of IVP computation
pub mod plots;
/// tiny module for profiling (might be useful for performance monitoring)
pub mod profiling;
/// tiny module to get system information - just a pretty-printing wrapper around famous sys-info crate (might be useful for performance monitoring)
pub mod sys_info;
/// parse document with structure like " title1 key1: value1, value2 key2: value2 title2 key3:value3, value4" into HashMap
pub mod task_parser;
///
mod task_parser_tests;

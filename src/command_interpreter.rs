pub mod bvp_dialogue;
pub mod ivp_dialogue;
/// parse document with structure like " title1 key1: value1, value2 key2: value2 title2 key3:value3, value4" into HashMap
pub mod task_parser;
pub mod task_parser_bvp;
pub mod task_parser_common;
pub mod task_parser_ivp;
///
mod task_parser_tests;
pub mod task_runner;

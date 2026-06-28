// Copyright (c)  by Gleb E. Zaslavkiy
//MIT License
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#[cfg(feature = "openblas-system")]
extern crate openblas_src as _;
pub mod Examples;
pub mod Utils;
pub mod command_interpreter;
pub mod global;
pub mod numerical;
pub mod somelinalg;
pub mod symbolic;
pub mod task_shell_cli;

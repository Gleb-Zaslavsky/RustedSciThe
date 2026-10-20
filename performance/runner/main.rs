//! Opt-in, repository-only performance regression runner.
//!
//! This binary is deliberately outside the production library module tree.
//! Enable it with `--features perf-gates`; ordinary library builds do not
//! compile or execute any performance scenario.

mod baseline;
mod cli;
mod comparison;
mod environment;
mod error;
mod measurement;
mod registry;
mod report;
mod scenarios;

fn main() {
    if let Err(error) = cli::run() {
        eprintln!("perf-gates: {error}");
        std::process::exit(error.exit_code());
    }
}

//! Compatibility namespace for the historical BVP solver implementation.
//!
//! Keeping this namespace in its own file gives rustc and rustfmt the same
//! base directory for the legacy test modules' source paths.

pub use super::BVP_sci_aot;
pub use super::BVP_sci_banded;
pub use super::BVP_sci_bordered_banded;
pub use super::BVP_sci_bordered_solver;
pub use super::BVP_sci_faer;
pub use super::BVP_sci_nalgebra;
pub use super::BVP_sci_numerical;
pub use super::BVP_sci_symb;
pub use super::BVP_sci_symbolic_functions;

#[cfg(test)]
#[path = "BVP_sci_aot_tests.rs"]
mod BVP_sci_aot_tests;
#[cfg(test)]
#[path = "tests/banded_story.rs"]
mod BVP_sci_banded_story_tests;
#[cfg(test)]
#[path = "BVP_sci_banded_tests.rs"]
mod BVP_sci_banded_tests;
#[cfg(test)]
#[path = "BVP_sci_bordered_banded_tests.rs"]
mod BVP_sci_bordered_banded_tests;
#[cfg(test)]
#[path = "BVP_sci_bordered_solver_tests.rs"]
mod BVP_sci_bordered_solver_tests;
#[cfg(test)]
#[path = "BVP_sci_faer_tests.rs"]
mod BVP_sci_faer_tests;
#[cfg(test)]
#[path = "BVP_sci_generated_compare_tests.rs"]
mod BVP_sci_generated_compare_tests;
#[cfg(test)]
#[path = "BVP_sci_nalgebra_tests.rs"]
mod BVP_sci_nalgebra_tests;
#[cfg(test)]
#[path = "BVP_sci_numerical_tests.rs"]
mod BVP_sci_numerical_tests;
#[cfg(test)]
#[path = "BVP_sci_story_tests.rs"]
mod BVP_sci_story_tests;
#[cfg(test)]
#[path = "BVP_sci_symb_tests.rs"]
mod BVP_sci_symb_tests;
#[cfg(test)]
#[path = "BVP_sci_symb_tests2.rs"]
mod BVP_sci_symb_tests2;
#[cfg(test)]
#[path = "tests/common.rs"]
pub(crate) mod test_common;

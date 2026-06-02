//! different utility modules used throughout the project
/// tiny module to save solution into file
pub mod logger;
/// tiny module to plot result of IVP computation
pub mod plots;
/// unified postprocessing facade for solver outputs
pub mod postprocessing;
/// tiny module for profiling (might be useful for performance monitoring)
pub mod profiling;
/// tiny module to get system information - just a pretty-printing wrapper around famous sys-info crate (might be useful for performance monitoring)
pub mod sys_info;

pub mod animation_2d;
pub mod animation_3d;

pub mod bevy_2d;

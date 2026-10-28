//! Per-solver logging scopes for numerical routines.
//!
//! Solver logs are opt-in. The first enabled scope installs a process-wide
//! `log` file sink; later solver scopes share that timestamped file while
//! retaining independent thread-local level filters. This avoids console-only
//! logging and keeps concurrent solver calls from changing each other's level.
//! Runtime sink failures remain queryable through [`solver_log_status`].

use std::cell::Cell;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use chrono::Local;
use log::{Level, LevelFilter, Log, Metadata, Record};

/// Maximum Rust `log` level a numerical solver may emit in its current scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum SolverLogLevel {
    /// Do not emit solver messages. This is the library default.
    #[default]
    Off,
    Error,
    Warn,
    Info,
    Debug,
}

thread_local! {
    static ACTIVE_SOLVER_LOG_LEVEL: Cell<SolverLogLevel> = const { Cell::new(SolverLogLevel::Off) };
}

static FILE_LOGGER: OnceLock<Result<LogSink, SolverLoggingError>> = OnceLock::new();
static LOGGER_FAILURE_REPORTED: AtomicBool = AtomicBool::new(false);

/// Failure while preparing or writing the persistent solver log sink.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolverLoggingError {
    /// The configured log directory or timestamped file could not be created.
    FileCreation(String),
    /// Another component already owns the process-wide `log` facade.
    LoggerAlreadyInstalled,
    /// Writing or flushing the active log file failed after setup.
    WriteFailure { path: PathBuf, message: String },
}

impl std::fmt::Display for SolverLoggingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileCreation(message) => write!(f, "cannot create solver log file: {message}"),
            Self::LoggerAlreadyInstalled => write!(
                f,
                "cannot install the solver file logger because the process already has a global log logger"
            ),
            Self::WriteFailure { path, message } => {
                write!(f, "cannot write solver log '{}': {message}", path.display())
            }
        }
    }
}

impl std::error::Error for SolverLoggingError {}

/// Installs a thread-local logging policy for one solver operation.
///
/// Dropping the guard restores the caller's prior policy, including during
/// early returns and unwinding. The first enabled scope owns the process-wide
/// logger slot; an already-installed host logger is reported as a setup error.
pub struct SolverLogGuard {
    previous: SolverLogLevel,
}

impl SolverLogGuard {
    /// Enters a solver log scope with the requested maximum level.
    ///
    /// Enabled logging lazily installs one process-wide file logger. If setup
    /// fails, logging is disabled for this scope and one diagnostic is written
    /// to stderr; callers that need to handle the failure should use
    /// [`Self::try_enter`].
    pub fn enter(level: SolverLogLevel) -> Self {
        match Self::try_enter(level) {
            Ok(guard) => guard,
            Err(error) => {
                if !LOGGER_FAILURE_REPORTED.swap(true, Ordering::Relaxed) {
                    eprintln!("RustedSciThe solver logging unavailable: {error}");
                }
                Self::enter_scope(SolverLogLevel::Off)
            }
        }
    }

    /// Fallible form of [`Self::enter`], exposing file/logger setup failures.
    pub fn try_enter(level: SolverLogLevel) -> Result<Self, SolverLoggingError> {
        if level != SolverLogLevel::Off {
            ensure_file_logger()?;
        }
        Ok(Self::enter_scope(level))
    }

    fn enter_scope(level: SolverLogLevel) -> Self {
        let previous = ACTIVE_SOLVER_LOG_LEVEL.with(|active| active.replace(level));
        Self { previous }
    }
}

/// Returns the path to the process-wide solver log, if logging has been enabled.
pub fn solver_log_path() -> Option<PathBuf> {
    FILE_LOGGER
        .get()
        .and_then(|result| result.as_ref().ok().map(|sink| sink.path.clone()))
}

/// Report whether the process-wide solver log sink is disabled, active or has
/// encountered an I/O failure. A runtime failure remains observable after the
/// numerical solve succeeds; it does not replace or discard that result.
pub fn solver_log_status() -> Result<Option<PathBuf>, SolverLoggingError> {
    sink_status(FILE_LOGGER.get())
}

fn sink_status(
    result: Option<&Result<LogSink, SolverLoggingError>>,
) -> Result<Option<PathBuf>, SolverLoggingError> {
    let Some(result) = result else {
        return Ok(None);
    };
    let sink = result.as_ref().map_err(Clone::clone)?;
    active_sink_status(&sink.path, &sink.health)
}

fn active_sink_status(
    path: &Path,
    health: &LogSinkHealth,
) -> Result<Option<PathBuf>, SolverLoggingError> {
    match health.write_failure.lock() {
        Ok(failure) => match failure.as_ref() {
            Some(error) => Err(error.clone()),
            None => Ok(Some(path.to_path_buf())),
        },
        Err(_) => Err(SolverLoggingError::WriteFailure {
            path: path.to_path_buf(),
            message: "log health state is poisoned".to_owned(),
        }),
    }
}

fn ensure_file_logger() -> Result<(), SolverLoggingError> {
    let result = FILE_LOGGER.get_or_init(|| {
        let (path, file) = create_log_file()?;
        let health = Arc::new(LogSinkHealth::default());
        if log::set_boxed_logger(Box::new(SolverFileLogger {
            path: path.clone(),
            file: Mutex::new(Box::new(file)),
            health: Arc::clone(&health),
        }))
        .is_err()
        {
            let _ = fs::remove_file(&path);
            return Err(SolverLoggingError::LoggerAlreadyInstalled);
        }
        log::set_max_level(LevelFilter::Debug);
        Ok(LogSink { path, health })
    });

    result.as_ref().map(|_| ()).map_err(Clone::clone)
}

fn create_log_file() -> Result<(PathBuf, File), SolverLoggingError> {
    let directory = match std::env::var_os("RST_SOLVER_LOG_DIR") {
        Some(path) => PathBuf::from(path),
        None => std::env::current_dir()
            .map_err(|error| SolverLoggingError::FileCreation(error.to_string()))?,
    };
    create_log_file_in(&directory)
}

fn create_log_file_in(directory: &Path) -> Result<(PathBuf, File), SolverLoggingError> {
    fs::create_dir_all(directory)
        .map_err(|error| SolverLoggingError::FileCreation(error.to_string()))?;
    let stamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    for sequence in 0..1000_u16 {
        let suffix = if sequence == 0 {
            String::new()
        } else {
            format!("_{sequence:03}")
        };
        let path = directory.join(format!("log_{stamp}{suffix}.txt"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(SolverLoggingError::FileCreation(error.to_string())),
        }
    }
    Err(SolverLoggingError::FileCreation(
        "could not allocate a unique timestamped log filename".to_owned(),
    ))
}

struct LogSink {
    path: PathBuf,
    health: Arc<LogSinkHealth>,
}

#[derive(Default)]
struct LogSinkHealth {
    write_failure: Mutex<Option<SolverLoggingError>>,
    failure_reported: AtomicBool,
}

impl LogSinkHealth {
    fn record_write_failure(&self, error: SolverLoggingError) {
        if let Ok(mut first_failure) = self.write_failure.lock()
            && first_failure.is_none()
        {
            *first_failure = Some(error.clone());
        }
        if !self.failure_reported.swap(true, Ordering::Relaxed) {
            eprintln!("RustedSciThe solver log write failed: {error}");
        }
    }
}

struct SolverFileLogger {
    path: PathBuf,
    file: Mutex<Box<dyn Write + Send>>,
    health: Arc<LogSinkHealth>,
}

impl Log for SolverFileLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Debug && metadata.target().starts_with(env!("CARGO_PKG_NAME"))
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let result = match self.file.lock() {
            Ok(mut file) => {
                let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
                writeln!(
                    file,
                    "{timestamp} {:<5} {}: {}",
                    record.level(),
                    record.target(),
                    record.args()
                )
                .and_then(|()| file.flush())
            }
            Err(_) => Err(std::io::Error::other("solver log file lock is poisoned")),
        };
        if let Err(error) = result {
            self.health
                .record_write_failure(SolverLoggingError::WriteFailure {
                    path: self.path.clone(),
                    message: error.to_string(),
                });
        }
    }

    fn flush(&self) {
        let result = self
            .file
            .lock()
            .map_err(|_| std::io::Error::other("solver log file lock is poisoned"))
            .and_then(|mut file| file.flush());
        if let Err(error) = result {
            self.health
                .record_write_failure(SolverLoggingError::WriteFailure {
                    path: self.path.clone(),
                    message: error.to_string(),
                });
        }
    }
}

impl Drop for SolverLogGuard {
    fn drop(&mut self) {
        ACTIVE_SOLVER_LOG_LEVEL.with(|active| active.set(self.previous));
    }
}

/// Returns whether a message at `level` is allowed by the active solver scope.
pub(crate) fn enabled(level: SolverLogLevel) -> bool {
    ACTIVE_SOLVER_LOG_LEVEL.with(|active| {
        let maximum = active.get();
        maximum != SolverLogLevel::Off && level <= maximum
    })
}

macro_rules! solver_log {
    ($level:expr, $macro:ident, $($arg:tt)*) => {{
        if $crate::numerical::logging::enabled($level) {
            log::$macro!($($arg)*);
        }
    }};
}

macro_rules! solver_error {
    ($($arg:tt)*) => {
        $crate::numerical::logging::solver_log!($crate::numerical::logging::SolverLogLevel::Error, error, $($arg)*)
    };
}

macro_rules! solver_warn {
    ($($arg:tt)*) => {
        $crate::numerical::logging::solver_log!($crate::numerical::logging::SolverLogLevel::Warn, warn, $($arg)*)
    };
}

macro_rules! solver_info {
    ($($arg:tt)*) => {
        $crate::numerical::logging::solver_log!($crate::numerical::logging::SolverLogLevel::Info, info, $($arg)*)
    };
}

macro_rules! solver_debug {
    ($($arg:tt)*) => {
        $crate::numerical::logging::solver_log!($crate::numerical::logging::SolverLogLevel::Debug, debug, $($arg)*)
    };
}

pub(crate) use solver_debug;
pub(crate) use solver_error;
pub(crate) use solver_info;
pub(crate) use solver_log;
pub(crate) use solver_warn;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_file_uses_timestamped_name_and_avoids_collisions() {
        let directory = std::env::temp_dir().join(format!(
            "rustedscithe-logging-test-{}-{}",
            std::process::id(),
            Local::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let (first_path, first_file) = create_log_file_in(&directory).unwrap();
        let (second_path, second_file) = create_log_file_in(&directory).unwrap();

        assert!(
            first_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("log_")
        );
        assert!(
            first_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".txt")
        );
        assert_ne!(first_path, second_path);

        drop((first_file, second_file));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn file_sink_writes_solver_records_and_filters_other_crates() {
        let directory = std::env::temp_dir().join(format!(
            "rustedscithe-logging-sink-test-{}-{}",
            std::process::id(),
            Local::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let (path, file) = create_log_file_in(&directory).unwrap();
        let logger = SolverFileLogger {
            path: path.clone(),
            file: Mutex::new(Box::new(file)),
            health: Arc::new(LogSinkHealth::default()),
        };
        let target = concat!(env!("CARGO_PKG_NAME"), "::test");
        let record = Record::builder()
            .args(format_args!("persistent marker"))
            .level(Level::Info)
            .target(target)
            .build();
        logger.log(&record);
        logger.flush();

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("INFO"));
        assert!(contents.contains("persistent marker"));
        assert!(
            !logger.enabled(
                &Metadata::builder()
                    .level(Level::Info)
                    .target("unrelated_dependency")
                    .build()
            )
        );

        drop(logger);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn runtime_sink_failure_is_retained_as_typed_health_without_solver_failure() {
        struct FailingWriter;

        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected sink failure",
                ))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let path = PathBuf::from("injected-solver-log.txt");
        let health = Arc::new(LogSinkHealth::default());
        let logger = SolverFileLogger {
            path: path.clone(),
            file: Mutex::new(Box::new(FailingWriter)),
            health: Arc::clone(&health),
        };
        assert_eq!(active_sink_status(&path, &health), Ok(Some(path.clone())));
        let record = Record::builder()
            .args(format_args!("this write fails"))
            .level(Level::Info)
            .target(concat!(env!("CARGO_PKG_NAME"), "::test"))
            .build();

        logger.log(&record);

        let expected = SolverLoggingError::WriteFailure {
            path: path.clone(),
            message: "injected sink failure".to_owned(),
        };
        assert_eq!(
            health.write_failure.lock().unwrap().as_ref(),
            Some(&expected)
        );
        assert_eq!(active_sink_status(&path, &health), Err(expected));
    }

    #[test]
    fn public_status_contract_distinguishes_disabled_setup_failure_and_active_sink() {
        assert_eq!(sink_status(None), Ok(None));

        let setup_failure = SolverLoggingError::FileCreation("permission denied".to_owned());
        assert_eq!(
            sink_status(Some(&Err(setup_failure.clone()))),
            Err(setup_failure)
        );

        let path = PathBuf::from("healthy-solver-log.txt");
        let sink = LogSink {
            path: path.clone(),
            health: Arc::new(LogSinkHealth::default()),
        };
        assert_eq!(sink_status(Some(&Ok(sink))), Ok(Some(path)));
    }

    #[test]
    fn scopes_default_off_nest_and_restore_after_drop() {
        assert!(!enabled(SolverLogLevel::Error));
        {
            let _outer = SolverLogGuard::enter_scope(SolverLogLevel::Info);
            assert!(enabled(SolverLogLevel::Warn));
            assert!(!enabled(SolverLogLevel::Debug));
            {
                let _inner = SolverLogGuard::enter_scope(SolverLogLevel::Off);
                assert!(!enabled(SolverLogLevel::Error));
            }
            assert!(enabled(SolverLogLevel::Info));
        }
        assert!(!enabled(SolverLogLevel::Error));
    }
}

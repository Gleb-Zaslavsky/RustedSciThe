use std::fmt;
use std::io;

#[derive(Debug)]
pub enum RunnerError {
    Configuration(String),
    UnknownSelection(String),
    UnknownMachine(String),
    MissingBaseline(String),
    IncompatibleBaseline(String),
    Scenario(String),
    Io(io::Error),
    Toml(toml::de::Error),
    TomlEncode(toml::ser::Error),
    Child(String),
    Batch { exit_code: i32, details: String },
}

impl RunnerError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Configuration(_)
            | Self::Scenario(_)
            | Self::Io(_)
            | Self::Toml(_)
            | Self::TomlEncode(_)
            | Self::Child(_) => 2,
            Self::MissingBaseline(_) | Self::UnknownMachine(_) => 3,
            Self::IncompatibleBaseline(_) => 4,
            Self::UnknownSelection(_) => 2,
            Self::Batch { exit_code, .. } => *exit_code,
        }
    }
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => write!(f, "invalid configuration: {message}"),
            Self::UnknownSelection(message) => write!(f, "unknown selection: {message}"),
            Self::UnknownMachine(message) => write!(f, "unknown machine: {message}"),
            Self::MissingBaseline(message) => write!(f, "no baseline: {message}"),
            Self::IncompatibleBaseline(message) => write!(f, "incompatible baseline: {message}"),
            Self::Scenario(message) => write!(f, "scenario failed: {message}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Toml(error) => write!(f, "invalid TOML: {error}"),
            Self::TomlEncode(error) => write!(f, "could not encode TOML: {error}"),
            Self::Child(message) => write!(f, "isolated scenario process failed: {message}"),
            Self::Batch { details, .. } => {
                write!(f, "one or more selected scenarios failed:\n{details}")
            }
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Toml(error) => Some(error),
            Self::TomlEncode(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for RunnerError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<toml::de::Error> for RunnerError {
    fn from(value: toml::de::Error) -> Self {
        Self::Toml(value)
    }
}
impl From<toml::ser::Error> for RunnerError {
    fn from(value: toml::ser::Error) -> Self {
        Self::TomlEncode(value)
    }
}

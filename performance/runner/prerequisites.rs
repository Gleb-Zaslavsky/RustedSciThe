use crate::registry::Scenario;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolStatus {
    pub name: &'static str,
    pub path: Option<PathBuf>,
}

impl ToolStatus {
    pub fn display(&self) -> String {
        match &self.path {
            Some(path) => format!("{}=available ({})", self.name, path.display()),
            None => format!("{}=MISSING on PATH", self.name),
        }
    }
}

pub fn required_tools(scenario: &Scenario) -> &'static [&'static str] {
    if scenario.execution.starts_with("AOT") {
        &["tcc"]
    } else {
        &[]
    }
}

pub fn status_for_scenario(scenario: &Scenario) -> Vec<ToolStatus> {
    required_tools(scenario)
        .iter()
        .map(|&name| ToolStatus {
            name,
            path: find_program(name),
        })
        .collect()
}

pub fn missing_required_tool(scenario: &Scenario) -> Option<&'static str> {
    required_tools(scenario)
        .iter()
        .copied()
        .find(|name| find_program(name).is_none())
}

#[cfg(test)]
fn missing_required_tool_with(
    scenario: &Scenario,
    mut resolve: impl FnMut(&str) -> Option<PathBuf>,
) -> Option<&'static str> {
    required_tools(scenario)
        .iter()
        .copied()
        .find(|name| resolve(name).is_none())
}

#[cfg(test)]
fn validate_scenario_with(
    scenario: &Scenario,
    resolve: impl FnMut(&str) -> Option<PathBuf>,
) -> Result<(), String> {
    match missing_required_tool_with(scenario, resolve) {
        Some(tool) => Err(format!(
            "{} requires `{tool}` on PATH; install it or exclude this AOT scenario",
            scenario.id
        )),
        None => Ok(()),
    }
}

fn find_program(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    let extensions = executable_extensions();
    find_program_in_path(program, &path, &extensions)
}

fn executable_extensions() -> Vec<String> {
    #[cfg(windows)]
    {
        env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into())
            .split(';')
            .filter(|extension| !extension.is_empty())
            .map(str::to_owned)
            .collect()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

fn find_program_in_path(
    program: &str,
    path: &std::ffi::OsStr,
    extensions: &[String],
) -> Option<PathBuf> {
    let has_extension = Path::new(program).extension().is_some();
    let mut names = vec![program.to_owned()];
    if !has_extension {
        names.extend(
            extensions
                .iter()
                .map(|extension| format!("{program}{extension}")),
        );
    }

    for directory in env::split_paths(path) {
        for name in &names {
            let candidate = directory.join(name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(windows)]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(any(windows, unix)))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::{find_program_in_path, required_tools, validate_scenario_with};
    use crate::registry::SCENARIOS;
    use std::ffi::OsStr;

    #[test]
    fn every_aot_scenario_requires_tcc_and_other_routes_do_not() {
        let mut aot_count = 0;
        for scenario in SCENARIOS.iter() {
            if scenario.execution.starts_with("AOT") {
                aot_count += 1;
                assert_eq!(required_tools(scenario), &["tcc"], "{}", scenario.id);
            } else {
                assert!(required_tools(scenario).is_empty(), "{}", scenario.id);
            }
        }
        assert!(aot_count > 0);
    }

    #[test]
    fn path_lookup_handles_windows_extensions_and_missing_tools() {
        let directory = tempfile::tempdir().unwrap();
        let compiler = directory.path().join("tcc.exe");
        std::fs::write(&compiler, "stub").unwrap();
        let path = std::env::join_paths([directory.path()]).unwrap();
        let extensions = vec![".exe".to_owned(), ".cmd".to_owned()];

        let resolved = find_program_in_path("tcc", &path, &extensions).unwrap();
        assert!(resolved.is_file());
        assert_eq!(
            resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_ascii_lowercase(),
            compiler
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_ascii_lowercase()
        );
        assert_eq!(
            find_program_in_path("missing-tool", &path, &extensions),
            None
        );
        assert_eq!(
            find_program_in_path("tcc", OsStr::new(""), &extensions),
            None
        );
    }

    #[test]
    fn missing_aot_compiler_is_reported_before_scenario_execution() {
        let scenario = SCENARIOS
            .iter()
            .find(|scenario| scenario.execution.starts_with("AOT"))
            .expect("registry has AOT scenarios");
        let message = validate_scenario_with(scenario, |_| None).unwrap_err();
        assert!(message.contains(scenario.id));
        assert!(message.contains("tcc"));
    }
}

use crate::error::RunnerError;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineProfile {
    pub alias: String,
    pub installation_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentFingerprint {
    pub id: String,
    pub target: String,
    pub profile: String,
    pub cpu: String,
    pub rustc: String,
    pub cpu_flags: String,
    pub worker_config: String,
    pub compiled_features: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub lockfile_fingerprint: String,
}

pub fn register(alias: &str) -> Result<MachineProfile, RunnerError> {
    validate_component(alias)?;
    let profile_path = machine_profile_path(alias)?;
    let local_path = local_identity_path()?;
    register_at_paths(alias, &profile_path, &local_path)
}

fn register_at_paths(
    alias: &str,
    profile_path: &Path,
    local_path: &Path,
) -> Result<MachineProfile, RunnerError> {
    validate_component(alias)?;
    let profile_id = read_profile_id(profile_path, alias)?;
    let installation_id = if local_path.exists() {
        let local = fs::read_to_string(&local_path)?.trim().to_owned();
        if profile_id
            .as_deref()
            .is_some_and(|profile| profile != local)
        {
            return Err(RunnerError::UnknownMachine(format!(
                "profile {alias:?} belongs to another installation; register this host with a new alias"
            )));
        }
        local
    } else {
        if profile_id.is_some() {
            return Err(RunnerError::UnknownMachine(format!(
                "profile {alias:?} already exists on another installation; choose a new alias"
            )));
        }
        generate_installation_id()
    };

    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_new_or_same(&local_path, &installation_id)?;

    if let Some(parent) = profile_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_new_or_same(
        &profile_path,
        &format!(
            "schema_version = 1\nalias = {:?}\ninstallation_id = {:?}\n",
            alias, installation_id
        ),
    )?;
    Ok(MachineProfile {
        alias: alias.to_owned(),
        installation_id,
    })
}

pub fn resolve(alias: &str) -> Result<MachineProfile, RunnerError> {
    validate_component(alias)?;
    let path = machine_profile_path(alias)?;
    let identity_path = local_identity_path()?;
    resolve_at_paths(alias, &path, &identity_path)
}

fn resolve_at_paths(
    alias: &str,
    profile_path: &Path,
    identity_path: &Path,
) -> Result<MachineProfile, RunnerError> {
    validate_component(alias)?;
    let profile_id = read_profile_id(profile_path, alias)?.ok_or_else(|| {
        RunnerError::UnknownMachine(format!(
            "profile {alias:?} is not registered; run `machine register`"
        ))
    })?;
    let local = fs::read_to_string(identity_path).map_err(|_| {
        RunnerError::UnknownMachine("this installation has no local machine identity".into())
    })?;
    let local = local.trim();
    if local != profile_id {
        return Err(RunnerError::UnknownMachine(format!(
            "profile {alias:?} belongs to a different installation"
        )));
    }
    Ok(MachineProfile {
        alias: alias.into(),
        installation_id: profile_id,
    })
}

pub fn fingerprint() -> EnvironmentFingerprint {
    let mut system = sysinfo::System::new();
    system.refresh_cpu_all();
    let cpu = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_owned())
        .filter(|brand| !brand.is_empty())
        .unwrap_or_else(|| "unknown-cpu".into());
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
    .to_owned();
    let target = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "rustc-version-unavailable".into());
    // Runtime RUSTFLAGS do not prove how this executable was compiled. Record
    // target features enabled in the binary rather than guessing from the shell.
    let cpu_flags = enabled_target_features();
    let worker_config = rayon::current_num_threads().to_string();
    let mut features = Vec::new();
    if cfg!(feature = "superlu") {
        features.push("superlu");
    }
    if cfg!(feature = "arrayfire") {
        features.push("arrayfire");
    }
    if cfg!(feature = "openblas-system") {
        features.push("openblas-system");
    }
    let compiled_features = features.join(",");
    let (source_revision, source_dirty) = source_state(Path::new(env!("CARGO_MANIFEST_DIR")));
    let lockfile_fingerprint = lockfile_fingerprint();
    // Stable FNV-1a over hard compatibility fields. Compiler version remains
    // visible provenance, but changing it does not hide the old baseline.
    let id = compatibility_id(
        &target,
        &profile,
        &cpu,
        &cpu_flags,
        &worker_config,
        &compiled_features,
    );
    EnvironmentFingerprint {
        id,
        target,
        profile,
        cpu,
        rustc,
        cpu_flags,
        worker_config,
        compiled_features,
        source_revision,
        source_dirty,
        lockfile_fingerprint,
    }
}

fn source_state(root: &Path) -> (String, bool) {
    let revision = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "git-unavailable".into());
    let dirty = source_is_dirty(root);
    (revision, dirty)
}

fn source_is_dirty(root: &Path) -> bool {
    // Baselines are generated measurement artifacts, not source inputs. Ignore
    // them so a multi-solver recording session does not require a commit after
    // every solver. Changes to runner code, scenarios, profiles, and all other
    // repository files remain part of the dirty-source check.
    Command::new("git")
        .args(["-C"])
        .arg(root)
        .args([
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)performance/baselines/**",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(true)
}

fn lockfile_fingerprint() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let Ok(contents) = fs::read(path) else {
        return "unavailable".into();
    };
    let hash = contents.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("fnv1a-{hash:016x}")
}

fn enabled_target_features() -> String {
    let mut features = Vec::new();
    macro_rules! push_feature {
        ($feature:literal) => {
            if cfg!(target_feature = $feature) {
                features.push($feature);
            }
        };
    }
    push_feature!("sse");
    push_feature!("sse2");
    push_feature!("sse3");
    push_feature!("ssse3");
    push_feature!("sse4.1");
    push_feature!("sse4.2");
    push_feature!("avx");
    push_feature!("avx2");
    push_feature!("avx512f");
    push_feature!("fma");
    push_feature!("bmi1");
    push_feature!("bmi2");
    push_feature!("neon");
    push_feature!("simd128");
    if features.is_empty() {
        "baseline-target-features".into()
    } else {
        features.join(",")
    }
}

fn compatibility_id(
    target: &str,
    profile: &str,
    cpu: &str,
    cpu_flags: &str,
    worker_config: &str,
    compiled_features: &str,
) -> String {
    let key =
        format!("{target}\n{profile}\n{cpu}\n{cpu_flags}\n{worker_config}\n{compiled_features}");
    let hash = key.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{}-{}-{hash:016x}", slug(target), profile)
}

fn machine_profile_path(alias: &str) -> Result<PathBuf, RunnerError> {
    validate_component(alias)?;
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("machines")
        .join(format!("{alias}.toml")))
}

pub fn find_local_profile() -> Option<MachineProfile> {
    let local_id = fs::read_to_string(local_identity_path().ok()?).ok()?;
    let profiles = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("machines");
    for entry in fs::read_dir(profiles).ok()? {
        let entry = entry.ok()?;
        if entry
            .path()
            .extension()
            .is_none_or(|extension| extension != "toml")
        {
            continue;
        }
        let contents = fs::read_to_string(entry.path()).ok()?;
        let value = toml::Value::Table(toml::from_str::<toml::Table>(&contents).ok()?);
        let id = value.get("installation_id")?.as_str()?;
        if id != local_id.trim() {
            continue;
        }
        let alias = value.get("alias")?.as_str()?.to_owned();
        return Some(MachineProfile {
            alias,
            installation_id: id.to_owned(),
        });
    }
    None
}

fn local_identity_path() -> Result<PathBuf, RunnerError> {
    #[cfg(windows)]
    let root = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| RunnerError::Configuration("APPDATA is not set".into()))?;
    #[cfg(not(windows))]
    let root = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(
            std::env::var_os("HOME")
                .ok_or_else(|| RunnerError::Configuration("HOME is not set".into()))?,
        )
        .join(".config")
    };
    Ok(root
        .join("RustedSciThe")
        .join("perf-gates")
        .join("installation-id"))
}

fn read_profile_id(path: &Path, expected_alias: &str) -> Result<Option<String>, RunnerError> {
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)?;
    let value = toml::Value::Table(toml::from_str::<toml::Table>(&contents)?);
    let alias = value
        .get("alias")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| RunnerError::Configuration(format!("{} lacks alias", path.display())))?;
    if alias != expected_alias {
        return Err(RunnerError::UnknownMachine(format!(
            "profile file {} declares alias {alias:?}, expected {expected_alias:?}",
            path.display()
        )));
    }
    value
        .get("installation_id")
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .map(Some)
        .ok_or_else(|| {
            RunnerError::Configuration(format!("{} lacks installation_id", path.display()))
        })
}

fn write_new_or_same(path: &Path, contents: &str) -> Result<(), RunnerError> {
    if path.exists() {
        let current = fs::read_to_string(path)?;
        let current_id = if path.extension().is_some_and(|ext| ext == "toml") {
            toml::from_str::<toml::Table>(&current)?
                .get("installation_id")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        } else {
            Some(current.trim().to_owned())
        };
        let requested_id = if path.extension().is_some_and(|ext| ext == "toml") {
            toml::from_str::<toml::Table>(contents)?
                .get("installation_id")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        } else {
            Some(contents.trim().to_owned())
        };
        if current_id != requested_id {
            return Err(RunnerError::UnknownMachine(format!(
                "refusing to overwrite machine identity at {}",
                path.display()
            )));
        }
        return Ok(());
    }
    fs::write(path, contents)?;
    Ok(())
}

fn validate_component(value: &str) -> Result<(), RunnerError> {
    if value.is_empty()
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
    {
        return Err(RunnerError::Configuration(format!(
            "invalid machine alias {value:?}"
        )));
    }
    Ok(())
}

fn generate_installation_id() -> String {
    use rand::RngExt;
    let random: u128 = rand::rng().random();
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:032x}{:016x}", random, time)
}

fn slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        compatibility_id, machine_profile_path, register_at_paths, resolve_at_paths,
        source_is_dirty, validate_component,
    };
    use std::path::Path;
    use std::process::Command;

    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    }

    #[test]
    fn compatibility_id_is_stable_for_equal_hard_environment_fields() {
        let first = compatibility_id(
            "windows-x86_64",
            "release",
            "Example CPU",
            "avx2,fma",
            "8",
            "superlu",
        );
        assert_eq!(
            first,
            compatibility_id(
                "windows-x86_64",
                "release",
                "Example CPU",
                "avx2,fma",
                "8",
                "superlu",
            )
        );
        assert_ne!(
            first,
            compatibility_id(
                "windows-x86_64",
                "release",
                "Example CPU",
                "avx2,fma",
                "4",
                "superlu",
            )
        );
    }

    #[test]
    fn machine_alias_rejects_path_components_and_platform_separators() {
        for alias in ["", "..", "../escape", "..\\escape", "nested/name", "a:b"] {
            assert!(validate_component(alias).is_err(), "accepted {alias:?}");
            assert!(
                machine_profile_path(alias).is_err(),
                "built path for {alias:?}"
            );
        }
        let valid = machine_profile_path("windows_dev-1").unwrap();
        assert!(valid.ends_with("windows_dev-1.toml"));
    }

    #[test]
    fn registration_rejects_an_alias_bound_to_another_installation() {
        let directory = tempfile::tempdir().unwrap();
        let profile_path = directory.path().join("machines/host.toml");
        let identity_path = directory.path().join("config/installation-id");
        std::fs::create_dir_all(profile_path.parent().unwrap()).unwrap();
        std::fs::create_dir_all(identity_path.parent().unwrap()).unwrap();
        std::fs::write(
            &profile_path,
            r#"schema_version = 1
alias = "host"
installation_id = "install-A"
"#,
        )
        .unwrap();
        std::fs::write(&identity_path, "install-B\n").unwrap();

        let error = register_at_paths("host", &profile_path, &identity_path).unwrap_err();
        assert!(error.to_string().contains("another installation"));
        assert_eq!(
            std::fs::read_to_string(&identity_path).unwrap(),
            "install-B\n"
        );
    }

    #[test]
    fn profile_alias_and_local_installation_must_both_match() {
        let directory = tempfile::tempdir().unwrap();
        let profile_path = directory.path().join("machines/requested.toml");
        let identity_path = directory.path().join("config/installation-id");
        std::fs::create_dir_all(profile_path.parent().unwrap()).unwrap();
        std::fs::create_dir_all(identity_path.parent().unwrap()).unwrap();
        std::fs::write(
            &profile_path,
            r#"schema_version = 1
alias = "different"
installation_id = "install-A"
"#,
        )
        .unwrap();
        std::fs::write(&identity_path, "install-A\n").unwrap();

        let error = resolve_at_paths("requested", &profile_path, &identity_path).unwrap_err();
        assert!(error.to_string().contains("expected \"requested\""));

        std::fs::write(
            &profile_path,
            r#"schema_version = 1
alias = "requested"
installation_id = "install-A"
"#,
        )
        .unwrap();
        std::fs::write(&identity_path, "install-B\n").unwrap();
        let error = resolve_at_paths("requested", &profile_path, &identity_path).unwrap_err();
        assert!(error.to_string().contains("different installation"));
    }

    #[test]
    fn generated_baselines_do_not_make_source_dirty_but_code_changes_do() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        git(root, &["init", "--quiet"]);
        git(
            root,
            &["config", "user.email", "perf-gates@example.invalid"],
        );
        git(root, &["config", "user.name", "perf-gates test"]);
        std::fs::write(root.join("Cargo.toml"), "[package]\nname='fixture'\n").unwrap();
        git(root, &["add", "Cargo.toml"]);
        git(root, &["commit", "--quiet", "-m", "fixture"]);

        assert!(!source_is_dirty(root));
        let baseline = root.join("performance/baselines/windows/scenario.toml");
        std::fs::create_dir_all(baseline.parent().unwrap()).unwrap();
        std::fs::write(&baseline, "samples = [1.0, 2.0, 3.0]\n").unwrap();
        assert!(!source_is_dirty(root));
        git(root, &["add", "performance/baselines"]);
        assert!(!source_is_dirty(root));

        std::fs::write(root.join("new-source.rs"), "fn changed() {}\n").unwrap();
        assert!(source_is_dirty(root));
    }
}

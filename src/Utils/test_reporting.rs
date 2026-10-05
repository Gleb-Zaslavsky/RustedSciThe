//! Small, opt-in reporting support for verbose integration and story tests.
//!
//! Reports are written after the measured operation has finished. The helper
//! is intentionally outside solver code so file-system work cannot enter a
//! solver timer or a production hot path. Each profile-qualified canonical
//! test name maps to a single canonical report file; a later run replaces that
//! profile's file and records a new UTC timestamp. Release writes also keep an
//! immutable copy below the profile's `archive/` directory.

use chrono::{SecondsFormat, Utc};
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tabled::{Table, Tabled};

static REPORT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static ACTIVE_CAPTURE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Writes one dated report for a named test and returns the resulting path.
///
/// RST_TEST_REPORT_DIR can override the repository-local test_reports
/// directory. Reports are partitioned by the Cargo profile (`debug` or
/// `release`) so a smoke run cannot replace a release baseline. Release
/// reports are additionally copied to `profile/archive/` with a UTC timestamp.
/// Set `RST_TEST_REPORT_ARCHIVE=always` to archive debug reports too, or use
/// `never` to disable archival for a local run. The optional
/// RST_TEST_REPORT_PROFILE override is useful for process-isolated harnesses.
pub fn write_test_report(
    suite: &str,
    canonical_test_name: &str,
    body: &str,
) -> io::Result<PathBuf> {
    let root = std::env::var_os("RST_TEST_REPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_reports"));
    write_test_report_in(&root, suite, canonical_test_name, body)
}

/// Writes a replaceable progress snapshot without creating an archive entry.
///
/// Long-running benchmark drivers use this for live compact tables. The final
/// report should still be written through [`write_test_report`] so release
/// archival remains atomic and immutable.
pub fn write_test_report_snapshot(
    suite: &str,
    canonical_test_name: &str,
    body: &str,
) -> io::Result<PathBuf> {
    let root = std::env::var_os("RST_TEST_REPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_reports"));
    let suite = sanitize_component(suite);
    let test_name = format!("{}__progress", sanitize_component(canonical_test_name));
    let profile = report_profile();
    write_test_report_in_profile(
        &root,
        &suite,
        &test_name,
        profile,
        canonical_test_name,
        body,
        false,
    )
}

/// Variant with an explicit root, primarily useful for utility tests.
pub fn write_test_report_in(
    root: &Path,
    suite: &str,
    canonical_test_name: &str,
    body: &str,
) -> io::Result<PathBuf> {
    let suite = sanitize_component(suite);
    let test_name = sanitize_component(canonical_test_name);
    let profile = report_profile();
    write_test_report_in_profile(
        root,
        &suite,
        &test_name,
        profile,
        canonical_test_name,
        body,
        archive_enabled(profile),
    )
}

fn write_test_report_in_profile(
    root: &Path,
    suite: &str,
    test_name: &str,
    profile: &str,
    canonical_test_name: &str,
    body: &str,
    archive: bool,
) -> io::Result<PathBuf> {
    let directory = root.join(&suite).join(profile);
    fs::create_dir_all(&directory)?;

    let path = directory.join(format!("{test_name}.md"));
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
    let payload = format!(
        "# Test report: {canonical_test_name}\n\n- suite: {suite}\n- profile: {profile}\n- recorded_at_utc: {timestamp}\n- canonical_test: {canonical_test_name}\n\n{body}\n"
    );

    // Write through a sibling temporary file so a reader never observes a
    // partially written report. Windows cannot rename over an existing
    // destination, therefore remove only this report's old file before the
    // final rename.
    let temporary = directory.join(format!(".{test_name}.tmp-{}", report_suffix()));
    fs::write(&temporary, &payload)?;
    if path.exists() {
        fs::remove_file(&path)?;
    }
    fs::rename(&temporary, &path)?;

    if archive {
        write_immutable_archive(&directory, test_name, &timestamp, payload.as_bytes())?;
    }

    Ok(path)
}

fn write_immutable_archive(
    directory: &Path,
    test_name: &str,
    timestamp: &str,
    payload: &[u8],
) -> io::Result<()> {
    let archive_directory = directory.join("archive");
    fs::create_dir_all(&archive_directory)?;
    let archive_stamp = timestamp.replace(':', "-");
    let base_name = format!("{test_name}__{archive_stamp}");

    for collision in 0..1000u32 {
        let suffix = if collision == 0 {
            String::new()
        } else {
            format!("__{collision}")
        };
        let path = archive_directory.join(format!("{base_name}{suffix}.md"));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(payload)?;
                file.flush()?;
                return Ok(());
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "too many report archive filename collisions",
    ))
}

/// Captures verbose test output and writes it as a dated report on drop.
///
/// The capture is thread-local and deliberately lives in the test utility
/// layer. It does not add any work to solver code or to measured benchmark
/// regions. The output is still forwarded to stdout immediately, so existing
/// `--nocapture` workflows keep their behaviour.
pub struct TestReportCapture {
    root: Option<PathBuf>,
    suite: String,
    canonical_test_name: String,
    previous: Option<String>,
}

impl TestReportCapture {
    /// Starts capturing output for one canonical test name.
    pub fn new(suite: impl Into<String>, canonical_test_name: impl Into<String>) -> Self {
        Self::new_with_root(None, suite, canonical_test_name)
    }

    /// Starts capturing output with an explicit report root.
    ///
    /// This is primarily useful for isolated utility tests; normal suites
    /// should use [`Self::new`] so `RST_TEST_REPORT_DIR` remains supported.
    pub fn new_in(
        root: impl Into<PathBuf>,
        suite: impl Into<String>,
        canonical_test_name: impl Into<String>,
    ) -> Self {
        Self::new_with_root(Some(root.into()), suite, canonical_test_name)
    }

    fn new_with_root(
        root: Option<PathBuf>,
        suite: impl Into<String>,
        canonical_test_name: impl Into<String>,
    ) -> Self {
        let previous = ACTIVE_CAPTURE.with(|active| active.borrow_mut().replace(String::new()));
        Self {
            root,
            suite: suite.into(),
            canonical_test_name: canonical_test_name.into(),
            previous,
        }
    }
}

impl Drop for TestReportCapture {
    fn drop(&mut self) {
        let body = ACTIVE_CAPTURE
            .with(|active| active.borrow_mut().take())
            .unwrap_or_default();
        let status = if std::thread::panicking() {
            "failed"
        } else {
            "passed"
        };
        let body = format!("status: {status}\n\n{body}");
        let result = self
            .root
            .as_deref()
            .map(|root| write_test_report_in(root, &self.suite, &self.canonical_test_name, &body))
            .unwrap_or_else(|| write_test_report(&self.suite, &self.canonical_test_name, &body));
        if let Err(error) = result {
            eprintln!(
                "[test report] unable to write {}::{}: {error}",
                self.suite, self.canonical_test_name
            );
        }
        ACTIVE_CAPTURE.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

/// Forwards one already formatted test line to stdout and the active report.
///
/// Set `RST_TEST_REPORT_STDOUT=off` (or `false`/`0`) for a large release run
/// when the report file, rather than the terminal, should be the diagnostic
/// output. The report capture remains active in that mode.
pub fn capture_test_line(args: fmt::Arguments<'_>) {
    capture_test_block(args.to_string());
}

/// Captures a preformatted multi-line block as one reporting event.
///
/// This is useful for tables: the caller can collect rows during the test and
/// emit one compact artifact after the measured work has completed.
pub fn capture_test_block(block: impl AsRef<str>) {
    let block = block.as_ref();
    if report_stdout_enabled() {
        std::println!("{block}");
    }
    ACTIVE_CAPTURE.with(|active| {
        if let Some(body) = active.borrow_mut().as_mut() {
            body.push_str(block);
            body.push('\n');
        }
    });
}

/// Captures a compact [`tabled`] table in the active report.
///
/// Rows should normally be accumulated in a local `Vec` while the expensive
/// operation runs. Formatting is performed only once, outside that operation,
/// and is therefore not part of a solver or callback timing scope.
pub fn capture_test_table<T: Tabled>(title: impl AsRef<str>, rows: &[T]) {
    let table = if rows.is_empty() {
        "(no rows)".to_owned()
    } else {
        Table::new(rows).to_string()
    };
    capture_test_block(format!("{}\n{}", title.as_ref(), table));
}

fn report_stdout_enabled() -> bool {
    !matches!(
        std::env::var("RST_TEST_REPORT_STDOUT").as_deref(),
        Ok("off") | Ok("false") | Ok("0")
    )
}

fn sanitize_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unnamed".to_string()
    } else {
        sanitized
    }
}

fn report_profile() -> &'static str {
    if let Ok(profile) = std::env::var("RST_TEST_REPORT_PROFILE") {
        if profile.eq_ignore_ascii_case("release") {
            return "release";
        }
        if profile.eq_ignore_ascii_case("debug") {
            return "debug";
        }
    }
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn archive_enabled(profile: &str) -> bool {
    match std::env::var("RST_TEST_REPORT_ARCHIVE") {
        Ok(value) if value.eq_ignore_ascii_case("always") => true,
        Ok(value) if value.eq_ignore_ascii_case("never") => false,
        Ok(value) if value.eq_ignore_ascii_case("debug") => profile == "debug",
        Ok(value) if value.eq_ignore_ascii_case("release") => profile == "release",
        Ok(value) if value.eq_ignore_ascii_case("true") => true,
        Ok(value) if value.eq_ignore_ascii_case("false") => false,
        _ => profile == "release",
    }
}

fn report_suffix() -> String {
    let sequence = REPORT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{}-{sequence}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::{
        TestReportCapture, capture_test_line, capture_test_table, sanitize_component,
        write_test_report_in, write_test_report_in_profile,
    };
    use std::fs;

    #[test]
    fn canonical_name_is_safe_and_report_is_replaced() {
        let root =
            std::env::temp_dir().join(format!("rustedscithe-test-report-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let first = write_test_report_in(&root, "BVP_Damp", "module::tests::story", "first")
            .expect("first report should be written");
        let second = write_test_report_in(&root, "BVP_Damp", "module::tests::story", "second")
            .expect("second report should replace the first");
        assert_eq!(first, second);
        assert!(fs::read_to_string(second).unwrap().contains("second"));
        assert_eq!(sanitize_component("a::b/c"), "a__b_c");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn capture_forwards_output_and_writes_report_on_drop() {
        let root = std::env::temp_dir().join(format!(
            "rustedscithe-test-report-capture-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        {
            let _capture = TestReportCapture::new_in(&root, "BVP_Damp_AOT", "module::tests::aot");
            capture_test_line(format_args!("AOT row: {}", 7));
        }
        let profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let report = root
            .join("BVP_Damp_AOT")
            .join(profile)
            .join("module__tests__aot.md");
        let contents = fs::read_to_string(report).expect("captured report should exist");
        assert!(contents.contains("status: passed"));
        assert!(contents.contains("AOT row: 7"));
        assert!(contents.contains("- profile: "));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn release_report_keeps_an_immutable_archive_copy() {
        let root = std::env::temp_dir().join(format!(
            "rustedscithe-test-report-archive-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);

        let canonical = write_test_report_in_profile(
            &root,
            "LSODE2_AOT",
            "module__tests__aot",
            "release",
            "module::tests::aot",
            "release row",
            true,
        )
        .expect("release report should be written");
        let archive_directory = canonical.parent().unwrap().join("archive");
        let archived: Vec<_> = fs::read_dir(&archive_directory)
            .expect("archive directory should exist")
            .map(|entry| entry.expect("archive entry should be readable").path())
            .collect();
        assert_eq!(archived.len(), 1);
        assert!(
            fs::read_to_string(&archived[0])
                .unwrap()
                .contains("release row")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn table_capture_writes_one_compact_block() {
        #[derive(tabled::Tabled)]
        struct Row {
            route: &'static str,
            elapsed_ms: &'static str,
        }

        let root = std::env::temp_dir().join(format!(
            "rustedscithe-test-report-table-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        {
            let _capture = TestReportCapture::new_in(&root, "Radau_Large", "table_story");
            capture_test_table(
                "summary",
                &[Row {
                    route: "ExprLegacy",
                    elapsed_ms: "1.25",
                }],
            );
        }
        let report = root
            .join("Radau_Large")
            .join("debug")
            .join("table_story.md");
        let contents = fs::read_to_string(report).expect("table report should exist");
        assert!(contents.contains("summary"));
        assert!(contents.contains("ExprLegacy"));
        assert!(contents.contains("elapsed_ms"));
        let _ = fs::remove_dir_all(root);
    }
}

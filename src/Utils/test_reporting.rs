//! Small, opt-in reporting support for verbose integration and story tests.
//!
//! Reports are written after the measured operation has finished. The helper
//! is intentionally outside solver code so file-system work cannot enter a
//! solver timer or a production hot path. Each canonical test name maps to a
//! single report file; a later run replaces that file and records a new UTC
//! timestamp.

use chrono::{SecondsFormat, Utc};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

thread_local! {
    static ACTIVE_CAPTURE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Writes one dated report for a named test and returns the resulting path.
///
/// RST_TEST_REPORT_DIR can override the repository-local test_reports
/// directory. The suite and test name are sanitized before becoming path
/// components, so Rust module paths remain safe on Windows and Unix.
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

/// Variant with an explicit root, primarily useful for utility tests.
pub fn write_test_report_in(
    root: &Path,
    suite: &str,
    canonical_test_name: &str,
    body: &str,
) -> io::Result<PathBuf> {
    let suite = sanitize_component(suite);
    let test_name = sanitize_component(canonical_test_name);
    let directory = root.join(&suite);
    fs::create_dir_all(&directory)?;

    let path = directory.join(format!("{test_name}.md"));
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
    let payload = format!(
        "# Test report: {canonical_test_name}\n\n- suite: {suite}\n- recorded_at_utc: {timestamp}\n- canonical_test: {canonical_test_name}\n\n{body}\n"
    );

    // Write through a sibling temporary file so a reader never observes a
    // partially written report. Windows cannot rename over an existing
    // destination, therefore remove only this report's old file before the
    // final rename.
    let temporary = directory.join(format!(".{test_name}.tmp-{}", std::process::id()));
    fs::write(&temporary, payload)?;
    if path.exists() {
        fs::remove_file(&path)?;
    }
    fs::rename(&temporary, &path)?;
    Ok(path)
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
/// Test modules normally expose this through a local `println!` macro so
/// existing diagnostic output needs no second formatting path.
pub fn capture_test_line(args: fmt::Arguments<'_>) {
    let line = args.to_string();
    std::println!("{line}");
    ACTIVE_CAPTURE.with(|active| {
        if let Some(body) = active.borrow_mut().as_mut() {
            body.push_str(&line);
            body.push('\n');
        }
    });
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

#[cfg(test)]
mod tests {
    use super::{TestReportCapture, capture_test_line, sanitize_component, write_test_report_in};
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
        let report = root.join("BVP_Damp_AOT").join("module__tests__aot.md");
        let contents = fs::read_to_string(report).expect("captured report should exist");
        assert!(contents.contains("status: passed"));
        assert!(contents.contains("AOT row: 7"));
        let _ = fs::remove_dir_all(root);
    }
}

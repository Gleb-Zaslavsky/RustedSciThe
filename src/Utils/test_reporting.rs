//! Small, opt-in reporting support for verbose integration and story tests.
//!
//! Reports are written after the measured operation has finished. The helper
//! is intentionally outside solver code so file-system work cannot enter a
//! solver timer or a production hot path. Each canonical test name maps to a
//! single report file; a later run replaces that file and records a new UTC
//! timestamp.

use chrono::{SecondsFormat, Utc};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

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
    use super::{sanitize_component, write_test_report_in};
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
}

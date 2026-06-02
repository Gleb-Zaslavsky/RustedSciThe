use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot, plots_terminal};
use nalgebra::{DMatrix, DVector};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct PostprocessDataset {
    pub axis_name: String,
    pub variable_names: Vec<String>,
    pub axis: DVector<f64>,
    pub values: DMatrix<f64>,
    pub metadata: Vec<(String, String)>,
}

impl PostprocessDataset {
    pub fn new(
        axis_name: impl Into<String>,
        variable_names: Vec<String>,
        axis: DVector<f64>,
        values: DMatrix<f64>,
    ) -> Result<Self, PostprocessError> {
        let dataset = Self {
            axis_name: axis_name.into(),
            variable_names,
            axis,
            values,
            metadata: Vec::new(),
        };
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    pub fn validate(&self) -> Result<(), PostprocessError> {
        if self.axis.len() != self.values.nrows() {
            return Err(PostprocessError::InvalidDataset(format!(
                "axis length {} does not match value rows {}",
                self.axis.len(),
                self.values.nrows()
            )));
        }
        if self.variable_names.len() != self.values.ncols() {
            return Err(PostprocessError::InvalidDataset(format!(
                "variable count {} does not match value columns {}",
                self.variable_names.len(),
                self.values.ncols()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostprocessAction {
    SaveTxt { path: PathBuf },
    SaveCsv { path: PathBuf },
    PlottersPng { output_dir: PathBuf },
    GnuplotPng { output_dir: PathBuf },
    TerminalPlot,
    WriteReport { path: PathBuf },
}

#[derive(Debug, Clone, Default)]
pub struct PostprocessPlan {
    pub actions: Vec<PostprocessAction>,
}

impl PostprocessPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_action(mut self, action: PostprocessAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn save_txt(self, path: impl Into<PathBuf>) -> Self {
        self.with_action(PostprocessAction::SaveTxt { path: path.into() })
    }

    pub fn save_csv(self, path: impl Into<PathBuf>) -> Self {
        self.with_action(PostprocessAction::SaveCsv { path: path.into() })
    }

    pub fn write_report(self, path: impl Into<PathBuf>) -> Self {
        self.with_action(PostprocessAction::WriteReport { path: path.into() })
    }

    pub fn plotters_png(self, output_dir: impl Into<PathBuf>) -> Self {
        self.with_action(PostprocessAction::PlottersPng {
            output_dir: output_dir.into(),
        })
    }

    pub fn gnuplot_png(self, output_dir: impl Into<PathBuf>) -> Self {
        self.with_action(PostprocessAction::GnuplotPng {
            output_dir: output_dir.into(),
        })
    }

    pub fn terminal_plot(self) -> Self {
        self.with_action(PostprocessAction::TerminalPlot)
    }

    pub fn execute(
        &self,
        dataset: &PostprocessDataset,
    ) -> Result<PostprocessReport, PostprocessError> {
        dataset.validate()?;
        let mut report = PostprocessReport::default();
        for action in &self.actions {
            match action {
                PostprocessAction::SaveTxt { path } => {
                    ensure_parent(path)?;
                    save_matrix_to_file(
                        &dataset.values,
                        &dataset.variable_names,
                        path_to_str(path)?,
                        &dataset.axis,
                        &dataset.axis_name,
                    )?;
                    report.record(action, PostprocessStatus::Done, Some(path.clone()), None);
                }
                PostprocessAction::SaveCsv { path } => {
                    ensure_parent(path)?;
                    save_matrix_to_csv(
                        &dataset.values,
                        &dataset.variable_names,
                        path_to_str(path)?,
                        &dataset.axis,
                        &dataset.axis_name,
                    )?;
                    report.record(action, PostprocessStatus::Done, Some(path.clone()), None);
                }
                PostprocessAction::WriteReport { path } => {
                    ensure_parent(path)?;
                    write_dataset_report(dataset, path)?;
                    report.record(action, PostprocessStatus::Done, Some(path.clone()), None);
                }
                PostprocessAction::PlottersPng { output_dir } => {
                    ensure_dir(output_dir)?;
                    let _guard = CurrentDirGuard::change_to(output_dir)?;
                    plots(
                        dataset.axis_name.clone(),
                        dataset.variable_names.clone(),
                        dataset.axis.clone(),
                        dataset.values.clone(),
                    );
                    report.record(
                        action,
                        PostprocessStatus::Done,
                        Some(output_dir.clone()),
                        None,
                    );
                }
                PostprocessAction::GnuplotPng { output_dir } => {
                    if !gnuplot_available() {
                        report.record(
                            action,
                            PostprocessStatus::Skipped,
                            Some(output_dir.clone()),
                            Some("gnuplot executable is not available in PATH".to_string()),
                        );
                    } else {
                        ensure_dir(output_dir)?;
                        let _guard = CurrentDirGuard::change_to(output_dir)?;
                        plots_gnulot(
                            dataset.axis_name.clone(),
                            dataset.variable_names.clone(),
                            dataset.axis.clone(),
                            dataset.values.clone(),
                        );
                        report.record(
                            action,
                            PostprocessStatus::Done,
                            Some(output_dir.clone()),
                            None,
                        );
                    }
                }
                PostprocessAction::TerminalPlot => {
                    plots_terminal(
                        dataset.axis_name.clone(),
                        dataset.variable_names.clone(),
                        dataset.axis.clone(),
                        dataset.values.clone(),
                    );
                    report.record(action, PostprocessStatus::Done, None, None);
                }
            }
        }
        Ok(report)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostprocessStatus {
    Done,
    Skipped,
    Failed,
}

#[derive(Debug, Clone)]
pub struct PostprocessReportEntry {
    pub action: PostprocessAction,
    pub status: PostprocessStatus,
    pub path: Option<PathBuf>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PostprocessReport {
    pub entries: Vec<PostprocessReportEntry>,
}

impl PostprocessReport {
    fn record(
        &mut self,
        action: &PostprocessAction,
        status: PostprocessStatus,
        path: Option<PathBuf>,
        message: Option<String>,
    ) {
        self.entries.push(PostprocessReportEntry {
            action: action.clone(),
            status,
            path,
            message,
        });
    }

    pub fn all_done(&self) -> bool {
        self.entries
            .iter()
            .all(|entry| entry.status == PostprocessStatus::Done)
    }
}

#[derive(Debug)]
pub enum PostprocessError {
    InvalidDataset(String),
    InvalidPath(PathBuf),
    Io(io::Error),
}

impl fmt::Display for PostprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDataset(message) => write!(f, "invalid postprocess dataset: {message}"),
            Self::InvalidPath(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::Io(error) => write!(f, "postprocess I/O error: {error}"),
        }
    }
}

impl std::error::Error for PostprocessError {}

impl From<io::Error> for PostprocessError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

fn ensure_parent(path: &Path) -> Result<(), PostprocessError> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    Ok(())
}

fn ensure_dir(path: &Path) -> Result<(), PostprocessError> {
    if !path.as_os_str().is_empty() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn path_to_str(path: &Path) -> Result<&str, PostprocessError> {
    path.to_str()
        .ok_or_else(|| PostprocessError::InvalidPath(path.to_path_buf()))
}

fn gnuplot_available() -> bool {
    Command::new("gnuplot").arg("--version").output().is_ok()
}

fn write_dataset_report(dataset: &PostprocessDataset, path: &Path) -> Result<(), PostprocessError> {
    let mut file = File::create(path)?;
    writeln!(file, "# Solver Result Report")?;
    writeln!(file)?;
    writeln!(file, "- axis: {}", dataset.axis_name)?;
    writeln!(file, "- points: {}", dataset.axis.len())?;
    writeln!(file, "- variables: {}", dataset.variable_names.join(", "))?;
    for (key, value) in &dataset.metadata {
        writeln!(file, "- {key}: {value}")?;
    }
    Ok(())
}

struct CurrentDirGuard {
    previous: PathBuf,
}

impl CurrentDirGuard {
    fn change_to(path: &Path) -> Result<Self, PostprocessError> {
        let previous = std::env::current_dir()?;
        std::env::set_current_dir(path)?;
        Ok(Self { previous })
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> PostprocessDataset {
        let axis = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let values = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        PostprocessDataset::new("x", vec!["y".to_string(), "z".to_string()], axis, values)
            .unwrap()
            .with_metadata("solver", "test")
    }

    fn single_series_dataset() -> PostprocessDataset {
        let axis = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let values = DMatrix::from_column_slice(3, 1, &[1.0, 3.0, 5.0]);
        PostprocessDataset::new("x", vec!["y".to_string()], axis, values)
            .unwrap()
            .with_metadata("solver", "test")
    }

    fn unique_output_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "rustedscithe_postprocess_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn postprocess_plan_writes_csv_txt_and_report() {
        let dir = unique_output_dir("exports");
        let csv = dir.join("solution.csv");
        let txt = dir.join("solution.txt");
        let report_path = dir.join("report.md");

        let report = PostprocessPlan::new()
            .save_csv(&csv)
            .save_txt(&txt)
            .write_report(&report_path)
            .execute(&dataset())
            .unwrap();

        assert!(report.all_done());
        assert!(csv.exists());
        assert!(txt.exists());
        assert!(report_path.exists());

        let csv_text = fs::read_to_string(csv).unwrap();
        assert!(csv_text.contains("x,y,z"));
        let report_text = fs::read_to_string(report_path).unwrap();
        assert!(report_text.contains("Solver Result Report"));
        assert!(report_text.contains("solver: test"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn postprocess_dataset_rejects_shape_mismatch() {
        let axis = DVector::from_vec(vec![0.0, 1.0]);
        let values = DMatrix::from_element(3, 1, 1.0);
        let err = PostprocessDataset::new("x", vec!["y".to_string()], axis, values).unwrap_err();
        assert!(err.to_string().contains("axis length"));
    }

    #[test]
    fn postprocess_plan_plotters_png_smoke_writes_images() {
        let dir = unique_output_dir("plotters");

        let report = PostprocessPlan::new()
            .plotters_png(&dir)
            .execute(&single_series_dataset())
            .unwrap();

        assert!(report.all_done());
        assert!(dir.join("y.png").exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn postprocess_plan_gnuplot_png_smoke_skips_when_binary_is_missing() {
        let dir = unique_output_dir("gnuplot");

        let report = PostprocessPlan::new()
            .gnuplot_png(&dir)
            .execute(&single_series_dataset())
            .unwrap();

        assert_eq!(report.entries.len(), 1);
        let entry = &report.entries[0];
        if gnuplot_available() {
            assert_eq!(entry.status, PostprocessStatus::Done);
            assert!(dir.join("y.png").exists());
        } else {
            assert_eq!(entry.status, PostprocessStatus::Skipped);
            assert!(
                entry
                    .message
                    .as_deref()
                    .unwrap_or_default()
                    .contains("gnuplot executable")
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn postprocess_plan_terminal_plot_smoke_reports_done() {
        let report = PostprocessPlan::new()
            .terminal_plot()
            .execute(&dataset())
            .unwrap();

        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].status, PostprocessStatus::Done);
    }
}

use crate::command_interpreter::task_runner::{parse_task_spec_from_str, render_task_check};
use regex::Regex;
use std::process::Command;

/// Function 1: calls Pandoc and returns DOCX contents as markdown text with LaTeX formulas
pub fn docx_to_markdown(input_path: &str) -> Result<String, String> {
    let output = Command::new("pandoc")
        .arg(input_path)
        .arg("--wrap=none")
        .arg("-t")
        .arg("markdown+tex_math_dollars")
        .output()
        .map_err(|e| format!("Failed to start Pandoc: {e}"))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Pandoc error: {err}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Converts the full document:
/// - preserves lines that start with `//`
/// - in formulas `$...$` and `$$...$$` transforms `=` to `:` and converts LaTeX to infix format
/// - leaves the rest unchanged
/// - preserves line structure (each input line -> separate output line)
pub fn transform_document(input: &str) -> String {
    let mut result = String::new();

    for line in input.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("//") || trimmed.starts_with("#") || trimmed.starts_with("\\#") {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        let processed = process_line(line);
        result.push_str(&processed);
        result.push('\n');
    }

    result
}

fn process_line(line: &str) -> String {
    let formula_re = Regex::new(r"(?:\$\$|\$)(?P<formula>.*?)(?:\$\$|\$)").unwrap();

    let mut result = String::new();
    let mut last_end = 0;

    for cap in formula_re.captures_iter(line) {
        let full_match = cap.get(0).unwrap();
        let formula_start = full_match.start();
        let formula_end = full_match.end();

        let formula_content = &cap["formula"];
        let starts_with_digit = formula_content.starts_with(|c: char| c.is_ascii_digit());

        let prefix = &line[last_end..formula_start];
        if starts_with_digit && prefix.ends_with(": ") {
            result.push_str(&prefix[..prefix.len() - 1]);
        } else {
            result.push_str(prefix);
        }

        let transformed = transform_nested_latex(formula_content);
        let transformed = replace_equals_with_colon(&transformed)
            .replace("\\-", "-")
            .replace("\\+", "+")
            .replace("\\*", "*")
            .replace("\\/", "/")
            .replace("\\=", "=")
            .replace('\\', "");
        result.push_str(&transformed);

        last_end = formula_end;
    }

    result.push_str(&line[last_end..]);
    if line.contains('$') {
        result = result.replace('\\', "");
    }
    result
}

fn transform_nested_latex(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut result = String::new();
    let mut i = 0;

    while i < chars.len() {
        if i + 5 <= chars.len() && chars[i..i + 5] == ['\\', 'f', 'r', 'a', 'c'] {
            i += 5;
            if let Some((num, next_i)) = extract_braces(&chars, i) {
                i = next_i;
                if let Some((den, next_i2)) = extract_braces(&chars, i) {
                    i = next_i2;
                    let trans_num = transform_nested_latex(&num);
                    let trans_den = transform_nested_latex(&den);
                    result.push_str(&format!("({})/({})", trans_num, trans_den));
                    continue;
                }
            }
            result.push_str("\\frac");
            continue;
        }

        if i + 5 <= chars.len()
            && chars[i..i + 5] == ['\\', 's', 'q', 'r', 't']
            && i + 5 < chars.len()
            && chars[i + 5] == '['
        {
            i += 5;
            if let Some((n_val, next_i)) = extract_square_brackets(&chars, i) {
                i = next_i;
                if let Some((x_val, next_i2)) = extract_braces(&chars, i) {
                    i = next_i2;
                    let trans_x = transform_nested_latex(&x_val);
                    let trans_n = transform_nested_latex(&n_val);
                    result.push_str(&format!("({})^(1/({}))", trans_x, trans_n));
                    continue;
                }
            }
            result.push_str("\\sqrt");
            continue;
        }

        if i + 5 <= chars.len() && chars[i..i + 5] == ['\\', 's', 'q', 'r', 't'] {
            i += 5;
            if let Some((x_val, next_i)) = extract_braces(&chars, i) {
                i = next_i;
                let trans_x = transform_nested_latex(&x_val);
                result.push_str(&format!("({})^0.5", trans_x));
                continue;
            }
            result.push_str("\\sqrt");
            continue;
        }

        if i + 2 <= chars.len() && chars[i] == '_' && chars[i + 1] == '{' {
            i += 1;
            if let Some((idx_val, next_i)) = extract_braces(&chars, i) {
                i = next_i;
                let trans_idx = transform_nested_latex(&idx_val);
                if trans_idx.chars().all(|c| c.is_ascii_alphabetic()) {
                    result.push_str(&format!("_{}", trans_idx));
                } else {
                    result.push_str(&format!("_{{{}}}", trans_idx));
                }
                continue;
            }
            result.push('_');
            continue;
        }

        if i + 2 <= chars.len() && chars[i] == '^' && chars[i + 1] == '{' {
            i += 1;
            if let Some((exp_val, next_i)) = extract_braces(&chars, i) {
                i = next_i;
                let trans_exp = transform_nested_latex(&exp_val);
                if trans_exp.chars().all(|c| c.is_ascii_digit() || c == '.') {
                    result.push_str(&format!("^{}", trans_exp));
                } else {
                    result.push_str(&format!("^{{{}}}", trans_exp));
                }
                continue;
            }
            result.push('^');
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }

    post_process_simple_tokens(&result)
}

fn extract_braces(chars: &[char], start_pos: usize) -> Option<(String, usize)> {
    if start_pos >= chars.len() || chars[start_pos] != '{' {
        return None;
    }

    let mut depth = 0;
    let mut end_pos = start_pos;

    for (idx, &ch) in chars.iter().enumerate().skip(start_pos) {
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
        }

        if depth == 0 {
            end_pos = idx;
            break;
        }
    }

    if depth == 0 {
        let content: String = chars[start_pos + 1..end_pos].iter().collect();
        Some((content, end_pos + 1))
    } else {
        None
    }
}

fn extract_square_brackets(chars: &[char], start_pos: usize) -> Option<(String, usize)> {
    if start_pos >= chars.len() || chars[start_pos] != '[' {
        return None;
    }
    let mut depth = 0;
    let mut end_pos = start_pos;

    for (idx, &ch) in chars.iter().enumerate().skip(start_pos) {
        if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
        }
        if depth == 0 {
            end_pos = idx;
            break;
        }
    }
    if depth == 0 {
        let content: String = chars[start_pos + 1..end_pos].iter().collect();
        Some((content, end_pos + 1))
    } else {
        None
    }
}

fn post_process_simple_tokens(input: &str) -> String {
    let re_greek = Regex::new(r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)").unwrap();
    let re_greek_upper =
        Regex::new(r"\\(Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)").unwrap();
    let re_partial = Regex::new(r"\\partial").unwrap();
    let re_nabla = Regex::new(r"\\nabla").unwrap();
    let re_cdot = Regex::new(r"\\cdot|\\times").unwrap();
    let re_div = Regex::new(r"\\div").unwrap();
    let re_escaped_ops = Regex::new(r"\\([+\-*/=])").unwrap();

    let mut out = re_greek.replace_all(input, "$1").into_owned();
    out = re_greek_upper.replace_all(&out, "$1").into_owned();
    out = re_partial.replace_all(&out, "partial").into_owned();
    out = re_nabla.replace_all(&out, "nabla").into_owned();
    out = re_cdot.replace_all(&out, "*").into_owned();
    out = re_div.replace_all(&out, "/").into_owned();
    out = re_escaped_ops.replace_all(&out, "$1").into_owned();
    out = out.replace('\\', "");
    let re_multi_space = Regex::new(r"  +").unwrap();
    out = re_multi_space.replace_all(&out, " ").into_owned();
    out.trim().to_string()
}

fn replace_equals_with_colon(input: &str) -> String {
    let re_eq = Regex::new(r"\s*=\s*").unwrap();
    re_eq.replace_all(input, ":").into_owned()
}

fn strip_docx_comment_lines(input: &str) -> String {
    let mut out = String::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("#") || trimmed.starts_with("\\#") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn strip_docx_bom(input: &str) -> &str {
    input.strip_prefix('\u{feff}').unwrap_or(input)
}

pub fn process_docx(input_path: &str) -> Result<String, String> {
    let markdown = docx_to_markdown(input_path)?;
    let markdown = strip_docx_bom(&markdown);
    let task_document = transform_document(&markdown).replace("\\", "");
    Ok(strip_docx_comment_lines(&task_document))
}

pub fn process_docx_check(input_path: &str) -> Result<String, String> {
    let task_document = process_docx(input_path)?;
    let spec = parse_task_spec_from_str(&task_document)
        .map_err(|err| format!("failed to parse converted task document: {err}"))?;
    Ok(format!(
        "{}\n[Convert check] task document preview\n{}",
        task_document.trim_end(),
        render_task_check(&spec)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_formula() {
        assert_eq!(transform_document("$x^2$"), "x^2\n");
    }

    #[test]
    fn transform_document_keeps_task_doc_structure_for_parser_roundtrip() {
        let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
y: $a*y$

where
gain: $2*t$

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0
"#;
        let converted = transform_document(input);
        let spec = crate::command_interpreter::task_runner::parse_task_spec_from_str(&converted)
            .expect("converted document should still parse as a task document");
        let check = crate::command_interpreter::task_runner::render_task_check(&spec);
        assert!(check.contains("[Task check] IVP"));
    }

    #[test]
    fn test_fraction() {
        assert_eq!(transform_document("$\\frac{a}{b}$"), "(a)/(b)\n");
    }

    #[test]
    fn test_greek_letters() {
        assert_eq!(transform_document(r"$\\alpha + \\beta$"), "alpha + beta\n");
    }

    #[test]
    fn test_equals_to_colon_in_formula() {
        assert_eq!(transform_document("$a=b$"), "a:b\n");
    }
}

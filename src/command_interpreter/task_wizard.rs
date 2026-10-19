use dialoguer::{Select, theme::ColorfulTheme};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Question {
    pub key: &'static str,
    pub prompt: &'static str,
    pub options: &'static [&'static str],
    pub default: usize,
}

/// Runs a retained-answer wizard with one-step back navigation and a final review.
pub fn run<F>(title: &str, mut build_questions: F) -> Option<HashMap<&'static str, usize>>
where
    F: FnMut(&HashMap<&'static str, usize>) -> Vec<Question>,
{
    let theme = ColorfulTheme::default();
    println!("{title}");
    let configuration_level = choose(
        &theme,
        "Configuration level",
        &[
            "Basic (recommended defaults)",
            "Advanced (all solver controls)",
        ],
        0,
    )?;
    let advanced = configuration_level == 1;
    let mut answers = HashMap::new();
    let mut cursor = 0;

    println!(
        "Mode: {}. Choose `<< Back` to revise the previous answer. Esc/Ctrl+C cancels without writing.\n",
        if advanced { "advanced" } else { "basic" }
    );

    loop {
        let all_questions = build_questions(&answers);
        let questions = all_questions
            .iter()
            .copied()
            .filter(|question| advanced || !is_advanced_question(question.key))
            .collect::<Vec<_>>();
        cursor = cursor.min(questions.len());

        if cursor == questions.len() {
            println!("\nCurrent selections:");
            let mut previous_section = None;
            for question in &questions {
                let section = section_for_key(question.key);
                if previous_section != Some(section) {
                    println!("  [{section}]");
                    previous_section = Some(section);
                }
                let index = answers
                    .get(question.key)
                    .copied()
                    .unwrap_or(question.default)
                    .min(question.options.len().saturating_sub(1));
                println!("  {}: {}", question.prompt, question.options[index]);
            }
            let inactive = inactive_keys(&answers, &all_questions);
            if !inactive.is_empty() {
                println!(
                    "\nSaved values from inactive branches: {}",
                    inactive.join(", ")
                );
                let stale_actions = [
                    "Keep them for possible branch restoration",
                    "Clear inactive values",
                    "Cancel without writing",
                ];
                match choose(&theme, "Inactive wizard values", &stale_actions, 0)? {
                    0 => {}
                    1 => {
                        for key in inactive {
                            answers.remove(key);
                        }
                    }
                    _ => return None,
                }
            }
            let review = [
                "Write task file",
                "Edit a section",
                "Cancel without writing",
            ];
            match choose(&theme, "Review configuration", &review, 0)? {
                0 => return Some(answers),
                1 if !questions.is_empty() => {
                    let mut sections = Vec::new();
                    for question in &questions {
                        let section = section_for_key(question.key);
                        if !sections.contains(&section) {
                            sections.push(section);
                        }
                    }
                    let section_index = choose(&theme, "Choose a section to edit", &sections, 0)?;
                    let section = sections[section_index];
                    cursor = questions
                        .iter()
                        .position(|question| section_for_key(question.key) == section)
                        .unwrap_or(0);
                }
                _ => return None,
            }
            continue;
        }

        let question = questions[cursor];
        let selected = answers
            .get(question.key)
            .copied()
            .unwrap_or(question.default)
            .min(question.options.len().saturating_sub(1));
        let mut options = question.options.to_vec();
        if cursor > 0 {
            options.push("<< Back");
        }

        match choose(&theme, question.prompt, &options, selected)? {
            index if cursor > 0 && index == question.options.len() => {
                cursor = previous_step(cursor).unwrap_or(0)
            }
            index => {
                answers.insert(question.key, index);
                cursor += 1;
            }
        }
    }
}

fn previous_step(cursor: usize) -> Option<usize> {
    cursor.checked_sub(1)
}

fn section_for_key(key: &str) -> &'static str {
    match key {
        "method" | "family" | "strategy" | "scheme" | "tolerance_profile" | "iteration_budget"
        | "step_profile" => "Solver",
        "assembly" | "symbolic" | "frontend" | "execution" | "route" | "profile" => "Symbolic",
        "structure"
        | "matrix"
        | "layout"
        | "linear_policy"
        | "banded_solver"
        | "sparse_factorization" => "Linear",
        "toolchain" | "native_execution" | "callback_policy" | "aot_policy" => "Execution",
        "continuation" | "restart" | "restart_each" => "Continuation",
        "output" | "output_policy" | "csv" => "Output",
        _ => "Other",
    }
}

/// Identifies controls that are useful for tuning a solver but should not
/// burden the normal task-template path. Their defaults remain encoded by the
/// dialogue, so hiding them never creates an incomplete task document.
fn is_advanced_question(key: &str) -> bool {
    matches!(
        key,
        "assembly"
            | "symbolic"
            | "frontend"
            | "execution"
            | "route"
            | "profile"
            | "structure"
            | "matrix"
            | "layout"
            | "linear_policy"
            | "banded_solver"
            | "sparse_factorization"
            | "toolchain"
            | "native_execution"
            | "callback_policy"
            | "aot_policy"
            | "continuation"
            | "restart"
            | "restart_each"
            | "output"
            | "output_policy"
            | "csv"
    )
}

fn inactive_keys<'a>(
    answers: &HashMap<&'static str, usize>,
    questions: &[Question],
) -> Vec<&'static str> {
    let mut active = questions
        .iter()
        .map(|question| question.key)
        .collect::<Vec<_>>();
    active.sort_unstable();
    let mut inactive = answers
        .keys()
        .copied()
        .filter(|key| active.binary_search(key).is_err())
        .collect::<Vec<_>>();
    inactive.sort_unstable();
    inactive
}

fn choose(theme: &ColorfulTheme, prompt: &str, items: &[&str], default: usize) -> Option<usize> {
    match Select::with_theme(theme)
        .with_prompt(prompt)
        .items(items)
        .default(default.min(items.len().saturating_sub(1)))
        .interact_opt()
    {
        Ok(choice) => choice,
        Err(error) => {
            eprintln!("Wizard cancelled: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_answer_keys_drive_dynamic_branches_and_retain_hidden_answers() {
        let mut answers = HashMap::new();
        answers.insert("route", 1);
        answers.insert("toolchain", 1);
        let questions = |answers: &HashMap<&'static str, usize>| {
            let mut result = vec![Question {
                key: "route",
                prompt: "route",
                options: &["Lambdify", "AOT"],
                default: 0,
            }];
            if answers.get("route") == Some(&1) {
                result.push(Question {
                    key: "toolchain",
                    prompt: "toolchain",
                    options: &["tcc", "gcc"],
                    default: 0,
                });
            }
            result
        };

        assert_eq!(questions(&answers).len(), 2);
        answers.insert("route", 0);
        assert_eq!(questions(&answers).len(), 1);
        assert_eq!(answers.get("toolchain"), Some(&1));
    }

    #[test]
    fn back_moves_one_step_without_discarding_answers() {
        let mut answers = HashMap::new();
        answers.insert("route", 1);
        answers.insert("toolchain", 2);

        assert_eq!(previous_step(2), Some(1));
        assert_eq!(previous_step(0), None);
        assert_eq!(answers.get("route"), Some(&1));
        assert_eq!(answers.get("toolchain"), Some(&2));
    }

    #[test]
    fn review_sections_group_current_and_future_settings() {
        assert_eq!(section_for_key("method"), "Solver");
        assert_eq!(section_for_key("frontend"), "Symbolic");
        assert_eq!(section_for_key("layout"), "Linear");
        assert_eq!(section_for_key("callback_policy"), "Execution");
        assert_eq!(section_for_key("restart_each"), "Continuation");
        assert_eq!(section_for_key("output_policy"), "Output");
    }

    #[test]
    fn inactive_branch_values_are_detected_without_being_dropped() {
        let mut answers = HashMap::new();
        answers.insert("route", 0);
        answers.insert("toolchain", 1);
        let questions = [Question {
            key: "route",
            prompt: "route",
            options: &["Lambdify", "AOT"],
            default: 0,
        }];

        assert_eq!(inactive_keys(&answers, &questions), vec!["toolchain"]);
        assert_eq!(answers.get("toolchain"), Some(&1));
    }

    #[test]
    fn advanced_controls_are_explicit_but_core_choices_stay_basic() {
        assert!(is_advanced_question("frontend"));
        assert!(is_advanced_question("sparse_factorization"));
        assert!(is_advanced_question("continuation"));
        assert!(!is_advanced_question("family"));
        assert!(!is_advanced_question("method"));
    }
}

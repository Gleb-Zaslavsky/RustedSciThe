use crate::symbolic::symbolic_engine::Expr;

use petgraph::algo::{kosaraju_scc, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction::Incoming;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Именованное символьное определение:
///
///     name = expression
#[derive(Clone, Debug)]
pub struct Definition {
    pub name: String,
    pub expression: Expr,
}

impl Definition {
    pub fn new(name: impl Into<String>, expression: Expr) -> Self {
        Self {
            name: name.into(),
            expression,
        }
    }
}

/// Ошибки анализа системы символьных определений.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicSystemError {
    DuplicateDefinition(String),
    UnknownTarget(String),
    CyclicDependency(Vec<String>),
}

impl fmt::Display for SymbolicSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateDefinition(name) => {
                write!(f, "переменная `{name}` определена более одного раза")
            }
            Self::UnknownTarget(name) => {
                write!(f, "целевая переменная `{name}` отсутствует в системе")
            }
            Self::CyclicDependency(names) => {
                write!(
                    f,
                    "обнаружена циклическая зависимость: {}",
                    names.join(" -> ")
                )
            }
        }
    }
}

impl Error for SymbolicSystemError {}

/// Результат анализа зависимостей одной целевой переменной.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyAnalysis {
    /// Целевая переменная.
    pub target: String,

    /// Все определённые переменные, необходимые для построения target.
    /// Сама target также включена.
    pub defined_dependencies: Vec<String>,

    /// Неопределённые переменные, остающиеся после полного раскрытия.
    pub free_variables: Vec<String>,
}

/// Система именованных символьных выражений.
///
/// Направление ребра:
///
///     lhs -> variable_from_rhs
///
/// То есть ребро `D -> a` означает, что выражение D зависит от a.
#[derive(Clone, Debug)]
pub struct SymbolicSystem {
    definitions: HashMap<String, Expr>,
    graph: DiGraph<String, ()>,
    node_by_name: HashMap<String, NodeIndex>,
}

impl SymbolicSystem {
    /// Строит систему и её граф зависимостей.
    pub fn new<I>(definitions: I) -> Result<Self, SymbolicSystemError>
    where
        I: IntoIterator<Item = Definition>,
    {
        let mut definition_map = HashMap::new();

        for definition in definitions {
            if definition_map
                .insert(definition.name.clone(), definition.expression)
                .is_some()
            {
                return Err(SymbolicSystemError::DuplicateDefinition(definition.name));
            }
        }

        let mut system = Self {
            definitions: definition_map,
            graph: DiGraph::new(),
            node_by_name: HashMap::new(),
        };

        system.build_graph();
        system.validate_acyclic()?;

        Ok(system)
    }

    /// Возвращает правую часть определения.
    pub fn definition(&self, name: &str) -> Option<&Expr> {
        self.definitions.get(name)
    }

    /// Проверяет, имеет ли переменная собственное определение.
    pub fn is_defined(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    /// Возвращает имена всех определённых переменных.
    pub fn defined_variables(&self) -> Vec<String> {
        let mut names = self.definitions.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Выходные переменные системы.
    ///
    /// Это определённые переменные, которые не встречаются
    /// в правых частях других определений.
    pub fn output_variables(&self) -> Vec<String> {
        let mut outputs = self
            .definitions
            .keys()
            .filter(|name| {
                let node = self.node_by_name[*name];

                self.graph
                    .neighbors_directed(node, Incoming)
                    .next()
                    .is_none()
            })
            .cloned()
            .collect::<Vec<_>>();

        outputs.sort();
        outputs
    }

    /// Анализирует подсистему, необходимую для вычисления target.
    pub fn analyze_target(&self, target: &str) -> Result<DependencyAnalysis, SymbolicSystemError> {
        let target_node = self
            .node_by_name
            .get(target)
            .copied()
            .ok_or_else(|| SymbolicSystemError::UnknownTarget(target.to_string()))?;

        let mut visited = HashSet::new();
        self.collect_reachable(target_node, &mut visited);

        let mut defined_dependencies = BTreeSet::new();
        let mut free_variables = BTreeSet::new();

        for node in visited {
            let name = &self.graph[node];

            if self.definitions.contains_key(name) {
                defined_dependencies.insert(name.clone());
            } else {
                free_variables.insert(name.clone());
            }
        }

        Ok(DependencyAnalysis {
            target: target.to_string(),
            defined_dependencies: defined_dependencies.into_iter().collect(),
            free_variables: free_variables.into_iter().collect(),
        })
    }

    /// Выполняет максимально полное раскрытие target.
    ///
    /// Все определённые зависимости подставляются.
    /// Неопределённые переменные остаются в выражении.
    pub fn expand(&self, target: &str) -> Result<Expr, SymbolicSystemError> {
        self.expand_until(target, &HashSet::new())
    }

    /// Fully expands every named definition with one shared memoization cache.
    ///
    /// This is the efficient compatibility route for small task-document
    /// systems: callers still receive ordinary [`Expr`] trees, while a shared
    /// dependency is resolved only once during this bulk operation.
    /// The returned map is ordered by definition name for reproducible parser
    /// diagnostics and tests.
    pub fn expand_all(&self) -> Result<BTreeMap<String, Expr>, SymbolicSystemError> {
        self.expand_all_until(&HashSet::new())
    }

    /// Fully expands every named definition except variables explicitly kept
    /// symbolic by `stop_variables`.
    pub fn expand_all_until(
        &self,
        stop_variables: &HashSet<String>,
    ) -> Result<BTreeMap<String, Expr>, SymbolicSystemError> {
        let mut memo = HashMap::<String, Expr>::new();

        for name in self.defined_variables() {
            let mut active = Vec::<String>::new();
            let _ = self.expand_variable(&name, stop_variables, &mut memo, &mut active)?;
        }

        Ok(memo.into_iter().collect())
    }

    /// Раскрывает target до заранее заданного множества переменных.
    ///
    /// Переменные из `stop_variables` не раскрываются,
    /// даже если для них имеются определения.
    pub fn expand_until(
        &self,
        target: &str,
        stop_variables: &HashSet<String>,
    ) -> Result<Expr, SymbolicSystemError> {
        if !self.node_by_name.contains_key(target) {
            return Err(SymbolicSystemError::UnknownTarget(target.to_string()));
        }

        let mut memo = HashMap::<String, Expr>::new();
        let mut active = Vec::<String>::new();

        self.expand_variable(target, stop_variables, &mut memo, &mut active)
    }

    /// Возвращает порядок вычисления зависимостей target:
    /// сначала независимые определения, затем зависящие от них.
    pub fn evaluation_order(&self, target: &str) -> Result<Vec<String>, SymbolicSystemError> {
        let target_node = self
            .node_by_name
            .get(target)
            .copied()
            .ok_or_else(|| SymbolicSystemError::UnknownTarget(target.to_string()))?;

        let mut reachable = HashSet::new();
        self.collect_reachable(target_node, &mut reachable);

        // При направлении lhs -> dependency toposort выдаёт lhs раньше
        // его зависимостей. Для порядка вычисления результат разворачиваем.
        let mut order = toposort(&self.graph, None).map_err(|_| self.cycle_error())?;

        order.reverse();

        Ok(order
            .into_iter()
            .filter(|node| reachable.contains(node))
            .map(|node| self.graph[node].clone())
            .filter(|name| self.definitions.contains_key(name))
            .collect())
    }

    fn build_graph(&mut self) {
        // Сначала добавляем все определённые переменные.
        let defined_names = self.definitions.keys().cloned().collect::<Vec<_>>();

        for name in &defined_names {
            self.ensure_node(name);
        }

        // Затем добавляем зависимости из правых частей.
        for lhs in defined_names {
            let lhs_node = self.node_by_name[&lhs];

            let variables = self.definitions[&lhs].all_arguments_are_variables();

            for rhs_variable in variables {
                let rhs_node = self.ensure_node(&rhs_variable);

                // Повторяющиеся появления одной переменной в выражении
                // не должны создавать параллельные рёбра.
                if self.graph.find_edge(lhs_node, rhs_node).is_none() {
                    self.graph.add_edge(lhs_node, rhs_node, ());
                }
            }
        }
    }

    fn ensure_node(&mut self, name: &str) -> NodeIndex {
        if let Some(node) = self.node_by_name.get(name) {
            return *node;
        }

        let node = self.graph.add_node(name.to_string());
        self.node_by_name.insert(name.to_string(), node);
        node
    }

    fn validate_acyclic(&self) -> Result<(), SymbolicSystemError> {
        toposort(&self.graph, None)
            .map(|_| ())
            .map_err(|_| self.cycle_error())
    }

    fn cycle_error(&self) -> SymbolicSystemError {
        let mut cycle = kosaraju_scc(&self.graph)
            .into_iter()
            .find(|component| {
                if component.len() > 1 {
                    return true;
                }

                let node = component[0];

                // Отдельная SCC также является циклом при наличии ребра v -> v.
                self.graph.find_edge(node, node).is_some()
            })
            .unwrap_or_default()
            .into_iter()
            .map(|node| self.graph[node].clone())
            .collect::<Vec<_>>();

        cycle.sort();

        SymbolicSystemError::CyclicDependency(cycle)
    }

    fn collect_reachable(&self, node: NodeIndex, visited: &mut HashSet<NodeIndex>) {
        if !visited.insert(node) {
            return;
        }

        for dependency in self.graph.neighbors(node) {
            self.collect_reachable(dependency, visited);
        }
    }

    fn expand_variable(
        &self,
        name: &str,
        stop_variables: &HashSet<String>,
        memo: &mut HashMap<String, Expr>,
        active: &mut Vec<String>,
    ) -> Result<Expr, SymbolicSystemError> {
        // Явная граница раскрытия.
        if stop_variables.contains(name) {
            return Ok(Expr::Var(name.to_string()));
        }

        // Свободная переменная.
        let Some(definition) = self.definitions.get(name) else {
            return Ok(Expr::Var(name.to_string()));
        };

        // Уже раскрыто ранее.
        if let Some(expanded) = memo.get(name) {
            return Ok(expanded.clone());
        }

        // Дополнительная защита от циклов.
        if let Some(position) = active.iter().position(|item| item == name) {
            let mut cycle = active[position..].to_vec();
            cycle.push(name.to_string());

            return Err(SymbolicSystemError::CyclicDependency(cycle));
        }

        active.push(name.to_string());

        let mut expanded = definition.clone();

        for dependency in definition.all_arguments_are_variables() {
            let replacement = self.expand_variable(&dependency, stop_variables, memo, active)?;

            expanded = expanded.substitute_variable(&dependency, &replacement);
        }

        active.pop();

        // Полный simplify здесь может оказаться дорогим.
        // На первом этапе можно оставить его вызывающей стороне.
        //
        // expanded = expanded.simplify();

        memo.insert(name.to_string(), expanded.clone());

        Ok(expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_system() -> SymbolicSystem {
        SymbolicSystem::new(vec![
            Definition::new("a", Expr::Const(5.0)),
            Definition::new("b", Expr::Const(2.0)),
            Definition::new("D", Expr::parse_expression("b^2 - 4*a*c")),
            Definition::new("x1", Expr::parse_expression("(-b + D^0.5)/(2*a)")),
        ])
        .unwrap()
    }

    #[test]
    fn finds_output_variables() {
        let system = quadratic_system();

        assert_eq!(system.output_variables(), vec!["x1".to_string()]);
    }

    #[test]
    fn finds_free_variables_of_target() {
        let system = quadratic_system();
        let analysis = system.analyze_target("x1").unwrap();

        assert_eq!(analysis.free_variables, vec!["c".to_string()]);
    }

    #[test]
    fn expands_all_defined_dependencies() {
        let system = quadratic_system();
        let expanded = system.expand("x1").unwrap();

        let variables = expanded.all_arguments_are_variables();

        assert_eq!(variables, vec!["c".to_string()]);
        assert!(!expanded.contains_variable("a"));
        assert!(!expanded.contains_variable("b"));
        assert!(!expanded.contains_variable("D"));
    }

    #[test]
    fn stops_at_selected_variable() {
        let system = quadratic_system();
        let stop = HashSet::from(["D".to_string()]);

        let expanded = system.expand_until("x1", &stop).unwrap();

        assert_eq!(
            expanded.all_arguments_are_variables(),
            vec!["D".to_string()]
        );
    }

    #[test]
    fn returns_dependency_evaluation_order() {
        let system = quadratic_system();
        let order = system.evaluation_order("x1").unwrap();

        let position = |name: &str| order.iter().position(|item| item == name).unwrap();

        assert!(position("a") < position("D"));
        assert!(position("b") < position("D"));
        assert!(position("D") < position("x1"));
    }

    #[test]
    fn rejects_direct_cycle() {
        let result = SymbolicSystem::new(vec![
            Definition::new("a", Expr::parse_expression("b + 1")),
            Definition::new("b", Expr::parse_expression("a + 1")),
        ]);

        assert!(matches!(
            result,
            Err(SymbolicSystemError::CyclicDependency(_))
        ));
    }

    #[test]
    fn rejects_self_reference() {
        let result =
            SymbolicSystem::new(vec![Definition::new("a", Expr::parse_expression("a + 1"))]);

        assert!(matches!(
            result,
            Err(SymbolicSystemError::CyclicDependency(_))
        ));
    }

    #[test]
    fn handles_shared_dependency_once_logically() {
        let system = SymbolicSystem::new(vec![
            Definition::new("a", Expr::Const(2.0)),
            Definition::new("b", Expr::parse_expression("a + 1")),
            Definition::new("c", Expr::parse_expression("a + 2")),
            Definition::new("result", Expr::parse_expression("b*c")),
        ])
        .unwrap();

        let expanded = system.expand("result").unwrap();

        assert!(expanded.all_arguments_are_variables().is_empty());
    }

    #[test]
    fn expands_all_definitions_with_deterministic_names() {
        let system = quadratic_system();

        let expanded = system.expand_all().unwrap();
        let names = expanded.keys().cloned().collect::<Vec<_>>();

        assert_eq!(names, vec!["D", "a", "b", "x1"]);
        assert!(!expanded["x1"].contains_variable("D"));
        assert!(!expanded["x1"].contains_variable("a"));
        assert!(!expanded["x1"].contains_variable("b"));
    }

    #[test]
    fn expands_multiple_quadratic_outputs_with_shared_dependencies() {
        let system = SymbolicSystem::new(vec![
            Definition::new("a", Expr::Const(5.0)),
            Definition::new("b", Expr::Const(2.0)),
            Definition::new("D", Expr::parse_expression("b^2 - 4*a*c")),
            Definition::new("x1", Expr::parse_expression("(-b + D^0.5)/(2*a)")),
            Definition::new("x2", Expr::parse_expression("(-b - D^0.5)/(2*a)")),
        ])
        .unwrap();

        let expanded = system.expand_all().unwrap();

        assert_eq!(
            expanded.keys().cloned().collect::<Vec<_>>(),
            vec!["D", "a", "b", "x1", "x2"]
        );
        for output in ["x1", "x2"] {
            assert_eq!(expanded[output].all_arguments_are_variables(), vec!["c"]);
            assert!(!expanded[output].contains_variable("D"));
        }
    }
}

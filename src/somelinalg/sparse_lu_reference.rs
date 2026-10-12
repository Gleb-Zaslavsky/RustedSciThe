//! Shared BVP-shaped sparse-LU fixtures for Rust tests and microbenchmarks.

use faer::sparse::{SparseColMat, Triplet};
use std::io;
use std::path::Path;

pub(crate) struct BvpSparseSystem {
    pub(crate) mesh_nodes: usize,
    pub(crate) state_dimension: usize,
    pub(crate) border_dimension: usize,
    pub(crate) matrix: SparseColMat<usize, f64>,
    pub(crate) triplets: Vec<Triplet<usize, usize, f64>>,
    pub(crate) rhs: Vec<f64>,
    pub(crate) exact_solution: Vec<f64>,
}

impl BvpSparseSystem {
    pub(crate) fn dimension(&self) -> usize {
        self.rhs.len()
    }
}

/// Build a deterministic, diagonally dominant block-tridiagonal matrix with
/// dense global border couplings, representative of collocation Jacobians.
pub(crate) fn bvp_sparse_system(
    mesh_nodes: usize,
    state_dimension: usize,
    border_dimension: usize,
) -> BvpSparseSystem {
    assert!(mesh_nodes >= 2 && state_dimension > 0);
    let state_size = mesh_nodes * state_dimension;
    let dimension = state_size + border_dimension;
    let mut triplets = Vec::new();

    for node in 0..mesh_nodes {
        let row_base = node * state_dimension;
        for local_col in 0..state_dimension {
            let col = row_base + local_col;
            let mut off_diagonal_sum = 0.0;
            for local_row in 0..state_dimension {
                if local_row == local_col {
                    continue;
                }
                let value = signed_value(node, local_row, local_col, 0.17);
                triplets.push(Triplet::new(row_base + local_row, col, value));
                off_diagonal_sum += value.abs();
            }
            for neighbor in [
                node.checked_sub(1),
                (node + 1 < mesh_nodes).then_some(node + 1),
            ]
            .into_iter()
            .flatten()
            {
                let neighbor_base = neighbor * state_dimension;
                for local_row in 0..state_dimension {
                    let value = signed_value(node, neighbor, local_row + local_col, 0.035);
                    triplets.push(Triplet::new(neighbor_base + local_row, col, value));
                    off_diagonal_sum += value.abs();
                }
            }
            for border_col in 0..border_dimension {
                let value = signed_value(node, local_col, border_col, 0.012);
                triplets.push(Triplet::new(state_size + border_col, col, value));
                off_diagonal_sum += value.abs();
            }
            triplets.push(Triplet::new(
                col,
                col,
                off_diagonal_sum + 2.0 + (local_col + 1) as f64 * 0.1,
            ));
        }

        for local_row in 0..state_dimension {
            let row = row_base + local_row;
            for border_col in 0..border_dimension {
                triplets.push(Triplet::new(
                    row,
                    state_size + border_col,
                    signed_value(node, local_row, border_col, 0.009),
                ));
            }
        }
    }

    for border in 0..border_dimension {
        let index = state_size + border;
        triplets.push(Triplet::new(index, index, 2.5 + border as f64));
        for state_index in [border % state_size, state_size - 1 - (border % state_size)] {
            triplets.push(Triplet::new(
                index,
                state_index,
                signed_value(border, state_index, 3, 0.01),
            ));
        }
    }

    let exact_solution = (0..dimension)
        .map(|index| ((index as f64 + 1.0) * 0.37).sin())
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; dimension];
    for entry in &triplets {
        rhs[entry.row] += entry.val * exact_solution[entry.col];
    }
    let matrix = SparseColMat::try_new_from_triplets(dimension, dimension, &triplets)
        .expect("generated BVP-like CSC matrix must assemble");

    BvpSparseSystem {
        mesh_nodes,
        state_dimension,
        border_dimension,
        matrix,
        triplets,
        rhs,
        exact_solution,
    }
}

fn signed_value(a: usize, b: usize, c: usize, scale: f64) -> f64 {
    let hash = a
        .wrapping_mul(31)
        .wrapping_add(b.wrapping_mul(17))
        .wrapping_add(c * 13);
    let centered = (hash % 19) as f64 - 9.0;
    scale * centered / 9.0
}

pub(crate) fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

pub(crate) fn relative_residual_inf(
    triplets: &[Triplet<usize, usize, f64>],
    solution: &[f64],
    rhs: &[f64],
) -> f64 {
    let mut residual = vec![0.0; rhs.len()];
    for entry in triplets {
        residual[entry.row] += entry.val * solution[entry.col];
    }
    let residual_inf = residual
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    residual_inf
        / rhs
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
            .max(1.0)
}

/// Write a shared fixture accepted by the Python SuperLU runner.
pub(crate) fn write_fixture(path: &Path, system: &BvpSparseSystem) -> io::Result<()> {
    let mut content = format!(
        "{} {} {} {} {}\n",
        system.dimension(),
        system.mesh_nodes,
        system.state_dimension,
        system.border_dimension,
        system.triplets.len()
    );
    for (index, value) in system.rhs.iter().enumerate() {
        if index > 0 {
            content.push(' ');
        }
        content.push_str(&format!("{value:.17e}"));
    }
    content.push('\n');
    for (index, value) in system.exact_solution.iter().enumerate() {
        if index > 0 {
            content.push(' ');
        }
        content.push_str(&format!("{value:.17e}"));
    }
    content.push('\n');
    for entry in &system.triplets {
        content.push_str(&format!("{} {} {:.17e}\n", entry.row, entry.col, entry.val));
    }
    std::fs::write(path, content)
}

/// Load either a shared five-field fixture or a BVP-captured three-field one.
pub(crate) fn load_fixture(path: &Path) -> Result<BvpSparseSystem, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read fixture {}: {error}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| "fixture has no header".to_owned())?
        .split_whitespace()
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("invalid fixture header: {error}"))?;
    let (dimension, mesh_nodes, state_dimension, border_dimension, entry_count, has_exact) =
        match header.as_slice() {
            [
                dimension,
                mesh_nodes,
                state_dimension,
                border_dimension,
                entry_count,
            ] => (
                *dimension,
                *mesh_nodes,
                *state_dimension,
                *border_dimension,
                *entry_count,
                true,
            ),
            [dimension, mesh_nodes, entry_count] => {
                (*dimension, *mesh_nodes, 0, 0, *entry_count, false)
            }
            _ => return Err("fixture header must contain 3 or 5 integer fields".into()),
        };
    let parse_vector = |line: Option<&str>, label: &str| -> Result<Vec<f64>, String> {
        line.ok_or_else(|| format!("fixture has no {label}"))?
            .split_whitespace()
            .map(|value| {
                value
                    .parse::<f64>()
                    .map_err(|error| format!("invalid {label} value: {error}"))
            })
            .collect()
    };
    let rhs = parse_vector(lines.next(), "RHS")?;
    if rhs.len() != dimension {
        return Err(format!(
            "RHS length {} does not match dimension {dimension}",
            rhs.len()
        ));
    }
    let exact_solution = if has_exact {
        let exact = parse_vector(lines.next(), "known solution")?;
        if exact.len() != dimension {
            return Err(format!(
                "known-solution length {} does not match dimension {dimension}",
                exact.len()
            ));
        }
        exact
    } else {
        Vec::new()
    };
    let triplets = lines
        .map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let [row, column, value] = fields.as_slice() else {
                return Err("fixture triplet must contain row, column, value".to_owned());
            };
            let row = row
                .parse::<usize>()
                .map_err(|error| format!("invalid fixture row index: {error}"))?;
            let column = column
                .parse::<usize>()
                .map_err(|error| format!("invalid fixture column index: {error}"))?;
            let value = value
                .parse::<f64>()
                .map_err(|error| format!("invalid fixture matrix value: {error}"))?;
            if row >= dimension || column >= dimension {
                return Err("fixture triplet index is outside matrix dimensions".into());
            }
            Ok(Triplet::new(row, column, value))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if triplets.len() != entry_count {
        return Err(format!(
            "fixture has {} triplets but header declares {entry_count}",
            triplets.len()
        ));
    }
    let matrix = SparseColMat::try_new_from_triplets(dimension, dimension, &triplets)
        .map_err(|error| format!("assemble fixture CSC matrix: {error:?}"))?;
    Ok(BvpSparseSystem {
        mesh_nodes,
        state_dimension,
        border_dimension,
        matrix,
        triplets,
        rhs,
        exact_solution,
    })
}

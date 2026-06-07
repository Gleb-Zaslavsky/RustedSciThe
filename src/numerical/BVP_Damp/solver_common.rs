use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use nalgebra::DVector;

pub(crate) const DEFAULT_FORWARD_SCHEME: &str = "forward";
pub(crate) const DEFAULT_SPARSE_METHOD: &str = "Sparse";
pub(crate) const DEFAULT_DENSE_METHOD: &str = "Dense";
pub(crate) const DEFAULT_MAX_ITERATIONS: usize = 25;

pub(crate) fn default_forward_scheme_name() -> String {
    DEFAULT_FORWARD_SCHEME.to_string()
}

pub(crate) fn default_sparse_method_name() -> String {
    DEFAULT_SPARSE_METHOD.to_string()
}

pub(crate) fn default_dense_method_name() -> String {
    DEFAULT_DENSE_METHOD.to_string()
}

pub(crate) fn default_placeholder_y() -> Box<DVector<f64>> {
    Box::new(DVector::from_vec(vec![0.0, 0.0]))
}

pub(crate) fn cleanup_registered_aot_artifacts(
    config: &mut GeneratedBackendConfig,
) -> std::io::Result<usize> {
    let Some(resolver) = config.resolver.as_mut() else {
        return Ok(0);
    };
    let problem_keys = resolver.registry().problem_keys();
    let mut removed = 0;
    for problem_key in problem_keys {
        if resolver.cleanup_artifact_by_problem_key(&problem_key)? {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Damped BVP treats `n_steps` as the number of intervals and stores both
/// endpoints, so the mesh length is `n_steps + 1`.
pub(crate) fn damped_interval_mesh(t0: f64, t_end: f64, n_steps: usize) -> DVector<f64> {
    let h = (t_end - t0) / n_steps as f64;
    DVector::from_vec((0..=n_steps).map(|i| t0 + i as f64 * h).collect())
}

/// Frozen BVP historically treats `n_steps` as the number of stored mesh
/// points, so the mesh length is exactly `n_steps`.
pub(crate) fn frozen_point_mesh(t0: f64, t_end: f64, n_steps: usize) -> DVector<f64> {
    let h = (t_end - t0) / (n_steps - 1) as f64;
    DVector::from_vec((0..n_steps).map(|i| t0 + i as f64 * h).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damped_interval_mesh_keeps_legacy_interval_count_contract() {
        let mesh = damped_interval_mesh(0.0, 1.0, 4);

        assert_eq!(mesh.as_slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn frozen_point_mesh_keeps_legacy_point_count_contract() {
        let mesh = frozen_point_mesh(0.0, 1.0, 5);

        assert_eq!(mesh.as_slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn default_placeholder_y_keeps_legacy_two_component_shape() {
        let y = default_placeholder_y();

        assert_eq!(y.as_slice(), &[0.0, 0.0]);
    }
}

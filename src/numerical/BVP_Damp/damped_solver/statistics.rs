/// Runtime counters, timers, and generated-backend diagnostics for Damped Newton.
#[derive(Clone, Debug)]
pub struct DampedBvpStatistics {
    pub counters: HashMap<String, usize>,
    pub timers: HashMap<String, String>,
    pub diagnostics: HashMap<String, String>,
    /// Typed telemetry for new consumers; legacy maps above remain compatible.
    pub telemetry: BvpTelemetrySnapshot,
}

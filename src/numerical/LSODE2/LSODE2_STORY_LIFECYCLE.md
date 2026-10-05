# LSODE2 Lifecycle And Telemetry Stories

Lifecycle reports explain when a prepared plan, callback, artifact or factor
is reused. Numeric parameter continuation may reuse unchanged symbolic shape;
schema, layout or Jacobian-pattern changes must invalidate it.

Telemetry is intentionally low-intrusion and stage-oriented. Cold stages
include validation, symbolic preparation, lowering, materialization, compile,
link and cache. Warm stages include binding, callback evaluation, chunks,
workers, copies, allocations, output writes and solver overhead.

`build_attempts`, `link_attempts`, cache hit/miss and `link_ms` must describe
the same lifecycle scope before being used for frontend comparisons. Parent
scopes are inclusive; exclusive remainders are named explicitly.

The report writer now partitions output by `debug` and `release` profile and
records that profile in the Markdown header. Historical reports are preserved
in [LSODE2_STORY_ARCHIVE.md](LSODE2_STORY_ARCHIVE.md); new reports are dated
and profile-aware under `test_reports/`.

The typed schema contract is executable in
`tests/telemetry_stage_story_tests.rs::lsode2_telemetry_schema_contract_story`.
It checks the stable cold/warm stage labels and the counters for binding,
copies, allocations, chunks, workers, output writes, cache resolution,
build/link attempts and solver requests.

## 2026-09-29 Release Lifecycle Evidence

The release lifecycle corpus passed for all completed reports. The normalized
`BuildIfMissing -> RequirePrebuilt` matrix covers ExprLegacy and AtomView on
Sparse and Banded, and the process-isolated release matrix covers Rust,
C/tcc, C/gcc and Zig. Numeric parameter rebinds reuse the producer artifact;
schema, layout and Jacobian-pattern changes invalidate it and are rejected.

The post-fix direct-versus-solver AOT report is now archived. AtomView reports
one build and one link in both direct and solver boundaries at
`512/1024/2048`; no native Jacobian/pattern preparation is repeated. ExprLegacy
solver rows report `2/2` because its compatibility residual and Jacobian
artifacts are separate. These counters describe different lifecycle scopes and
must not be presented as linker-speed comparisons.

The AOT chunk-policy report also separates the one-time Auto calibration from
the callback interval. Its `2250.700 ms` calibration is not part of the
`0.011142 ms` Auto residual measurement. This closes the earlier apparent
1000x callback anomaly, while portable Parallel break-even remains open.

The long Criterion parameter-continuation bench is still running and is not
included in the completed release evidence. Story-level continuation parity
and short fair-performance reports are complete and dated separately.

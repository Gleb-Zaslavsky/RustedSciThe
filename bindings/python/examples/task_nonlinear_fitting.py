"""Parse, inspect and run nonlinear and fitting task documents."""

from rustedscithe import render_table, run_task_batch, task, task_dict


root_source = """{"format_version":1,"problem":{"formulation":"root","variables":"x",
"equations":"x - 2","initial_guess":0.0},"solver":{"method":"newton"}}"""

fit_source = """data
arg: x
x: 0.0, 1.0, 2.0
y: 1.0, 3.0, 5.0
model
method: lm
parameters: a, b
initial_guess: 0.0, 0.0
equations: a*x + b
"""

root_task = task(root_source)
assert root_task.kind == "nonlinear"
assert task_dict(root_source)["problem"]["formulation"] == "root"
root_result = root_task.run_nonlinear()
if not root_result.successful:
    raise RuntimeError(f"root solve did not converge: {root_result.termination}")
print("root solution:", root_result.solution)
print("root telemetry:", root_result.telemetry_counters())

fit_task = task(fit_source)
fit_result = fit_task.run_optimization()
print("fitted parameters:", fit_result.solution_map())
coordinates, values = fit_result.evaluate_at([0.5, 1.5])
print("fitted values:", list(zip(coordinates.tolist(), values[:, 0].tolist())))

batch = run_task_batch([root_source, fit_source])
if any(row["status"] != "ok" for row in batch):
    raise RuntimeError("a nonlinear/fitting task failed in batch dispatch")
print(render_table(
    [{"kind": row["kind"], "status": row["status"]} for row in batch],
    columns=["kind", "status"],
))

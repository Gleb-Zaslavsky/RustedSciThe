"""Run independent text tasks and render a compact failure-tolerant table."""

from rustedscithe import render_table, run_task_batch


task_text = """task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""

rows = run_task_batch([task_text, "task\nmethod: BDF\n"])
print(render_table(rows, columns=["source", "status", "kind", "error_type"]))

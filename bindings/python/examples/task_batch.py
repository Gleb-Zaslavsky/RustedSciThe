"""Run independent text tasks and render a compact failure-tolerant table.

The first document is a valid scalar BDF task. The second deliberately omits
the required solver declaration. Batch execution should return one row per
input and preserve the successful result even when another task fails; callers
can inspect ``status`` and ``error_type`` instead of parsing terminal output.
"""

from rustedscithe import render_table, run_task_batch


# Task documents are parsed by the native Rust parser. The second input is
# intentionally incomplete to demonstrate per-task failure isolation.
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

# The valid task remains available in the result even though its neighbor fails.
rows = run_task_batch([task_text, "task\nmethod: BDF\n"])
# The valid job must survive its neighbor's parse failure, and the failure is
# typed so callers need not inspect unstable diagnostic text.
if len(rows) != 2 or rows[0]["status"] != "ok":
    raise AssertionError("the valid task did not produce a result row")
if rows[1].get("error_type") != "TaskDocumentError":
    raise AssertionError("the invalid task did not retain its parser error type")
print(render_table(rows, columns=["source", "status", "kind", "error_type"]))

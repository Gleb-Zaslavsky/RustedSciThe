# Trajectory Animation in Python

The optional interactive animation helpers display numerical trajectories
already returned by an IVP solver. They do not integrate equations, mutate a
solver, or evaluate symbolic callbacks. Use them for exploration and teaching,
not as a numerical postprocessor or a replacement for saved solution data.

## Input contract

```python
from rustedscithe import animate_trajectory_2d, animate_trajectory_3d

animate_trajectory_2d(points_xy, times, axis_names=("x", "z"), speed=1.0)
animate_trajectory_3d(points_xyz, times, axis_names=("x", "y", "z"), speed=1.0)
```

`points_xy` has shape `(samples, 2)`, `points_xyz` has shape `(samples, 3)`,
and `times` has shape `(samples,)`. The Python API uses sample-major NumPy
layout; the binding performs one copy/transposition into the native animation
layout. Shape mismatch, empty trajectories, non-finite coordinates/times,
coordinates outside finite `f32` range, non-monotone times and non-positive or
non-finite speed are rejected before Bevy opens a window. A single sample is
allowed. Time may be strictly increasing or strictly decreasing.

Adjacent time differences determine playback intervals, divided by `speed`; for
example, `speed=2` plays twice as fast. The clock carries render-frame
overshoot across sample boundaries and loops using the last observed interval.
The rendered current-point marker is sampled at display FPS and may jump over
several very closely spaced adaptive solver samples; the drawn path still
contains the trajectory segments up to the current sample.
Intervals below one microsecond are clamped for display. Visualization stores
coordinates as `f32`, so it is not a precision-preserving export path.
Intervals above `1e9` playback seconds are rejected rather than creating a
window that appears frozen for an impractical duration.
Each coordinate range must also remain representable as finite `f32` geometry.

## Animate a solver result

`examples/lsode2_lorenz_animation.py` solves the Lorenz system and opens the 3D
trajectory by default. Select another view or use `--no-animate` for a
headless numerical run:

```console
python examples/lsode2_lorenz_animation.py --no-animate
python examples/lsode2_lorenz_animation.py --animate 2d
python examples/lsode2_lorenz_animation.py --animate 3d --speed 1.5
python examples/lsode2_lorenz_animation.py --animate both
```

The no-argument command opens the 3D `(x,y,z)` view; the 2D view uses the
`(x,z)` projection. The
`both` mode closes the 2D child window and then opens 3D in a new child process,
without solving the model a second time. The
three-body animation example opens its selected body's planar path by default.
Use `--no-animate` for a headless numerical run:

```console
python examples/lsode2_three_body_animation.py --no-animate
python examples/lsode2_three_body_animation.py --animate --body 1
```

Animation coordinates are uniformly normalized for display so small or
offset trajectories fit the viewport; the position HUD still reports the
original solver coordinates.

For an existing IVP result, `animate_solution_2d(solution, columns=(0, 2), ...)`
and `animate_solution_3d(solution, columns=(0, 1, 2), ...)` read the detached
NumPy state/time arrays and select columns. They do not call `solve()` or
re-evaluate callbacks.

## Runtime behavior

Opening a Bevy window is synchronous and blocks until the window closes; press
Esc or use the window close control to exit. Each call runs one native event
loop in a short-lived Python child process. This isolates winit's one-event-loop-
per-process restriction, so a caller may close a 2D view and later open a 3D
view in the same Python session. The parent waits for the child and relays typed
input/runtime failures. Call playback from Python's main thread; a worker-thread
call raises `RuntimeError` before a process is started. Window creation requires
a working desktop renderer and is intentionally not part of ordinary headless
tests. Importing `rustedscithe` or solving an IVP does not open a window.

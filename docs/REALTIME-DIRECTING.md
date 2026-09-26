# Buffered realtime directing

Run `./run-realtime.command` in an installed checkout, with the authenticated
realtime worker reachable at `http://127.0.0.1:8769`. The viewer listens on
`http://127.0.0.1:2350`. Set `STAGEZERO_PYTHON`, `STAGEZERO_TOKEN_FILE`,
`STAGEZERO_REALTIME_URL`, or `STAGEZERO_REALTIME_PORT` to override those defaults.
The launcher does not provision a Pod or download checkpoints.

The isolated worker is `experiments/serve_realtime_backend.py`. It loads Core
once and publishes synchronized 40-frame windows at 20 fps through authenticated
job endpoints. Use a private SSH tunnel; the backend defaults to loopback.
InterGen is disabled unless the worker starts with `--research-intergen` and
explicit cached repository/checkpoint paths. See
[the release gates](REALTIME-PRODUCTION-GATES.md) for the research boundary.

The viewer's **Generate / redirect** runs Core from the last committed native
history. Generation runs in a background worker. Playback holds the last real
pose on an underrun and reports the failure instead of fabricating animation.
**Retry failed generation** resumes a failed free-direction request.

**Run complete scene** executes a geometry-grounded scene through the same
buffered director, measures outcomes, and saves both the exact timeline and
its report. Paired scenes require the research checkbox. During a coordinated
scene, free direction waits until completion; the lower-level scheduler can
retain a continuous cached action through its release before redirecting.
An arbitrary fresh paired sample is never treated as a continuation.

**Save exact project** saves locally and downloads an NPZ archive. **Open exact
project** reloads committed frames without regenerating them. A loaded project
plays its exact recorded motion; generation starts only after an explicit new
direction. Load a saved project at startup with `--project /absolute/file.npz`.
Scene geometry and initial actor placements are saved with new projects. Use `--scene /absolute/plan.json` to supply geometry for older motion-only archives.

Complete scene runs use `experiments/generate_scene_showcase.py --scenario`
with `gate_meet_handshake`, `staged_fight`, or `object_reach_inspect`. The default
uses a real wall clock and records startup, buffering, generation timing,
continuity, and task-specific outcome checks. `--offline-assembly` is for CPU
integration tests and is not realtime evidence. A failed quality check retains
the output with a failure label; it must not be advertised as a passing scene.

The original studio and its current character/object work are separate from
this isolated viewer. This milestone must pass complete real-model scene
validation before its components are adopted into that studio.

**Load scene layout** loads geometry without starting paired generation. Select
an actor, enter an exact object name (for example `Stone gate`), and choose
`go_through` or `approach`. **Navigate to object** generates dense native root
constraints along the planned route, keeping the other actor stationary and
avoiding its planning footprint. The planner does not infer a passage through
an unknown mesh. Previous committed motion is backed up when replacing a layout.

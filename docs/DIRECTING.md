# Single-actor directing workflow

This development branch extends StageZero with recorded takes, branching,
scrubbing, and project files for one G1 actor. The existing `preview.py` and
`live_viewer.py` remain available. The current Pod is reachable and serving the
private ARDY inference backend; credentials, connection details, model weights,
and generated project files stay private.

## Start

For the local viewer only:

```sh
.venv/bin/python director_viewer.py
```

Open http://127.0.0.1:2336/. To start or reconnect the existing Pod backend and
its authenticated SSH tunnel before opening the viewer, use:

```sh
./run-director.command
```

The launcher uses `.runtime/pod.env`, `.runtime/api-token`, the configured SSH
key, and `.runtime/known_hosts`. It reuses the existing Pod and does not create
or resize resources. If the Pod or tunnel is unavailable, the viewer still
opens and recorded playback and project controls remain usable. The former live
viewer uses port 2335. The existing private Tailscale Serve proxy on port 2334 now targets port 2336.

## Workflow

- Choose Live ARDY, enter an instruction, and generate. ARDY returns a complete
  4.16-second segment before playback begins; generation is not streamed.
- Generating at the end of a take extends it. Pause and scrub to an earlier
  moment to make an alternate ending. The original remains available and the
  branch keeps the exact shared prefix. The first three frames are too short
  for continuation history; use New take to start fresh.
- Use the Take selector to compare performances. Rewind returns to frame zero
  without deleting motion. New take starts another performance while keeping
  prior takes in the project.
- Save / Open stores every take, the playhead, scene settings, prompts,
  generation provenance, and recorded events in a `.stagezero.npz` file.
  Saves are written atomically to `.runtime/projects` and can also be
  downloaded from the browser. Saving during new generation records a
  consistent snapshot; the status indicates when newer changes remain unsaved.
- New project automatically saves an `automatic-backup` when the current
  project has takes, then clears the session. If the backup write fails, the
  existing project remains loaded. Opening a project replaces the loaded
  project, so save first if you need to keep its latest edits.
- A saved project can be opened locally or uploaded from a downloaded file.
  Replay and scrubbing use stored motion and need no Pod or inference.

If generation fails or the Pod connection drops, the viewer reports the
failure and retains already recorded takes. Retry after restoring the existing
Pod/tunnel with `./run-director.command`; the viewer's local playback controls
do not depend on inference. A stale result arriving after New project, Open, or
a seek is discarded. Returning to recorded preview is also available as a
fallback.

The teal circle is a gate activation area, not a motion constraint. Entering
it records a gate-open event. Scrubbing before that event closes the gate;
branches retain only earlier events and calculate new events from their motion.
This is deterministic scene logic, not AI prop manipulation or collision
physics. Explicit checkpoint-conditioned destination editing is not implemented.

Limits: 12 takes, ten minutes per take, and twenty minutes of total stored
motion. There is no take deletion or middle-of-take replacement. Branching
preserves the past and replaces the future. All browser tabs share one session.
The visual review and partial instruction-compliance findings are documented
in the results below; the model does not guarantee a destination or exact pose.

## Regression checks and evidence

Run the controller and persistence regression suite with:

```sh
.venv/bin/python -m unittest -v test_live_motion test_directing test_director_edges
```

All 24 offline tests pass. They use controlled synthetic transport and verify
controller behavior, branch preservation, project round trips, backup failure
retention, cancellation of late results, and save snapshot status. They do not
measure real inference or motion quality. Browser visuals and the sustained
Pod run are reported separately in [DIRECTING-RESULTS.md](../review/DIRECTING-RESULTS.md)
with measured results and screenshots.

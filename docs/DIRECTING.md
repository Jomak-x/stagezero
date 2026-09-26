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

## Studio interface

The viewer now has a collapsible studio sidebar with **Direct**, **Takes**,
**Scene**, **Camera**, and **Project** tabs. Playback and a seconds-based playhead
remain above the tabs; the bottom timeline shows the generated action segments
and supports frame scrubbing. Segment blocks describe stored motion and are
read-only; choose Replace ending in Direct to generate an alternate version.

- **Direct:** choose Create, Extend, or Replace ending; describe motion; set an
  Auto, new-motion, or total duration; and cancel an in-flight generation.
- **Takes:** rename, duplicate, trim through the playhead into a new take, jump to
  action boundaries, reuse an action's prompt, loop, and play at 0.25–2× speed.
  Duplication and trimming preserve the original performance.
- **Scene:** show/hide the grid and platform, enable the gate, and edit its floor
  position and trigger radius. Enable the move handle to drag the gate. Gate
  events are recalculated for stored takes when its geometry changes.
- **Camera:** perspective/front/side/top views, focus actor, orbit/pan/dolly
  buttons, and reset. Drag or two-finger scroll to pan, pinch to zoom, select Orbit or Look, and use
  WASD/QE for movement. Follow is off by default, preserves manual camera offsets,
  and does not sweep the camera on playhead jumps. Rewinding preserves the view.
- **Project:** saves and downloads all takes. Opening another project first backs
  up current takes. An existing project can also be opened at startup with
  `--project /path/to/project.stagezero.npz`; use `--port` for an isolated preview.

Studio regression checks:

```sh
.venv/bin/python -m unittest discover -v
```


### Touchpad studio client

Build the versioned browser client with `cd studio_client && npm ci && npm run build`. The launcher builds it when missing. Restart the viewer after Python changes and reload the browser after rebuilding the client.

Plain drag and two-finger scroll pan. Pinch zooms. The viewport toolbar switches drag between Pan, Orbit, and Look; Alt-drag or middle-drag also orbits. Navigation stays upright. Click the viewport before using WASD and Q/E to move, Space to play/pause, arrows to step, and Home/End to seek. Focus actor and Reset camera recover your view.

The timeline uses seconds and exact frame positions, reserves its own visible row, and previews scrubbing locally. The time field is an explicit seek input; it is not overwritten during playback. Playback status and rendered pose update together, and idle controls avoid repeated network updates.

### Create, extend, and revise motion

The Motion tab separates three operations. **New take** opens a fresh draft while keeping existing takes. **Add action** appends generated motion after the selected take. Click an action card or choose **Edit action** to load that action's direction, rewrite it, and replace only that action; later actions are retained and repositioned to join the result. **Delete selected action** closes the gap and updates the duration. Take options contains rename, duplicate, trim, and delete take. **Undo last edit** restores the latest edit, deletion, generation, or placement while no subsequent project change has superseded it.

In **Character**, choose **Move character start**, drag the red or blue floor arrow, and press **Done placing**. Exact start position also controls X, Z, and facing. Placement changes the stored motion and generation starting pose, so playback, continuation, and saved projects agree. Moving an existing take moves its entire motion path; it does not reroute that path around obstacles. Sequence joins align position and facing but remain cuts, not animation blends.

Duration planning runs at 25 frames per second. An explicit duration is rounded to the nearest frame. Requests longer than one model segment are generated as consecutive conditioned segments, with the final segment trimmed to the planned length. The controller installs the complete result only after every segment succeeds; cancellation or failure leaves the stored original intact.

Auto uses a duration written in the prompt when one is present, otherwise a labelled estimate based on instruction length and sequence words. It is an editing convenience, not an AI prediction of when an action is complete. Each request is limited to 30 seconds of new motion. Scene props and effects follow the take's timeline; this duration control concerns the motion take, not a separate scene clip.

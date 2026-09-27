# Background navigation demo (opt-in Native Core)

**Continuation checkpoint:** see [current status and next steps](handoff/BACKGROUND-INTERACTION-CONTINUE.md). The full temple route is unfinished.

This adds ordered spatial commands to the existing **Motion → Advanced scene
motion → Scene direction · Core → Move to a scene object** panel. It does not
replace the main G1 generator, AI cast, paired interactions, scene authoring,
character generation, or their archives.

Examples:

- `walk two metres forward then walk one metre left`
- `approach Streetlamp` (use the exact object name or stable ID)
- `open Checkpoint door then go through Checkpoint door`
- `go through Arch then walk one metre forward`

Choose the actor, type into **Spatial commands**, and press **Run spatial
commands**. Directions mean **turn and walk** relative to the actual committed
heading; “back” does not promise backpedalling. Each leg starts from the actual
completed previous motion. Up to four actions; relative distances 0.25–8 m.
Plain free-motion prompts still use **Generate / redirect**. This small command
grammar makes no LLM call and is not wired into the main AI-cast prompt parser.

## Motion and reactions

ARDY Core still generates every character frame at its native 20 fps. A large
initial direction change receives a two-second native stationary turn before
walking. The explicit spatial profile uses 1.2 m/s XZ targets without dense travel heading and a short eased final settle. Ordinary/default navigation scheduling is unchanged. Native generated arrays are not pose-warped. Optional terrain fitting and native terrain experiments now exist separately, are not activated by ordinary generation, and do not yet solve stair traversal.
The existing Navigate to object button retains its target schedule; the new spatial-command workflow also inserts a native turn before object travel when needed.

The controller checks collision proxies and continuous authored floor support
before committing each horizon. It measures terminal arrival (at most 0.30 m,
or half the requested displacement for short relative walks) and actual
passage crossing before advancing. A failed route or generation retains all
committed motion, with an explicit status. Arrival checks are not physical
contact or gait-quality guarantees.

The first successfully submitted spatial command opts this Core timeline into
versioned scene reactions. Only ground-level built-in doors configured with
`open` + `proximity`, and grounded proximity lamps, participate. A door lifts
its authored assembly smoothly over 0.8 seconds after actual root proximity;
`open` means operating this automatic door, not touching a handle. Subsequent
passage planning uses its actual raised geometry. Geometry checks and rendering
use the same evaluated transforms. Door history starts at activation, persists
through the full committed trajectory, and rewinds correctly when seeking.
Rejected candidate motion cannot retain events.

Old Core projects default to static props. New archives retain reaction version
and activation frame, exact native arrays, scene and command measurements.
Unfinished saved commands load for replay; they never resume inference by
loading the archive. Scene edits cancel pending commands as before.

## Demo and boundaries

See [complete videos, fresh tests and rejected attempts](../review/background-demo/README.md).
The checkpoint is an authored variation with a real wall opening; the original
Industrial yard preset has a solid wall behind its decorative door and is not
silently modified. Scene metadata supplies object location, size and rotation;
a custom mesh's name does not establish a usable hole or walkable terrain.

**The full temple stairs → elevated bridge → shrine door route is not solved.**
The temple example uses its ground courtyard only. Elevated floors, stairs,
hand-operated doors, arbitrary mesh articulation, grasping and physical contact
remain unsupported. Some hunched posture, foot glide and stiffness remain even
in the selected clips. This is a bounded demo release, not a claim that every
background or unrestricted instruction works reliably.

## Separate replay or live preview

After the normal project dependencies and client build:

```sh
PYTHONPATH=vendor/ardy .venv/bin/python director_viewer.py \
  --reference-only --port 2392 \
  --core-project review/background-demo/open-door-seed33/motion.core.stagezero.npz
```

This starts paused on an exact saved take. An absent private token permits
replay without contacting a GPU. For live commands add your existing
`--core-backend-url` and `--core-token-path`. This command never provisions a
Pod or restarts a worker. Do not replace the working main demo until this
separate opt-in feature has been reviewed.

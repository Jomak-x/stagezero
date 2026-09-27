# Scene direction in the Studio

The existing Studio has an optional **Motion → Scene direction · Core** panel
for one or two actors. It uses native ARDY Core motion and the current authored
scene. The Scene, Character, camera and ordinary G1 take workflows remain in
the same application.

Core clips have 27 joints at 20 fps. Ordinary Studio takes use G1's 34 joints at
25 fps. They have separate timelines and project archives: a Core archive is
not a G1 take, and switching modes does not convert either skeleton. Core mode
renders its own two supplied human character assets; selecting or generating a
G1/GLB character does not change the native Core cast.

For prompt-driven one-to-three-person scenes in the existing studio, use **Motion → Direct → AI cast · 1–3 people**; see [Main studio cast integration](MAIN-CAST.md).

For paired acting with separate starts and a shared meeting, use **Motion → Direct → Two characters**; see [Paired direction](PAIRED-DIRECTION.md). The existing native Core panel below is now inside **Advanced scene motion**. Core and paired native projects remain separate.

## Use the current UI

1. In **Scene**, load or create a layout. Start with a spacious set and a clear
   area near its centre. The cast-placement search avoids conservative prop
   footprints and spaces two actors apart. It fails explicitly when no suitable
   placement is found within the worker's bounds.
2. Open **Motion → Scene direction · Core** and enable **Use Core scene
   direction**. Choose **One actor** or **Two actors**, then press **Start /
   change cast** to apply a cast change. Replacing a populated Core timeline
   first saves a local backup.
3. Use **Direction idea → Use idea**, or write a separate direction for each
   actor. Select 2, 6 or 12 seconds and press **Generate / redirect**. The
   generator appends complete two-second native windows to the committed
   timeline. A redirect changes pending work while preserving committed frames.
4. Use **Play**, **Pause**, **Restart**, the frame slider or the Studio timeline
   to inspect the result. Playback holds the last available real frame if the
   generator falls behind. **Cancel pending motion** stops further requested
   work; **Retry failed generation** retries a failed pending request.
5. Use **Save exact Core project + download** and **Core projects → Open saved Core project**
   (or **Open Core project file**) inside this panel. The archive includes native motion, cast IDs, scene
   metadata and playback state. Loading does not start inference. The ordinary
   Studio **Save project + download** button saves the separate G1 project.
6. Disable **Use Core scene direction** to return to the G1 timeline and cast.

When a bundled example is available, **Play included motion example** loads a
recorded native clip without contacting the worker. It is a recorded example,
not a fresh result for the current prompt. Generated work and loaded recordings
should still be watched through before use.

## Navigate through known geometry

In **Move to a scene object**, choose the actor, a named object with its stable
ID, and an available action. **Navigate to object** compiles `approach` or
`go_through` into dense root-position and heading targets. The other actor holds
its position while the planner avoids its footprint. This is staged movement;
the model does not jointly reason about two people interacting.

The catalog preserves the current scene's IDs, transforms and dimensions. It
supports the Studio's 64-object limit and the low custom ground slabs used by
generated city scenes. Rotated custom footprints retain their rotation. Ground
navigation stays on the Y=0 plane; raised slabs and stairs do not become
walkable just because they look like floors. Core target coordinates must stay
within ±25 metres, and a navigation request including its final hold must fit
within 30 seconds.

A procedural arch has a known opening. A built-in door is passable only when
its actual raised geometry leaves enough clearance; the normal closed door is
solid. The adapter can read evaluated open-door states, but Core mode currently
uses authored, static object positions, so it does not automatically open doors.
The separate legacy G1 gate trigger is not a Core navigation target. A custom
object named “gate”, “door” or “portal” supplies no verified hole. The Studio
catalog therefore does not offer passage through arbitrary custom meshes.

The UI selects exact IDs, avoiding duplicate-name ambiguity. The adapter also
supports exact names for programmatic callers and rejects ambiguous or unknown
names. Planning through a blocked, narrow, low or unverified opening fails
instead of substituting another target.

Before a generated window is committed, the controller checks sampled body
spheres against conservative prop boxes and checks two actors' planar root
discs with a 0.65-metre combined threshold. A failed window leaves prior motion
intact. These checks can reject visually acceptable close encounters and do
not prove watertight mesh clearance or physical contact. Planned waypoints are
conditioning targets; seeing them in a plan does not establish goal completion.

## Object reactions and current limits

Core mode keeps props at their authored positions so motion planning and
rendering use the same scene geometry. Scene particle effects follow the Core
playhead. Existing proximity, opening, activation, pickup and sit-style prop
reactions remain in G1 mode; they are not silently driven from Core joint
indices. The static scene can still be edited, and a scene change invalidates
pending Core generation.

There is no general visual scene understanding, grasp solver, prop attachment,
object ownership transfer, shared handoff or contact physics in this panel.
Finger and facial acting are not generated here. Two independently conditioned
actors are not a learned handshake, fight or conversation system. InterGen
paired-contact research remains excluded from this UI and its production
archive format; its research licensing and quality limits are not bypassed by
calling it a Studio feature. Swing experiments are not exposed by these
controls.

## Useful directions to explore

These are candidate prompts, not a tested success matrix. Start with short
beats, inspect the motion, then extend the sequence.

| Intent | Practical direction | Boundary |
| --- | --- | --- |
| Solo martial arts | “Perform a controlled martial arts form, settle into a guard, then relax.” | Solo action; no opponent or impact claim. |
| Dance or expressive acting | “Dance with rhythmic steps,” or “notice something surprising, pause, then celebrate.” | Text controls broad movement; precise acting beats can vary. |
| Staged opponents | One actor feints from a safe distance; the other retreats or raises a guard. | Separate performances; contact or synchronized reactions are not guaranteed. |
| Conversation | One person gestures while speaking; the other listens and responds. | Blocking and body language, without speech, lip sync or gaze guarantees. |
| Celebration | Give each actor a different celebratory direction with room to move. | Two actors share timing, but do not learn social coordination together. |
| Story blocking | Approach a known prop, pause for a short gesture, then navigate through a verified arch. | Use object navigation for spatial intent and short directions for acting. |

The underlying earlier interaction experiments measured dense Core gate routes,
slow two-actor staging and reference-conditioned hand targets. Those bounded
results do not establish arbitrary scene or prompt reliability in this new UI.
The current panel exposes navigation and free direction; it does not expose a
generic hand-contact command.

## Launch against an existing service

Use the installed viewer environment, built `studio_client/build`, initialized
ARDY submodule, authorized private G1 recording CSV, and existing private token
files described in [the setup guide](../README.md#local-viewer-setup-authorized-recording-required).
The default Studio launch loads the G1 CSV and token. For offline native replay,
use `--reference-only` as described below; that mode needs neither private file.

Run from the repository root, adapting the paths to an existing installation:

```sh
.venv/bin/python director_viewer.py \
  --port 2336 \
  --recording /absolute/path/to/recorded_g1.csv \
  --token-path /absolute/path/to/existing-g1-token \
  --core-backend-url http://127.0.0.1:8769 \
  --core-token-path /absolute/path/to/existing-core-token
```

`--core-token-path` defaults to `--token-path`. If the Core token is absent,
the Core session supports archive/example playback without generation; the
separate default G1 startup token requirement still applies unless `--reference-only` is set. A token file alone does
not establish service health. A refused or unavailable worker produces an
explicit generation failure.

These arguments connect to an already running, authenticated private Core
service, normally through an existing loopback tunnel. This command does not
provision a Pod, download model weights, start a worker or restart the G1
service. The historical `run-director.command` launcher has its own G1 startup
and tunnel behaviour; use the direct command above when those resources are
already managed.

## Verification and measurements

The final release check passed 684 Python tests, 29 client tests, TypeScript
checking and the production client build.

CPU verification covers the actual 50-object generated city, custom floor
handling, rotated footprint placement, the 64-object limit, blocked arches,
closed and evaluated raised doors, ambiguous names, unknown custom openings,
and Core coordinate bounds. Controller and renderer checks cover their own
contracts; fake generation cannot establish motion quality or model latency.

The integrated branch was verified with the existing warm Core worker and twelve
serial GPU jobs; no worker was restarted and no Pod was rented. See
[the complete measured report](../review/studio-core/live-results.json) and
`experiments/verify_studio_core.py` for reproduction.

| Real case | Result |
| --- | --- |
| Approach a streetlamp in main's actual 50-object generated city | 160 frames; first new commit 0.706 s; final planned-mark error 3.0 mm; zero sampled solid-proxy overlap frames |
| Cross the procedural arch | 120 frames; first new commit 0.881 s; measured aperture crossing; final mark error 3.8 mm; zero sampled solid-proxy overlap frames |
| Buffering in those two controller runs | No observed post-start holds; these are controller observations, not browser frame-rate measurements |
| Root steps at horizon boundaries | Maximum 4.10 cm in city, 2.82 cm at arch |
| Two actors generated through the actual main UI | 120 synchronized frames; minimum root separation 0.937 m; no root-disc overlap; maximum boundary root step 4.01 cm |
| Transport failure then retry | Real failed connection; repaired service continued with exact old positions, rotations and native-feature prefix |
| Saved projects | Exact array round trips; actual studio local project picker restores native motion and saved generated geometry |

Floor support is checked continuously over a 0.28 m actor footprint. Visual review
invalidated the original arch attempt because its exit left the platform. The
corrected run uses the same generated recipe with the arch moved from Z=-2.6 m
to Z=-1.5 m, leaving supported floor on both sides. The report retains the rejected
attempt and its measurements; the shipped arch replay is the corrected run.

The root agent visually reviewed the textured two-actor result and the city and
arch projects inside the preserved main studio. These checks establish bounded
mechanics, not guaranteed action semantics, realistic fighting or physical
contact. The first-commit values are not command-to-visible-action latency.
The browser's file-upload automation was unavailable because the extension
lacked file-URL access; the local saved-project picker was tested instead.


## Further work

Contact-aware learned controllers need shared actor/object state, explicit
contact timing, physical feasibility checks and measured entry/exit behaviour.
A bounded, reviewed motion library with motion matching could provide dependable
repeated gestures and transitions before trying unrestricted paired prompts.
Foot and hand IK can improve placement when evaluated against actual geometry;
finger articulation, facial expression and gaze need their own rigs and
animation sources. Each addition needs complete-scene review so local pose
corrections do not conceal sliding, collisions or broken motion boundaries.

## Replay on a clean checkout

After the normal Python/client dependency setup, the native sample needs no private CSV,
API token, or GPU:

```bash
git submodule update --init --recursive
(cd studio_client && npm ci && npm run build)
.venv/bin/python director_viewer.py --reference-only --port 2371
```

Open Motion → Scene direction · Core → Play included motion example.
`--reference-only` starts G1 on a static reference pose; it does not pretend that
pose is generated animation. Live Core direction additionally needs the existing
service URL and token described above.

## Two-character choreography update

The integrated **Together · experimental** controls now include a reviewed
`pose_duet_v1` recipe and separate AI candidate planning. See the
[two-character review](TWO-CHARACTER-REVIEW.md) for the complete selected video,
UI steps, reproduction commands, 730-test verification, measured timing and
clearance, preserved rejected trials, and remaining transition/foot-slide limits.


## Separate joint-pair research workflow

The independent Core duet was rejected in user review for spacing and timing. A separate opt-in joint-model workflow now sits under Together, with separate paired archives and playback; it does not change the native Core contract above. See [Joint two-character performance review](TWO-CHARACTER-PAIRED.md) for the replacement, videos, rejected trials and reproduction commands.


### Native paired recovery review

The retargeted paired preview was rejected by the user. See [NATIVE-PAIR-RECOVERY.md](NATIVE-PAIR-RECOVERY.md) for the measured conversion defects, restored original InterGen preview, direct authored-rig experiment, actual InterMask trials, full captures and candid rejections. This research has not been promoted into the ready Core/G1 workflow.

## Native cast integration review

See [NATIVE-CAST.md](NATIVE-CAST.md) for native InterGen cast selection, exact archives, studio video export and optional ARDY approach/exit. The approved native character rendering is retained. Sparring is a research preview; physical strikes, foot locking and multi-party joint generation are not solved. ARDY composition explicitly labels its model sources and authored transitions. All earlier rejected arch/platform and paired-rendering evidence remains preserved.

## Prompt-scene speed and cast review

The separate `prompt_scene_viewer.py` preview accepts one prompt for 1–3 performers
and automatic action durations. It combines exact native paired sources with real
Core travel, explicit authored transitions, and stationary inactive performers.
It uses the existing generated city/market/industrial backgrounds. Warm inference
and intent caching improve measured latency without reusing generated motion.
See [prompt-scene evidence](../review/prompt-scenes/README.md) for complete videos,
measurements, reproduction commands, and preserved failures. This does not replace
main's current UI or promote arbitrary contact/falls/three-body motion as solved.

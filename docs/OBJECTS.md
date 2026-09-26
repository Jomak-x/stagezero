# Scene generation: props, effects and lighting

Describe a scene → validate a scene document → build procedural 3D geometry →
replay actor reactions and effects. Use **Scene → Scene generator** in the studio.
The UI binding is replaceable; generation, validation and playback do not depend
on the studio layout or browser client.

## What works

- **16 prop kinds:** door, lamp, ball, chair, table, sofa, crate, barrel, pillar,
  wall, arch, plant, tree, rock, console and platform. Up to 40 props per scene.
- **Six effects:** rain streaks, drifting snow, fireflies, sparks, smoke-like
  particles and a rotating portal. Up to eight emitters, with bounded particle
  counts. Same seed and playhead always produce the same effect state.
- **Five light palettes:** neutral, warm, moonlight, neon and sunset.
- **Five complete starter sets:** Neon research lab, Enchanted grove, Cozy living
  room, Industrial yard and Winter plaza.
- **Editing:** move, duplicate and remove props; add/clear effects, adjust effect
  density when adding, switch lighting, frame the whole scene, download/import
  scene JSON. Scene files work without generating a motion take first.
- **Project persistence:** props, effects and lighting save with all takes.
  Old object files and projects remain compatible. Invalid files or failed
  generation preserve the current scene; late AI results cannot overwrite a
  project edited during the request.

Effects follow the motion playhead: press **Play** to animate, **Pause** to hold,
and scrub to reproduce an exact moment. They do not run on an independent clock.
**Frame whole scene** fits a larger set into the current client's camera. Starter
sets and successful generation also frame the result automatically.

## Three generation sources

**Recipes · offline** is immediate and makes no network calls. Use a set name or
phrases such as “enchanted forest with fireflies”, “industrial yard with rain”,
“three red crates and two blue barrels”, or “snow and fireflies”. Named sets use
a deliberate layout; counted prop lists use a grid. These recipes match keywords,
counts and a small color vocabulary; they do not understand arbitrary language,
negation or spatial instructions. Variation seed changes particle patterns.

**AI gateway · Neon** sends your description and the supported catalog to a text
model. The model chooses the composition, sizes, palette, effects and lighting.
It creates a structured scene, not an arbitrary textured mesh. All output is
validated. A schema failure gets at most one correction attempt (a second model
call); network/HTTP failures are not automatically retried. Failure preserves the
current scene and never silently switches to recipes.

**Local AI · Ollama** supports a locally running Ollama instance at
`127.0.0.1:11434`, using `qwen3:4b` by default. Set `STAGEZERO_LOCAL_MODEL` in the
viewer process environment to select another installed model. This mode sends no
credentials and does not contact Neon. It never installs or downloads a model.
The local request adapter is tested with simulated responses; no local model was
installed or benchmarked in this implementation. See
[local and mesh-model options](SCENE-GENERATION-OPTIONS.md) for hardware, licenses
and an optional setup command.

## Neon credentials

The gateway reads recognized entries from the private, Git-ignored
`.runtime/objects.env` file automatically; process environment overrides the file.
The parser accepts assignments and optional `export`, not executable shell code.
AWS keys in that file are ignored by scene generation.

```sh
NEON_AI_GATEWAY_BASE_URL='https://YOUR_BRANCH_GATEWAY_HOST'
NEON_AI_GATEWAY_TOKEN='YOUR_GATEWAY_TOKEN'
STAGEZERO_OBJECT_MODEL='gpt-5-mini'
```

The gateway token needs `ai_gateway:invoke`. The code appends `/v1` to Neon's bare
host, then calls `/chat/completions`. Alternative OpenAI-compatible gateways can
set `STAGEZERO_OBJECT_API_BASE` (including `/v1`) and `STAGEZERO_OBJECT_API_KEY`.
Keep tokens out of scene files, chat messages and source control. Credentials
never enter browser controls, saved projects or generated scene documents.

Live Neon authentication, model discovery and scene generation were verified.
The reusable `examples/scenes/ai-observatory.json` was produced by a live gateway
request; the other five scene examples are deterministic procedural recipes.

## Command line

```sh
.venv/bin/python scene_generation.py 'neon lab with sparks' --source recipe --output .runtime/scene.json
.venv/bin/python scene_generation.py 'A mysterious observatory with plants and a teal portal' --source gateway --output .runtime/ai-scene.json
.venv/bin/python scene_generation.py 'A small room with a sofa and lamp' --source local --output .runtime/local-scene.json
.venv/bin/python director_viewer.py --port 2340 --objects .runtime/scene.json
```

`object_generation.py` retains the original four-prop offline CLI for compatibility.
The expanded workflow is `scene_generation.py`.

## Interaction and visual limits

Walking and stair commands now use scene geometry in the studio. The motion
controller reads the same normalized, scaled and rotated custom meshes as the
renderer. Stair commands select a reachable flight, approach it, and place feet
on the rendered treads using leg inverse kinematics. Corrected positions,
rotations and normalized motion features are stored together and reused for
continuation. Auto length accounts for the stair route (up to 30 seconds);
explicit durations can end partway through the approach or ascent. Changing the
scene during generation cancels the pending action. Existing saved takes keep
their original motion; regenerate an action to apply scene grounding.

Doors lift, lamps brighten, balls attach to the first hand that touches them,
chairs highlight on approach, and console screens activate. These remain
kinematic reactions. A chair cue does not force a sitting pose, and there is
no general grip, throw/release, or rigid-body physics solver. Locomotion stops
at unsupported ground or blocking object bounds; it does not perform general
navigation around arbitrary obstacles. Stair flights must have identifiable
horizontal box treads and a reachable approach. Unsupported routes report an
error instead of generating an unconstrained climb.

Effects and geometry are stylized mockups. Smoke uses opaque/dim particles rather
than volumetric simulation. Lighting adds colored point lights to the studio's
existing lights; this is not a postprocessing bloom or material system. Props
support yaw rotation. Scene scale/placement can be edited through JSON or
prop-position controls. Large set dressing may obscure the actor from some
angles; use the camera controls or **Frame whole scene**.

## Integration contract

- `scene_composition.validate_scene(doc)` validates version 2 documents with
  exactly `version`, `name`, `objects`, `effects`, `lighting`.
- `make_preset(name, seed)` / `generate_recipe(prompt, seed)` produce offline scenes.
- `SceneGenerator.generate(prompt)` and `LocalSceneGenerator.generate(prompt)`
  return the same scene shape.
- `ObjectDirectorSession.generate_scene`, `set_scene`, `scene_document`,
  `edit_object`, `duplicate_object`, `remove_object`, `load_scene_document` handle
  controller integration without a UI dependency.
- `object_states()` returns a render bundle: `objects` (evaluated states),
  `effects` (specs), `seconds` (playhead), `lighting` (preset).
- `ObjectSceneLayer.update(objects, bundle)` renders props and delegates effects
  and lighting to `SceneAtmosphereLayer`. Legacy state-list callers still work.
- `add_object_controls(gui, session)` attaches the scene authoring panel.

Coordinates are meters, +Y up, with center-based positions. Prop sizes are
0.05–12 m; effect sizes are 0.1–8 m; positions are bounded to ±20 m. G1 hand
endpoints are joints 25 and 33. Generated code, external asset URLs and unknown
behavior fields are rejected.

## Verification

```sh
.venv/bin/python -m unittest -v test_scene_objects test_scene_effects test_scene_composition test_object_generation
```

Coverage includes geometry bounds, static and interactive props, deterministic
seeking, effect resource reuse/cleanup, strict input validation, presets, counted
recipes, legacy migration, save/load, stale generation and simulated gateway/local
responses. Browser checks use an isolated preview, not the main viewer process.

## Original props and architectural sets

The Neon scene workflow now has two stages: design a reusable pack of original
props, then lay out a scene using that pack. Cities and interiors use stable architectural layouts populated with the new geometry; other settings use AI composition. In the Scene tab, describe your set
and select **AI gateway · Neon**. **Generate scene + custom props** does both stages automatically;
**Prepare custom props · AI** lets you prepare first and compose afterward with
exactly that prompt's pack. **Reuse saved prop library** uses the most recent
saved assets (up to 16). Prepared assets are stored privately under
`.runtime/scene-assets`; scene/project exports embed geometry and remain portable.

Custom props are assemblies of colored boxes, spheres, cylinders and cones, with
rotation and repetition for details such as window grids, shelves and floorboards.
This is stylized generated geometry, not textured image-to-3D reconstruction.
The model can invent new assemblies instead of selecting only catalog objects.
Generated geometry is fitted to normalized local bounds before strict validation.
The renderer compiles each custom instance to a colored GLB and caches meshes.

Version 3 scenes add `assets` and an optional `camera` to the version 2 document.
Custom objects reference an asset ID. All props support optional yaw, including rotated interaction bounds. Limits are 16 assets,
64 parts per asset, 512 expanded shapes per asset, 64 object instances and 12,000
expanded shapes and 250,000 generated triangles per scene. Shapes have bounded coordinates, colors and dimensions;
generated code and remote asset URLs are not executed or loaded. Existing v1/v2
scenes remain supported. Imports and compact exports share a 1 MB limit.

The offline **City boulevard** starter includes layered buildings, framed windows,
storefronts, sidewalks, benches, trees and lamps. **Designed apartment** includes
an open-front room, paneled walls, windows, sofa, table, bookshelves and details.
**Scene camera** uses the authored view; **Frame whole scene** fits the full set.
Prop editing supports position, dimensions, duplication, removal and yaw.
Local Ollama remains an optional catalog-scene composer; no model was downloaded.

CLI:

```sh
.venv/bin/python scene_generation.py 'A detailed art deco city street' --source gateway --output examples/my-city.json
.venv/bin/python scene_generation.py 'city boulevard' --output examples/city.json
```


## Scene refinement and model selection

Neon scene asset design and AI layout default to `gpt-5-6-sol`. The model ID was
verified against this gateway's model catalog and exercised with real city and
room requests. Small functional-object requests retain `STAGEZERO_OBJECT_MODEL`.
Optional environment overrides are `STAGEZERO_SCENE_ASSET_MODEL` and
`STAGEZERO_SCENE_LAYOUT_MODEL`; no additional credentials are required.

The single generation action designs props, checks and repairs common geometry
faults, saves the reviewed pack, and stages the scene. Repeated panes must fit
within their spacing; facade details must sit in front of opaque backing; small
unsupported details and coplanar overlays are corrected. Facade repairs at the local bounds reserve a small margin before retrying; world
size is preserved by renderer normalization. Remaining errors get a bounded model
repair attempt. These checks also run on reused packs. They are
geometry checks, not an automatic aesthetic rating. Original city/interior assets
use deterministic architectural placement; open-ended AI layout is used for
other environments. Saved packs avoid another design call when reused.

Static object state and transforms are cached instead of serialized and resent
every frame. Editing one prop rebuilds only its affected geometry. Broad custom
GLBs (any face area at least 4 square meters) cast shadows but do not receive them to
avoid the renderer's visible self-shadow bands. Smaller props still receive shadows.
This trades some shadows on large surfaces for stable, clean rendering.

Visual acceptance evidence, the 30 fps render captures, live generation metadata,
and a server-side performance benchmark are in `review/scene-refinement/`.
The benchmark measures Python update overhead, not interactive browser FPS.


## Larger scenes and stress testing

City boulevard now lays out 24 buildings in street-facing rows and a distant
skyline, with sidewalks, trees, lamps and benches (50 objects total).
Residential neighborhood adds 16 houses, lawns, picket fences, mailboxes and
street furniture (56 offline objects; the generated version can add a cafe and
benches for 59). Market square and Warehouse workshop have dedicated staging
with an open central route, rather than using living-room placement. These are
available as starter sets and through automatic scene-prompt routing.

Fresh scene packs request 6–8 reusable assets with sufficient response space;
malformed JSON gets one bounded repair attempt. Market stall variants and
warehouse machinery have distinct roles. The generated examples under
`examples/scenes/` embed all geometry and are editable after import.

Limits remain 64 objects, 16 assets, 512 shapes per asset and 12,000 shapes per
scene. A second 250,000-triangle budget reflects curved primitives' actual cost:
a sphere has 168 triangles while a box has 12. Over-budget scenes are rejected
before replacing the active scene. The UI exports compact JSON and both file
load paths accept up to 1 MB, fixing large exported scenes failing re-import.

`review/scene-scale/` contains the live generation metadata, scene metrics,
server stress diagnostic, multi-angle WebGL captures, and a captioned 36-second
demo. The 64-object / 12,000-box-primitive test rendered successfully. A former
legal sphere-heavy scene exceeded 2 million triangles; it was captured for
diagnosis and is now rejected by the triangle limit. The video frame rate and
capture timings are not an interactive-browser FPS benchmark. Shared-asset edits
can still briefly rebuild many instances, even though static updates are cheap.

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
Local generation uses a compact JSON schema for the model's prop/effect choices,
positions, dimensions and colors. Python adds IDs, interaction defaults and seeds,
then validates the normal scene document. Requests allow up to 300 seconds for
model loading and generation, with a distinct message for a read timeout.

Verified on an M4 MacBook Air with 16 GB RAM, Ollama 0.34.4 and `qwen3:4b`:
the loaded model used about 3.3 GB, and a detailed six-prop scene took about
27 seconds after loading. General prompts can still produce poor placement;
explicit object counts, positions and dimensions improve control. If the viewer
runs on a remote machine, its loopback Ollama address must reach the Mac through
a private SSH reverse tunnel. The model runs on the Mac, not in the browser.
See
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

Doors lift, lamps brighten, balls attach to the first hand that touches them,
chairs highlight on approach, and console screens activate. Static scenery has
no interaction trigger. Balls remain attached for the rest of the take. A chair
cue does not force a sitting pose. Props are kinematic reactions, not physics:
there is no collision response, obstacle avoidance, grip solver, or throw/release.
The scene generator itself does not constrain the actor's motion.

Effects and geometry are stylized mockups. Smoke uses opaque/dim particles rather
than volumetric simulation. Lighting adds colored point lights to the studio's
existing lights; this is not a postprocessing bloom or material system. Props
have no rotation field yet. Scene scale/placement can be edited through JSON or
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

# Character generator

The **Character** tab uses **Neon AI Gateway and a self-hosted TRELLIS worker**.
It does not call Meshy or require Meshy credits.

1. Describe a character's appearance, outfit, and personality.
2. Neon **gpt-6-astra** develops a coherent visual design.
3. Neon's image-generation tool renders a detailed, photographic-quality human reference in an A-pose.
4. Our GPU reconstructs a mesh, bakes a 2048-pixel color texture, and restores visible face and clothing detail from the reference.
5. The studio fits the body to motion and immediately uses it as the actor.

Choose **Create & use character**, then press **Play** or describe an action in
**Direct**. The generated person replaces the default actor. **Your cast** keeps
characters for reuse; **Use this character** switches the actor and **Use default
actor** restores the robot. The selected character is restored after restart.
No download or manual import is required.

Example: `A silver-haired adult space mechanic in a worn orange flight suit,
rolled sleeves, dark utility boots, and a warm, confident face`.

The **Live design preview** shows the reference while the GPU works. Elapsed time and the current stage remain visible, and failures leave your current actor intact.
Generation takes a few minutes, depending on image generation and GPU load.
A verified original rescue-medic run took 81 seconds for the Neon reference
and 88 seconds for GPU reconstruction and download on the current setup.

## Connections

The reference stage reuses `NEON_AI_GATEWAY_BASE_URL` and
`NEON_AI_GATEWAY_TOKEN` from `.runtime/objects.env` or the environment. It uses
the gateway's OpenAI Responses route with its image-generation tool. Keys stay
on the server. `STAGEZERO_CHARACTER_IMAGE_MODEL` optionally selects the GPT
image orchestrator; the tested default is `gpt-6-astra`.
`STAGEZERO_CHARACTER_DESIGN_MODEL` selects the design model; its default
`gpt-6-astra` was confirmed in the gateway's model list and tested live.

The geometry stage uses the existing GPU Pod, in its own environment at
`/workspace/stagezero-characters`. It listens only on `127.0.0.1:8770`, requires
the existing private backend token, and is reached through an SSH tunnel.
Run `./run-character-backend.command` to deploy the worker source and recover
that connection after the model environment has been installed.
See [GPU installation and recovery](CHARACTER-GPU-INSTALL.md) for the verified
runtime, model versions, and setup steps.

The current Pod has limited disk space. Its character checkpoints are cached
in `/dev/shm/stagezero-characters/hf`, a memory-backed filesystem. This cache
and the readiness marker disappear when the Pod restarts. Run the backend
launcher to restore them: it starts a background download and smoke test using
the saved verification reference. Progress is logged in `warmup.log` on the Pod.
A worker without its readiness marker refuses jobs rather than showing a false ready
state. The Mac's saved character library is independent of that cache.

Neon usage follows the user's gateway plan. Self-hosted generation has no
Meshy per-asset fee, but still uses the existing GPU's compute and hosting.
The launcher does not create, resize, or stop paid resources.

## Jobs and library

Only one GPU character job runs at a time. **Cancel generation** stops the local
request and asks our worker to terminate its reconstruction process. A Neon
image request already in flight may finish remotely. Existing saved assets
remain intact on failure or cancellation. If a completed design is followed by
a GPU failure, retrying the same description reuses that design once in memory
instead of making another Neon image request. Changing the description requests
a new design.

Characters are stored automatically under `.runtime/characters/` as GLB and
metadata files, with their PNG reference. The `.active.json` marker remembers
the current actor. The original mesh stays intact; motion fitting happens in
the studio. Motion project files do not currently bundle the character library.

An optional **Import existing model** section supports self-contained GLB 2.0
files up to 40 MiB. Automatic body fitting expects an upright, front-facing
human-shaped model with separated limbs. External resources are refused.

## Quality and animation

This is image-conditioned 3D reconstruction, not a trained character-specific
anatomy system. The back of the character is inferred. Thin fingers, hair,
hidden details, and facial likeness can need cleanup. A strong reference image
does not guarantee an equally detailed mesh.

Body fitting uses an approximate human rig driven by the existing motion
skeleton. Fitted joints and hierarchical rotations preserve limb lengths;
mesh connectivity separates arm and leg weights even when hands sit near hips.
World X/Z travel stays aligned with scene targets. A deterministic two-bone
leg correction clears boots without lifting the whole body; vertical motion
uses fitted pelvis height. Seeking a frame gives the same pose in either direction. This is suitable for quick character creation and motion previews;
it is not a production artist's rig. Fingers and faces do not animate
independently. Long coats, skirts, bulky silhouettes, fused limbs, and extreme
poses can distort. Prefer one upright human-shaped subject with separated
arms and legs. There is no cloth simulation or facial performance rig.

Front-detail projection is visibility-tested and blended into the original bake;
UV padding prevents pale island seams. Side and back detail still depend on
reconstruction quality. The renderer preserves the full 2048 color texture
and smooth normals while the body animates. Authored normal and metallic/roughness
maps and material factors are preserved when present; the engine does not invent
missing material maps. The studio uses its bundled studio environment for lighting.
Only bone transforms change during playback, rather than transferring an
entire model for every frame.

Sources: [Neon Responses API](https://neon.com/docs/ai-gateway/openai-responses),
[Microsoft TRELLIS](https://github.com/microsoft/TRELLIS).

## Rejected local costume prototype

The saved Spider-Man character was built locally with `character_costume.py`: a
fitted red/blue costume, masked head, white lenses, web pattern and spider emblem
over an existing AI-generated human mesh. Both image services failed on the
Spider-Man reference request, so this asset is explicitly marked
`procedural-costume`; it is not represented as a successful Neon reconstruction.
It uses the same animated actor path, but it did not meet the requested visual
quality. Its donor silhouette and simplified mask are not a realistic anatomy
solution. Passing motion checks did not validate its appearance.

New candidates must pass the [visual review](CHARACTER-VISUAL-REVIEW.md), including
neutral-clay anatomy, multiple views, closeups, and motion poses. Provider failures
must not be represented as successful character generation.

Final test evidence and remaining limits: [character verification](CHARACTER-VERIFICATION.md).

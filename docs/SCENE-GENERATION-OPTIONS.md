# Scene generation options

Implementation update: Neon now designs original bounded geometry packs, saves them locally, and composes version 3 scenes from their reusable assets. See [OBJECTS.md](OBJECTS.md). The model comparisons below are background research; no mesh model has been installed.

Checked 2026-09-26 against upstream documentation. This Mac mini has an Apple M6 and 32 GB unified memory. `ollama` 0.34.3 is installed, but its local server was not reachable during this check. No model manifests were present in the default `~/.ollama/models/manifests` directory, and `OLLAMA_MODELS` was not set in this shell. No local model availability or inference speed has been verified. The existing motion Pod has an RTX 6000 Ada GPU, but its ARDY environment and VRAM are already used for motion generation.

## Best next step: a local scene planner

Use a small text model to turn a prompt into a **validated scene specification**, then build objects, rooms, lighting, and effects with deterministic renderer code. This keeps edits, seeking, saving, and contact behavior controllable. [Ollama supports JSON-schema constrained output](https://ollama.com/blog/structured-outputs) and [local OpenAI-compatible chat completions](https://docs.ollama.com/quickstart). [Qwen3:4b](https://ollama.com/library/qwen3) is a reasonable first trial on 32 GB unified memory: Ollama lists a 2.5 GB model download, while [Qwen's upstream 4B model is Apache 2.0 licensed](https://huggingface.co/Qwen/Qwen3-4B). Model download size is not peak runtime memory, and quality/latency must be measured in this studio. Try 8B only if 4B regularly fails the scene schema or prompt intent.

Optional setup, when ready to download the model: `ollama pull qwen3:4b`. Open the Ollama app or start its server before selecting **Local model** in the studio. This setup has not been run here.

The current `GatewayGenerator` already validates prop JSON, but it **requires HTTPS** and only knows four prop kinds. A local Ollama connection therefore needs a separate loopback adapter or an explicit, narrowly scoped localhost exception, plus a richer versioned schema. Keep hard limits on prop count, coordinates, dimensions, allowed materials, light intensity, particle counts, and effect duration. Let the model select catalog items and parameters; do not execute generated code or load arbitrary file paths from its output.

## Mesh models for an optional asset pipeline

| Model | Upstream capability and requirements | Fit here |
| --- | --- | --- |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR/blob/main/README.md) | Single image → 3D mesh; MIT code and weights. README gives about 6 GB VRAM for one image and allows CPU fallback for a mesh extraction dependency. | Good small **offline experiment** for a prop from a reference image. It does not turn a whole text scene into editable objects or supply interaction logic. Apple Silicon performance is unverified. |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | Image → mesh, optional texturing; can export GLB and has an API server. README claims macOS support and lists 6 GB VRAM for shape, 16 GB for shape plus texture; texture setup builds custom rasterizers. [Community license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE) excludes EU, UK, and South Korea and has other conditions. | The Mac's 32 GB unified memory may hold some configurations, but CPU/MPS compatibility, speed, and the texture build need a real trial. A separate Pod environment is more predictable. Review the license before shipping assets or a service. |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2/blob/main/README.md) | High quality image → textured 3D/GLB. Official setup requires Linux, CUDA, and an NVIDIA GPU with at least 24 GB VRAM; model/code are MIT, with separate dependency terms. | Technically suited to the existing RTX 6000 Ada Pod **as a separate batch worker**, after checking available VRAM, disk, dependencies, and its [third-party licenses](https://github.com/microsoft/TRELLIS.2/blob/main/README.md#%EF%B8%8F-license). It is not a Mac-local interactive dependency or an ARDY drop-in. |

These tools reconstruct **individual static assets** from images. A full scene still needs layout, scale, collisions, lighting, animation, and editable identities. An attractive GLB does not provide a door hinge, grasp point, sit target, or particle behavior. For mockup speed, a parametric object catalog is more useful than generating every chair or wall as a new mesh. Add mesh import later for hero props and keep a simple collision/interaction proxy beside each imported asset.

## Effects should begin as procedural presets

Expose a bounded effect vocabulary in the scene spec: fog/haze, dust motes, sparks, smoke, rain, snow, fireflies, glow, light flicker, impact rings, trails, and color grading. Define emitter position, shape, color, rate, lifetime, speed, size, seed, and start/end times. Seeded simulation or time-derived particle positions make timeline seeking and project replay repeatable. Trigger effects from validated scene events such as `door_open`, `lamp_on`, `ball_touch`, and `foot_contact`. The planner can choose a preset and parameters from prose while the renderer guarantees safe counts and predictable behavior.

For later mesh integration, prefer an offline queue: reference image → model output → review/decimate/normalize → GLB asset with metadata → import into studio with an authored interaction proxy. Keep this separate from live motion inference. The first useful evaluation is whether a generated prop looks better than a parametric one **at the same scene scale**, loads quickly, and remains editable; reported upstream quality or GPU times are not measurements on this project.

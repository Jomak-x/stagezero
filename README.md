# StageZero — Single-actor directing

The Studio now supports [GLB character import and humanoid rig mapping](docs/GLB-CHARACTERS.md).
Use `director_viewer.py --glb /path/to/character.glb` for GLB characters with the
normal **Direct** prompt/Generate controls, takes and timeline. The **Character**
tab loads or switches models. Live generation uses the existing authorized ARDY
backend; see the [GLB startup guide](docs/GLB-CHARACTERS.md#use-glbs-with-prompts-and-live-generation).
`asset_viewer.py` is a separate synthetic diagnostic lab and has no prompt generation.

Real ARDY G1 generation is connected to the existing viewer. The original recorded preview is retained in a separate, clearly labeled mode. This is **complete-segment generation, not streaming**: each instruction produces 104 fresh frames (4.16 seconds at 25 fps), then playback begins.

## Clone and collaborate

```sh
git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git
cd stagezero
```

This repository excludes the gated recorded CSV, model weights, installed
environments and all credentials. Start with the [offline controller checks in
CONTRIBUTING.md](CONTRIBUTING.md#offline-controller-checks): they need no Pod,
GPU, token, recording or viewer dependencies. See [docs/QA.md](docs/QA.md) for
reproducible checks and the limits of offline validation.

### Local viewer setup (authorized recording required)

From the clone root, with Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
installed:

```sh
git submodule update --init --recursive
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-live.txt
```

`uv` can install Python 3.11 if it is missing. The requirements install public
Python packages and the pinned viewer fork, not model weights. The viewer uses
Torch for skeleton/rendering utilities; this does not run local inference.
Do not install all upstream ARDY inference dependencies for this viewer.

The standalone preview, live viewer and directing viewer load
`assets/recorded_g1.csv` at startup, including Live ARDY's reference pose.
See [CONTRIBUTING.md](CONTRIBUTING.md#viewer-and-private-demo-prerequisites)
for the authorized source member. Without that private file, viewer startup
is expected to fail; passing controller tests does not make a demo runnable.
Once it is present, `./run-preview.command` opens standalone recorded playback
at http://127.0.0.1:2334/ with no token, SSH configuration or Pod.

### Existing private live demo

Both the live and directing viewers additionally read `.runtime/api-token` at
startup, even in Recorded preview mode. Their launchers (`run-live.command` and
`run-director.command`) also require SSH configuration.
For an existing authorized installation, prepare the configuration directory:

```sh
mkdir -p .runtime
cp pod.env.example .runtime/pod.env  # First setup only; keep existing settings.
```

Edit `.runtime/pod.env` with the current connection details. The core owner
must privately provision matching local/Pod bearer tokens and verified
`.runtime/known_hosts` entries before live startup. The example config does
not supply them. Existing installations retain their private files. The
addresses below are placeholders, not public demo endpoints.

## Directing demo

The current viewer adds stored takes, exact-prefix alternate endings, scrubbing,
project save/load, automatic backups and a deterministic gate reaction.
Run `./run-director.command` and open http://127.0.0.1:2336/.
For the private remote demo, use the existing Tailscale address on port 2334.
See [the directing guide](docs/DIRECTING.md) for the workflow and
[measured results and screenshots](review/DIRECTING-RESULTS.md) for verification.

## Earlier live viewer (retained locally)

The earlier live viewer remains available on the mini at http://127.0.0.1:2335/.
Start it with `./run-live.command`. The private Tailscale address on port 2334
now opens the directing viewer described above; it no longer routes to this
earlier viewer.

1. Select **Live ARDY** under Motion source.
2. Enter `A person waves with their right hand.` and click **Generate next 4 seconds**.
3. During playback, change the instruction to `A person slowly walks forward.` and click Generate again. The actor holds its displayed pose while the next segment is generated using recent motion history.
4. Pause/resume locally. Generate again to extend the motion. Reset clears the generated history and restores the reference pose and camera.
5. Select **Recorded preview** to use the original 12-second shadow-boxing fallback. That mode uses no inference.

Drag to orbit, right-drag to pan, scroll to zoom. Follow actor keeps horizontal root travel in view; turn it off for a stationary camera. Editing an instruction while generation is pending invalidates that request; click Generate to submit the replacement. Pause while generating keeps the new segment paused when it arrives.

The Mac mini must remain awake and connected; the existing Pod must be running for generation. Multiple browser tabs share one actor and playback state. This is a private single-user development viewer.

## Start or recover on the Mac mini

Double-click `run-live.command`, or run:

```sh
cd /path/to/stagezero
./run-live.command
```

This reuses/starts the backend on the existing Pod, restores an SSH tunnel if absent, and reuses/starts the local viewer. It does not provision resources. Cached model loading historically took about 70 seconds; the configured viewer can show recorded playback while loading. If generation reports unavailable, run the launcher again and retry. An expired/replaced Pod address requires updating `STAGEZERO_SSH_HOST` and `STAGEZERO_SSH_PORT` in `.runtime/pod.env`.

The existing private Tailscale forwarding is configured as:

```sh
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg --http=2334 http://127.0.0.1:2336
```

The Pod listens only on 127.0.0.1:8765 and requires a bearer token stored in private `.runtime/api-token` files. The Mac's tunnel also binds only loopback. Browser clients never receive the backend token. No public inference endpoint or Tailscale Funnel is used. `.runtime` must not be committed or shared.

### Stop

Ctrl+C in the viewer terminal stops the local viewer. Close the managed tunnel with:

```sh
ssh -S /path/to/stagezero/.runtime/pod-ssh -O exit "$STAGEZERO_SSH_HOST"
```

To stop the backend process without changing the Pod itself:

```sh
ssh -i ~/.ssh/id_ed25519 -p "$STAGEZERO_SSH_PORT" -o UserKnownHostsFile=/path/to/stagezero/.runtime/known_hosts "$STAGEZERO_SSH_HOST" "pkill -f '^.venv/bin/python -u pod_backend.py$'"
```

**Stopping these processes does not stop RunPod billing.** Stop the Pod yourself in RunPod when finished; no paid-resource changes are automated here. Backend logs: `/workspace/stagezero/backend.log`. Restart: run the launcher again.

For the SSH commands above, first run `source .runtime/pod.env` from your clone
and replace `/path/to/stagezero` with its actual path.

## Original fallback

`preview.py`, `run-preview.command` and the original G1 assets remain unchanged;
the recorded CSV is private and must be supplied separately. `./run-preview.command`
runs the original standalone preview at http://127.0.0.1:2334/. See
`review/MILESTONE-1.md` for the historical preview report and source provenance.
The configured live viewer's Recorded preview mode works without a responding
Pod, but still requires the startup files described above. Port 2334 is also
used by the documented Tailscale forwarding setup; the live viewer itself
listens on loopback port 2335.

## Implementation and dependencies

Official upstream source: `vendor/ardy`, commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`. Reuses its G1 skeleton, supplied mesh rig, rendering helper, checkpoint loader, text encoder, autoregressive inference, and motion decoder. No human-skeleton retargeting is performed.

Model: [nvidia/ARDY-G1-RP-25FPS-Horizon52](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52), 34 G1 joints, 25 fps, 414 motion features, four frames per token. Each request freshly encodes the instruction and generates two 52-frame horizons. Follow-up history ends at the displayed pose (up to 52 frames, aligned to four-frame tokens). In the first three frames there is not yet sufficient history, so an immediate replacement starts fresh.

Pod: Python 3.12.3, torch 2.8.0+cu128, RTX 6000 Ada. Exact installed packages are recorded in `review/pod-packages.txt`. The existing isolated environment is `/workspace/stagezero/.venv`, with official source in `/workspace/stagezero/ardy`. `start-backend.sh` sets PYTHONPATH explicitly to avoid an editable-install assets namespace collision. Checkpoints and the local Llama/LLM2Vec encoder are cached. Required gated access was verified; no terms were accepted by this implementation.

The backend launcher preserves an explicit `HF_HOME`. Without one, it uses `/workspace/hf` when `.runtime/hf-token` is present, or the existing `/workspace/.cache/huggingface` directory otherwise. This keeps restarts pointed at the cached model and credentials used by the original deployment.

Mac: Python 3.11 in `.venv`; install `requirements-live.txt` when recreating that environment. The Mac renders and manages playback; it does not run inference. No new local inference compatibility investigation was performed.

```sh
.venv/bin/python -m unittest discover -v -p 'test_*.py'
# Optional real Pod soak test: 24 fresh requests, about 100 seconds
.venv/bin/python measure_backend.py
```

See `review/MILESTONE-2.md` for measurements, visual evidence, and limitations. Glasses and multiple actors are not started and require review approval.

## Scene generation

The studio's **Scene → Scene generator** composes 16 procedural prop types,
six animated effects, five lighting palettes and five complete starter sets.
Use instant offline recipes, the connected Neon AI gateway, or an optional local
Ollama model. Move/duplicate props and import/export standalone scene JSON.
See [scene generation and setup](docs/OBJECTS.md) for examples, the UI-independent
integration API and current contact limitations. Reusable examples live in
`examples/scenes/`; `ai-observatory.json` was generated through Neon.

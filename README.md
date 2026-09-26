# StageZero — Milestone 2

Real ARDY G1 generation is connected to the existing viewer. The original recorded preview is retained in a separate, clearly labeled mode. This is **complete-segment generation, not streaming**: each instruction produces 104 fresh frames (4.16 seconds at 25 fps), then playback begins.

## Clone and collaborate

```sh
git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git
cd stagezero
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for independent tasks, tests without a
GPU, and required asset access. This repository excludes the gated recorded
CSV, model weights, installed environments and all credentials. A fresh clone
needs an authorized `assets/recorded_g1.csv` before either viewer can start.
Use Python 3.11 and `uv pip install --python .venv/bin/python -r requirements-live.txt`
after creating `.venv` with `uv venv --python 3.11 .venv`.

Existing installations retain their private files. Copy `pod.env.example` to
`.runtime/pod.env` for connection configuration; the bearer token and verified
SSH host keys must already be provisioned privately on the trusted machines.
The addresses below are placeholders, not public demo endpoints.

## Try it now

On the MacBook, connect Tailscale with the existing account and open:

**http://YOUR-MINI.YOUR-TAILNET.ts.net:2334/**

Private IP alternative: http://YOUR-TAILSCALE-IP:2334/. On the mini directly: http://127.0.0.1:2335/.

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

This reuses/starts the backend on the existing Pod, restores an SSH tunnel if absent, and reuses/starts the local viewer. It does not provision resources. Cached model loading takes about 70 seconds; the viewer can show recorded playback while loading. If generation reports unavailable, run the launcher again and retry. An expired/replaced Pod address requires updating the script's SSH host/port.

The existing private Tailscale forwarding is configured as:

```sh
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg --http=2334 http://127.0.0.1:2335
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

`preview.py`, `run-preview.command`, original G1 assets and recorded CSV remain unchanged. `./run-preview.command` runs the original standalone preview at http://127.0.0.1:2334/. See `review/MILESTONE-1.md` for the historical preview report and source provenance. The live viewer's Recorded preview mode is available without a working Pod connection.

## Implementation and dependencies

Official upstream source: `vendor/ardy`, commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`. Reuses its G1 skeleton, supplied mesh rig, rendering helper, checkpoint loader, text encoder, autoregressive inference, and motion decoder. No human-skeleton retargeting is performed.

Model: [nvidia/ARDY-G1-RP-25FPS-Horizon52](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52), 34 G1 joints, 25 fps, 414 motion features, four frames per token. Each request freshly encodes the instruction and generates two 52-frame horizons. Follow-up history ends at the displayed pose (up to 52 frames, aligned to four-frame tokens). In the first three frames there is not yet sufficient history, so an immediate replacement starts fresh.

Pod: Python 3.12.3, torch 2.14.0+cu130, RTX 6000 Ada. Exact installed packages are recorded in `review/pod-packages.txt`. The existing isolated environment is `/workspace/stagezero/.venv`, with official source in `/workspace/stagezero/ardy`. `start-backend.sh` sets PYTHONPATH explicitly to avoid an editable-install assets namespace collision. Checkpoints and the local Llama/LLM2Vec encoder are cached. Required gated access was verified; no terms were accepted by this implementation.

Mac: Python 3.11 in `.venv`; install `requirements-live.txt` when recreating that environment. The Mac renders and manages playback; it does not run inference. No new local inference compatibility investigation was performed.

```sh
.venv/bin/python -m unittest -v test_live_motion.py
# Optional real Pod soak test: 24 fresh requests, about 100 seconds
.venv/bin/python measure_backend.py
```

See `review/MILESTONE-2.md` for measurements, visual evidence, and limitations. Milestone 3 (glasses) is not started and requires review approval.

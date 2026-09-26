# StageZero — Milestone 1

A local, recorded-motion preview using NVIDIA ARDY's supplied Unitree G1
humanoid and Viser renderer. **No live AI generation is connected.**

## Run

Double-click `run-preview.command`, or run from this directory:

```sh
./run-preview.command
```

Open **http://127.0.0.1:2334**. The viewer is bound to this Mac only.
If a server is already running, use its existing browser tab rather than
starting a second instance. Stop the server with Ctrl+C in its terminal.

- **Play:** play/resume; replay from frame zero after the clip ends.
- **Pause:** freeze the current pose; camera controls remain usable.
- **Reset:** pause at frame zero and restore the initial camera.
- Drag to orbit, right-drag to pan, scroll to zoom.
- The recording stops at its end; it does not loop across a discontinuity.

## Open remotely from your MacBook

Connect Tailscale on the MacBook using the same existing account, then open:

**http://YOUR-MINI.YOUR-TAILNET.ts.net:2334/**

Fallback private IP: **http://YOUR-TAILSCALE-IP:2334/**.
Tailscale Serve forwards this private-network address to the local viewer;
it is not published to the public internet. The Mac mini must remain awake,
connected to Tailscale, and running the preview. The proxy runs in the background;
the preview itself must be relaunched after a reboot.

To disable only this preview proxy on the mini:

```sh
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --http=2334 off
```

To re-enable it:

```sh
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg --http=2334 http://127.0.0.1:2334
```

## What is being shown

One approximately 12-second shadow-boxing motion-capture recording from
[BONES SEED](https://huggingface.co/datasets/bones-studio/seed), retargeted by
the dataset providers to G1. Source archive member:

`g1/csv/230509/shadow_boxing_R_003__A361.csv`

The original CSV is retained in `assets/recorded_g1.csv`. Playback uses every
second source frame (120 fps -> 60 fps), converts centimeters to meters and
the coordinate basis using ARDY's demo conventions, and subtracts the initial
horizontal root position. No new motion is synthesized or foot correction
applied. Source foot sliding/ground penetration can remain; this preview
does not establish ARDY generation quality.

## Reuse and dependencies

Official ARDY: `vendor/ardy`, revision
`693f74d13b3d04a0a22ce127ee79c929dd89756b`.
`preview.py` adapts its recorded-viewer pattern and CSV conversion and reuses
`Character`, `G1MeshRig`, skeleton kinematics, and supplied meshes directly.
The upstream source is unchanged. Its Apache-2.0 license and Unitree mesh
attribution are retained in `vendor/ardy/LICENSE` and `vendor/ardy/ATTRIBUTIONS.MD`.
Dataset terms are retained in `assets/BONES-SEED-LICENSE.md`.
Dataset files have separate access/redistribution terms from the viewer code.

The isolated `.venv` uses Python 3.11 and `requirements-preview.txt`.
To recreate it with uv:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-preview.txt
```

The first launch builds the pinned Viser browser client and requires network
access for its Node/npm dependencies. Subsequent launches use the local build.
No model checkpoints, Llama encoder, TensorRT, or motion-correction extension
are required for this milestone. Normal playback requires no Hugging Face login.

## Review gate

Verified on this Mac in the browser: the mesh renders; recorded poses change;
Play, Pause, resume/replay, end-of-clip stop, orbit, zoom, and Reset work.
Reset restores both frame zero and the initial camera. A screenshot is saved
in `review/stagezero-preview.png`. Right-drag panning is supplied by Viser but
was not separately exercised by the automated UI check. Motion arrays contain
718 finite frames with 34 joints. Foot-joint positions reach approximately
2.6 cm below the stage plane in the source recording; foot contact is not
corrected in this preview.

Review the actor, recorded movement, stage, and camera before authorizing
Milestone 2. Live text-to-motion and VITURE integration have not been built.
The Pod was connection-tested only. Llama access was still pending at the
latest check; it is not required for this recorded preview.

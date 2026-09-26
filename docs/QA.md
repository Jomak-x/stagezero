# MVP verification and onboarding

Historical real-inference measurements are in [Milestone 2](../review/MILESTONE-2.md).
They do not show that a Pod is currently serving requests. The checks below use
simulated controller transport or public dependency installation; they do not
validate motion quality, inference latency or the private deployment.

## Repeatable offline checks

Follow [CONTRIBUTING.md](../CONTRIBUTING.md#offline-controller-checks) to create
`.venv-tests` with Python 3.11, NumPy 1.26.4 and requests 2.34.2. From the clone root:

```sh
.venv-tests/bin/python -m unittest discover -v -p 'test_*.py'
.venv-tests/bin/python -m unittest -v tests.test_controller_edges
.venv-tests/bin/python -m compileall -q live_motion.py live_viewer.py preview.py pod_backend.py measure_backend.py test_live_motion.py tests
```

The first command is the CI test command. It includes the original eight tests
in `test_live_motion.py` and the additional cases in `tests/test_controller_edges.py`.
Keep `tests/__init__.py` so Python 3.11 discovery descends into `tests/`.
The test environment deliberately has no viewer or inference dependencies.
All controller responses and failures are simulated; no test needs the ARDY
submodule, recording, token, SSH, network service or GPU.

The original tests cover basic stale results, draft editing, pause, reset,
recorded-mode switching, retry, history length, wrong skeleton and non-finite
positions. The six new methods add:

- Burst replacement: queued intermediate prompts are skipped; a late success
  or failure cannot overwrite the newest pending request.
- Repeated pause/resume while pending: the last playback choice is preserved.
- Reset after generated playback: pending history is discarded, the reference
  pose remains visible, and the next request starts without history.
- Connection failure/recovery: the displayed clip survives and retry receives
  history ending at the displayed frame.
- Twelve malformed-response cases through the controller: missing fields,
  incompatible request/model/fps/shapes, non-finite motion/rotations, scaled and
  reflected rotations; each retains the scene and permits a valid retry.
- Empty, whitespace-only and oversized prompts are rejected before transport;
  a trimmed 500-character prompt is accepted.

Their event-gated synthetic transport deliberately ignores cancellation so
stale-result rejection is exercised locally. Synthetic positions identify which
request won; they are not examples of AI-generated motion.

## Clean-clone setup check

Run in a new directory, separate from an existing private demo:

```sh
git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git
cd stagezero
git submodule status
git ls-files --stage vendor/ardy
git ls-files assets
git ls-files '*.csv' '*.pt' '*.pth' '*.safetensors' '*.key' '*.pem' '.runtime/*' '.venv/*' '.venv-tests/*'
git check-ignore .runtime/pod.env .runtime/api-token .runtime/known_hosts assets/recorded_g1.csv weights.pt weights.pth model.safetensors .venv/bin/python .venv-tests/bin/python .env private.key private.pem
```

Expected: ARDY resolves to `693f74d13b3d04a0a22ce127ee79c929dd89756b`
(no leading `-` or `+` in submodule status), the parent repository tracks only
`assets/source_path.txt` under `assets/`, the private-file listing is empty,
and all representative private paths are ignored. These filename checks are
not a complete secret scan or a check of Git history.

For local viewer dependency verification, use the [README setup](../README.md#local-viewer-setup-authorized-recording-required):

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-live.txt
uv pip check --python .venv/bin/python
```

This downloads public packages and the pinned viewer fork, not gated weights or
recordings. On a fresh clone without the CSV, `.venv/bin/python preview.py` and
`.venv/bin/python live_viewer.py` should reach `load_recording` and fail with
`FileNotFoundError` for `assets/recorded_g1.csv`, before opening a viewer server.
That verifies imports and the private-asset boundary, not successful rendering.
Do not fabricate a recording or fetch gated assets just to make this check pass.

## Results from this QA pass

Base revision: `8e9ae12`. Environment: macOS arm64, CPython 3.11.15,
uv 0.11.24. A new recursive clone was fetched from GitHub independently of the
working checkout.

| Check | Result |
| --- | --- |
| Public recursive clone / pinned ARDY commit | Passed |
| Fresh test environment, pinned NumPy / requests | Installed; original eight tests passed |
| Combined suite, root discovery | All 14 tests passed, including six new methods and their subcases |
| Full `requirements-live.txt` install | Passed, 53 packages; `uv pip check` passed |
| Viewer imports / missing recording boundary | Both entrypoints reached `load_recording`, then failed on the absent CSV as expected |
| Tracked private paths and representative ignore rules | Passed checks listed above |
| Pod, SSH tunnel, browser rendering, gated assets, real inference | Not exercised |

The full install retained the pinned Torch 2.14.0, SciPy 1.17.1, Pillow 12.3.0
and viewer commit `7c82ad8f8640bad9dff8ded5c5eee908eeb08f11`. Imports emitted
an upstream `torch.jit.script` deprecation warning but reached the recording
loader. This installation result applies to the tested macOS arm64 environment;
controller CI runs separately on Ubuntu/Python 3.11.

## Reproducible issues and core-owner follow-ups

### Live launcher exits silently when its token is absent

In a fresh clone with no `.runtime/api-token`:

```sh
bash -x ./run-live.command
```

Observed: exit status 1 at `test -s .runtime/api-token`, before any SSH call or
the "Pod unavailable" fallback message. Without `-x`, there is no explanation.
The README now distinguishes standalone recorded playback from the configured
live viewer and explains all startup prerequisites. Proposed owner fix: add an
actionable preflight message in `run-live.command`; decide separately whether
the live viewer should start in recorded-only mode without a backend token.
The launcher and viewer are unchanged in this PR.

### Runtime metrics append prompts to a tracked review artifact

`live_viewer.py` constructs the session with `review/live-metrics.jsonl` as its
metrics path. `pod_backend.py` includes the full instruction in response metadata,
and `MotionSession.record_ack` appends that metadata to the path after rendering.
`git ls-files review/live-metrics.jsonl` confirms the file is tracked.

On an already configured private demo, one generated segment with a browser
render acknowledgement will append a record; `git diff -- review/live-metrics.jsonl`
then exposes that instruction as a local change. A broad commit could publish
it. An offline probe confirmed that synthetic prompt metadata is persisted by
`record_ack` to a temporary JSONL file; the tracked destination is confirmed by
source inspection. The live reproduction was not run in this offline pass. Proposed owner fix:
write future metrics to an ignored runtime path, preserving the checked-in file
as historical review evidence. Review metric diffs before committing demo work.

## When the user restarts the existing Pod

These remain core-owner/user checks, outside the offline pass:

1. Confirm current SSH host/port, update `.runtime/pod.env` if necessary, and
   check whether `/workspace/stagezero` and the checkpoint cache survived.
2. Restore the authenticated loopback backend and SSH tunnel. Record cold
   loading separately from warm generation; do not provision new resources.
3. On the actual MacBook, measure several fresh instructions through Tailscale.
   Check visible action changes, root travel, feet, camera and pose continuity.
4. Run a 10-minute generation/playback session; record failures, latency
   distribution, GPU memory and process memory. The prior soak was 100 seconds.
5. Interrupt/reconnect the tunnel; test rapid prompt changes, pause/reset while
   pending, recorded fallback, browser reload and backend restart.
6. Save results and screenshots, inspect logs for private prompts, list
   limitations, then stop for review.

Acceptance: no stale motion appears after cancellation/reset; controls remain
responsive during generation/failure; recovered backend accepts new commands;
reported live motion is freshly generated; recorded and simulated cases remain
explicitly labeled. Motion quality still requires the user's visual review.

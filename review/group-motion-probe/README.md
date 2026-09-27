# Optional group motion experiment

Original research decision: keep the broad experiment optional. The subsequent three-person prompt integration is documented in `../group-prompt-integration/README.md`; ten-person capture remains research-only. The reviewed main demo is sufficient. No existing production Python, UI, planner, pair pipeline or playback cap changed.

Watch [the complete 1:58 comparison](group-comparison.mp4) or serve this directory and open [the review page](index.html). All source takes run at 1x without internal cuts; chapter titles identify the exact-source sequential timing control (not a production planner run).

## What worked and what did not

- Six fresh three-person cases: wave, dance and celebrate, each on two seeds, across City, Market and Industrial with line and triangular starts. All generated tracks passed geometry checks. Waves and celebration read clearly; dance is understated, and foot glide/float remains. These finite probes do not establish reliability for arbitrary prompts.
- Ten independently generated celebrations: readable loose group activity, distinct seeds and private continuation history. Offline native mesh capture only; not ten-person Studio support or a real-time performance benchmark.
- Existing handshake plus third wave: the pair arrays remain byte-identical after archive roundtrip. Full matched-camera before/after takes are included.
- Mid-scene third celebration during Industrial spar: rejected for endpoint velocity/proportion mismatch; exact original project returned. Raw candidate, requests and diagnostics retained under `overlays/industrial-spar-s45/attempt-001`. Do not bypass the bridge gate for a demo.
- Physically coordinated three-person contact is unsupported. Independent solo motion does not model shared contact constraints, forces or synchronization. This experiment deliberately does not introduce a joint model.

## Reproduction

From the repository root with its normal Python environment and pinned submodule initialized (`git submodule update --init vendor/ardy`):

```sh
PYTHONPATH=.:vendor/ardy python -m unittest test_independent_group_motion
PYTHONPATH=.:vendor/ardy python -m unittest discover
# Replay accepted three-person output without GPU generation:
python director_viewer.py --reference-only --native-project review/group-motion-probe/fresh/celebrate-three-s75/attempt-001/scene.cast.stagezero.npz --port 24978
# After coordinating/reserving an existing Core worker lane and checking shared costs:
python experiments/trial_group_motion.py --case all --token /path/to/private-token --core-url http://127.0.0.1:8769 --model-lane-reserved
python experiments/trial_group_overlay.py --token /path/to/private-token --model-lane-reserved
# Rebuild the comparison from retained full captures (Pillow + ffmpeg):
python review/group-motion-probe/build-reel.py
```

The regular director requires its normal built client/character assets. Generation attempts are immutable and numbered. Explicit prompts, layouts and seeds are frozen in `cases.json`; no external planner was used. Each actor uses `(seed + 1009 * actor_index) % 2**32` and its own native330 history. Core generates at 20fps; the existing semantic subset and interpolation produce 30fps native22 display data. No new model or model training.

Capture: use the repository's existing `experiments/capture_cast_review.py` harness with `capture.json` / `capture-overlay.json` (see its help). Ten-person capture uses `experiments/capture_group_probe.py --help`, an existing built Studio client and the saved research NPZ. The research renderer is separate and does not lift production actor limits. `prepare-baselines.py` and `prepare-capture.py` document authored comparison preparation; archives are already retained so rebuilding baselines is unnecessary.

## Evidence and validation

`results.json` links every first attempt and timing. Seven fresh group cases used 84 Core horizons; two overlay probes used seven more. Total model wait about 80 seconds, sequential requests, no new Pods. Observed shared rate was $1.90/hour below the $4/hour cap; shared live Pods were preserved and this lane released afterward.

`fresh/*/attempt-001/metrics.json` records feet, root and body measurements for three-person probes. Source manifests contain body clearance/scene geometry including ten actors. These checks are diagnostics, not proof of animation quality. Full takes were reviewed at normal speed and with continuous filmstrips. `video/*` retains frame hashes, captures and provenance. `reel-manifest.json` records source hashes and chapter times.

1,372 Python tests passed (`tests/python-suite.txt`), including eight new focused tests covering independent histories, seeds, malformed source retention, cancellation, overlap rejection, exact pair preservation and exact fallback. The first suite attempt failed because this reused worktree lacked the pinned ARDY submodule; initializing it resolved that setup issue. No production code was changed to fix it.

Neon GPT-6 Astra provided the bounded design review (`tests/neon-design-review.txt`); its proposed maximum spacing constraint was not adopted. Codex GPT-6 Sol high implemented the isolated module/tests; Sol medium implemented the research capture tool. Main agent integrated, ran actual models and visually reviewed results. A later Neon Sol review truncated and is not counted as approval.

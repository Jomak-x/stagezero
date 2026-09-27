# Background navigation demo review

## Selected examples

- [Automatic door and entry, complete 12 s video](door-seed33/video/performance.mp4).
  [Explicit open-command archive](open-door-seed33/motion.core.stagezero.npz),
  [measurements](open-door-seed33/report.json). `open Checkpoint door then go through Checkpoint door`.
  Both commands completed; automatic opening and passage crossing measured.
  Fresh explicit-open run: 7.92 s end-to-end, 5.29 s total service waits.
  Final errors 6.9 mm / 2.6 mm. Original approach-command video has exactly
  identical positions, rotations and native features to the explicit open run:
  [verified equality](open-command-video-equivalence.json).
- [Temple courtyard directions, complete 10 s video](temple-turn-seed33/video/performance.mp4).
  [Exact archive](temple-turn-seed33/motion.core.stagezero.npz),
  [measurements](temple-turn-seed33/report.json). `walk 1 metre forward then walk 1 metre left`.
  Both commands completed, 5.77 s end-to-end / 4.21 s service waits;
  final mark errors 1.9 mm / 2.8 mm. This uses the actual temple preset,
  **only its ground courtyard**. It does not climb stairs, cross the ravine,
  open the shrine door, or establish generalized terrain locomotion.

The root agent reviewed complete captures/contact sheets and normal-speed
browser replay. Timing, turning and posture remain imperfect: visible foot
sliding and hunched/stiff poses mean this is a bounded demo, not animation
quality parity for every native free-motion request. All generated character
positions, rotations and native features are byte-identical to accepted raw
service clips. The existing main renderer is used; no added pose correction,
interpolation or retiming. Every capture lists all native frame indices.

## Other tests and retained failures

| Attempt | Result |
| --- | --- |
| `door-seed33` | Approach → automatic opening → passage; 240 frames, completed |
| `door-seed42` | Independent seed, same route completed; 12 s motion / 7.62 s wall time |
| `open-door-seed33` | Explicit open → passage, both measured, exact old video arrays |
| `grove-seed33` | First move completed; next endpoint overlapped a prop, stopped explicitly with 80-frame prefix intact |
| `grove-clear-seed33` | Clear starting point in unchanged grove, two commands completed (pre-turn-polish controller), 160 frames |
| `temple-courtyard-seed33` | Completed mechanically; rejected abrupt turn, 27.2° maximum hip-to-head lean |
| `temple-turn-seed33` | Selected native turn prefix; maximum hip-to-head lean 21.7°, final14.0°; still visually imperfect |
| `temple-sparse-seed33` | Diagnostic all-sparse conditioning; upright torso but unwanted deep squat during stop (pelvis minimum0.668 m); rejected |
| `temple-easy-walk-seed33` | Diagnostic construction error before any GPU call; retained |
| `temple-easy-walk-seed33-v2` | Sparse walking/dense stop comparison completed; final lean17.8°, no clear overall improvement; not promoted |

Each directory retains its scene, report, complete committed archive, and raw
native clips. Measurements prove bounded endpoint/collision-proxy mechanics;
they do not establish natural motion by themselves. No new models, Pods or
physics runtime were installed for this finish. Existing services remain intact.

## Verification

Before updating to latest main: 1,345 Python tests passed,45 client tests passed,
TypeScript and production client build passed. Initial full-suite errors were
missing pinned ARDY submodule files; after initializing the exact submodule,
the final complete suite passed. Neon `gpt-5-4-mini` reviewed integration points;
Neon `gpt-6-astra` reviewed reaction semantics and the final command/lifecycle
delta. Its short-move false-success and malformed-archive issues were fixed
and tested; the final delta review found no concrete P1/P2 issue.

Focused tests cover object ambiguity, blocked/unsupported routes, rotated doors,
future-frame isolation, smooth opening, full-history latching, opt-in defaults,
exact archives, safe rejection, cancellation/retry, subsequent actual pose and
heading, actual aperture crossing, small-distance no-op rejection, and native
turn scheduling/wraparound. Fake providers test orchestration, not visual quality.

## Reproduce

```sh
PYTHONPATH=vendor/ardy .venv/bin/python -m unittest discover -p 'test_*.py' -q
.venv/bin/python experiments/verify_background_demo.py \
  --case door --seed 33 --token-file /path/to/private/core-token \
  --output /new/review/directory
PYTHONPATH=vendor/ardy .venv/bin/python experiments/capture_core_performance.py \
  --archive /new/review/directory/motion.core.stagezero.npz \
  --output-dir /new/video/directory --port 24980
```

The two optional
sparse flags are diagnostic comparisons only and are not production defaults.
The live harness uses an existing authenticated Core service and caps each case
at18 native horizons /180 seconds. It does not provision or restart services.

See [feature usage and explicit boundaries](../../docs/BACKGROUND-NAVIGATION-DEMO.md).

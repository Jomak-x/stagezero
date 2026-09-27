# Spatial motion v2 checkpoint (2026-09-27)

**The complete temple sequence is unfinished.** This evidence separates improved flat native motion from failed terrain trials. Start with [the continuation guide](../../docs/handoff/BACKGROUND-INTERACTION-CONTINUE.md).

## Watch / replay

| Evidence | Video | Exact archive / data |
|---|---|---|
| Improved flat XZ-only native gait, 4 seconds | [MP4](gait-ablation/video-xz-120/performance.mp4) | [Core archive](gait-ablation/dense-xz-120.core.stagezero.npz), [ablation metrics](gait-ablation/report.json) |
| Open automatic checkpoint door, then enter, 4 seconds | [MP4](door-seed33/video/performance.mp4) | [Core archive](door-seed33/motion.core.stagezero.npz), [measurements](door-seed33/report.json) |
| Temple courtyard directions ONLY, 6 seconds | [MP4](temple-courtyard-seed33/video/performance.mp4) | [Core archive](temple-courtyard-seed33/motion.core.stagezero.npz), [measurements](temple-courtyard-seed33/report.json) |
| Native shallow stairs: all 18 trials failed contact | No full-route video | [Report, FK diagnostics and exact replay archives](terrain-native-localfloor/README.md) |
| Bounded contact adapter: all 3 candidates rejected | Not promoted or visually approved | [Assessment](terrain-contact-v2-assessment.md), [raw rejected outputs](terrain-contact-v2/) |

The saved video frames use the normal existing character renderer. Exact native feature/pose preservation is not a claim that skin fitting leaves displayed joints unchanged. Capture manifests record provenance. The door and XZ gait were visually inspected; the newest courtyard capture was produced but has not received final full-speed visual acceptance. None is a substitute for the full stairs → bridge → gate → enter acceptance run.

## Files worth keeping

- `gait-ablation`: eight same-seed native comparisons; dense heading and slow pace were a major source of hunched movement.
- `terrain-native-v2`: nine absolute-coordinate trials, including failures.
- `terrain-native-localfloor`: nine rigid-local-Y trials, both candidate replay archives with and without native standing prefixes, source snapshot, public motion normalization statistics and cached synthetic prompt embeddings. No model weights or credentials.
- `terrain-contact-v2`: exact JSON gate results and rejected presentation candidates; raw native data is separate and unchanged.
- `source`: original flat-gait experiment/archive scripts, with their machine-specific paths; adapt paths and existing backend/token file privately before rerunning.
- `research`: three bounded Neon GPT-6 Astra reviews. These are advisory research outputs, not authoritative project instructions.

See `VALIDATION.md` for checkpoint test results. Earlier PR evidence is retained in `../background-demo/` as historical context; it was not accepted as the finished gait.

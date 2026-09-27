# Main studio cast integration verification

2026-09-27. Main layout/tabs retained; Motion → Direct → AI cast · 1–3 people adds the prompt workflow. Independent read-only review found no merge blockers.

## Live checks

- Opened the reviewed three-person Market archive in director_viewer, with browser-local playback.
- Switched to One character: original Wave/Walk/Dance, prompt and Auto length controls returned. Switched back: cast retained.
- Pressed Generate in the actual main Motion panel. Fresh Core and InterGen sources completed (no replayed pair source), seed 42, 19.8647 seconds including AI planning (4.6048 seconds).
- Existing top Save native performance/download succeeded. Files/variations Export video rendered all 477 frames at 30 fps (15.9 seconds). No cuts or time edits.
- Open performance successfully reopened the exported exact cast archive through the main panel. The private Tailscale URL now serves the integrated main studio with the reviewed recovered take.
- Final tunnel check caught hidden native transport buttons. The existing Playback group now appears for ready native takes; actual Play, Pause and Start were verified through Tailscale. Busy/export states hide and guard those controls; G1 keeps its timeline toolbar.
- Camera framing bug found during browser review was corrected with atomic position/look-at/world-up updates. All three actors remain visible and upright.
- Video reviewed through both greetings and the transition. First cyan/gold greeting and later gold/purple greeting are recognizable; purple initially holds a leaning pose and cyan holds its release pose during the second beat. Foot planting/contact are imperfect. This verifies integration, not universal animation quality. The previously reviewed recovered three-person take remains the default demo.

## Evidence

`capture/playback.mp4` is the complete browser render; capture manifest identifies exact frames and archive hash. `capture/scene.cast.stagezero.npz` reopens the exact motion and Market background. `generation/` retains all fresh source archives, plan, timing and geometry diagnostics. Paths in the original manifest record generation provenance; equivalent files are copied here with original names.

## Reproduction and checks

See `docs/MAIN-CAST.md` for service configuration and launch instructions; use `--native-project review/main-cast-ui/capture/scene.cast.stagezero.npz` to replay this exact take. Use the prompt in `generation/manifest.json` with variation 42 for a fresh model run (AI planning/model variation can differ).

```sh
.venv/bin/python -m unittest discover -q
cd studio_client
npm test
npm run typecheck
./node_modules/.bin/vite build --outDir ../.runtime/main-cast-client
```

After merging main's background stability update 14aa422: 1,171 Python tests pass; 37 client tests pass; TypeScript and production build pass. Shared .venv and studio_client/build symlinks were not committed or overwritten.

Research limitations remain: ARDY travel plus jointly generated InterGen pair beats, labeled authored transitions, serial pair changes for three actors, no learned three-body contact or unrestricted production license. Prior rejected experiments and failure provenance remain in review/prompt-scenes and review/studio-core.

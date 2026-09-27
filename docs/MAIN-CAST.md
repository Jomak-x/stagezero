# Multi-person motion in the main studio

The existing main studio layout is preserved. In **Motion → Direct**, choose **AI cast · 1–3 people**. Describe the people and actions, then Generate. Cast size, starting places and duration are automatic. Use the existing **Scene** tab to generate or choose the background, the Playback buttons and bottom timeline to review, and the existing Save project/download and Open controls for the active performance.

Examples:

- “One person waves hello, then celebrates.”
- “Two people start at x -3 z 0 and x 3 z 0, meet at x 0 z 0, then perform a controlled boxing spar with dodges.”
- “Three people greet each other in turn. Person 1 shakes hands with person 2 and releases. Then person 2 shakes hands with person 3.”

Plan/timing and Files/variations are folded inside the Motion panel. They expose variation, retry, exact archives, video export and framing without another top-level studio. **One character**, **Two characters**, advanced Core direction, full-scene creation, imported characters and existing G1 takes remain available. Switching modes keeps their motion separate. Starting a new G1 take performs an explicit mode handoff; generation and capture must finish or be cancelled first.

## Existing services

The feature uses existing Core and native paired services; it does not rent a Pod. `run-director.command` accepts `STAGEZERO_NATIVE_PAIR_CONFIG` from private `.runtime/pod.env`, or uses `.runtime/prompt-native-provider.json` when present. Keep worker tokens/configuration out of Git. Configure the existing Core service through the director's `--core-backend-url` and `--core-token-path` flags. Without providers, saved archive playback still works.

For a direct launch, build the current client first and supply your private configuration:

```sh
(cd studio_client && npm ci && npm run build)
.venv/bin/python director_viewer.py --reference-only --port 2383 \
  --token-path /path/to/private/api-token \
  --native-pair-config /path/to/private/prompt-native-provider.json \
  --native-project review/prompt-scenes/ui-three-heading-recovery/scene.cast.stagezero.npz
```

The normal studio launcher remains `run-director.command`. No recorded motion file is needed with `--reference-only`. `--native-project` recognizes both native paired and composed cast archives. The saved scene is restored exactly on open; a retained take continues to use its saved background, while new generation uses the current Scene tab document.

## Formats and limits

Cast archives use `.cast.stagezero.npz`, 1–3 native22 display tracks at 30 fps. They do not overwrite G1’s 34-joint/25-fps or Core’s 27-joint/20-fps archives, histories or timelines. Native paired sources retain their own exact features. Browser-local playback transfers prepared motion once.

ARDY Core generates solo actions and travel. InterGen generates each active pair together. Three-person scenes sequence pairs while nonparticipants receive labeled authored observer continuation: initial meeting-facing staging, eased arm relaxation, small attention turns, and upper-body settling. Pelvis, legs, and feet remain exact during each waiting span; active InterGen frames are unchanged. Authored transitions remain labeled and are not learned contact dynamics. Waiting stance, foot glide and contact remain variable. See [interaction quality evidence](../review/interaction-quality/README.md) for complete before/after captures and rejected ankle-planting trials. Complex dance/fall/help-up/hug composition is still unvalidated. InterGen's noncommercial research license continues to apply.

Full earlier source archives, failed cases, visual captures and timings are preserved in `review/prompt-scenes/`. Main-UI integration verification is in `review/main-cast-ui/`. Do not treat numerical gates as animation quality guarantees.

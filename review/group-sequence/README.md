# Group sequence UI audit and bounded experiment

**Decision: not demo-approved for compound group actions.** The exact request to meet, dance, then have all three backflip still fails semantic visual review. Keep the reviewed demo and pair pipeline. New multi-action group generation is opt-in; no new model, worker deployment or Pod was introduced.

`comparison.mp4` contains complete original and experimental performances plus actual UI regressions, with title cards and labels only. `comparison-chapters.json` identifies the original clips and durations. This is an honest diagnostic comparison, not a success reel. The original take was reconstructed from its unchanged saved candidate joints and rendered through Studio; its original screenshot and generation remain alongside it.

## What changed

- Integrated PR34 head `6b11074225b3d3aadd082b4d67333e5f58139feb` on an isolated branch based on main `4fbbc02`. Explicit starting locations and initial yaw survive independent-group routing.
- Ordered all-solo group stages preserve all performers and the ending. A bounded semantic plan review checks explicit ordered group requests before generation. Pair-plus-third remains its existing single-interval path; physical three-person contact remains unsupported.
- Group starts require at least 2m spacing during planning, allowing the existing repair attempt to fix AI-selected marks before motion generation. Scene-aware placement reserves explicit marks, tries alternate noncrossing formations, and prefers action room along the street.
- Experimental Core sequences retain full source horizons, actual navigation, private per-actor histories, real backgrounds, geometry gates and a 1000-frame budget. Four-frame action carry matches the existing runtime. An alternate fresh-action policy uses separately labeled authored bridges; common 21/24/27/30-frame candidates must pass every actor's existing mechanical gates. No source action frames are trimmed, replaced, or fabricated to manufacture a successful ending.
- Studio progress identifies group generation accurately. Background-picker copy now explains that a saved take keeps its original scene and a newly generated take uses the selected background.

## Actual model and UI findings

| Case | Input / seed | Result |
|---|---|---|
| Original user take | Exact meeting → dance → all backflip prompt, 42 |10s single compound clip. Triangle arrangement, no real meeting, missing sequence. Failed. |
|40-frame history | Exact prompt through Motion UI, City, 42 |22s approach/dance/flip timeline; later actions largely repeat idle/arrival. Failed despite geometry acceptance. |
|4-frame history | Same recorded plan/starts, City, 42/43 |42 hit sidewalk during finale;43 passed geometry but only one actor inverted. Neither fulfilled all backflips. |
| Fresh action stages | Same plan, roomier City formation, 42/43 |42 completed 23.4s but one actor never inverted and finale was not three clean backflips.43 rejected a >0.30m transition height gap. |
| Final exact prompt UI | Replanned original prompt, City, 42, final fresh policy |25.4s, all stages generated and displayed. Dance is modest; all three requested flips absent. Failed visual review. |
| Winter plaza spacing failure | Full scene; all three wave, 42 |AI chose marks 1.4m apart. Preserved rejection led to planning validation/repair fix. |
| Winter plaza repaired UI | Same request; picker → Full scene, 42 |Three performers all raise/wave arms over 4s, correct snow/background, playback and full export verified. |
| Winter two-stage UI | All wave 4s then all dance 4s,43 |Fresh transition rejected excessive joint speed; prior successful take retained. Exact source replay under 21/24/27/30 search still rejected. |
| Pair regression UI | Opposite street starts → handshake 4s → release,42 |Two actors, 12s, real approach and native pair segment, complete playback/export. Contact/release remains imperfect; this is a routing regression check, not a new quality endorsement. |
| Isolated flip probes | Explicit standing-backflip prompt, City, 90(2s),91/92(4s) |91 visually rolled backward near ground;92 performed aerial rotations and recovered. Results are seed-sensitive; numerical inversion alone is not acceptance. |

Every failure remains in its named folder with sources, plans and raw requests. Source manifests use the working paths at capture time; the corresponding files are retained by basename in each copied directory. UI generation used real Core/InterGen workers; source-replay cases are explicitly named. No batch failure was silently substituted into a success.

`measurements.json` records route errors, native history seam jumps, pelvis excursion, torso-inversion counts, relative movement, and a near-floor foot-speed proxy. These numbers do not prove foot planting, a real airborne backflip, correct contact, or good animation. All complete playable group and pair UI outputs above were watched; final actions were also scrubbed closely. Geometry-accepted metadata intentionally retains `visual_acceptance: unverified` and is not presented as animation approval.

## Run the optional experiment

Use the project's configured Python environment and existing Core worker. Keep server credentials private.

```sh
PYTHONPATH=.:vendor/ardy python experiments/review_group_sequence.py \
  --scene review/prompt-scenes/backgrounds/city.json \
  --plan review/group-sequence/request-plan.json \
  --seed 42 --policy fresh \
  --token-file .runtime/api-token \
  --output .runtime/my-group-sequence-review
```

`--policy continuous` compares native-history carry. Outputs must be a new directory. Review the archive in Studio via Open performance; play and export the full take. The preserved archive is the authoritative replay for a historical run, because planner output and placement policy can differ in a fresh run.

To exercise new requests through the actual Studio, explicitly launch the existing director with `STAGEZERO_GROUP_SEQUENCE_MODE=fresh` or `continuous`. With the variable unset, multi-action groups fail visibly before Core generation and retain the previous take; ordinary one-action groups and existing pair paths keep their existing behavior. Do not enable this automatically in the live demo based on these results.

## Validation and resources

Neon GPT-6 Astra reviewed the bounded plan and implementation. Codex Astra/high owned sequence code/tests; Sol/high owned planner and adapter test work. Main agent integrated, repaired placement, ran actual models and operated the UI. Four concrete Neon findings were addressed: reserved explicit starts, alternate route feasibility, yaw-only arrivals, and cancellation.

Full Python regression suite and final focused results are recorded in `validation.json`. No TypeScript implementation changed. Main, the dirty original workspace, previous live 2399 Studio, and shared workers were preserved. Existing Pods were checked at $0.53+$0.84=$1.37/hour total; no new Pods were provisioned. Existing shared Pods were not stopped because they serve the demo and other work.

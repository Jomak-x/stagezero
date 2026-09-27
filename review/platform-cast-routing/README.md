# Platform cast-routing regression review

## Reproduce and verify

On main before this fix, enter Full scene, leave Length Auto, and submit: `Two people meet and shake hands for four seconds.` The workflow generated 6.48 seconds of single-actor motion and Review scene changed Direct to One character. `two-person-before.png` preserves that failure.

The fixed UI classifies the original direction before single-actor planning or rewriting, routes 2/3 performers to the existing cast builder, and checks that the scene plan retains that exact count. Single-actor Full scene keeps its original duration/workflow. Uncertain/unsupported casts and explicit multi-person duration selections fail visibly rather than silently generating one actor. Select Auto for cast generation; describe timing in the prompt.

## Actual model runs through UI

| Entry | Direction | Result |
|---|---|---|
| Full scene | Two people meet and shake hands for four seconds. | Two performers, 7.7s; generated in 8.75s after connection recovery, full playback/save/export verified |
| Motion, default One character | Three people celebrate together in place, each waving both arms happily. | Three performers, 4s simultaneous independent Core motion; generation 9.10s; playback/save/export verified |
| Full scene | One person waves hello with their right hand. | Existing single-actor workflow, 2s; Review scene/playback verified |
| Motion | A person waves hello. | Existing assistant + G1 generation,4.16s, one complete chunk |
| Full scene | Three people: two meet and shake hands for four seconds while the third waves happily nearby. | Three performers, 7.7s; pair contact and independent third wave watched through completion |
| Full scene | Ten people dance together. | Visible unsupported-cast message; previous take retained |
| Two characters, custom | Two people face each other, give a high five, and lower their hands. | Rejected for unsafe entry transition; source and failure preserved |
| Two characters, custom | Two people shake hands warmly and release their hands. | 12.97s playable/exported, but looked more like an embrace; NOT a demo-quality semantic success |

Seed 42 for cast/manual cases. Manual starts changed to (-1.5,-1) and (2,1), meeting (0,0). Automatic starts and placements are in provenance manifests. All these tests used the existing stage scene; this integration fix does not re-qualify every environment/model prompt. Earlier multi-background motion evidence remains unchanged on main.

`platform-comparison.mp4` includes the before screenshot, complete successful pair/group exports, and the imperfect manual sample explicitly labelled. Individual videos and exact archives remain in their named folders. Quantitative duration/timings describe execution, not animation quality. Paired contacts remain model-dependent; independent group motion does not solve physically coordinated three-person contact.

## Runtime repair

The private Studio had no native pair provider configuration and its existing 8782 SSH tunnel had stopped. Restored only that tunnel, preserving remote workers. Provider config now resolves explicit CLI, STAGEZERO_NATIVE_PAIR_CONFIG, then ignored .runtime/prompt-native-provider.json. Missing configuration has an actionable UI message. Keep private credentials out of Git.

The original dirty workspace and previous live processes were preserved. No Pods provisioned; existing running total $1.37/hour, below shared $4/hour cap.

## Checks

Full Python suite: 1,415 passed before final stale-message-only fix; focused 86 UI/routing tests passed after it. Neon GPT-6 Astra reviewed routing; fixes include prompt provenance across duration changes and cancellation isolation. Codex Astra/high implemented bounded actor routing; Sol/high implemented provider configuration and manual-control gating. Main agent integrated and tested actual UI outputs.

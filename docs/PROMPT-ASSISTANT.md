# Motion prompt assistant and turn interpretation

The prompt assistant helps turn an ambiguous instruction into an explicit motion description. A phrase such as **“turn back”** may mean rotating to face the opposite direction, walking backward while keeping the same facing, or returning along a previous path. These actions need different prompts. The assistant should ask which meaning is intended before choosing one, and preserve a direction, angle, distance, or timing that the user already supplied.

For a confirmed turnaround in place, an example is: “Turn your whole body 180 degrees in place to face the opposite direction, then hold that facing.” Add left or right only when the user chooses a direction. For backward travel, describe backward steps and the retained facing explicitly. A prompt suggestion changes the instruction sent to the motion model; it is not a measured or enforced rotation.

## What the reported partial turn revealed

The inspected source contains no application rule mapping “turn back” to a 45-degree right turn. `live_motion.py` forwards the prompt text to `/generate`, and `pod_backend.py` passes the stripped text to ARDY's text encoder. The model samples motion from that text and the recent motion history.

`motion_policy.py::recognize_action` recognizes only overhead arm raises, right-hand waves, squats, and stop-and-stand-still instructions. “Turn back,” “turn around,” “turn 180 degrees,” and “turn left again” all return no recognized action. Their candidates therefore receive no turn-completion proxy. Finite poses, valid rotation matrices, toe-floor clearance, and continuity checks can accept a smooth partial turn. Passing those checks does not establish that the requested facing was reached.

`motion_constraints.py` currently supports a root X/Z waypoint, not a facing-angle goal. `motion_action_goals.py` provides reference-assisted overhead and squat goals only. The default ARDY decoder reconstructs posed joints from the generated rotation channels; its separate `global_root_heading` feature is not a standalone command that rotates the displayed skeleton.

Local `review/live-metrics.jsonl` entries for “he turns left” and “turn left again” record `selection.action: null`, `pose_goal: null`, and physically accepted candidates. No matching “turn back” entry was found in the inspected local JSONL/log files. Those records do not contain enough motion data to measure the reported 45-degree result, so the exact angle and direction of that particular clip remain unverified.

## Reproducible evidence

The CPU-only probe at `scratch/prompt-assistant-01a0dd7b/diagnosis/2026-09-26_121000_turn_probe.py` checks two behaviors:

1. The four turn phrases above have no supported action proxy.
2. Given a synthetic safe 45-degree candidate followed by a synthetic safe 180-degree candidate, the current selector chooses the first candidate even for “Turn around 180 degrees to face the opposite direction.” Both have `proxy.met: null`.

Both checks passed against the inspected source. Their output is saved in `scratch/prompt-assistant-01a0dd7b/diagnosis/synthetic-policy-evidence.log`. These are controlled policy examples, not a recreation of the user's generated clip.

## Behavior boundary

Clarification and explicit wording address ambiguity in the submitted instruction. They do not guarantee ARDY will execute an exact angle, complete every action, or maintain an exact final pose. The existing quality thresholds and candidate selection remain unchanged by this diagnosis.

Long actions are generated as multiple 104-frame requests, each receiving the action prompt again. A future measured turn controller would need one action-wide reference facing and a completion policy across those chunks, consistent position/rotation conditioning, and candidate validation. A per-request “rotate another 180 degrees” target could make a long action turn repeatedly. This requires separate implementation and validation; prompt wording alone does not provide that guarantee.

## Using the assistant

Press **Generate** once. Studio refines the direction before submitting motion.
If the direction is materially ambiguous, answer the short question and continue;
the same Generate operation resumes automatically. The interface shows refinement,
waiting-for-answer and motion-generation stages. The original and validated
model-facing directions remain visible for inspection.

A clear request such as “Öne doğru bir kere zıpla” must retain forward direction
and one repetition. “Make a nice big jump” may need a height-versus-travel question;
“big” never supplies a numeric distance or duration. Missing destinations and
lateral movement without a direction frame also require clarification. A selected
prop in the object editor is not automatically a movement destination.

Duplicate clicks do not create duplicate operations. Editing the direction,
changing its context or cancelling invalidates pending work. Late responses may
neither overwrite the new text nor start generation with an old direction.
Refinement failure preserves the original, shows one error, and never silently
submits the raw direction. The Generate flow has no separate **Improve prompt**
button: refinement runs automatically. The action-edit assistant retains its
manual refinement control.

## Model connection and failure handling

The old implementation always called local Ollama, even when the application had a configured text gateway, and caught provider/validation failures as successful offline guidance. For a clear English prompt that guidance could be identical to the input. This concealed model failures and made the button appear ineffective.

The assistant now uses the existing configured text gateway when present, preserving the configured model and credentials. Without a gateway it retains the local Ollama provider. It does not install a model, start a service, change private configuration, or silently switch providers after a failed request. Gateway calls have bounded timeouts and response sizes, carry an explicit JSON contract, and validate output before display. Errors produce a retryable failure rather than an unchanged successful suggestion.

The refiner keeps the original prompt, accumulated answers, question history, and optional scene context separate. Scene context can supply explicit `reference_target`, `street_end_target`, or `direction_reference`, plus resolved named `targets` and scene `objects`. Model-facing text must preserve supplied actions, directions, quantities, units, target and reference information. Deterministic checks reject detected changed or invented constraints; they are conservative checks, not a complete natural-language equivalence proof.

Known clarification questions work without network access. Unsupported-motion explanations must retain the requested action and explain the limitation rather than silently substitute another movement. The assistant does not certify that ARDY can execute a direction, distance, count, or duration precisely; motion execution and its measured validation remain separate.

## Validation

Focused tests cover clear commands, bilingual ambiguity, explicit scene context, answer roundtrips, intent constraints, API failure/retry, concurrent clicks, cancellation, stale replies, motion-mode changes, and both Studio generation entry points. Live UI inspection confirmed that the main Generate flow has no separate Improve prompt button. Real provider-to-motion end-to-end and video acceptance remain unverified because the video task was stopped by the user. Test doubles do not establish real model success.

## Automatic pipeline diagnostics

The automatic pipeline repair is tracked in ADR0007. Its acceptance record lives
under `scratch/auto-prompt-01a0df25/`. Diagnostics distinguish response format,
missing or invalid fields, transport failures and semantic constraint rejection.
Each attempt retains original input, answers, raw provider response, parsed result
and exact failed validation rule. Malformed response repair uses that rule and has
a finite attempt ceiling; genuine constraint changes remain failures.

The earlier 2389 failure retained only request inputs and the final generic
warning. Its exact raw response and reject exception cannot be reconstructed from
those logs. New observed-provider traces and deterministic old/new validator
replays must be reported separately from that historical failure.

Enable the standard `prompt_assistant` logger at DEBUG when investigating a
provider rejection. It emits one `prompt_refinement` JSON record per refinement
result, including every bounded attempt. The QA runtime additionally persists
these records beside its request ledger. Treat diagnostic prompts as user data;
transport headers, credentials and arbitrary transport exception text are not
included in the trace.

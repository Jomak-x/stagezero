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

Both the new-motion and action-edit Direction fields include **Prompt assistant**. **Improve prompt** asks the local model to clarify or rewrite the current instruction. Answer the questions using a suggested answer or your own words, press **Continue and refine**, review/edit **Refined direction**, then select **Use prompt**. Applying a suggestion changes only that field; generation still requires its normal button. A bare ambiguous turn command also opens clarification when Generate or Update motion is pressed.

The optional AI provider uses the existing local Ollama service at `127.0.0.1:11434` and `STAGEZERO_LOCAL_MODEL` (default `qwen3:4b`). It does not install models or start services. Connection/read timeouts are 3/55 seconds, response size is capped at32KiB, and output must satisfy a bounded structured schema. If AI is unavailable or its output loses explicit angle/side/in-place constraints, the interface shows a warning and offline guidance. This constraint check is deliberately lexical, not a general proof that every natural-language detail survived.

Offline guidance can clarify the known turn meanings; it is labeled separately from AI output. Unknown or unresolved references may still need a user's fuller description. Prompt and take changes invalidate pending suggestions, and the apply callback rechecks the target under the session lock. Multiple question rounds retain question text and answers.

Validation uses deterministic provider/transport doubles and a private browser preview with offline guidance and no motion backend. It does not spend shared inference capacity or verify the quality of a new real ARDY turnaround.

The completed change passed18 core,8 helper lifecycle, and5 Studio integration tests; the combined focused suite passed90 tests and full unittest discovery passed369. Independent review found no remaining actionable scoped issues. Browser verification confirmed the clarification→180° choice→preview→Use prompt path. The temporary verification process was stopped; running user applications were not restarted.

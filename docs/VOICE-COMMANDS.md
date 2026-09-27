# Voice commands

Activate **Voice tools** in the studio sidebar, choose **Automatic**, then hold
the microphone button while speaking. Release to submit immediately. The
microphone is only active while held; disabling Voice tools stops recording.
Typed commands use the same routing and queue.

- `Generate a scene: walk to the door, turn, then wave.` plans a full sequence.
- `Generate a short: wave hello.` generates one motion action.
- `Edit scene "Walk and wave" to walk slowly.` revises the full sequence,
  preserving the original take and its total duration.
- `Edit "Walk and wave" to walk slowly.` creates an alternate take and preserves
  the original.
- `Edit action 2 in "Walk and wave" to wave with both hands.` replaces that action.

Use the take name shown in the interface. Ambiguous or missing names are rejected
with guidance, rather than editing the current selection. Commands operate on
one named take or action at a time. This version exposes generation and motion
editing, not arbitrary application tools or multi-character control.

The queue runs commands in order and waits while another action is generating.
Cancel an individual request in the queue. Changing the project invalidates
queued work; cancelled or superseded generations cannot install a late result.
Full-scene results also appear in the existing Full scene interface.
New scenes use the planner's automatic duration; short commands use the current
single-action duration estimator.

## Configuration

Set `ELEVENLABS_API_KEY` in the server environment (the ignored
`.runtime/pod.env` is supported by the existing launcher). The key needs Speech
to Text permission. No voice ID is needed for command transcription. Keep the
key out of browser code and saved projects.

Microphone capture requires localhost or HTTPS and browser microphone permission.
Recordings are limited to 30 seconds and 8 MiB and sent to ElevenLabs for
transcription. The application does not persist raw recordings. Typed commands
remain available when microphone access or transcription fails. Existing motion
backend and full-scene planning configuration are required for generation.

Character speech generation, synchronized captions, and bundled dialogue audio
are a separate future integration; this PR controls the existing scene and action
workflows through speech-to-text.

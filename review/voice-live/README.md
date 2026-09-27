# Real voice-command review

[Watch the video](voice-live-system-demo.mp4)

The video plays prerecorded command audio, shows the exact ElevenLabs transcript, and renders actual saved character motion at native 25 fps. Waiting periods are condensed. This is a rendered review of real results, not a continuous recording of the studio interface.

| Spoken command | Verified result |
| --- | --- |
| Generate a scene: walk forward, stop, then wave. | Three-action scene completed. |
| Generate a short: bow. | Queued behind the scene, then generated a separate short take. |
| Edit Copper Finch action two to bow. | Replaced the named action atomically, retaining the surrounding prompts. |

The initial edit using the automatically generated title was rejected when transcription misheard its name. Renaming that same saved take **Copper Finch** made the edit succeed. No fuzzy matching guessed which take to edit. The action edit preserves its original duration: replacing a 0.48-second stop makes a very brief bow. The separate short shows a longer bow.

Queued cancellation and unknown-name rejection also passed. The live test exposed two issues, now fixed: spoken punctuation/action numbers were rejected by the parser, and socket coalescing could drop the previous request's completion when the next request began. Validation after fixes: 61 Python voice/planning tests and 30 client tests passed.

Physical microphone capture and browser permissions were not exercised. Character dialogue, lip sync, and multi-character control are not demonstrated.

## Evidence and reproduction

[The report](report.json) includes real WebSocket status events, the rejected transcript, and successful edit checks. Request durations include transcription, queue wait, planning, and generation. The recovery GPU count covers only the resumed edit process. The initial rejected command WAV was recreated later and is explicitly labeled in the report.

The local review runner sends macOS `say` WAVs through Viser into `VoiceDirecting.recording`, using real ElevenLabs Scribe, StoryPlanner, StoryWorkflow, and the existing ARDY backend. It does not start external services. Local WAVs and project snapshots are saved in `.runtime/voice-live/`. The run resumed from the saved scene/short project for the named-edit retry instead of generating them again.

# Character speech live review

The [offline report](offline-report.json) covers a real Viser WebSocket with a
local speech adapter: a retryable failure, a successful retry using an
already-recorded MP3, removal, and cancellation while synthesis is blocked.
It made no provider or motion generation calls.

The [live report](live-report.json) covers four real ElevenLabs text-to-speech
requests on the saved generated G1 take. Bella speaks lines 1 and 3; Roger
speaks lines 2 and 4. Lines start at frames 0, 35, 75, and 141 in the
untouched live result. Their aligned cue durations are 0.72, 1.48, 2.68, and
3.12 seconds; request wall times were 0.253, 0.302, 0.358, and 0.354 seconds.
The fourth line extended the take with a stationary final pose. Repeating
line 1 at a new frame completed from memory: synthesis count stayed 4 → 4.
No GPU motion request occurred, and the original generated motion frames are
unchanged.

The [render manifest](render-manifest.json) points at the clean four-line
project and its standalone MP3s. The original live four-line result had one
frame (0.04 seconds) of overlap between lines 3 and 4. The clean review
project shifts line 4 one frame later and extends the stationary hold by one
frame; it preserves every generated audio byte and every original motion
frame. The untouched live and cached-repeat projects remain in
`/Users/jakob/Desktop/Shellhacks/.runtime/character-voice/` and are linked
from the live report. Aligned cue durations include decoder padding, so
`ffprobe` reports slightly shorter MP3 container durations.

Reproduce the offline socket checks with:

```sh
/Users/jakob/Desktop/Shellhacks/.venv/bin/python review/character-voice/run_character_voice.py
```

The live run requires an explicit `--live --allow-provider-calls` pair and an
authorized provider account. The runner only uses an existing saved motion
project; it does not ask the GPU service for more motion.

## UI verification

The actual Voice control center generated `Hello from Sarah.` and `Hello from Charlie.` successfully, completing real TTS tests for all four available voices. Both temporary lines were removed through the UI. Playback advanced captions with the saved take; pause retained the correct line. The local browser build and all 45 client tests pass. The complete Python suite passed 1,314 tests before the final additional regression cases; CI verifies the final commit.

[Watch the four-line character voice video](character-voices-demo.mp4). The same character demonstrates alternating female/male voices; this is not simultaneous multi-character dialogue or lip sync.

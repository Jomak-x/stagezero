# Stage One demo, revised cut

The delivery is a curated set of separate silent clips plus one concatenated review reel. No captions, narration, music, or descriptive title cards are burned into the footage. The CSV and gallery identify clips for editing.

The UI footage was recaptured using the live application WebGL canvas and timeline, composited with event-driven interface snapshots. Playback runs at its recorded speed; generation waits and idle navigation are removed with hard cuts. No optical-flow interpolation or invented frames are used. The capture targets 25 fps; delivery is standardized to 30 fps for editing, so container frame rate alone is not evidence of motion smoothness. Actor-region cadence checks were performed on the final workflow clips.

The scene-only section keeps one selected example per capability: custom mechanic, two-person interaction, three simultaneous actions, a door opening on contact, and a continuous terrain route with two stair ascents and a bridge crossing. Repeated door and staircase takes are excluded.

Coverage limits: a fresh character-generation attempt encountered a busy GPU backend and is excluded. Rooftop jump, third-flight ascent, and descent probes were rejected and are not represented as successful capabilities. Saved generated characters are demonstrated through the picker. 

The character-picker race and robot-switch deadlock were fixed and checked with 43 relevant tests. Live Alex-to-woman-to-robot switching was then verified in the studio. This PR contains the final media package only. Recorder and picker fixes, raw captures, and the first delivery remain in the working demo archive.

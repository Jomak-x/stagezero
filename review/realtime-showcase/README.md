# Verified realtime scene recordings

These recordings render the exact saved native poses at 20 fps, without motion interpolation or authored pose replacement. Each scene is 600 frames / 30 seconds; the combined reel joins three separate tests.

- [Gate, handshake, departure](gate_meet_handshake.mp4)
- [Staged sparring and disengagement](staged_fight.mp4)
- [Console inspection and departure](object_reach_inspect.mp4)
- [Combined 90-second reel](realtime-showcase-reel.mp4)
- [Live private viewer screenshot](live-private-viewer.png)
- [Deliberate connection failure](live-recovery-failure.png)

Contact sheets show scene boundaries and interaction details. Manifests record source/report hashes and render properties. Raw projects remain in the ignored local `.runtime/realtime-showcase-v1` directory; independent measurements are in [realtime-validation.json](../realtime-validation.json).

Paired motion uses research-only InterGen. Contacts are proximity-conditioned choreography, not grasping or physical collision simulation. Scene navigation uses declared geometry and verified affordances. See [measured results and limitations](../../docs/REALTIME-RESULTS.md).

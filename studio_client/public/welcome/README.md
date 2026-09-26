# Welcome victory dance and backflip

`victory-dance.mp4` is a decorative G1 animation viewed from the front, independent of the user's project and camera. It plays a recorded animation; opening Welcome does not run inference.

- Dance, frames 0–103: original ARDY output preserved exactly. Model `nvidia/ARDY-G1-RP-25FPS-Horizon52`, expressive profile, seed 17, one candidate, no history. Prompt: “A person performs a joyful victory dance in place, bouncing from foot to foot and pumping both fists in celebration, facing forward.”
- Ending, frames 104–223: **procedural keyframed backflip**, not model-generated choreography. Fixed-bone G1 forward kinematics and local-rotation interpolation combine a crouch, a single backward 360° rotation, an airborne arc, landing absorption and an upright hold. Poses reference the original dance and a generated backflip attempt; the generated attempt did not provide a usable complete landing.
- Source animation: 224 frames at 25 fps (8.96 seconds), captured at 900×900 using the official G1 mesh rig and a fixed frontal camera fitted to the complete motion arc.
- Presentation: composited over `#0b1011`, forward playback only, with a short final hold and fade before the loop restarts (9.56 seconds total). The backflip is never played in reverse. H.264, yuv420p, no audio.
- Poster: dance frame at 1.8 seconds. Reduced-motion users start with a still image and can opt into playback.

The procedural animation is an opening-screen visual, not evidence of inference quality or a physically validated robot-control trajectory. Existing upstream ARDY/G1 asset attribution and licenses continue to apply. The gated source recording is not included in these assets.

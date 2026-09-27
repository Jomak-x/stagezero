# Character start placement review

`placement-review.mp4` is an 18-second, 1280×720 placement preview with three live
Neon GPT-6 Astra plans. It shows one explicit start/facing, two independent
starts/opposite facings, and automatic three-person staging. The explicit cases
returned the requested marks and angles; the automatic case chose three separate
marks at x=-2,0,2, all facing +Z. Exact prompts, raw responses, validated plans,
resolved marks and source provenance are checked in beside the video.

The poses/motion are replayed from the existing solo-city archive and rigidly
placed using the production initial-pose helper. This is a schematic staging
preview, not a new Core/InterGen performance or motion-quality acceptance.
The local checkout had no private motion backend configuration; the Mac mini was
online through Tailscale but refused SSH connections on port 22. No remote
services were restarted or changed.

Validation: 109 focused CPU tests passed across prompt planning/building, paired
meetup, cast archives, observer motion and observer staging. Tests cover separate
position/yaw intent, invalid yaw, one/two/three actors, explicit zero, first-route
heading, native source orientation, Core request radians, paired first-target
heading ramp and legacy observer/position-only behavior.

Re-render without gateway access:

```sh
python3 experiments/review_initial_placement.py --replay-plans --output review/initial-placement
```

To obtain new plans, omit `--replay-plans` and supply normal gateway environment
variables or `--gateway-env /path/to/private/.env.neon-worker`. The script needs
numpy, requests, Pillow and ffmpeg. It never saves credentials.

## Live interactive demo

The follow-up demo runs `initial_placement_demo.py` with the project's actual
Xbot skin renderer and a live Neon planner. In the browser, clicking **Place
with Neon** chose Alice=(-2,0), facing90°, Bob=(2,0), facing-90°, and Carol=(0,-2),
facing0°. Dragging Alice's X handle moved her to(-4.1,0); dragging the Y rotation
ring changed her heading to4°. Bob and Carol stayed unchanged. The browser
export succeeded and the exported JSON revalidated to the exact displayed marks.

`interactive/ai-and-edited-state.json` includes the original live plan, model
metadata and edited state. `interactive/events.json` records the actual UI
operations; `interactive/edited-plan.json` is a builder-compatible exported plan.
The preview is live and interactive, but it does not generate new body motion.
112 focused CPU tests passed including atomic rollback, independent actor
editing, reset, 1–3 actors and exported-plan round trip.

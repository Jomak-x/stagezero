# Final demo stress review

[Play the 1:52 review reel](demo-review.mp4) or open [the review page](index.html). Five complete, normal-speed takes: four new heavier trials and the previously reviewed Market48 observer example. No cuts inside performances, no retiming. Title/failure cards are editorial additions. Exact source videos, source hashes, every frame index, raw motion, rejected attempts and measurements are preserved.

**Assessment:** ready for a prepared demo with the verified cases. Arbitrary three/four-beat rotations are still unreliable, and close-contact/foot detail remains rough. A successful generation is not a guarantee of attractive animation. This final pass adds a compatibility recovery, not a new animation approach.

| Test | Seed | Result | Visual finding |
|---|---:|---|---|
| Long City greetings, three actors, road starts 5m apart | 62 | 876 frames / 29.2s | Partner change readable; restrained observer, imperfect feet/contact |
| Industrial spar then greet, wide starts | 63 | 581 frames / 19.37s | Full sequence completes; close-contact body overlap remains |
| Market partner dance | 64 | 321 frames / 10.7s after compatibility recovery | Coherent dance, rough contact/feet |
| City solo walk/turn/wave | 65 | 240 frames / 8s | Walking completes; wave is understated |
| Market three-beat round robin | 61 | Rejected on candidate and main | Existing endpoint continuity rejection retained |
| Market four-beat rotation | 66 | Rejected on candidate and main | Existing endpoint continuity rejection retained |
| Original City explicit layout | 62 | Preflight rejection | Invalid start; corrected layout above retains prompt/actions/seed |

## Compatibility with main

The first dance attempt exposed a real regression: the new 8cm arrival-margin policy rejected three initial sampled body-sphere frames (worst about -5.83cm), while latest main accepted the same InterGen source. The final builder retries **once**, with the same seed/raw pair and original arrival targets, only for that new clearance failure. All original geometry, continuity, height and cancellation/error behavior remains enforced. Both attempts and their source lists are recorded in metadata. Other failures do not trigger this recovery.

The final dance has one sampled noncontact overlap frame, -2.07cm, with positive torso clearance. This is a proxy measurement, not mesh penetration. It is not claimed to pass the new zero-overlap policy. Native contact motion is not edited to make measurements look better.

`baseline-main/market-dance/attempt-001` and `market-round-robin/attempt-001` replay archived InterGen with fresh Core on main. Four-beat baseline attempt001 lacks a later archived pair because the candidate cancelled it; it is inconclusive. Baseline attempt002 uses fresh same-seed Core/InterGen and also rejects the existing endpoint gate. Original logs remain alongside each comparison.

## Verification

The main agent played all four complete new takes at 1x and inspected continuous 2fps filmstrips, including approach, handoff and endings. The reel also includes the previously reviewed complete Market48 performance. [Machine-readable results](final-results.json), [reel manifest](reel-manifest.json), and per-attempt `metrics.json` retain measurements and limitations.

**1,364 Python tests pass**, including four bounded-recovery tests. The unchanged client previously passed 45 tests, TypeScript and production build. Fresh post-fix City42 and Market48 reruns are exactly equal to the previously reviewed displayed arrays: [comparison](regression-comparison.json). Neon GPT-5-6-Sol reviewed stress coverage and GPT-6 Astra reviewed the bounded compatibility design; Codex integrated and tested the patch.

No fresh external AI planner was used: these are explicit frozen test plans driving actual motion models. No new Pods were rented; last live total was $1.90/hour across both chats. Main UI, original dirty checkout, other workers and live port2380 remain untouched.

## Reproduce

From the checkout, reserve the shared model lane first, then use existing private credentials (never copy them into review artifacts):

```sh
PYTHONPATH=.:vendor/ardy .venv/bin/python experiments/stress_interaction_demo.py \
  --generate --case city-long-supported --model-lane-reserved \
  --config /path/to/existing-provider-config.json --token /path/to/existing-token
```

`--case all` retains the original invalid layout and rejected stress seeds intentionally. Each attempt gets a new immutable directory. Plans and case seeds are in `cases.json`. Replay any accepted `.stagezero.npz` with `experiments/run_interaction_review.sh`; no model calls are needed for playback. The existing six-case matched before/after evidence remains in [interaction-v2](../interaction-v2/index.html).

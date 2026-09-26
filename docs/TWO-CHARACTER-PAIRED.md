# Joint two-character performance review

**Rejected by user.** The retargeted close-exchange preview was rejected for motion quality and malformed joints. Native-source recovery and direct authored-rig experiments supersede this approach; retain the evidence as failure provenance.

The previous independent Core duet was rejected by the user for excessive separation and mistimed actions. Its evidence is retained in `review/two-character/`; it is not the selected result.

This replacement generates both people together in one InterGen sample, then retargets that complete sample to the existing textured studio characters. It does not stitch independently generated actions or hide a failing ending. The existing city from `review/scene-integration/live-city.json` is reused.

## Use

Open Motion → Scene direction · Core → Together · experimental. Choose Close exchange or Partner dance, use the preset, then Generate joint pair · research. The paired controls provide play/restart, a frame slider, cancellation, and separate `.paired.stagezero.npz` save/load. Native Core and G1 clips remain separate. Switch modes after pending generation finishes or is cancelled.

The close exchange is a six-second block/shove/step-apart performance. Partner dance is a small shared turn with a long setup, not a rich dance routine. These are reviewed fixed prompt/seed presets; arbitrary new prompts are not guaranteed to work.

## What changed

- Joint generation supplies shared timing and partner-relative movement. No imposed 2.25 m separation.
- Paired character fitting preserves reachable hand targets and applies one shared height/floor translation. An initial fitting attempt raised hands relative to the torso; that attempt was rejected and the height correction fixed.
- Complete scenes commit atomically. Cancellation, transport failure, scene/floor failure, or gross torso overlap retains the previous scene.
- All-frame torso-centre proximity below 0.35 m rejects an output; this is a proxy, not mesh collision proof. Existing authored-floor and scene-solid checks also apply. Hands may intentionally meet.
- Paired archives preserve exact retargeted arrays, scene, seed and source provenance; they do not invent native ARDY features.

## Evidence and rejected cases

See `review/two-character/paired-research/independent-review.json` and the per-run reports/captures. Six fresh joint-model trials were generated serially on the existing worker: exchange42, high-five7301, dance7302, boxing7303, boxing7304, push37. UI verification adds its own actual generation.

The high-five did not make a clear high-five. Boxing7303 had a real body intersection near the ending and is rejected in full, not shortened into a successful demo. Boxing7304 avoids that intersection but its second half is less readable. Push37 does not give a clean push/backward stumble. None is a reviewed default.

The selected exchange has a minimum/median root gap of 0.635/0.836 m, compared with 1.644/1.941 m in the rejected duet. Its minimum torso-centre distance is 0.467 m. Peak single-frame joint movement is 0.270 m versus 0.568 m in the old duet. These measurements support review; they do not establish animation quality on their own.

## Reproduce

Run from this worktree with the existing private Core token path in `CORE_REVIEW_TOKEN_FILE`. The worker at 8769 must expose paired research. No new Pod or worker restart was needed.

```sh
.venv/bin/python experiments/trial_paired_scene.py \
  --prompt 'Two people perform a choreographed martial arts exchange: sidestep dodge, forearm block, controlled push, then step apart.' \
  --seed 42 --frames 120 --token-file "$CORE_REVIEW_TOKEN_FILE" \
  --name exchange-rerun --output .runtime/paired-rerun

.venv/bin/python experiments/capture_core_performance.py \
  --canonical .runtime/paired-rerun/exchange-rerun.intergen.canonical.npz \
  --scene review/scene-integration/live-city.json --paired-retarget \
  --output-dir .runtime/exchange-recapture --port 24892 \
  --camera-position 2.5 1.85 3.5 --look-at 0 0.95 0 --sheet-frames 24
```

Enter the printed local capture URL and click Start exact research capture. Every saved frame is rendered at 20 fps; the 30 fps InterGen source is retargeted/resampled by the model bridge. A Studio-saved paired project can instead be captured with `--paired-project path.paired.stagezero.npz --paired-retarget`, which uses its saved scene.

## Limits

This is a distinct research workflow using an InterGen checkpoint marked CC BY-NC-SA 4.0, not a production replacement for ARDY. Scene geometry is checked after generation, not perceived by the model. The fitted character rigs have rigid fingers and approximate joint placement; contact, foot locking and mesh physics are not solved. The close exchange reads as a coordinated scuffle, not polished fight choreography.

## Verified studio result

The default Close exchange was freshly generated through the studio, replayed, saved, switched back to G1, and reopened at its saved playhead. Exact positions, rotations, scene and frame survived archive round-trip. All 828 Python tests passed after integrating main PR #19.

The full UI-generated video is `review/two-character/paired-research/ui-close-exchange-capture/performance.mp4`; the replay is `ui-close-exchange.paired.stagezero.npz` in the same evidence directory. `results.json` records metrics, review and rejections. Browser playback and the full-scene contact sheet show closer coordinated movement, but artificial hand and arm poses remain. This is a research preview for review, not production-quality animation acceptance.

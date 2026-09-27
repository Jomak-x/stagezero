# Native cast and ARDY interaction integration

The approved native InterGen character rendering now runs inside StageZero's existing Motion tab. It bypasses the rejected InterGen → Core27 → approximate skin path. Existing G1, Core, character authoring and scene generation remain available.

## Use it

The current unified workflow is documented in [PAIRED-DIRECTION.md](PAIRED-DIRECTION.md) and runs at `http://127.0.0.1:2380/`. Choose **Motion → Direct → Two characters** for separate starts and a shared meeting. The lower-level native controls described below are under **Cast, playback and files**:

1. Add and name characters; choose two different performers and press **Use this pair**. The UI supports up to six cast members. Other actors hold static poses at separate marks; this is not jointly generated group motion.
2. Press **Play reviewed handshake** or **Play sparring preview**, or describe a new interaction and press **Generate pair motion**. The first two are reproducible library sources; generation runs the actual InterGen checkpoint on the configured Pod.
3. Use **Place the pair in the scene** to translate/rotate both performers together. Relative partner motion stays unchanged. **Frame cast** frames the current cast.
4. The optional **Closed fists** hand pose is authored finger articulation. It does not change body joints or imply that the model generated finger motion or verified a strike.
5. **Add ARDY approach and exit · experimental** makes four bounded Core requests around the selected native interaction. Source segments and authored transitions are labeled on the timeline. It preserves the original interaction and checks continuous authored floor support plus sampled scene solid proxies before committing.
6. Save the performance using the primary Save button or the native archive controls. Reopen it under **Saved native performances**. **Export playback video** renders every frame and downloads an MP4.

## Models and continuity

InterGen generates two actors jointly at 30 fps. ARDY Core generates its own native27 motion at 20 fps. Core history never contains fabricated InterGen features. Composed playback uses an explicit Core-only anatomy retarget, 20→30 fps interpolation, and separately authored entry/exit transitions. The native interaction frames remain unchanged in local coordinates. Exact Core positions, rotations and features, plus the original pair archive, are retained under `.runtime/native-pair-context/<run>/`.

Initial direct Core→pair seams moved body joints by about 30 cm despite centimetre-level root target accuracy. That candidate was rejected. Matching only the Core segment proportions to the pair reduced the visible anatomy jump. This is an explicit display transformation, not native model conditioning or a learned continuous rollout. Foot planting, palm/fist contact, character-to-character mesh collisions and arbitrary combat timing are still not solved. The optional ankle-plant experiment remains separate pending visual acceptance.

The actual generated-city backdrop is reused. Geometry checks are conservative scene metadata proxies, not visual perception or physical simulation. Pair generation is not scene-conditioned. Shared placement controls alone do not guarantee collision-free staging; the ARDY composed builder rejects unsupported or colliding results.

## Run

From this checkout, after building `studio_client` and installing the existing project environment:

```sh
.venv/bin/python director_viewer.py --reference-only --port 2378 \
  --token-path /path/to/private/api-token \
  --native-pair-config /path/to/private/native-pair-provider.json \
  --objects review/scene-integration/live-city.json
```

Omit `--native-pair-config` for library playback without remote generation. The provider configuration contains `ssh_host`, `ssh_port`, and `known_hosts`; optional model/probe paths and timeouts are documented by `NativePairConfig` in `native_pair_provider.py`. The configured host must already contain the InterGen probe/checkpoint. No model download, worker restart or Pod rental happens on studio startup. Credentials and configuration are not committed.

The bundled authored rig has source/hash attribution in `assets/paired/README.md`. InterGen remains a noncommercial research model (CC BY-NC-SA 4.0); integration does not change its license.

## Validation and evidence

`review/two-character/platform-integration/` contains actual new model samples, full studio exports, exact project archives and review decisions. Earlier rejected outputs remain in their original evidence folders. Tests cover native archive round trips, cast selection, cancellation/stale responses, placement, capture leases, geometry rejection and mixed-source timeline coverage. Tests do not establish animation quality; full playback review is separate.

## Review milestone

The integrated sparring preview uses seed 37 and the exact prompt in `review/two-character/platform-integration/block37/provenance.json`. Generation took 13.918 seconds end to end, including 0.900 seconds reported model sampling. Closed fists are separately authored. The reviewed ARDY handshake composition is 492 frames (16.4 seconds): approach, authored entry, original 210-frame interaction, authored exit and departure. Approach root errors were 1.85 cm and 0.39 cm; departure errors were 8.80 cm and 1.12 cm. These numbers do not verify hand contact or animation quality.

Use `--native-project /path/to/scene.native-pair.stagezero.npz` to reopen an exact exported scene on startup. Full videos and raw source archives are listed in `review/two-character/platform-integration/results.json`. Serve that folder’s parent using `python3 -m http.server 24893 --directory review/two-character` and open `/platform-integration/watch.html` for the comparison player. Run regressions with `.venv/bin/python -m unittest discover`.

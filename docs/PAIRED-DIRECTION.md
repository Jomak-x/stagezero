# Direct a two-person scene

The **Direct a scene with two people** panel builds one complete take from two named cast members, separate start markers, a shared meeting point, and an interaction. Choose the provided handshake or sparring source, or describe another interaction for fresh InterGen generation. **Preview places** checks the routes and meeting area without running either motion model. **Generate scene** submits the bounded motion jobs; **Cancel scene** stops publication, and **Retry scene** repeats the captured request while its scene and cast remain unchanged.

The planner keeps actor IDs and roles fixed. It routes each start to that actor's own entry position, schedules a shared arrival, checks continuous authored floor and conservative scene-solid proxies, and rejects routes that cross too closely. A landmark choice resolves to clear ground beside that object; its name does not create a passage. The approach uses the existing authenticated ARDY Core service: one 40-frame, 20 fps Core27 window per planned horizon, with Core native features carried only between Core windows. Its display positions are resampled to 30 fps and fitted to each native22 actor's proportions. A separately labeled 21-frame authored transition joins the approach to the **original 210-frame, 30 fps InterGen pair**, whose source joints are restored exactly. The composed take has no unified model features and does not include a generated exit.

These checks reject missed root targets, unsupported floor, sampled scene-solid overlap, excessive approach proximity, large arrival errors, and failed transition bounds (including an entry root-height mismatch above 0.30 m) before replacing the current take. They do not establish mesh clearance, foot locking, partner contact, or convincing choreography. Close partners are allowed during the interaction so intended contact is not rejected by the approach spacing rule. `result.json` records each source segment, plan, measurements, geometry report, and `animation_accepted: false`.

## Reproduce the recorded city requests

Run from the repository root with the existing virtual environment, the already running authenticated Core service at `127.0.0.1:8769`, and a local file containing its token. Set `CORE_TOKEN_FILE` to that file's absolute path. Use a **new, nonexistent** output directory for each command; the trial refuses to overwrite evidence.

```sh
CORE_TOKEN_FILE=/absolute/path/to/existing/core-token
.venv/bin/python experiments/trial_paired_direction.py \
  --request review/two-character/city-meetup/requests/sparring.json \
  --scene review/scene-integration/live-city.json \
  --token "$CORE_TOKEN_FILE" \
  --output /private/tmp/paired-direction-sparring-repro
```

```sh
.venv/bin/python experiments/trial_paired_direction.py \
  --request review/two-character/city-meetup/requests/crossed-sparring.json \
  --scene review/scene-integration/live-city.json \
  --token "$CORE_TOKEN_FILE" \
  --output /private/tmp/paired-direction-crossed-repro
```

Fresh paired motion additionally needs the configured existing Pod, its installed published InterGen checkpoint and probe, and the interaction prompt planner's configured gateway. The private `--config` JSON supplies SSH host, known-hosts file, and remote asset paths; it is not checked in. This path uses the original native30 probe, not the older Core endpoint's 20 fps paired retargeting. It neither downloads checkpoints nor restarts workers.

```sh
.venv/bin/python experiments/trial_paired_direction.py \
  --request review/two-character/city-meetup/requests/embrace.json \
  --scene review/scene-integration/live-city.json \
  --token "$CORE_TOKEN_FILE" \
  --config .runtime/native-pair-provider.json \
  --output /private/tmp/paired-direction-embrace-repro
```

The recorded runs under [`review/two-character/city-meetup`](../review/two-character/city-meetup) produced 471 frames from four Core windows for sparring, 411 frames from three windows plus a fresh InterGen embrace, and 531 frames from five windows for crossed sparring. All are 30 fps composed review candidates; their reported arrival errors were below 5 cm. Those measurements alone are not visual acceptance. Final capture decisions and the replacement standing-hug run are in [the milestone review](../review/two-character/city-meetup/README.md). The first embrace is now rejected by the entry-height check; its original output remains as failure evidence. Use `requests/standing-hug.json` with the same custom-generation command to reproduce the replacement.

Each successful trial writes a separate native project (`scene.native-pair.stagezero.npz`), `result.json`, and a timestamped `sources/` folder containing the request, plan, exact native pair joints and features, every completed raw Core window, and the composed review array. A failed or cancelled run writes `failure.json` and retains completed source archives; it does not publish a partial project. In Studio, **Save performance + download** and the primary **Save** control write the separate native project format with cast IDs, scene, placement, and playhead. Retry is available only while the saved request remains valid for the same scene and cast. Existing G1 and Native Core takes remain separate.

InterGen's published checkpoint is licensed **CC BY-NC-SA 4.0** for noncommercial research. Fresh sampling runs serially on the existing Pod with a GPU free-memory preflight and temporary-file cleanup. The pod's disk was tight during this work; do not add checkpoints there for this workflow. The authored Xbot skin and optional hand poses are display choices, not generated finger contact or a physics solve. Review the complete rendered take before treating any candidate as usable animation.

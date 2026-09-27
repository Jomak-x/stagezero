# Interaction quality milestone — observer continuation

The selected improvement replaces frozen nonparticipants with bounded authored upper-body continuation. Initial observers face the meeting; after their own greeting, arms ease down and the torso settles. Greeting travel uses a relaxed entry cue while sparring retains a guard. Existing native pair frames, roots, waiting legs/feet, source archives and main UI are preserved.

This is a visible **observer-release improvement**, not acceptance of fully natural three-person acting. The clearest exact-source comparison is the original main-UI take: cyan no longer holds an extended release arm through the next greeting, and purple settles while waiting. Whole-body weight shifts, stepping to follow a new partner, exact palm contact and reliable planted feet remain unresolved.

## Watch and run

Every linked MP4 contains the complete take at 30 fps, with no cuts, retiming or joined takes. Each capture directory includes the exact cast archive, every captured frame index, source hash, camera provenance, first/last frames and a contact sheet.

| Comparison | Before | Selected after | What it isolates |
|---|---|---|---|
| Recorded main-UI take, 15.9 s | [Before](video/recorded-before/playback.mp4) | [Observer-only](video/observers-only/playback.mp4) | Identical model motion and camera; only authored observer continuation |
| Fresh Market, seed 42 | [Before](video/baseline-market-three-s42-fixed-plan/playback.mp4) | [Final](video/final-market-three-s42-fixed-plan/playback.mp4) | Fresh Core + InterGen on both sides; same saved plan/seed, Core output differs |
| Fresh Market, seed 48 | [Before](video/baseline-market-three-s48-fixed-plan/playback.mp4) | [Final](video/final-market-three-s48-fixed-plan/playback.mp4) | Fresh Core + InterGen; also includes initial meeting-facing staging and relaxed entry |
| Fresh City, seed 42 | [Before](video/baseline-city-handshake-s42-fixed-plan/playback.mp4) | [Final](video/final-city-handshake-s42-fixed-plan/playback.mp4) | Relaxed greeting entry, unchanged paired source |
| Fresh Industrial, seed 45 | [Before](video/baseline-industrial-spar-s45-fixed-plan/playback.mp4) | [Final, retained with overlap concern](video/final-industrial-spar-s45-fixed-plan/playback.mp4) | Explicit x ±3 m starts, spar guard cue |

Held-out final runs: [City seed 43](video/final-city-handshake-s43-fixed-plan/playback.mp4), [Market seed 49](video/final-market-three-s49-fixed-plan/playback.mp4). The latter is retained with a tight-entry overlap concern. No failed or awkward case was replaced by a better seed.

From the repository root, with the normal Python dependencies and built Studio client:

```sh
# Replay the selected Market seed 48 in the existing main UI, without models.
PYTHONPATH=.:vendor/ardy sh experiments/run_interaction_review.sh
# Open http://localhost:24971 ; Start / Play / Pause and timeline work as usual.
# Optional arguments: saved-cast-path port. STAGEZERO_PYTHON overrides .venv/bin/python.
```

`STAGEZERO_CLIENT_BUILD` may point to an existing compatible client build. The review used the unchanged main client from the live-demo worktree, read-only; the UI layout and live-demo services were preserved. The existing research-preview sentence now identifies authored observer motion.

To reproduce complete captures into a new directory after building the current client:

```sh
PYTHONPATH=.:vendor/ardy .venv/bin/python experiments/capture_cast_review.py \
  --manifest review/interaction-quality/capture-final.json \
  --output-root /tmp/interaction-final-recapture --port 24971 --wait-seconds 180
# Open the printed local URL, enter Studio, then click Start cast batch capture.
```

Stop only your own preview on that port before capture. The capture tool refuses existing output directories, preserves partial failures, and uses one fixed camera for each complete take. `camera: closer` uses 0.75 of the full-take camera distance; all actors remain visible in these captures.

## Visual review and acceptance

The primary agent played the final performances at normal speed, inspected start/contact/release/end views, and supplemented the three-person and spar reviews with complete 0.5-second filmstrips. [Visual review records](visual-review.json) distinguish the narrow accepted change from remaining defects. The observer-only and observer-plus-feet alternatives were also compared on the same recorded take.

- **Accepted:** arm relaxation and upper-body settling remove the most obvious frozen release pose without changing active pair motion. Initial meeting-facing staging improves the first waiting actor's orientation. Reentry uses a smooth envelope back to the exact original boundary pose; no observer-induced hard reset was visible.
- **Still awkward:** a source stance can hold one heel/foot raised throughout waiting. Small attention turns cannot rotate an entire body to follow a second greeting happening behind it. Source approach stepping/glide and backwards torso lean remain visible. Some entry/exit movements are brisk despite continuous positions.
- **Contact is not solved:** paired action coordinates are intentionally unchanged. The City wrist minima remain 18.4/19.7 cm; these are wrist distances, not distances between rendered palms/fingers. Market contact can look like a handshake, but close bows/hand crossings do not establish correct grasp physics.
- **Retained concerns:** Industrial seed 45 has eight unintended sphere-overlap frames at 4.10–4.33 s (minimum −3.84 cm); Market seed 49 has ten at 2.30–2.60 s (minimum −0.94 cm). Close entry views are retained in `inspection/`. These overlaps predate refinement and are not worsened. Existing scene gates pass; this does not make those takes collision-free or visually clean.
- **Rejected default:** ankle planting is disabled. Exact-source trials in `ablation/` and the [feet experiment video](video/observers-feet/playback.mp4) had mixed support-speed results and no consistent visual win. `--refine-feet` remains an explicit offline experiment only.

## Model evidence and measurements

[Experiment matrix](experiment-matrix.md), [final matrix](final-matrix.json), and [comparison](comparison.json) record four fresh baselines, four historical observer-plus-ankle trials, and six final runs across City, Industrial and Market backgrounds; automatic and explicit starts; greeting and spar prompts; repeated and held-out seeds. All final cases completed on their first motion-generation attempt. Warm final motion generation took 4.08–13.28 s for 7.4–16.0 s performances, excluding external planning and video rendering.

Final waiting-character motion exceeds 0.1 mm on roughly 70–95% of held-frame transitions, versus 0% in the baseline. This measures movement only. Roots and active InterGen frames remain exact relative to each unrefined composition; waiting lower bodies remain exact too. The full metric files include transition joint/root/heading changes, low/slow ankle support-speed proxies, wrist proximity, route/scene validation and all-frame body sphere clearance. None is a full mesh, physical contact or perceptual-quality proof.

Every final source folder includes `unrefined-cast.npz` and `refined-cast-candidate.npz`. `ablation-observer-only/` retains exact-source before/after archives for the selected refinement. The repeated fresh baseline/final runs share numerically identical seeded InterGen pair arrays, but their Core arrays differ; they must not be described as exact-source ablations.

The first local planner attempt failed for missing gateway configuration and remains at `baseline/city-handshake-s42/sources/`. Automatic approval review then rejected sending the scene to the configured external AI gateway. Existing saved public plans were used for all runs; Core and InterGen motion was freshly generated. New AI planning was not verified. Private provider credentials are not included.

The generation records predate the runner's new code-hash field. Historical source hashes were not retroactively invented. Future runs record code SHA-256 and Git dirty state before launch. The baseline implementation was main `71232729189e2993ffa40e89a735632156cfa520`; the selected implementation is this branch on main `c297551` plus the files in this PR. See the matrix for fresh-model and offline ablation commands. Re-reserve shared compute before fresh-model runs; this milestone provisioned no Pods.

## Validation

1,273 Python tests passed on the integrated branch, including active-frame/lower-body preservation, observer continuity/anatomy, contextual entry cues, initial-facing scene gates, and rejected-candidate provenance. [Test output](validation/python-tests.txt). Main client code is unchanged. The isolated existing-UI replay was checked with the selected saved archive; production port 2380 and the dirty original workspace were preserved.

Refinement failures retain unrefined and candidate joints plus the diagnostic report before publication. Newly worsened unintended body-proxy overlap rejects a candidate; existing overlaps are explicitly reported rather than silently relabeled as safe. InterGen's noncommercial research terms still apply.

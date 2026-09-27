# Interaction quality model matrix

Four baseline, four historical observer-plus-ankle experiments in `after/`, and six selected observer-only performances in `final/` were generated with live ARDY Core and InterGen requests. The same saved plan and motion seed were used for repeated cases; two final seeds were held out. The first attempt to run fresh external AI planning was rejected by automatic approval review because it would send the scene document to the configured external gateway. All motion runs therefore use plans from earlier prompt-scene evidence. The plan provenance is recorded in `baseline-matrix.json`; no motion source was replayed.

| Case | Background | Start | Seed | Baseline frames | Selected final frames |
|---|---|---|---:|---:|---:|
| City handshake | generated city | automatic | 42 | 222 | 222 |
| City handshake, held out | generated city | automatic | 43 | — | 225 |
| Controlled spar | generated industrial | explicit x = -3 m / +3 m, meeting x = 0 | 45 | 378 | 372 |
| Alternating greetings | generated market | automatic | 42 | 478 | 478 |
| Alternating greetings | generated market | automatic | 48 | 480 | 480 |
| Alternating greetings, held out | generated market | automatic | 49 | — | 456 |

`baseline/`, `after/`, and `final/` retain exact cast archives, newly generated Core and InterGen sources, source manifests, `result.json`, and measured `metrics.json`. The paired source numeric arrays match between baseline and final for the same seed; Core source arrays differ. Each final source folder also preserves `unrefined-cast.npz` before observer motion. `comparison.json` maps its hash and proves exact root and native paired frame preservation. `ablation-observer-only/` is the selected exact-source comparison; `ablation/` retains historical observer-plus-ankle trials. Every refined segment passed `check_cast_geometry` against the saved background.

The repeated market cases show held performer motion on 74–95% of held-frame transitions over 0.1 mm, up from 0% in baseline. This measures movement, not whether it looks natural. Wrist gaps are unchanged for the exact InterGen pair sources. The experimental ankle solver had mixed foot support results and was excluded from selected final behavior. The primary agent reviewed normal-speed playback and supporting filmstrips; see `visual-review.json` and `README.md` for the narrow accepted milestone and retained concerns. `final-matrix.json` and `comparison.json` hold the per-case measurements and explicit limitations.

Run the fixed-plan fresh model matrix from this worktree when the shared Core and InterGen lanes are reserved:

```sh
.venv/bin/python experiments/review_interaction_quality.py --phase final
```

The runner creates a new numbered attempt if an output directory already exists, preserving prior successes and failures. `--case market-three-s48` narrows a run. `--phase baseline --measure-only` recalculates metrics without making model requests. New generation is limited to `--phase final`; the older baseline and observer-plus-ankle experiment are historical. To reproduce the baseline implementation, check out clean main commit `71232729189e2993ffa40e89a735632156cfa520` in a separate worktree and run `experiments/trial_prompt_scene.py` with the saved plan, scene, seed, existing private provider and token paths. The original baseline and final generation records predate the runner's code-file hash field; future reruns record file SHA-256 and Git dirty status before launch. The existing private provider config and token are read by the trial script but never copied into the review output. To reproduce an exact-source offline ablation without GPU calls:

```sh
.venv/bin/python experiments/review_interaction_quality.py \
  --refine-archive review/interaction-quality/baseline/market-three-s42-fixed-plan/scene.cast.stagezero.npz \
  --refine-output review/interaction-quality/ablation-observer-only/market-three-s42-new \
  --refine-seed 42
```

`--refine-feet` opts into the historical ankle experiment for an offline test. If refinement rejects a candidate, the runner preserves the before archive, candidate joints, detailed refinement report, and failure record. Geometric gates remain active; sampled body proxy clearance and ankle support speed are not exact mesh collision, palm contact, or physics tests.

# Working together

Clone with `git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git`.
ARDY is an unchanged submodule pinned to the tested upstream commit. Keep its
licenses and mesh attributions intact; do not commit installed environments,
model weights, access tokens, private SSH details, or gated source data.

For controller and project regression tests only (no Pod, meshes, dataset
access, or GPU needed):

```sh
python3 -m venv .venv-tests
.venv-tests/bin/python -m pip install numpy==1.26.4 requests==2.34.2
.venv-tests/bin/python -m unittest -v test_live_motion test_directing test_director_edges
```

Use Python 3.11 for this pinned NumPy version. Full viewer installation uses
`requirements-live.txt`; see README for setup and required private assets.
The recorded CSV is deliberately absent from Git. Obtain authorized access to
BONES SEED through its official Hugging Face page, then provide the source
member named in `assets/source_path.txt` as `assets/recorded_g1.csv`.
Do not copy another person's credentials or redistribute gated source data.

Work on a short feature branch and open a pull request. Keep each PR within one
agreed milestone. Include how it was tested, screenshots for visual changes,
and whether any behavior was simulated. Do not merge unreviewed changes into
the demonstration setup. One person owns a file at a time to limit conflicts.

## Current milestone and proposed follow-up

The single-actor directing demo milestone is complete and ready for user
review. It includes multiple stored takes, extension and alternate endings,
scrub/replay, save/load, gate reaction, and recovery when inference or its
connection fails. The existing Pod is reachable. `docs/DIRECTING.md` has the
local and Pod-backed launch commands. The 24 controller and persistence tests,
browser review, and sustained live-inference results are documented separately
in `review/DIRECTING-RESULTS.md`. Stop for user review before starting another
milestone.

**Proposed friend task after user review: independent QA and onboarding.** If
approved as a follow-up, add tests in a separate
`tests/test_controller_edges.py` and maintain `docs/QA.md`. The current suite
already covers late completions across project changes and seeking, new-project
backup success/failure, and concurrent save snapshots; avoid duplicating those
cases. Add coverage for empty/long prompts, repeated pause/resume, malformed
output, rapid replacement, and mode changes using explicitly simulated
transport. Validate clean-clone setup and record precise bugs/screenshots from
the private demo if access is provided. Do not modify inference or the existing
viewer in the same PR. This work needs no paid GPU. Submit a focused PR with
reproducible failures and proposed fixes. This proposal does not authorize work
to begin before user review and approval.

**Core integration ownership for future approved work:** `live_motion.py`,
`pod_backend.py`, and launch scripts. Maintain existing-Pod/tunnel recovery,
repeated instructions, stale result handling, and accurate latency/memory
reporting. Preserve official G1 inference and recorded fallback. Keep
credentials and private runtime assets out of Git.

**User review:** inspect the actual MacBook interaction and motion quality.
This milestone does not include glasses, multiple actors, new models, or broad
scenery expansion. Inviting a friend to the private tailnet or granting
repository write access remains a separate user action.

UI polish can remain a later friend-owned task once visual direction is agreed.

# Working together

Clone with `git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git`.
ARDY is an unchanged submodule pinned to the tested upstream commit. Keep its
licenses and mesh attributions intact; do not commit installed environments,
model weights, access tokens, private SSH details, or gated source data.

For controller tests only (no Pod, meshes, dataset access or GPU needed):

```sh
python3 -m venv .venv-tests
.venv-tests/bin/python -m pip install numpy==1.26.4 requests==2.34.2
.venv-tests/bin/python -m unittest -v test_live_motion.py
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

## Proposed parallel split for the next review

**Core integration owner:** `live_motion.py`, `pod_backend.py`, launch scripts.
Validate Pod restart, tunnel recovery, repeated instructions, stale results,
latency and memory. Preserve official G1 inference and recorded fallback.

**Friend: independent QA and onboarding:** add tests in a separate
`tests/test_controller_edges.py` and maintain `docs/QA.md`. Test empty/long
prompts, repeated pause/resume, malformed output, rapid replacement and mode
changes using explicitly simulated transport. Validate clean-clone setup and
record precise bugs/screenshots from the private demo if access is provided.
Do not modify inference or the existing viewer in the same PR. This work needs
no paid GPU. Submit a focused PR with reproducible failures and proposed fixes.

**User:** review motion quality and the interaction on the actual MacBook;
restart the existing Pod only for the coordinated live test. Approve the next
major milestone before glasses work begins. Inviting a friend to the private
tailnet or granting repository write access remains a separate user action.

UI polish can be a later friend-owned task once visual direction is agreed.
Glasses, timelines, multi-actor features and new models are not part of this
repository handoff or QA pass.

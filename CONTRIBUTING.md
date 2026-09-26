# Working together

Clone with `git clone --recurse-submodules https://github.com/Jomak-x/stagezero.git`.
ARDY is an unchanged submodule pinned to the tested upstream commit. Keep its
licenses and mesh attributions intact; do not commit installed environments,
model weights, access tokens, private SSH details, or gated source data.

## Offline controller checks

From the clone root, use **Python 3.11** for the pinned NumPy version. With
Python 3.11 already installed:

```sh
python3.11 -m venv .venv-tests
.venv-tests/bin/python -m pip install numpy==1.26.4 requests==2.34.2
.venv-tests/bin/python -m unittest discover -v -p 'test_*.py'
.venv-tests/bin/python -m compileall -q live_motion.py live_viewer.py preview.py pod_backend.py measure_backend.py test_live_motion.py tests
```

Alternatively, with [uv](https://docs.astral.sh/uv/getting-started/installation/)
installed, replace the first two commands with:

```sh
uv venv --python 3.11 .venv-tests
uv pip install --python .venv-tests/bin/python numpy==1.26.4 requests==2.34.2
```

`uv` can install the interpreter if needed. Installation needs internet access
for public packages; the tests themselves use **explicitly simulated transport**
and synthetic arrays, with no Pod, network calls, meshes, recording, credentials
or GPU. The controller tests also work without initializing the ARDY submodule.
Compilation checks syntax only; it does not import dependencies or run inference.

Use discovery above, as CI does, to run both `test_live_motion.py` and
`tests/test_controller_edges.py`. Keep `tests/__init__.py`: Python 3.11 root
discovery needs it to recurse into that directory. For just the added cases:

```sh
.venv-tests/bin/python -m unittest -v tests.test_controller_edges
```

## Viewer and private demo prerequisites

Full viewer installation uses `requirements-live.txt`; see [README.md](README.md)
for the commands. A recursive clone includes public ARDY source and rig assets,
but not the recorded CSV. To run either viewer, obtain authorized access to
[BONES SEED](https://huggingface.co/datasets/bones-studio/seed), then provide the
source member named in `assets/source_path.txt` as `assets/recorded_g1.csv`.
Do not copy another person's credentials or redistribute gated source data.
No dataset or model access is needed to contribute controller tests.

Standalone `./run-preview.command` needs the authorized recording and viewer
environment, but no token or Pod. The live viewer requires the recording and
privately provisioned `.runtime/api-token` even for its Recorded preview mode;
the live launcher additionally needs SSH settings (normally in `.runtime/pod.env`,
or supplied as environment variables), an SSH key and verified
`.runtime/known_hosts`. Coordinate private demo setup with the core owner.

## Contribution scope

Work on a short feature branch and open a pull request. Keep each PR within one
agreed milestone. Include how it was tested, screenshots for visual changes,
and whether any behavior was simulated. Do not merge unreviewed changes into
the demonstration setup. One person owns a file at a time to limit conflicts.

## Proposed parallel split for the next review

**Core integration owner:** `live_motion.py`, `pod_backend.py`, launch scripts.
Validate Pod restart, tunnel recovery, repeated instructions, stale results,
latency and memory. Preserve official G1 inference and recorded fallback.

**Friend: independent QA and onboarding:** maintain the separate
`tests/test_controller_edges.py` and `docs/QA.md`. Test empty/long
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

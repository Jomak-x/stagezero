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
.venv-tests/bin/python -m unittest -v test_live_motion test_directing test_director_edges tests.test_controller_edges
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

The command above runs controller, directing, persistence and edge-case tests
with only NumPy and requests. Keep `tests/__init__.py`: Python 3.11 root
discovery in the full CI suite needs it to recurse into that directory.
For just the added controller cases:

```sh
.venv-tests/bin/python -m unittest -v tests.test_controller_edges
```

## Full regression checks

Root discovery also includes motion and studio tests that import Torch, public
ARDY source and the viewer library. Initialize the submodule and add these
dependencies before running the same discovery command as CI:

```sh
git submodule update --init --recursive
uv pip install --python .venv-tests/bin/python einops==0.8.2 torch==2.8.0 'viser @ git+https://github.com/nv-tlabs/kimodo-viser.git@7c82ad8f8640bad9dff8ded5c5eee908eeb08f11'
.venv-tests/bin/python -m unittest discover -v -p 'test_*.py'
```

On Linux, CI installs Torch from `https://download.pytorch.org/whl/cpu`.
These tests use public assets and simulated inputs; they require no gated
recording, model weights, credentials, Pod or GPU.

The separate frontend CI job uses Node 24. From `studio_client/`, run:

```sh
npm ci
npm run build
node --experimental-strip-types --test src/*.test.mjs tests/*.test.mjs
```

The client-local `.npmrc` enables legacy peer resolution for the pinned React
canary, whose version falls outside dependencies' stable React peer ranges.

## Viewer and private demo prerequisites

Full viewer installation uses `requirements-live.txt`; see [README.md](README.md)
for the commands. A recursive clone includes public ARDY source and rig assets,
but not the recorded CSV. To run any viewer, obtain authorized access to
[BONES SEED](https://huggingface.co/datasets/bones-studio/seed), then provide the
source member named in `assets/source_path.txt` as `assets/recorded_g1.csv`.
Do not copy another person's credentials or redistribute gated source data.
No dataset or model access is needed to contribute controller tests.

Standalone `./run-preview.command` needs the authorized recording and viewer
environment, but no token or Pod. The live and directing viewers require the
recording and privately provisioned `.runtime/api-token` even in Recorded preview
mode; their launchers additionally need SSH settings (normally in `.runtime/pod.env`,
or supplied as environment variables), an SSH key and verified
`.runtime/known_hosts`. Coordinate private demo setup with the core owner.

## Contribution scope

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

**Friend: independent QA and onboarding:** maintain the separate
`tests/test_controller_edges.py` and `docs/QA.md`. The added tests cover empty/long
prompts, repeated pause/resume, malformed output, rapid replacement and reset
using explicitly simulated transport. The original controller suite covers
mode changes. The directing suite also covers
late completions across project changes and seeking, new-project backup
success/failure, and concurrent save snapshots; avoid duplicating those cases.
For future approved QA work, validate clean-clone setup and
record precise bugs/screenshots from the private demo if access is provided.
Do not modify inference or the existing viewer in the same PR. This work needs
no paid GPU. Submit a focused PR with reproducible failures and proposed fixes.
Further milestone work remains subject to user review and approval.

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

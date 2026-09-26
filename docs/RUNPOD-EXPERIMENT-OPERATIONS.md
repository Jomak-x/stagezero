# RunPod experiment operations

Checked 2026-09-26 with `runpodctl 2.14.0-dd55bcf`. This page covers temporary research Pods for motion, character interaction, and scene constraints. The existing StageZero inference Pod and public demo are separate production resources.

## Current access

- The CLI is installed at `/opt/homebrew/bin/runpodctl`. Its `pod list --all --output=json` command currently returns `no_credentials`.
- `RUNPOD_API_KEY` is absent, `.runtime/runpod-api-key` is absent, and the CLI's existing config does not authenticate. Do not put a key in a command argument, log, or tracked file. A privately stored key can be read into `RUNPOD_API_KEY` for the process running the CLI.
- The existing StageZero Pod remains reachable with the project's private SSH configuration. CLI access is needed to inventory the user-mentioned `onboarding` Pod and to create, stop, or delete Pods. There is no verified ID, GPU, status, or price for `onboarding` yet.
- No Pod was created, modified, stopped, or deleted in this check.

## Experiment budget and placement

1. Once CLI authentication works, list all Pods and the account balance. Inspect `onboarding` first. Reuse it if it is available, has sufficient GPU memory and disk, and does not interrupt another workload.
2. Otherwise use at most **two new Pods**, with total new GPU rate at most **$3/hour** and total incremental experiment spend at most **$15** including storage. Record each Pod ID, start time, GPU rate, disk allocation, and stop deadline before beginning inference. Keep at least $1 of the session cap unallocated for storage and shutdown delay.
3. Prefer one 48 GB GPU for the first isolated motion or two-person batch experiment. Add an 80 GB GPU only if measured peak memory, model compatibility, or parallel workload requires it. Published Secure Cloud rates on 2026-09-26 include RTX 6000 Ada 48 GB at $0.84/hour, A40 48 GB at $0.49/hour, and A100 80 GB at $1.59/hour; live availability and quoted account price must be checked before creation. One RTX 6000 Ada plus one A100 would be $2.43/hour for GPU compute. At that rate, five hours is $12.15 before storage; use a shorter deadline if the actual quote is higher.
4. Use on-demand billing, one GPU per Pod, the minimum sufficient disk, and SSH access. Do not expose a model HTTP port publicly; use SSH forwarding or an authenticated endpoint for tests. Keep credentials out of `--env` and command-line arguments.

RunPod bills compute and storage by the second, but retained volume disk continues charging while stopped. Container disk is erased on stop. Export evidence before stopping; delete a disposable research Pod after its artifacts are copied out and verified. Stopping can also release the GPU slot, so a restart may have no GPU available.

## CLI procedure after private authentication

These subcommands and flags were verified against the installed CLI's `--help`. Replace placeholders only after reading the live inventory and quote.

```sh
runpodctl pod list --all --output=json
runpodctl user --output=json
runpodctl gpu list --output=json
runpodctl pod get POD_ID --output=json
runpodctl pod create --help
runpodctl pod stop POD_ID
runpodctl pod get POD_ID --output=json
runpodctl pod delete POD_ID
```

Creation should record the returned Pod ID immediately, including if `--wait` times out: the CLI says a timeout **keeps** the Pod. Use `pod get` to verify its actual state, then start research. Before and after each experiment, record GPU memory, generation time, quality metrics, and saved output paths. On completion or budget deadline, run `pod stop POD_ID` and verify `runtimeStatus` is stopped. After exporting artifacts, run `pod delete POD_ID` only for research Pods created in this session, then confirm absence with `pod list --all`.

The [RunPod CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods) describes `--stop-after` and `--terminate-after`, but **neither flag appears in this installed 2.14.0 CLI's `pod create --help`**. Do not rely on those flags for a spending limit. A local timer can be a reminder or secondary stop attempt; it is not a guaranteed cloud cutoff if the Mac sleeps or the process exits. Active deadline checks and final remote status verification remain necessary. The [RunPod Pod management guide](https://docs.runpod.io/pods/manage-pods) documents `pod stop`, data retention, and deletion. [RunPod Pod pricing](https://docs.runpod.io/pods/pricing) gives storage charges, and the [current GPU pricing page](https://www.runpod.io/pricing) gives published list rates.

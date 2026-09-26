# Kimodo RP v1.1 offline probe: resources and provenance

Measured 2026-09-26 on the **existing** RTX 6000 Ada Pod. This was a bounded, offline sample-generation experiment, not an integrated streaming worker or a live replacement for ARDY Core. No Pod was created or stopped; the two pre-existing GPU workers remained active throughout and still occupied 15,882 and 17,142 MiB afterward.

## Exact experiment

The probe used the official `nv-tlabs/kimodo` source at commit `58e781898b3d7e328a676a75d3e338c45dce3ad9`, the `kimodo-soma-rp-v1.1` checkpoint, 100 denoising steps, 30 fps, and an 8-second/240-frame cap. The two prompts and seeds exactly match the 8-second Core comparison in `.runtime/grounded-experiments/expressive-manifest.json`. The Python environment was the Pod's existing `/workspace/ardy-env` (PyTorch `2.8.0+cu128`, Transformers `5.8.1`); no package was installed into it. The source checkout and Hugging Face cache were staged under `/dev/shm/grounded-kimodo-probe`, away from the workers and the nearly full overlay disk. The cache was populated offline from files downloaded under existing local access; no authentication credential was copied to the Pod.

The inference command, run on that Pod after the official source and the four required Hugging Face snapshots had already been staged, was:

```sh
HF_HOME=/dev/shm/grounded-kimodo-probe/hf \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=local \
PYTHONPATH=/dev/shm/grounded-kimodo-probe/kimodo \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
timeout 240 /workspace/ardy-env/bin/python -u \
  /dev/shm/grounded-kimodo-probe/kimodo_probe.py \
  --profile expressive8 --seconds 8 --steps 100 \
  --output-dir /dev/shm/grounded-kimodo-probe/output/expressive8 \
  --gpu-fraction 0.10
```

The repo-owned [probe script](../../experiments/kimodo_probe.py) saves native `[240,77,3]` joint positions and `[240,77,3,3]` global rotation matrices, plus prompt/seed/timing metadata. The raw outputs are `.runtime/grounded-experiments/kimodo_hiphop_8s.npz` (SHA-256 `7f4436e78988f7b5004cc447509a8e7d744e16e7e2471acbe74497ff149e6da4`) and `kimodo_martial_combo_8s.npz` (`7040da737abb041ebf04bcd1d8173284949b36efe0ac66b253976860f1b0482c`); see `kimodo-expressive-manifest.json`. Separate `*_core27.npz` files are named-skeleton retargets for the common renderer. The source outputs were not pose-edited. The current official code uses SOMA77, although the model card still describes the earlier 30-joint SOMA format; the saved array shapes and official checkout are the operative schema.

## Measured footprint and latency

| Resource | Observation |
| --- | ---: |
| GPU | NVIDIA RTX 6000 Ada, 49,140 MiB total |
| GPU baseline after probe | 33,039 MiB used; 15,473 MiB free; original worker PIDs `11186` and `23565` only |
| PyTorch peak allocated during 8-second inference | 1,236,401,664 bytes (hip-hop), 1,236,270,592 bytes (martial); this is allocator accounting, **not** total process or system memory |
| GPU allocator cap set by probe | 10% of device VRAM |
| RAM-backed Hugging Face model/text-encoder cache | 17,547,819,440 bytes (17.55 GB decimal / 16.34 GiB) |
| Official source checkout in RAM | 126,224,239 bytes |
| Cold model load with CPU text encoder | 58.72 seconds for this run |
| 100-step 8-second generation | 3.53 seconds hip-hop; 3.47 seconds martial after load |
| Pod overlay / RAM disk after probe | 1.9 GB overlay free; 46 GB `/dev/shm` free |

`TEXT_ENCODER_DEVICE=cpu` is the key distinction: it keeps the large Llama/LLM2Vec text encoder and its roughly 17.5 GB of cache files out of GPU memory. NVIDIA's [official Kimodo README](https://github.com/nv-tlabs/kimodo) says fully GPU-resident inference needs about 17 GB VRAM, whereas CPU text encoding can reduce GPU use below 3 GB, at a slower load/encoding stage. Our measured 1.24 GB peak is PyTorch *allocated* GPU memory during generation; it must not be confused with the 17.55 GB of cached model files or with the process's total resident memory. The comparison reloaded the model for each probe process. No persistent Kimodo process remained after generation, and the two original workers were left untouched. No additional Pod was provisioned; the existing Pod's actual hourly charge was not available from its local configuration.

## Visual decision and limits

The saved [martial comparison](kimodo-vs-core-martial-8s.png) shows the clearest complete action: Kimodo gives a guarded advance, punches, a high front kick around 3.45 seconds, and a recovered guard by 7 seconds. Core's matched seed has a deeper initial stance but leaves its kick near the end of the 8-second window. The [hip-hop comparison](kimodo-vs-core-hiphop-8s.png) is modest for both models; it is not a showcase candidate. These are qualitative judgments from the rendered saved poses, not a model-wide ranking. The raw motion still needs contact/foot and scene checks: descriptive diagnostics are in `.runtime/grounded-experiments/kimodo-expressive-diagnostics.json`. Neither Kimodo nor its retarget has been wired into a continuously responding worker or two-person interaction controller. A direct Kimodo→Core continuation attempt currently fails the Core neutral-skeleton position/rotation consistency check, so the interactive `Continue` action must use a native Core clip until a valid conversion is established. NVIDIA's [model card](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1) explicitly lists single-character output, foot skating, imperfect text adherence, and no awareness of scene objects.

## Four further 8-second stunt prompts

The same process/environment/100-step/GPU-cap command above was repeated with `--profile stunts8` and output directory `/dev/shm/grounded-kimodo-probe/output/stunts8`; the [probe source](../../experiments/kimodo_probe.py) contains the four exact prompts and seeds. The model loaded in 58.29 seconds. All four raw native outputs are preserved under `.runtime/grounded-experiments/` with matching hashes in `kimodo-stunts-manifest.json`; named Core27 renderer conversions are separate files. `kimodo-stunts-diagnostics.json` describes motion and inferred contact without assigning a quality pass.

| Seed and requested action | Generation | Rendered observation |
| --- | ---: | --- |
| 6301, steps → cartwheel → balanced landing | 3.45 s | The native body does invert, but the fitted avatar's hands sink roughly 0.23 m below the display floor; the recovery leans backward. Reject this render pending contact-aware retarget/floor work. |
| 6302, spinning roundhouse → land → two punches → guard | 3.34 s | A clear high kick around 3.65 s and guard recovery around 4.4 s. No clearly complete spin or two follow-up punches: usable kick beat, partial prompt adherence. |
| 6303, breakdance footwork → low floor sweep → rise | 3.38 s | Crouch and arm gestures, but no floor sweep. Reject as a breakdance beat. |
| 6304, duck → forward shoulder roll → rise → run | 3.32 s | Shuffling/running motions without the requested roll. Reject as a stunt beat. |

This screen review is a seed-specific rejection, not proof that the model cannot generate the actions. The high-kick take is the only displayable new clip; the earlier 8-second martial combo remains the more complete mini-sequence. After this run the probe process exited, the original two workers remained active, and GPU free memory was 15,525 MiB. No new Pod or persistent model service was created.

## Post-trial resource cleanup

After locally rechecking all eight raw NPZ files against their three generation manifests (SHA-256, array shapes, and finite values), the isolated Pod directory `/dev/shm/grounded-kimodo-probe` was deleted. It contained only this trial's Hugging Face cache, official source checkout, temporary virtual environment/package cache, script, and remote output copies; no active Kimodo process was present. Its measured size immediately before deletion was **23,607,231,174 bytes** (about 23.6 GB). `/dev/shm` free space rose from 49,112,260,608 to 72,785,354,752 bytes. GPU free memory remained **15,525 MiB**, and the same original worker PIDs `11186` and `23565` remained at 15,882 and 17,090 MiB. The source revision is recorded above, the raw outputs remain in this worktree, and the local Mac model cache was left untouched. No shared workspace package, model, or service was removed.

## Source and license boundaries

- The [official Kimodo source](https://github.com/nv-tlabs/kimodo) is Apache-2.0; its checkpoints are separately licensed. The [Kimodo-SOMA-RP-v1.1 checkpoint](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1) lists the NVIDIA Open Model License and describes this version as ready for commercial use.
- The required [Meta-Llama-3-8B-Instruct text encoder](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is gated and carries Meta's separate Llama 3 Community License. Its license/access obligations are distinct from Kimodo's source and motion-checkpoint terms. Other text-encoder adapters/dependencies should likewise be reviewed before distributing a packaged runtime.
- The paired InterGen research samples elsewhere in this lab use [CC BY-NC-SA 4.0](https://github.com/tr3e/InterGen); they were not combined into or used to produce these single-character Kimodo outputs.

# InterGen pair-generation feasibility

Checked 2026-09-26 against the [official InterGen repository](https://github.com/tr3e/InterGen) at `b1750df4a859443b8a69329f759ce3a7503ae179` and the [official CLIP source](https://github.com/openai/CLIP) at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. The checkpoint and runtime work on the existing 48 GB pod GPU. Eight real paired samples were generated; raw geometry measurements below are a screening result, with visual and collision review still needed.

## What the official demo actually needs

The [demo inference script](https://github.com/tr3e/InterGen/blob/master/tools/infer.py) takes a text caption, sets a fixed 210-frame window, generates both people together, denormalizes a 262-feature representation per person, and renders the first 22 joint positions at 30 fps. The repository already includes `data/global_mean.npy` and `data/global_std.npy`, which `MotionNormalizer` reads. The InterHuman dataset is **not read by this inference path**; it is needed for training/evaluation, not a text-only demo.

The [published checkpoint download script](https://github.com/tr3e/InterGen/blob/master/prepare/download_pretrain_model.sh) points to `intergen.ckpt`, whose response advertised **2,701,617,525 bytes**. The [model constructor](https://github.com/tr3e/InterGen/blob/master/models/intergen.py) calls `clip.load("ViT-L/14@336px")`; that [official CLIP file](https://github.com/openai/CLIP/blob/main/clip/clip.py) advertises **934,088,680 bytes** and would be downloaded automatically if absent. Thus an unmodified fresh demo needs about 3.64 GB of model files. Its `tools/infer.py` also opens `.\prompts.txt`, a Windows-style literal that does not resolve to the supplied `prompts.txt` on Linux.

The released InterGen checkpoint was downloaded once into `/workspace/stagezero/intergen-lab/intergen.ckpt` (SHA-256 `bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`). Its Lightning `state_dict` contains all CLIP **text** parameters used by `InterGen`: a `(49408, 768)` token embedding, `(77, 768)` positional embedding, transformer blocks 0–11, and final normalization. The model only retains these text modules from CLIP. [`experiments/intergen_probe.py`](../experiments/intergen_probe.py) can therefore construct the same official CLIP text modules without downloading the unused visual CLIP weights, then load the checkpoint. On CPU, strict key accounting found **346 state keys, zero missing model keys, and zero unexpected model keys**. Eight GPU samples completed through this path and produced finite joint arrays.

The official repository was cloned on the existing pod to `/workspace/stagezero/intergen-lab/InterGen`; official CLIP source was cloned to `/workspace/stagezero/intergen-lab/CLIP`. Missing small Python modules (`yacs`, `ftfy`, `matplotlib` and its absent support packages) were installed in `/workspace/stagezero/intergen-lab/deps`, separate from the live StageZero environment. The probe uses the existing `/workspace/stagezero/.venv` for PyTorch 2.8.0/CUDA 12.8, NumPy, SciPy and Pillow. The authors specify Python 3.8/PyTorch 1.13.1/CUDA 11.7 in their [requirements](https://github.com/tr3e/InterGen/blob/master/requirements.txt); inference succeeded on the pod's newer runtime. After checkpoint and dependency setup, approximately 1.52 GB of pod filesystem space remained; disk use is shared with other experiments.

## Measured pair samples

All eight samples use **210 frames at 30 fps** and preserve both people in the same native coordinate system. The raw `.npz` outputs and full per-sample CPU metrics are in `/Users/jakob/Desktop/Shellhacks/.runtime/intergen-lab/`. No actor was moved independently to make contact look better. The first sample ran while other models were resident, with 16,739 MiB GPU free at launch and a 20% CUDA allocator cap. Across samples, generation took **0.862–0.897 seconds** after model load; peak PyTorch allocation was **1.280 GB** (1.357 GB reserved). A separate `nvidia-smi` watchdog observed up to **1,798 MiB** for the InterGen process; polling is too coarse to treat this as an exact peak. No OOM or watchdog stop occurred.

| Prompt / seeds | Raw joint observation | Limit |
| --- | --- | --- |
| “Two people shake hands and step apart.”, seeds 37–42 | In all **6/6** samples, one fixed wrist pair stayed within 15 cm for **42–96 contiguous frames** (1.4–3.2 s). Seed 42 has 96 frames, with median closest-wrist gap 1.44 m in the first 15 frames and 1.43 m in the last 15; its minimum root and upper-spine separation are 0.671 m and 0.646 m. | Seeds 38 and 40 bring roots and upper spines to about 0.18–0.24 m; these are possible penetrations. Wrist proximity alone does not prove a plausible handshake or clean foot contact. |
| “Two people embrace each other.”, seed 37 | Median root separation closes from 1.43 m in the first 15 frames to 0.34 m in the last 15. | A fixed wrist pair is within 15 cm for only 3 frames; whether the pose reads as an embrace needs visual review. |
| “One person pushes another person backward.”, seed 37 | Median root separation grows from 0.47 m to 0.90 m; one wrist pair stays within 15 cm for up to 24 frames. | Direction, balance and physical plausibility need visual review. |

The geometry summary uses raw `joints`, wrist indices 20/21 and upper-spine index 9. A same-pair dwell requires the *same* two wrists to remain within 15 cm across consecutive frames; it does not switch hands per frame. The summary is recorded in `interaction_metrics.json`. The probe also stores `smoothed_joints` separately for viewer rendering; those are not used for the measurements above.

## Reproduce the check and one sample

Run from a shell on the existing pod when the shared GPU has room for another model. The probe does not download assets or silently substitute generated poses. Its `check` and `validate` commands are CPU-only.

```bash
lab=/workspace/stagezero/intergen-lab
export PYTHONPATH="$lab/deps:$lab/CLIP"
export MPLCONFIGDIR="$lab/mpl-cache"
python=/workspace/stagezero/.venv/bin/python

"$python" "$lab/intergen_probe.py" check \
  --repo "$lab/InterGen" --checkpoint "$lab/intergen.ckpt" --text-only-clip
"$python" "$lab/intergen_probe.py" validate \
  --repo "$lab/InterGen" --checkpoint "$lab/intergen.ckpt" --text-only-clip
"$python" "$lab/intergen_probe.py" sample \
  --repo "$lab/InterGen" --checkpoint "$lab/intergen.ckpt" --text-only-clip \
  --prompt 'Two people embrace each other.' --seed 37 --frames 210 \
  --cuda-memory-fraction 0.20 \
  --output "$lab/results/embrace_seed37.npz"
```

`sample` writes `features` with shape `(frames, 2, 262)`, raw `joints` and render-smoothed `smoothed_joints` with shape `(frames, 2, 22, 3)`, plus prompt, seed, source revision, generation time and peak CUDA allocation in JSON metadata. These are **InterGen's joint coordinates**, not StageZero rig animation. Retargeting, ground/orientation alignment, contact scoring and visual inspection remain necessary before calling a handshake or embrace successful. The probe defaults to one deterministic prompt/seed and refuses to overwrite an output unless `--force` is supplied.

## Scope and usage rights

InterGen's [repository license](https://github.com/tr3e/InterGen#licenses) is CC BY-NC-SA 4.0; this checkpoint and its output are confined here to a noncommercial research preview. The authors separately prohibit redistribution of the InterHuman dataset. The probe stores no dataset files and uses only the published checkpoint and included normalization arrays.

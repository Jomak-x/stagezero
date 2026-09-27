# Isolated InterMask feasibility probe

The official InterHuman checkpoints load and infer successfully on the existing
RTX6000 Ada Pod. These two raw samples do **not** establish an improvement over
the recovered InterGen baseline. No further seeds were generated after their
unfavorable point-distance and bone-length measurements. Main visual review is
separate; this folder does not claim mesh quality or verified physical contact.

## Reproducible inputs

- Official source: https://github.com/gohar-malik/InterMask
- Pinned revision: `5100c555de9839b325d0f3d6904669698c5c87f5`.
- Official checkpoint folder linked by `prepare/download_models.py`:
  https://drive.google.com/drive/folders/1WCFR7Opc5S3cke26cjEhdvSOH_CXL2Ut
- Transformer: `interhuman/trans_default/model/best_fid.tar`,651,546,446bytes;
  SHA256 `392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665`.
- VQ: `interhuman/vq_default/model/best_fid.tar`,518,555,706bytes;
  SHA256 `af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272`.
- Prompts: `Two people shake hands and step apart.` and
  `Two people embrace each other.`; seed42 for each.
- 208frames at30FPS (6.933s); native temporal stride4 prevents an exact210frame
  comparison.20 mask steps, guidance2, temperature1, top-k filter threshold0.9.
- Raw denormalized features `(208,2,262)` and joints `(208,2,22,3)` are preserved.
  There is no Gaussian smoothing, foot IK, root repositioning, independent actor
  scale, Core bridge, or mesh retarget. VQ emits zero normalized foot channels;
  the final four denormalized features are not predicted contact labels.

## Loading and runtime

The Pod's existing Python3.12.3/PyTorch2.8.0+cu128 was used without upgrading its
packages. New lab-only dependencies: torch-geometric2.6.1,
positional-encodings6.0.3, gdown5.2.0. Existing CLIP source and InterGen dependency
folder were read through PYTHONPATH. This differs from the authors' older
Python3.7/PyTorch1.13/torch-geometric2.3.1 environment.

The released Transformer omits frozen CLIP weights. The same official
ViT-L/14@336px text weights were reused from the existing, hash-verified InterGen
checkpoint; no visual encoder download was needed. All354 Transformer state
keys and67 VQ keys loaded, with no missing or unexpected keys. No model-source
edits were needed. `inspection.json` records hashes and provenance.

| Sample | Generation | Peak PyTorch allocation |
| --- | ---: | ---: |
| Handshake42 |0.749s|954,516,992bytes|
| Embrace42 |0.562s|955,171,328bytes|

These are generation timings, excluding CPU model loading and checkpoint
hashing. A0.5s watchdog observed at most1,500MiB for this process and at least
12,558MiB globally free. The probe imposed a6GiB allocator/process ceiling,
8GiB global free-memory floor and180s GPU-phase time bound. Coarse watchdog
sampling is not an exact process peak. After exit, GPU usage returned to
34,448MiB used/14,064MiB free; the shared30GB filesystem retained719MiB free.
No existing worker was stopped, restarted, or reconfigured.

## Source measurements

| Point/skeleton proxy | InterMask handshake42 | Original InterGen handshake42 | InterMask embrace42 | Original InterGen embrace37 |
| --- | ---: | ---: | ---: | ---: |
| Minimum any wrist-pair gap |17.94cm|1.32cm|5.90cm|Not a useful hug metric|
| Longest fixed wrist pair below15cm |0s|3.2s|0.433s|Not a useful hug metric|
| Minimum root distance |69.59cm|67.09cm|16.71cm|31.64cm|
| Minimum spine9 distance |62.54cm|64.63cm|12.04cm|24.18cm|
| Bone length deviation from per-actor/bone median, p95 |16.16%|3.70%|20.81%|4.20%|

Original InterGen baselines have210frames; the embrace seed differs. These are
small diagnostic comparisons, not distribution-level model rankings. Wrist
centers do not establish palm contact. Close torso points do not measure signed
mesh penetration. All InterMask frame-level source arrays remain available for
independent rendering and review. See `source_metrics.json` and `run_summary.json`.

## Runner

`experiments/intermask_probe.py` defaults to CPU inspection. The exact run was:

```bash
PYTHONPATH=/workspace/stagezero/intermask-lab/deps:/workspace/stagezero/intergen-lab/deps:/workspace/stagezero/intergen-lab/CLIP \
/workspace/stagezero/.venv/bin/python /workspace/stagezero/intermask-lab/intermask_probe.py \
  --repo /workspace/stagezero/intermask-lab/InterMask \
  --clip-checkpoint /workspace/stagezero/intergen-lab/intergen.ckpt \
  --output /workspace/stagezero/intermask-lab/results --seeds 42 --run
```

Existing sample paths intentionally cause a refusal to overwrite. Use a fresh
output directory only when a further experiment is actually requested. Model
repository MIT licensing does not erase InterHuman dataset/research conditions;
this remains an explicitly experimental research path.

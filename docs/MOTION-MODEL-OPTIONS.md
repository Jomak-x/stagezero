# Motion model options

Checked against NVIDIA's public ARDY and Kimodo releases on 2026-09-26. This is a model-selection note, not a claim that one checkpoint has been benchmarked in this app.

## What the current G1 model means

The ARDY G1 checkpoint is trained on Bones Rigplay 1 but emits the 34-joint Unitree G1 robot skeleton at 25 fps. The model card says each trained model emits one skeleton, and ARDY's prompt following is imperfect: it documents foot skating, actions it cannot perform, and no awareness of objects in the scene. It lists locomotion, gestures, combat, dancing, and everyday actions as its strongest areas. The G1 rig therefore is a real mismatch for scenes centered on expressive human characters; changing checkpoints can address body shape and joint layout, but is not evidence of better acting or object-aware choreography. ARDY documents pose sequences, not facial animation.

## Options

| Option | Skeleton and release | Fit for person-focused scenes | Tradeoff |
| --- | --- | --- | --- |
| **ARDY Core Horizon40** | 27-joint Core, 20 fps, 40-frame horizon; trained on Bones Rigplay 1. A short Horizon8 variant is also released. | Best first comparison if the app should keep ARDY's online text updates and interactive constraints while animating a human-shaped character. The vendored ARDY registry and viewer already recognize Core. | Core is its own skeleton, not a universal avatar rig; the app's target character still needs a compatible rig or retargeting. The official card gives the same broad prompt-following and realism limitations as G1, so treat this as a skeleton-fit experiment, not a guaranteed expressivity upgrade. |
| **ARDY G1 Horizon52** | 34-joint Unitree G1, 25 fps, 52-frame horizon; Horizon8 also exists. | Keep for robot scenes or assets already rigged to G1. | Robot proportions/joint layout are awkward for human characters. ARDY's short autoregressive windows support streaming but should not be confused with one-shot, scene-aware choreography. |
| **Kimodo SOMA RP v1.1** | 30-joint SOMA model; converts to the 77-joint SOMA output used for visualization/export. Trained on the full Bones Rigplay 1 set. | Strong official human-body option to evaluate when body/rig quality matters more than preserving ARDY's streaming loop. NVIDIA identifies SOMA as its default human skeleton. | Separate Kimodo runtime and output/rig integration; not a drop-in ARDY checkpoint. The official authoring demo generates clips from prompts and constraints rather than ARDY's autoregressive online prompt stream. |
| **Kimodo SMPL-X RP v1** | 22-joint SMPL-X body skeleton; trained on Bones Rigplay 1 retargeted to SMPL-X. | Consider only when a downstream SMPL-X pipeline is specifically valuable. | Hugging Face requires accepting access conditions and sharing contact information; the card limits it to non-commercial research. NVIDIA's Kimodo skeleton guidance says this option can show particularly severe retargeting artifacts and is not the recommended default. |

The released ARDY menu currently contains Core and G1 only. Its repository says a SOMA ARDY model is planned; the existence of Kimodo SOMA weights does not make them ARDY-compatible.

## Prompting and interaction limits

ARDY is the better fit when a person should react to changing prompts while motion is playing: the project describes online text prompting and streaming generation. Both G1 and Core were trained on the same 630-hour Bones Rigplay 1 corpus, whose captions cover locomotion, daily actions, and gestures. The official card says motions can ignore text, certain actions are out of distribution, motion is realistic rather than cartoon-like, and characters do not perceive scene objects. Detailed multi-person blocking, facial acting, or precise hand-object contact therefore needs additional authored constraints, retargeting, or a different animation layer; a more vivid prompt alone cannot remove those structural limits.

Kimodo offers a concrete human-body route: its SOMA-RP v1.1 model is recommended by NVIDIA for full-data use, and its interactive demo supports prompts and kinematic constraints. Its standard generation takes a prompt, duration, and constraints to generate a clip, with a documented maximum duration of 10 seconds; it does not retain ARDY's live autoregressive prompt-update behavior.

## Access and integration

- ARDY's official model cards use the NVIDIA Open Model Agreement and do not show a separate file-access gate. The official ARDY setup still requires Hugging Face approval and a token for the **gated `meta-llama/Meta-Llama-3-8B-Instruct` text encoder** at inference time.
- Kimodo SOMA RP v1.1 is listed under NVIDIA's Open Model license. Kimodo also requires approval/token access to that same gated Llama text encoder.
- Kimodo SMPL-X RP v1 has an additional Hugging Face access condition (contact-information sharing) and a non-commercial research-only license. Do not select it for a commercial product without resolving those terms.
- NVIDIA's published ARDY setup targets Linux with an NVIDIA GPU. Kimodo's official docs also primarily test Linux and recommend GPU-capable PyTorch; its checkpoint cards list NVIDIA GPU families. These models are most directly deployable on the RunPod GPU host, not the local Apple Silicon machine.

## Practical recommendation

Compare ARDY Core against the current G1 setup using the same short prompt set and the same camera/character scale. Core is the lowest-change experiment for a human-shaped character while preserving ARDY's live interaction model. If its 27-joint skeleton still cannot deliver the needed body silhouette or performance style, evaluate Kimodo SOMA RP v1.1 as a separate integration. Keep G1 where a robot is the intended actor. Before committing, compare actual clips: NVIDIA's model cards acknowledge prompt-following and foot-contact errors, and do not publish a head-to-head expressive-acting score for G1 versus Core.

## Bounded Core probe on the existing Pod

On 2026-09-26, `experiments/core_motion_probe.py` loaded the official Core Horizon40 checkpoint on the existing RTX 6000 Ada Pod. It generated one Core-native standing prefix, then 24 clips: four prompts (right-hand wave, both arms overhead, squat, startled turn), text CFG 2 and 4, and three seeds per setting. Each clip contains two 40-frame horizons at 20 fps. The [rolling report](../review/motion-core-probe.json) records the exact prompts, seeds, model metadata, timings, joint names, and every case result. Saved pose arrays remain under the ignored `.runtime/motion-research/core-v1/` directory; `experiments/review_core_motion.py` plays them with NVIDIA's CoreSkin renderer.

All 24 cases completed with finite 27-joint poses and valid rotation matrices. The checkpoint loaded in 70.94 seconds; individual 80-frame generation and decode calls took 0.14–0.17 seconds after loading. The isolated process used about 14.93 GiB peak GPU allocation. These figures exclude network/download time and do not measure an integrated app request.

The measured geometry is promising for some actions: both hands stayed at least 0.15 m above the shoulders for 15 or more frames in all six overhead cases; all six squat cases lowered the hips by 0.56–0.66 m from the shared prefix; the startled-turn cases rotated the root by 75–157 degrees. Right-hand waving was weaker: only two of six cases put the right hand 0.10 m above its shoulder for five or more frames, and that height proxy does not establish a back-and-forth wave. The mean history-to-generation joint seam was 0.0013–0.0020 m. These are geometric proxies, not ratings of acting, realism, object contact, foot locking, or facial expression. Visual review of the actual CoreSkin playback is required before a product decision.

Core is **not a drop-in checkpoint** for the current StageZero G1 character. Its 27 joint names, 330 motion features, 20 fps, and 40-frame horizon differ from G1's 34 joints, 414 features, 25 fps, and 52-frame horizon. A human character and compatible viewer/export/retargeting path must be integrated and tested before Core motion can appear in the directing demo. This small probe also lacks a like-for-like G1 comparison from the same initial human pose, so the numbers alone do not establish that Core is the better overall acting model.

## Official references

- [ARDY project page](https://research.nvidia.com/labs/sil/projects/ardy/)
- [ARDY source and released checkpoint table](https://github.com/nv-tlabs/ardy)
- [ARDY Core Horizon40 model card](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40)
- [ARDY G1 Horizon52 model card](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52)
- [Kimodo source and model table](https://github.com/nv-tlabs/kimodo)
- [Kimodo quick start and model variants](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/quick_start.html)
- [Kimodo skeleton compatibility notes](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/skeleton.html)
- [Kimodo generation controls](https://research.nvidia.com/labs/sil/projects/kimodo/docs/interactive_demo/generation.html)
- [Kimodo SOMA RP v1.1 model card](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
- [Kimodo SMPL-X RP model card and access terms](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1)
- [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)

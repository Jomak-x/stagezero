# Realtime release gates

The production profile uses ARDY Core. Coordinated InterGen scenes require an
explicit research profile: the [InterGen repository](https://github.com/tr3e/InterGen)
licenses its material under CC BY-NC-SA 4.0. A research demo is not a commercial
release of that model.

The [ARDY Core model license](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40/blob/main/LICENSE)
permits commercial use subject to its terms and redistribution notices. The
ARDY code is separately Apache-2.0; component and asset terms remain applicable.

A release must pass real warm-model generation, bounded buffering, synchronized
actors, cancellation/stale-result rejection, exact project reload, and recovery
from service errors. Tests using fake generators establish scheduler behavior,
not model speed, motion quality, contact accuracy, or scene understanding.

Navigation is constrained by supplied scene geometry and verified affordances.
A named object alone does not establish a passable opening, reachable contact,
grasp, attachment, or physical collision response. Unknown actions must fail
explicitly. Root/hand targets guide generation; they are not a physics guarantee.

Paired motion must use one shared world transform and fixed skeleton geometry.
Entry/exit seams, wrist residuals, floor penetration, speed, obstacle clearance,
and actual playback underruns must be measured on complete generated scenes.
A continuous-looking montage of independent takes does not satisfy these gates.

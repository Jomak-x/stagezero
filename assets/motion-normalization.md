`motion-normalization.npz` contains the small motion normalization statistics
from the existing NVIDIA `ARDY-G1-RP-25FPS-Horizon52` checkpoint, revision
`059b8007df0ba194a006a877b59a563955ac7b70`, `stats/motion/mean.npy` and `std.npy`.
It is needed to place saved takes locally while preserving their normalized
continuation features; it contains no generated performance or model weights.

The 418 checkpoint channels are sliced to the 414 global feature channels as
`[0:5] + [9:418]`, matching the vendored `MotionRepBase` implementation. Statistics
are cast to float32 before computing `sqrt(std**2 + 1e-5)`, matching ARDY inference.
The asset records its source, revision, model and epsilon in JSON metadata.

Alignment follows `ArdyMotionRep.rotate` and `translate_2d`: root XZ and facing
match the previous take, while the original floor-relative height, poses, foot
contacts and duration are preserved. A join is a cut, not an AI-generated blend.

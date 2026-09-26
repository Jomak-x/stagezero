# Generated city used by the swing experiment

The live swing scene uses the existing `review/scene-refinement/strong-city.json`
from the main Shellhacks checkout. Its exact bytes are snapshotted at
`swing_assets/strong-city.json` (SHA-256
`a478cbf6c51cba6ddd22e75fecc24203f1725f51c50bb2eadf6ccc5b886dac00`).
The original `asset_geometry.py` compiler is snapshotted at
`swing_assets/asset_geometry.py` (SHA-256
`09881c0db7e741978090da4404848a869e26abde474ef1062581bd201a028ab6`).
Both files are data/compiler snapshots inside the isolated experimental worktree;
the main checkout was only read.

The six landmark assets were produced by the configured `gpt-5-6-sol` gateway in
one recorded 73-second request. `review/scene-refinement/strong-city-metadata.json`
reports 14–18 parts for each building, 11 parts for the street furniture, and zero
quality issues after repair. The captured scene's layout was composed by the
pipeline's deterministic architectural staging and visually refined in the prior
scene review. Four road, sidewalk, and bench recipes are authored environment
components. Thus the building geometry is genuinely model generated, while the
street layout and ground surfaces are existing pipeline composition. The
50-object `review/scene-scale/city-generated.json` was inspected but not used:
it is an expanded scale fixture, and would overstate the provenance of the
original scene.

`swing_scene.load_swing_scene()` exports exactly the compiled city triangles. It
follows `ObjectSceneLayer._build_custom`: compile primitive recipes and repeats,
fit each asset to its actual mesh bounds, scale by object size, rotate by yaw,
then translate to world position. Source vertex colors become 155 indexed meshes
grouped by object and color (10,548 triangles total). The adapter does not build
an alternate city backdrop. Units are metres and +Y is up.

The seven buildings occupy x≈−8.1..9.75, y=0..9.1, z≈−24.35..−2.0.
The roadway is centered on x=0 and runs approximately z=−24..6. Full mesh
bounds provide conservative building collision AABBs. Roof discs come from
upward facing boxes in the generated building recipes; they are generally
1.2–2.4 metres across. Six web anchors use actual upper facade mesh vertices,
selected near the street-facing, approach-side corner so a line avoids the
building's conservative collision box until its endpoint. Each anchor exposes
the measured short terminal allowance to reach that vertex. These are gameplay
surfaces and collision proxies; they do not
claim physical rigid-body collision against every decorative facade detail.

Spider-Man starts at `[0, 0.96, 3.5]` (hip position). MJ is on the generated
terracotta roof of `city-4` at approximately `[-6.74, 8.14, -4.07]` (hip), above
its 7.18-metre roof plane. The intended destination is the generated glass
tower roof of `city-10` at approximately `[3.55, 9.03, -17.98]` (surface), across
the street and 14 metres deeper into the city. The controller adds 0.96 metres
to a landing surface to get hip height. An additional tall skyline building
continues to z≈−23 for depth.

The scene is a stylized block set, not a photorealistic open-world map. The
route has room for roughly two visible swing arcs and a roof pickup/landing;
its seven buildings and 20-metre run do not provide an indefinite traversal.

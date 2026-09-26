# Reusable cinematic sets

In **Scene → Starter set**, choose a set and click **Build starter set**. Use
**Scene camera** for its designed view, or **Frame whole scene** to inspect its
full extent. Variation seed changes the repeatable dressing and colors.

| Set | Contents | Future demonstration |
| --- | --- | --- |
| Rooftop swing district | 72 × 72 m, eight blocks, 32 buildings, varied facades, roof equipment, two roof bridges | Spider-Man-style aerial chase, roof landings, wall climbs, bridge vaults |
| Harbor chase | Stacked shipping containers, two gantries, loading decks, catwalks, crane, water, working console | Container parkour, crane attachment, deck-to-deck chase, console activation |
| Jungle temple | Terraces, stairs, chasm crossing, towers, gateways, jungle trees, working door | Adventure entrance, bridge crossing, ledge climbs, door opening |

These are procedural geometry recipes with individual editable props. Their
generation is immediate and offline; no AI call or new credentials are needed.
The existing Neon custom-prop generation remains available separately.

## Save and reuse

Portable seed-zero scenes are in `examples/scenes/`: `rooftop-swing-district.json`,
`harbor-chase.json`, and `jungle-temple.json`. Import them through **Scene files**,
or export the current scene after editing. Save project also preserves the set.
Scene JSON embeds all geometry definitions, placements, lighting, camera and
interaction targets, so it does not depend on a private prop library.

**Interaction targets → Show target markers** displays cyan swing anchors,
green landings, orange climb points, and purple vault points. Select a target
and click **Focus target** for a close camera view and its world coordinates.
Targets follow prop position, size and yaw edits; deleting a prop removes its
targets. They are named planning points, not a collision solver, swing physics,
or a guarantee that generated motion will reach them. The harbor console and
temple door use the existing proximity interaction system.

## Integration

`cinematic_scenes.make_cinematic(name, seed)` returns a validated v3 scene.
`scene_targets.resolve_targets(scene['targets'], scene['objects'])` resolves
each prop-relative target to a world-space position for a future motion planner.
Presets, JSON, and target resolution are independent of the current UI.

Targets contain `id`, `name`, `object_id`, `kind`, and `local_position`.
The latter is in the rendered object's normalized bounds, between -0.5 and 0.5,
before size and yaw transforms. It is not a raw asset recipe coordinate.

## Verification

All three sets were validated across 30 seeds. The city stays within the
existing 64-prop, 16-asset, 12,000-primitive and 250,000-triangle limits. Tests
cover round trips, geometry budgets, bridge alignment, target attachment and
deletion, and the existing scene/controller regressions. Actual WebGL captures
and a captioned camera-orbit preview are under `review/cinematic-scenes/`.
The preview shows set geometry and recorded reference-character playback;
it does not demonstrate an implemented superhero swing simulation.

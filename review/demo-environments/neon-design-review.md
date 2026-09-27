# Draft architectural art brief and implementation recipe

Create three **fictional, unoccupied Three.js environments** for a later animated crowd demonstration. The visual language is restrained contemporary urban realism: convincing scale, layered storefronts, ordinary infrastructure, and selective weathering. The crossing is Tokyo-inspired in density and composition, not a reconstruction. All businesses and transport identities are invented; no real trademarks or geographic accuracy are implied.

This is a proposed design for review, not an implemented or verified scene.

## 1. Shared construction language

Use **meters**, with `Y` up, `X` east–west, and `Z` north–south. Keep each environment near its own origin.

Build a reusable kit containing:

- Road surfaces, curbs, sidewalk slabs, tactile paving, crossing paint, drains, and utility covers.
- Building shells, facade bays, recessed windows, storefront assemblies, doors, awnings, and rooftop equipment.
- Streetlights, pedestrian signals, benches, planters, and fictional sign panels.
- Interior modules: floors, ceilings, rear walls, counters, shelves, tables, and luminaires.

Reuse construction logic, not complete compositions. Give each environment a different skyline, frontage rhythm, material palette, and lighting setup.

### Ground and street continuity

Use asphalt meshes approximately **0.12 m thick**, sidewalks **0.15 m above road level**, and curb pieces with a lightly beveled upper edge. At crossings, provide **2.4–4 m-wide curb ramps** flush with the roadway.

Roads must extend **at least 120 m beyond the principal activity area**, with continuous sidewalks and simplified distant frontage. Hide their eventual ends behind bends, buildings, atmospheric depth, or camera limits—not exposed rectangular edges.

Represent painted markings with thin, slightly elevated geometry and `polygonOffset`. Use subtle asphalt roughness variation rather than oversized cracks. Keep drains outside the main walking strips.

## 2. Environment A — Harumi Exchange Scramble

### Composition and dimensions

A dense commercial intersection framed by four contrasting corner buildings. Its strongest visual features are broad diagonal crossings, layered upper-floor signs, and a readable vertical hierarchy from shopfront to rooftop.

Two perpendicular roads are each **28 m wide**, centered on the axes. Their overlap forms a **28 × 28 m intersection**, bounded by `X/Z = ±14`.

Provide **8 m-wide sidewalks**, with building lines beginning at approximately `X/Z = ±22` in the four corner quadrants. Widen selected corner setbacks to create **10 × 10 m waiting areas**.

Road construction continues at least **180 m from the origin** along each arm.

### Crossings and clear space

Provide:

- Four **5 m-wide conventional crosswalks**, positioned across the road approaches immediately outside the intersection.
- Two **5 m-wide diagonal crosswalks**, linking opposite intersection corners.
- **0.45 m-wide white stripes** separated by approximately **0.45 m gaps**.

Generate diagonal stripe geometry in crossing-local coordinates, then rotate it **±45°** and clip it to the intended crossing footprint. At the center, merge overlapping paint coverage rather than stacking coplanar surfaces.

Reserve unobstructed pedestrian routes from the diagonal ends into the corner waiting areas. Signal poles, signs, planters, and drainage structures stay outside both crossing footprints and their landing zones. No bollards, traffic islands, parked vehicles, or decorative objects interrupt the scramble.

### Architecture

Use four deliberately different buildings:

1. **Northeast:** **38 m** tall, pale ceramic panels, regular **3 m facade bays**, and a softly chamfered corner.
2. **Northwest:** **25 m** tall, charcoal masonry, narrow vertical window groups, and restrained projecting signs.
3. **Southwest:** **18 m** tall, warm concrete, broad retail glazing, and a deep second-floor canopy.
4. **Southeast:** **31 m** tall, muted aluminum cladding, one vertical sign stack, and a recessed ground-floor entrance.

Ground floors are **4.5 m high**; upper floors approximately **3.2 m**. Roof parapets conceal most mechanical equipment, leaving only a few believable silhouettes.

Fictional tenants include **Morrow Books**, **Koma Tea Room**, **Ninefold Optics**, and **Aster Housewares**. Limit each facade to one dominant sign plus smaller tenant panels. Favor cream, charcoal, brick red, and faded blue—not saturated neon.

### Light and materials

Use bright overcast daylight with a weak directional sun. Ceramic facades should read as slightly reflective; concrete remains matte. Keep asphalt mostly dry, with only localized darker patches near drains.

The image should be rich because of architectural layering, not excessive props.

## 3. Environment B — Lantern Court Block

### Composition and dimensions

This environment emphasizes connected walking routes and visible occupied-looking—but empty—interiors.

Place surrounding street centerlines at `X = ±64 m` and `Z = ±44 m`. Each street is **16 m wide**, producing a central curb-bounded block approximately **112 × 72 m**.

Provide **6 m-wide sidewalks** around that block. The principal building envelope is therefore approximately **100 × 60 m**.

Cut an **8 m-wide north–south pedestrian passage** through the block at `X = 0`, connecting both street frontages. Open its middle into a **24 × 20 m courtyard**. Give every entrance and passage junction clear sightlines.

At the surrounding junctions, use **4 m-wide conventional crossings**. Continue all streets and sidewalks beyond the block; neighboring masses establish an ongoing district rather than an isolated diorama.

### Architecture and interiors

Use a lower skyline of **12–22 m**, with brick, buff render, pale stone bases, and occasional painted metal panels. Vary shop widths between **5 and 9 m** while retaining shared floor heights.

A corner café, **Juniper Table**, occupies approximately **14 × 10 m**, with **3.6 m clear interior height**. Model its actual floor, ceiling, rear wall, counter, shelving, pendant fixtures, and sparse table arrangement. Maintain **1.5 m clear interior circulation** and a **2 m entrance landing**.

Adjacent shops:

- **Paper Harbour:** stationery, shallow display shelving, warm off-white interior.
- **Fieldglass Goods:** household objects represented by simple grouped cylinders and boxes.
- **North Thread:** fabric rolls, folded stock, and wall display rails; no mannequins.

Interiors should extend **6–10 m behind the glass**. Strong depth cues come from side-wall returns, ceiling planes, lighting, and objects at several distances—not flat shop-window pictures.

Provide a **2.5 m-deep arcade** along one courtyard edge. Place café furniture only in a designated courtyard furnishing strip, preserving a continuous **4 m-wide through-route**.

### Character

Use late-afternoon light, with sunlight grazing brick and entering the passage. A few interiors can be brighter than the street, but avoid uniformly glowing windows.

Weathering is selective: darkened sill edges, slightly faded awning fabric, and small tonal differences between neighboring masonry sections.

## 4. Environment C — Meregate Station Plaza

### Composition and dimensions

A broad station forecourt at dusk, distinguished by a long canopy, reflective glazing, and a calm horizontal composition.

Create a **120 × 64 m plaza**, spanning `X = −60…60` and `Z = −10…54`. The station sits along its north edge, with a **100 × 22 m footprint** extending from `Z = 54…76`.

An east–west boulevard occupies `Z = −38…−18`, giving **20 m road width**. Provide an **8 m-wide plaza-side sidewalk**, plus a continuous opposite sidewalk. Extend the road beyond `X = ±220 m`.

A central **6 m-wide crosswalk** aligns with the station entrance. Keep the approach centered on `X = 0` free of obstacles for at least **12 m width** across the entire plaza.

### Station architecture

Use a **9 m-high concourse**, with a shallow upper volume reaching **15 m**. The main canopy projects **8 m**, supported by slender columns outside entrance desire lines.

The entrance comprises three **4 m-wide door groups**, separated by solid piers. Behind them, model a visible **10 m-deep lobby** with ceiling strips, ticket-machine silhouettes, and a broad opening toward implied platforms.

Use fictional identity **Meregate Transit**, platform labels **A–D**, and restrained timetable graphics. Timetable text is decorative, not a claim of live information.

Keep planting and seating in two lateral furnishing bands. Trees use instanced trunks and simple clustered canopy geometry; prioritize convincing silhouette over botanical detail.

### Dusk treatment

Use a cool blue-gray environment, a faint warm horizon, and **3000–3500 K-looking station lighting**. Favor pools of light beneath the canopy and at entrances.

Slightly damp paving may reflect broad luminous shapes, but use high enough roughness to avoid mirror-like ground. Emergency signs and small indicators provide accents; the scene is not neon-lit.

## 5. Facade, storefront, and atlas recipe

Build facades as depth layers:

1. Structural wall plane.
2. Window recesses **0.12–0.25 m deep**.
3. Frames projecting **0.04–0.08 m**.
4. Glass behind the frame face.
5. Interior geometry or recessed opaque backing.
6. Exterior signs, lintels, sills, and canopy edges.

Use **0.3–0.6 m-deep storefront recesses**, doors approximately **1.0 × 2.2 m**, and believable mullion spacing. Avoid placing transparent glass directly over an opaque facade wall.

Generate separate canvas atlases for signage, facade detail, and road markings. Suggested size: **2048²**, with **8–16 px gutters** and duplicated border colors to reduce mip bleeding. Use higher-resolution dedicated panels only for signs close to cinematic cameras.

Set color atlases to `SRGBColorSpace`; keep roughness and normal data linear. Prefer `MeshStandardMaterial`, reserving `MeshPhysicalMaterial` glass for important close views. Use moderately reflective, restrained-opacity glazing elsewhere.

Instance repeated windows, frames, lights, and paving modules by geometry/material family. Atlas variation on instances requires custom per-instance UV rectangles or separately batched UV variants; a shared texture offset alone will not vary each instance.

## 6. Six cinematic camera views

Coordinates are `(X,Y,Z)` in meters; FOV values are vertical.

| View | Position → target | FOV | Purpose |
|---|---|---:|---|
| A1 | `(0,9,-64)` → `(0,3,0)` | 48° | Street-level approach; diagonal paint and layered signs |
| A2 | `(52,58,46)` → `(0,0,0)` | 42° | Elevated scramble geometry and four-corner composition |
| B1 | `(0,7,-60)` → `(0,2,0)` | 45° | Connected frontage, passage, and courtyard depth |
| B2 | `(10,1.65,-23)` → `(20,1.4,-15)` | 55° | Reserved café interior composition; counter and window layers |
| C1 | `(0,8,-48)` → `(0,4,54)` | 46° | Boulevard crossing aligned with the station |
| C2 | `(34,2,28)` → `(0,4,58)` | 50° | Oblique canopy, dusk glazing, and paving reflections |

Reserve these sightlines during layout; framing and clipping remain review tasks.

## 7. Future navigation and implementation sequence

Store non-rendered anchors for entrances, crossing endpoints, corner waiting areas, shop counters, seating approaches, courtyard junctions, and station portals. Each anchor should include an ID, position, facing direction, usable radius, and interaction category.

Maintain separate walkable polygons, crossing polygons, and static obstacle footprints. Expand obstacle footprints by a proposed **0.35 m agent clearance** during future navigation preprocessing. Keep crossing permissions as metadata for later signal logic, not present-day actors or behavior.

Suggested build order:

1. Establish dimensions, road continuity, and protected pedestrian corridors.
2. Add building masses and camera reservations.
3. Construct layered facades and selected real-depth interiors.
4. Generate atlases and batch repeated components.
5. Add lighting, restrained weathering, and distant context.
6. Review collision clearances, transparency ordering, draw calls, and all six compositions.

Deliver the initial draft with **no people, vehicles, animated actors, or visible navigation markers**.
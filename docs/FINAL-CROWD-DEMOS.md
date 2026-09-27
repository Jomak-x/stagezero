# Three saved StageZero demonstration films

These are curated, reproducible scene presets, not requests that must generate during a live presentation. Generation, retries, motion review and collision repair happen beforehand. Opening a demo later loads its exact scene, actors, motion, schedule and camera edit. The presenter can play the director's cut, pause, switch to free camera or follow a character.

## 01 — One hundred stories

**Hero:** a detailed, fictional Tokyo-inspired scramble crossing. Compact streets continue out of frame, with dense layered shopfronts, upper-floor glazing, restrained signage and long views down each approach. It is inspired by Shibuya rather than a geographic reconstruction or branded replica.

**Proposed film:** 150 seconds, two or three crossing cycles. Begin at pedestrian eye level with a waiting group, rise to reveal the crowd, cut to a few individuals, then widen as their paths separate. Arrivals and departures continue beyond the camera; nobody teleports at the edge. Preset spawn/despawn regions must remain outside every active shot and preserve scene continuity when the presenter changes camera.

Plan for 64–100 active pedestrians. The roadway is now 25.52 m wide, with 4.06 m painted paths and approximately 4.64 m corner sidewalk planning corridors. The new environment still requires combined actor/performance validation. The final number should serve the image and remain stable in presentation. More people are not automatically a better demo.

At peak scramble, provisionally allocate 52 people crossing, 28 on nearby sidewalks, 16 at corner approaches, and four at storefronts. Stagger arrivals and use small groups; keep future actors at normal human scale and outside the environment group’s horizontal compression. Use close street-level and low elevated shots for density; overhead shots explain routes. Do not hide simulation failures with camera choices. This allocation is a composition proposal, not validated playback.

Feature six readable stories amid the crowd: two friends spot and greet each other; a commuter takes a diagonal route; a shopper leaves a storefront and joins the next phase; a tourist pauses safely to look at a sign; a pair walks together then separates; a late arrival waits for the next signal. Every actor has an intent, route, destination, timing, motion source and fallback. No combat in the opening flagship scene.

## 02 — Every window, a life

**Hero:** a connected city block with actual usable shop interiors, a café, bookstore, alley and courtyard. This is the technology-limit demonstration: several simultaneous, spatially separated activities that remain coherent when the camera changes.

**Proposed film:** 180 seconds. Begin overhead with the block's movement, follow one customer through a door to order, cut to a bookstore exchange, return outside as another character passes, then reveal all stories continuing at once. Camera cuts show concurrent world time; they do not restart each interaction. Interior actors remain simulated while offscreen.

Target 64–100 active people across exterior and interiors after profiling, with 8–12 featured roles and a smaller number of carefully reviewed simultaneous interactions. The exact capacity is unvalidated. Start with buying, greeting, asking directions, queuing and sitting. A confrontation can be an optional later vignette if the interaction system supports it; it is not necessary to prove independence.

Use separate interior visibility/render groups and efficient animation sharing without reducing near-camera motion quality. Navigation links connect real doors, thresholds, counters and seats. Furniture, doors and walls constrain motion; no camera angle is used to hide invalid body contact.

## 03 — Before the last train

**Hero:** a broad station plaza and glazed concourse at dusk. This complements the first two with a clear narrative, warm interior light, quieter closeups and waves of arriving passengers.

**Proposed film:** 120–150 seconds. A train arrival is suggested by a departure board and passenger flow, followed by a reunion, a kiosk queue, an information exchange and people dispersing into the district. A final elevated view shows the flow thinning while one featured pair remains. Transport signage is fictional and decorative in the background study.

Target 64–100 active people only after validation. Main featured roles: a greeter, returning friend, hurried commuter, kiosk customer and attendant, and someone receiving directions. Transit portals need real offscreen staging space; characters must not appear directly behind glass or on a stair they cannot navigate. Flat accessible circulation is required even if architectural stairs are shown.

## Saved demo contract

Each final bundle should retain:

- Versioned environment geometry/material assets, lighting, seed, world units, scene bounds and source provenance.
- Static obstacle footprints, walkable regions, doorway/level links, spawn/exit gates and interaction anchors, checked against the rendered geometry.
- Stable actor IDs; appearance, personal prompt/intent, timeline, navigation result, pair-role bindings, generated motion sources and accepted/rejected generation records.
- One shared world clock and exact cached motion arrays. Changing cameras never triggers generation or changes the story.
- Saved camera positions/targets/FOVs and an editorial timeline with holds, moves and cuts; follow-camera targets by actor ID; a reset that restores the exact beginning.
- Measured browser performance and asset/load budgets, full street/overhead/interior playback videos, overlap/deadlock/contact/foot-sliding diagnostics, and a human visual review record.

Precomputation buys time for quality control. It does not remove the need for real-time browser playback to remain smooth when the presenter explores freely. A exported film is a useful backup, labeled separately from interactive playback.

## Quality gates before people are generated

1. Background visual approval at eye level, elevated views and every intended interior. Look for scale, topology, doorway clearance, facade depth, floor support, occlusion and camera clipping.
2. Integrate the completed movement and surroundings-awareness work. Ground navigation in the exact rendered scene, then validate one actor per route and one pair per interaction.
3. Compose the actor schedule offline, run bounded model jobs, inspect all featured characters and then the whole scene. Repair local failures and retain evidence; never silently substitute unrelated motion.
4. Scale 16→32→64→100 with final materials, shadows and cameras enabled. Measure frame times, cold-load time, total available memory measures, generation time, overlaps, deadlocks and sliding. Include lower-power target hardware if available.
5. Save the exact accepted bundle, record its director's cut and rehearse loading it from a fresh browser session. Final sign-off comes from viewing the complete films, not numerical thresholds alone.

## Current delivery boundary

This phase builds three background studies and a separate viewer with saved shots, free camera, simple camera tours and actual canvas captures. It contains no generated crowd or finalized surroundings-awareness integration. The first crossing's crowd performance measurements do not apply to these richer environments until rebenchmarked with actors.

The backgrounds are AI-assisted procedural Three.js architecture: authored scene geometry, generated canvas material/sign textures and a Neon GPT-6 Astra architectural review. They are not photographs, scans or photogrammetry, and are not yet photoreal production environments. The next refinement should follow actual browser images rather than an unverified promise of realism.

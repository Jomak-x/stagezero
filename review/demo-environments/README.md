# Saved background studies

Three new 3D environments for the final crowd demonstration: a Tokyo-inspired scramble crossing, a connected city block with shop interiors, and a station plaza at dusk. These are background studies awaiting visual approval and completed movement integration. No people or model motion are generated in this phase.

## Review the result

| Preset | Opening still | Full 36-second camera tour | Saved settings |
|---|---|---|---|
| Crossing | [View](preview/crossing-hero.png) | [Play](preview/crossing-tour.mp4) | [JSON](preview/crossing-preset.json) |
| City lives | [View](preview/city-hero.png) | [Play](preview/city-tour.mp4) | [JSON](preview/city-preset.json) |
| Last train | [View](preview/station-street.png) | [Play](preview/station-tour.mp4) | [JSON](preview/station-preset.json) |

All 18 selected shots are in `preview/`. The video is actual 30 FPS browser canvas capture, transcoded without retiming; it is not a prerendered substitute for interactive playback. Raw and rejected captures remain local under `captures/`, outside Git to avoid duplicate media. See [capture index](capture-index.json) and [video provenance](video-provenance.json).

Final checks: TypeScript and production build pass. All 18 saved camera positions agree with their browser capture metadata. Visual inspection corrected roof/tree/pole occlusion, shop/camera alignment, billboard clipping, stretched sign text and overlapping crossing paint. The crossing now uses one merged paint mask. Observed browser HUD remained around 60 FPS without actors on this machine; this is not a measured crowd benchmark.

## Open the actual scenes

```sh
cd studio_client
./node_modules/.bin/vite --config environments.vite.config.mts
```

Open http://127.0.0.1:24974/environments.html. The three preset buttons switch full 3D worlds. Drag/scroll to explore; shot buttons restore saved positions. Play camera tour moves through saved compositions. Save still and Record tour capture the actual browser canvas into `review/demo-environments/captures` while the local Vite server is running. The local capture endpoint accepts only its exact loopback origin and uses sanitized server-created file names.

A production build is available with `vite build --config environments.vite.config.mts`; capture-to-disk endpoints are development-server features, not part of the static bundle. Billboard assets are bundled locally. No model service, credentials or GPU Pod is required to open these presets.

## What was generated

- Neon **GPT-6 Astra** supplied a focused architectural review, retained in [neon-design-review.md](neon-design-review.md). The worker received only a standalone generic design brief; no internal project documentation was exported.
- The scene geometry and canvas material/sign textures are AI-assisted authored procedural Three.js assets. These are not photos, scans or a reconstruction of real Shibuya. Design choices are physically scaled but not surveying data.
- Original raster billboard artwork and four glazing variants were created with the built-in image-generation tool and saved at `studio_client/src/environments/assets/billboards.png`; [billboard prompt](billboard-prompt.txt) and [window prompt](window-prompt.txt). Upper-floor window detail is baked imagery, not simulated interiors or live reflections.
- Public architectural reference: [Shibuya architectural interview photographs](https://naito-shibuya2025.shibuyabunka.com/interview/1/). No reference photograph is copied into the project. Store and transport identities are fictional.

## End goal

See [the three saved-demo specification](../../docs/FINAL-CROWD-DEMOS.md) for proposed films, character stories, camera behavior, saved-bundle requirements and validation gates. Population ranges there are design targets, not tested capacities.

Movement generation, crowd placement, surrounding-aware interactions and Pod provisioning are deliberately deferred until the movement system is ready. Do not apply the earlier crowd benchmark's 60 FPS result to these richer backgrounds: lighting, geometry, shadows and future actors need a combined measurement. The current scene anchor metadata is a starting point for integration, not a navigation guarantee.

## Remaining boundaries

These are detailed architectural presets, not finished populated films or photoreal digital twins. The foreground café/bookshop and station are real modeled rooms; most other storefronts and upper floors are scenic geometry. Reflections, foliage, distant façades and area lighting are approximations. City/station surroundings are less extensive than the crossing. Free-camera exploration can expose scenic boundaries; the saved shots are the reviewed compositions. Planning bounds describe the activity region, while decorative scenery may extend farther. No navigation/collision guarantee follows from the metadata. Main Studio integration and finished actor choreography remain future work.

# Cinematic effects

In **Scene → Cinematic effects & lighting**, select `explosion` or `energy_burst`, set intensity, position and scale, and click **Add effect**. Press Play. The selected take's playhead drives the effect; pausing freezes it and scrubbing reconstructs the same phase. Clear effects removes them from the scene. Scene JSON and project saves preserve all settings.

- **Explosion:** a turbulent volume of fire and cooling smoke, with ballistic embers and an expanding ground wave.
- **Energy burst:** a luminous torus, twisting strands, orbiting sparks and a radial wave.

Both are six-second looping visual effects. They do not damage props or simulate fluid dynamics. There is no independent cue start time yet: all cinematic effects use the scene playhead. Position is the effect volume's center in meters, Y up. Place an explosion at roughly half its height above the floor. Defaults place it behind the actor. A short take can end before the full six-second decay.

Offline recipes recognize “explosion”, “fireball”, “energy burst” and “implosion”. AI scene catalogs include both effects; actual AI selection depends on its response. Examples: `examples/scenes/cinematic-explosion.json` and `examples/scenes/cinematic-energy_burst.json`.

## Renderer

The custom studio client draws bounded shader volumes, seeded particles and shockwaves. No textures, external assets or GPU inference are required. The existing Viser float32 point-cloud transport carries four triplets under reserved `/effects/<id>/cinematic_<kind>` nodes: time/intensity/seed, size, normalized RGB, reserved. Standard scene transforms and node cleanup remain authoritative. The custom studio client is required to display these descriptors correctly; ordinary Viser clients do not render the cinematic shaders. Pure Python `evaluate_effects` retains a particle approximation for consumers.

Build with `cd studio_client && npm run build`, then reload the studio browser. Python changes require restarting the studio process.

## Reproduce the visual review

Run `.venv/bin/python review/cinematic-effects/preview.py`, open http://127.0.0.1:24901/, and choose the effect. The preview includes the recorded actor at its initial pose, lights and a simple set. **Save review frames** exports both effects at .45, 1.2 and 2.8 seconds using the actual WebGL renderer. Captures are not generated concept images.

Validation: `.venv/bin/python -m unittest test_scene_effects test_cinematic_effects test_scene_composition test_scene_performance test_studio_server test_studio_editing test_studio_ui test_adaptive_scene_generation`; client typecheck/build and `node --experimental-strip-types --test tests/*.test.mjs`.

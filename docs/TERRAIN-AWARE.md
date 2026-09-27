# Terrain-aware movement in Studio

Terrain-aware movement is an explicit one-actor option in the normal app. It uses rendered scene geometry to plan supported routes and a separate Human17 foot-contact solver to display them. Ordinary ARDY Core direction and G1 takes keep their existing generation and rendering paths.

## Use it

1. In **Scene → Scene generator**, build the **Traversable temple** or **Industrial switchback** starter set. The offline recipe generator also recognizes those names. Other authored Scene3 backgrounds can use the same geometry pipeline; a picture or decorative backdrop does not provide walkable support.
2. Open **Motion → Advanced scene motion → Scene direction · Core → Move to a scene object**. Set the terrain start X/Z and yaw, then click **Start terrain actor here**. The temple starts at X `0`, Z `0.85`, yaw `3.14159`; the industrial set starts at X `0`, Z `2`, yaw `3.14159`. The starting point must have one unambiguous rendered support surface.
3. Confirm **Terrain-aware movement** is checked. Enter at most four ordered actions in **Spatial commands**, then click **Run spatial commands**. Use authored object names or IDs. For example: `walk up temple stairs then cross temple bridge`, followed by `open temple gate`, then `enter`.
4. A submission becomes visible after all its actions pass validation. Later submissions append from the last committed native history and displayed foot contacts, regardless of the playback cursor. **Cancel pending motion** preserves the prior complete take. A failed display solve can retry its saved native result without repeating GPU inference.
5. Use **Core projects → Save exact Core project + download** to save native motion, display poses, scene geometry, and reaction history together. Reopen that project for playback or further commands. Turning terrain-aware movement off restores the separate ordinary Core take. **Start terrain actor here** deliberately starts a new terrain take.

The industrial route uses genuinely different topology: approach stairs, a turn and detour around a cargo crate, an X-axis gantry, a rotated workshop door, and a separate descending stair flight. Its command sequence is:

```
walk up foundry approach stairs, cross service gantry bridge
open workshop door, enter
```

`walk down loading exit steps` is experimental: a saved native trial passes CPU contact repair, but the final fresh UI trial rejected a direction-reversing pivot. It is not part of the verified demo. For the demo, use the two commands above or replay the included industrial archive.

Scene-tab edits do not silently move an existing terrain take onto different geometry. Save the take, build the new scene, then use **Start terrain actor here** for it. Saved projects retain their own authored scene.

For adaptive AI Scene3 generation, explicitly request walking up/down stairs or crossing a bridge. Such requests now ask for connected solid support, a single stair asset with broad shallow box treads, and upper-body obstacle clearance. They bypass the visual-only architectural layout shortcut. This guidance produces candidate geometry, not a guarantee: the motion planner and contact checks still accept or reject the actual result. Live AI scene generation was not part of the recorded UI runs; those used the two offline starter recipes.

## Supported scope

| Interaction | Behavior |
| --- | --- |
| Walking, approaching, ascending shallow stairs, crossing a bridge | Uses actual supported surfaces and obstacle clearance; rejects missing, ambiguous, blocked, or unsuitable routes. |
| Descent and difficult pivots | Experimental. The final industrial descent UI trial rejected safely; not verified for the demo. |
| Static Scene3 geometry | Uses rendered triangles, boxes, and yaw transforms; no invisible ramp or floor through a gap. Stair aliases select detected stair flights rather than decorative names alone. |
| Configured automatic doors | Lift from measured actor proximity and height, retain the trigger during replay, and permit crossing only after opening. |
| Repeated submissions and save/load | Preserve committed native features and display poses exactly. Relative directions use the displayed actor's committed facing, regardless of playback cursor. |
| Ordinary ARDY | Separate take, native history, controls, and renderer; terrain correction is opt-in. |

The supported display rig is the built-in Human17 character. The terrain solver preserves native root XZ and the upper body's relative joint motion, but assists world-space facing, foot placement, leg cadence, and pelvis height. Facing follows the supported route so the torso does not remain turned away from walking feet. These are assisted steps, not native ARDY stair generation. Native features are never reconstructed from the assisted display.

Jumping, climbing, ladders, hand-operated doors, grabbing/pushing objects, moving platforms, multi-actor terrain choreography, and arbitrary imported rigs are unsupported. Unconstrained natural-language intent is not an LLM scene router: use the bounded spatial grammar or switch back to ordinary direction. Steep or irregular terrain, overlapping floors, narrow clearances, and turns that exceed the solver's limits can reject. The system is reusable across suitable authored backgrounds; it does not make every generated background traversable automatically.

Contact checks include the rendered shoe samples, swept sole envelope, bone reach, knee/rotation bounds, observed route progress, and finite body-clearance proxies. They do not certify full physical dynamics or every possible skinned-mesh collision. [Normal-app evidence and reproduction](../review/terrain-assisted/normal-app/README.md) distinguish successful runs from rejected trials.

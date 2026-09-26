# Scene generation integration verification

The isolated integration starts from main's current studio, upload, character,
and camera implementation. It adds custom scene geometry and a separate object
append flow without importing unrelated changes from the older development
checkout.

Live Neon verification (gpt-5-6-sol):

- A single art-deco city prompt produced eight custom assets and 50 placed props
  in 68.46 seconds.
- A separate vending-machine prompt produced one original 18-part asset in
  49.73 seconds.
- The object was appended to the city, producing 51 objects while retaining its
  existing objects, assets, camera, lighting and effects.

The validated source results are in this directory. They contain geometry and
placement only, never credentials. Credentials were read server-side from the
existing private configuration.

Actual WebGL checks covered the cinematic city, the new live city and the single
object. The corrected city no longer has triangular floor flicker: authored
scene ground hides the generic studio floor, grid and demo platform. Individual
object scenes retain the studio floor. Browser checks exercised the latest-main
Scene tab, its separate replace/add actions, automatic selection of an appended
lamp, and disabled grid/platform controls with the authored-ground explanation.

`render_review.py` reproduces captures with an authorized recording passed via
`--recording`; its default is the repository's private recorded CSV. Optional
`--video` records a six-second camera orbit. These are scene previews, not a
motion/contact or swing-physics demonstration. Generated props are stylized
primitive assemblies, and arbitrary custom props are static by default.

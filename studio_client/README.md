# StageZero browser client

This is a source snapshot of the browser client shipped by the pinned
`nv-tlabs/kimodo-viser` dependency at
`7c82ad8f8640bad9dff8ded5c5eee908eeb08f11` (Viser 1.0.16).
Upstream Apache-2.0 license texts and NVIDIA notices are retained in LICENSE and LICENSE_NV.
StageZero modifications are maintained here, never patched into site-packages.

## Build

Use Node 22.6+ (Node 24 recommended). The checked-in `.npmrc` keeps npm compatible with the pinned upstream React canary:


```sh
cd studio_client
npm ci
npm run build
node --experimental-strip-types --test tests/*.test.mjs
```

The current development checkout can reuse the installed Viser node_modules;
a fresh checkout uses `npm ci` against the checked-in lockfile. Build output
and node_modules are ignored. `director_viewer.py` serves this build on its
existing private Viser HTTP/WebSocket port through `studio_server.py`.
The Python message schema must remain compatible with the pinned dependency.

## Changes from upstream

- Browser-local touchpad navigation with explicit pan/orbit/look modes and
  upright pitch limits, cleaned-up viewport-scoped keyboard listeners.
- Stable per-control WebSocket coalescing; independent GUI edits are preserved.
- Seconds-based timeline with local scrub preview and exact frame bounds.
- Bounded render resolution on Retina displays; floor/depth configuration lives
  in the Python viewer.

Python session state remains authoritative for generated/stored animation.
The UI does not simulate playback ahead of the displayed actor pose.

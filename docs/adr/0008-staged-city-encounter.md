# Staged city encounter on native Core

Status: Proposed
Date: 2026-09-26
Specification: user city walk, encounter, fight and departure request.

## Context

PR #18 added a two-actor Core panel to Motion Studio. Its actors are generated
independently, even when they share a clock. The production archive excludes
InterGen paired motion because that research source has a noncommercial license.
Text alone does not tell ARDY about street geometry, partner state or contact.

## Decision

The city example compiles a shared event sequence into bounded 40-frame Core
stages. Each actor receives its own text prompt and native root-position and
heading targets. Route planning rejects unsupported ground, solid proxies and
unsafe planned separation before it replaces a take. A candidate record keeps
model, prompt, seed, conditions, output and measured outcome separately.

The production UI calls the sequence a **staged/no-contact fight**. It does not
label proximity as a strike or physical reaction. The native clip remains
unchanged; generated poses are measured after production for route, facing,
floor, collision and continuity. A planned path or passing proxy test does not
establish convincing acting. Visual review is required before a candidate can
be marked successful.

The G1 timeline and its projects remain separate. InterGen is not imported
into this production flow. The preset uses a fresh two-actor Core timeline and
backs up an existing one before replacement.

## Verification boundary

CPU tests cover plan validity, unsafe layout rejection, archive isolation and
controller handoff. A private offline Studio preview tests replay and user
controls. Real ARDY output, visual acting and physical contact require separate
GPU and UI acceptance. Until that evidence exists, action candidates remain
`planned_unobserved` and the city example is not a validated generated scene.

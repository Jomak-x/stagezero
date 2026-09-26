# Scene command planning experiment

`scene_ai_planner.py` translates a text direction into a small list of scene
actions. Its AI adapter is optional and has been exercised through the
project's existing configured Neon AI Gateway. A separate rule matcher handles
one precise command at a time and returns `source: "rule_based"`.

The model receives known actor and object IDs, names, kinds, positions, sizes,
and an enumerated list of allowed action triples. It returns only:

```json
{"actions":[{"actor_id":"alex","verb":"go_through","target_id":"arch-0","start_seconds":0}]}
```

The adapter validates the entire result again. Unknown IDs, extra fields,
unknown verbs, invalid times, and ungrounded passages fail closed. An invalid or
failed AI response does not silently turn into a rule-based success. No model
coordinate, URL, code, or tool call is passed to the motion or scene planner.
The selected target must also be named in the user's direction. This extra
check was added after a real model trial replaced a nonexistent “rocket hatch”
with the known gate.

## Current action boundary

| Direction | Grounded intent | Executable scene path |
| --- | --- | --- |
| `go_through` an arch | Yes, if actor fits the procedural opening | Yes, through `interaction_planner.plan_action` |
| `go_through` a custom gate | Yes, only with trusted `verified_open` passage metadata | Yes, after collision and clearance checks |
| `approach` a scene object | Yes | Yes, around scene obstacles |
| `approach` another actor | Yes | No shared actor-aware path yet |
| `face` / `reach` a scene object | Yes | Symbolic until facing and hand targets compile into motion |
| `handoff` to another actor | Yes | Symbolic until object ownership, shared contact, and two-actor timing exist |

`compile_actions` rejects a plan containing any symbolic action before returning
routes. A route is geometric intent, not proof that a generated character followed
it. `motion_following_verified` remains false in route assumptions.

The procedural arch's opening comes from its known construction. A decorative
`door` is not automatically a passable gate. A custom mesh's visible shape or
name does not prove a hole; a trusted scene record must include an opening that
fits inside its bounds and reaches the ground:

```python
affordances = {
    "custom-gate-1": {
        "kind": "passage", "verified_open": True,
        "width_m": 2.0, "height_m": 2.5, "depth_m": 0.4,
        "floor_y_m": 0.0,
    }
}
```

Object recognition here means matching **stored scene metadata**: object IDs,
names, kinds, geometry, and trusted affordances. There is no camera perception,
mesh topology inspection, or image recognition in this planner. An image model
could propose labels and locations later, but the scene importer would still
need to verify scale, collision geometry, and passage openings before a route
could use them. The official OpenAI [vision guide](https://developers.openai.com/api/docs/guides/images-vision)
explicitly cautions that vision models can be wrong about spatial localization,
descriptions, and object counts.

## Optional AI configuration

Set `STAGEZERO_SCENE_AI_MODEL` and a private `OPENAI_API_KEY` or
`STAGEZERO_SCENE_AI_API_KEY`. The default endpoint is
`https://api.openai.com/v1`; an OpenAI-compatible HTTPS endpoint can be selected
with `STAGEZERO_SCENE_AI_API_BASE`. Values can also live in the untracked
`.runtime/scene-ai.env`. The existing `.runtime/objects.env` is also recognized:
`NEON_AI_GATEWAY_BASE_URL`, `NEON_AI_GATEWAY_TOKEN`, and
`STAGEZERO_OBJECT_MODEL` are used when scene-specific settings are absent. A
private `.runtime/ai-api-key` file is read if no key environment variable
exists. Do not place keys in scene files or browser UI. Neon's own
[AI Gateway example](https://neon.com/blog/llms-belong-in-your-backend) uses the
same branch-host variable, token, and `/v1` Chat Completions path.

```python
from scene_ai_planner import SceneAIPlanner, compile_actions, plan_local

actors = [{"id": "alex", "name": "Alex", "position": [0.0, 0.0, -3.0]}]
plan = SceneAIPlanner.from_env().plan("Alex go through the gate", scene, actors, affordances)
# Or, without an API call:
plan = plan_local("Alex go through the gate", scene, actors, affordances)
routes = compile_actions(plan, scene, actors, affordances)
```

Actor positions are ground anchors in metres; a 3D position's Y must stay near
zero. A pelvis joint position must first be converted to its ground anchor.
When one actor has several navigable actions, compilation starts each later
route at the preceding endpoint and no earlier than its ending time. An
explicitly overlapping start is rejected. Routes for different actors have
independent clocks; joint collision avoidance is a further experiment.

The official OpenAI [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
documents `json_schema` for Chat Completions and notes that JSON mode alone
does not enforce a schema. The OpenAI endpoint uses a strict JSON schema;
generic compatible gateways use JSON mode. Both still undergo application-side
validation. Network requests have a bounded timeout and response size, reject
redirects, and require a completed non-refusal response. No configured key
means no AI request.

## Product experiments worth running next

1. **Directed ensemble staging:** Ask a model for actor roles and beat timing,
   then compile each actor's routes against the same scene. Add inter-actor
   collision and temporal separation, since independent paths can collide.
2. **Object use through target constraints:** For a ball, console, chair, or
   door, derive a reachable hand/body goal from the object's measured surface.
   Require measured contact and a persistent object state before claiming a
   pickup, activation, or sit.
3. **Shared prop handoff:** Represent who holds the prop at every frame and a
   common contact window. Generate both actors against that shared target and
   verify minimum hand-to-prop distance and collision clearance.
4. **Scene-to-action loop:** Ask an AI director for a symbolic action plan,
   generate motion, measure goal/contact errors, then let it revise the next
   symbolic beat. Keep geometry and physics checks deterministic, so a fluent
   description cannot conceal a failed movement.

These are proposed experiments, not delivered capabilities. In particular,
neither AI planning nor text similarity establishes that a character reached an
object or passed through a gate. Tests use a simulated transport and the real
deterministic route compiler. A separate live gateway experiment is reported
below.

## Real gateway trial, September 26, 2026

[`experiments/verify_scene_ai.py`](../experiments/verify_scene_ai.py) sent seven
synthetic scene directions to the existing configured Neon gateway using
`gpt-5-mini`; one ambiguous direction was rejected locally before a request.
The final run matched the expected outcome in **8/8 cases**, with **1.248 s
median API trial time** across seven calls. The [sanitized per-case
record](../review/scene-ai-results.json) includes returned action IDs, route
waypoints when compiled, timings, and failure reasons. No key or raw provider
response is saved.

| Case | Observed outcome |
| --- | --- |
| “Go through the gate” | One grounded `go_through` action compiled through the procedural arch |
| Rotated custom gate by exact ID | One grounded `go_through` action compiled using verified passage metadata |
| Two objects both named gate | Rejected as ambiguous without an API call |
| Approach, then pass | Two routes chained from the first endpoint and end time |
| Nonexistent rocket hatch | Rejected because the selected gate was not named in the direction |
| Two actors staging | Two independent `approach` routes; no inter-actor collision claim |
| Face partner and handoff | Symbolic actions returned; executable route compilation refused |

In the first run, the model inserted unrequested facing and approach steps for
the gate. A minimal-action instruction corrected that in the repeated trial.
The nonexistent-object substitution prompted the deterministic mention check.
These are one-run observations on synthetic metadata, not a reliability rate or
evidence that generated movement followed the paths.

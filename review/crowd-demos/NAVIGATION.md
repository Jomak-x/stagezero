# Saved crowd navigation

The viewer loads cached trajectories; it does not invoke a planner, motion model, or network service during playback. All actors remain present for the full 120 seconds. Cameras share one world clock.

## Rebuild

Run these commands from the repository root with Python and NumPy available. The checked-in motion atlas must exist before building; omitting it would produce navigation without the paired actions and gesture cues.

```sh
CROWD_DEMO_PYTHON=/Users/jakob/Desktop/Shellhacks/.venv/bin/python
test -s review/crowd-demos/assets/manifest.json
$CROWD_DEMO_PYTHON experiments/build_crowd_demos.py --scenes crossing city station --counts 112 128 --output review/crowd-demos
$CROWD_DEMO_PYTHON -m unittest test_crowd_demo_navigation test_crowd_demo_cues
```

This uses seed `20260927`, duration `120.0`, sample interval `0.1`, a `0.65 m` planning grid, `0.5 s` reservation ticks, and `0.28 m` root-disc radii. The 112- and 128-person takes are planned independently. Generation is deterministic for the same code, configuration and motion assets. Source atlas provenance is recorded in `assets/manifest.json` and `assets/motion-summary.json`; rebuilding navigation does not generate new model motion. JSON formatting and postprocessing can change file hashes without changing trajectories, so browser recordings retain the actual loaded input hashes.

The builder applies native pair cues, the café attendant and stationary crossing-queue cues before final validation. The crossing additionally applies an explicit initial curb placement: actor 2 starts at its first safe curb point and holds until its continuous reserved path joins at 0.5 seconds. This avoids beginning the film with that actor already in the road; it creates no playback jump. The correction is recorded as `initial_curb_placements` and is collision-checked with the whole population.

### Same-take performance subsets

The supplied 16/32/64/100-person archives retain the first N stable actor IDs from the 112-person take. They preserve trajectories, motion cues and timing. They are controlled renderer-load comparisons, not independently replanned lower-density crowd simulations. Removing actors cannot introduce a new pair collision, but every subset is still validated.

To reproduce the subsets after building the 112-person takes:

```sh
$CROWD_DEMO_PYTHON - <<'PY'
import copy, json
from pathlib import Path
from crowd_demo_navigation import validate
root = Path('review/crowd-demos')
for scene in ('crossing', 'city', 'station'):
    source = json.loads((root / f'{scene}-112.json').read_text())
    for count in (16, 32, 64, 100):
        data = copy.deepcopy(source)
        data['agents'] = data['agents'][:count]
        for frame in data['frames']:
            frame['people'] = frame['people'][:count]
        data['config']['count'] = count
        data['subset_provenance'] = {
            'source': f'{scene}-112.json',
            'method': 'First N stable actors from the same saved take; paths unchanged.'
        }
        for event in data['events']:
            if 'slot_xz' in event:
                pairs = [(a, slot) for a, slot in zip(event['actor_ids'], event['slot_xz']) if a < count]
                event['actor_ids'] = [a for a, slot in pairs]
                event['slot_xz'] = [slot for a, slot in pairs]
                event['crew_count'] = len(pairs)
            else:
                event['actor_ids'] = [a for a in event['actor_ids'] if a < count]
        cues = data['motion_cues']
        if 'flashmob_applied' in cues:
            cues['flashmob_applied']['actor_ids'] = [a for a in cues['flashmob_applied']['actor_ids'] if a < count]
        for field, count_field in (('dwell_cues', 'dwell_gesture_events'),
                                   ('queue_cues', 'queue_gesture_events')):
            if field in cues:
                cues[field] = [c for c in cues[field] if c['actor_id'] < count]
                cues[count_field] = len(cues[field])
        if cues.get('stationary_staff', {}).get('actor_id', -1) >= count:
            cues.pop('stationary_staff')
        data['metrics'] = validate(data)
        (root / f'{scene}-{count}.json').write_text(
            json.dumps(data, separators=(',', ':'), allow_nan=False))
metrics = {}
for scene in ('crossing', 'city', 'station'):
    for count in (16, 32, 64, 100, 112, 128):
        path = root / f'{scene}-{count}.json'
        metrics[path.name] = validate(json.loads(path.read_text()))
(root / 'navigation-metrics.json').write_text(json.dumps(metrics, indent=2))
PY
```

## Geometry and behavior

`crowd_demo_layout.py` mirrors ground footprints from `EnvironmentScene.ts` and `StreetDressing.ts`. Crossing architecture is horizontally scaled by `0.58`; actor dimensions are not. Conservative rectangles cover buildings, transparent shop glazing, counters, tables, chairs, planters, lamps, kiosks, ticket equipment and relevant street furniture. City shop entrances use the actual 2.3 m portals. Station circulation uses the flat approach and excludes decorative stairs.

The city uses the north and south sidewalks and the actual painted 4 m zebras centered at `x=-6.5` and `x=7`, spanning `z=-12..12`. The broad side-street carriageway is excluded. Starts and greeting pairs are on pedestrian pavement. Fixed roles replace random road wandering: high-street errands, eight café/bookshop customers (IDs30–37),12 courtyard viewers (38–49),24 flashmob performers (6–29), and café attendant63. Ordinary dwell motion uses subtle idle/look cues, never random waves.

The crew approaches grid-aligned marks, settles by38s, eases facing before40s, performs the synchronized25-second routine during40–65s, then disperses. The6×4 formation is centered at `(-48.1,-53.95)` with2.6m spacing. Each source motion stays inside a reviewed1.2m slot envelope; source root excursion is0.4014m. Other pedestrians are excluded from the entire formation rectangle. Viewers use a west courtyard lane so the alley exit stays available. The speaker at `(-57.5,-59.5)` is represented by its exact1.0×0.7m footprint. The dance retains native roots and applies an explicitly recorded vertical floor correction (maximum9.53cm), with one-second entry/exit height ramps. This does not turn the general navigation path into a foot-contact solver.

The16-person load subset contains10 of the24 performers;32/64/100 contain all24. Subsets are derived **after** full motion integration, with event IDs, slots and cue metadata pruned together. Do not feed the reduced16-person event back through the full routine-authoring helper, which correctly requires24 performers.

Actor63 remains at `(-29,-23.53)`, behind the actual café counter, physically present for the full film. Its zero root travel is an intentional work role, reported separately and exempt only from mobile-travel requirements. Ten-second task segments describe attending the counter; generated listening acknowledgements occur only when another actor is actually nearby, plus subtle look motion. It exists in64/100/112/128 profiles. Both112 and128 city takes contain eight moving shop customers plus the attendant. The rejected earlier road-wandering112/128 archives are retained under `rejected/road-wandering-city/`.

Actors 0–5 form three greeting pairs. Their reserved windows are 12–20 seconds, with the original native paired motion at 14–17.9667 seconds. Source root tracks and a shared pair rotation preserve role placement. Authored entry/exit blends stay in reserved 2.15 m pockets. Stationary browsing, watching and signal queues use subtle generated idle/look motion with settle/departure margins. Conspicuous gestures are restricted to the reviewed paired encounters, dance and proximity-triggered café acknowledgements.

## Crossing composition and phase audit

The final crossing routes prioritize opposite corners through the two painted 4.06 m diagonal corridors. The earlier peripheral take is rejected and retained under `rejected/peripheral-crossing/`: its broad road-occupancy count obscured an almost empty painted center. At 31.7 seconds that rejected112-person take had only one actor inside the central intersection. The rejected source snapshot,128-person archive and original112-person browser input hash are retained.

Planner and viewer phase boundaries agree exactly:

| Clock modulo54s | Viewer | Planner |
| --- | --- | --- |
| 0 ≤ t < 10 | WAIT | red |
| 10 ≤ t < 46 | CROSS | walk |
| 46 ≤ t < 54 | CLEAR | clearance |

New road entry requires CROSS plus a predicted completion budget. Reserved road occupancy is forbidden during WAIT. Final112-person audit: zero road samples during WAIT and zero entries outside CROSS, including at both54s and108s transitions. Central occupancy means **both** `abs(x) < 12.76` and `abs(z) < 12.76`; it excludes the outer arterial arms.

The final128-person central peak is128 at29.7s, with a mean of75.861 during CROSS; all128 reach opposite corners. Its maximum explicit signal queue is42s and maximum continuous stationary span is53s.

The final112-person central peak is109 at28.7s, with a mean of65.722 during CROSS. Forty actors enter the central region by13.2s. At31.7s there are99 in the center, and at67.6s there are47. All112 visit the central10 m square and reach an opposite corner. These density and opposite-corner completion checks are now validation gates.

## Measured results and limits

`navigation-metrics.json` contains every saved population; `navigation-archive-hashes.json` records the exact final archive hashes and byte sizes. The combined navigation/cue unit suite passes18 tests, including invalid-collision rejection, pedestrian-region adherence, formation arrivals and dispersal. All final takes pass sampled static-disc, sampled and swept pair-disc, walkable-region, root-jump and unscheduled-stall checks. The final112-person mobile travel minima are96.891 m (crossing),32.435 m (city) and94.460 m (station). City112/128 respectively have53/65 actors entering painted crossings and35/45 completing a move to the opposite sidewalk; unpainted-road actor samples and outsider formation samples are zero. All24 performers are settled at39s and have moved at least9.9m from their mark by120s in112. City112/128 maximum explicitly queued waits are48/60s; their27/29 stationary spans over30s include scheduled audience viewing and the permanent staff role. These are disclosed scheduled holds, not a claim that every root moves continuously. Minimum swept pair separation is0.65 m. Maximum cached root step is0.1838 m at0.1s, or1.8385 m/s.

Zero unscheduled deadlocks does **not** mean no long pauses. In crossing112, seven actors have a stationary span longer than30 seconds; the maximum combined dwell/queue span is50.5s and the maximum explicitly labeled signal-queue span is42s. All resume and complete an opposite-corner crossing. The viewer uses subtle idle/look motion during these waits. Metrics report these waits separately from unscheduled walking/yielding stalls.

Root paths are continuous piecewise-linear grid segments, with abrupt velocity changes at corners and dwell boundaries. The measured peak root acceleration proxy is36.77 m/s². Navigation heading is eased, but this is not a physical acceleration controller or a foot-contact solver. Conservative root-disc clearance does not certify full-body, hand, clothing or shoe collision avoidance. Foot sliding, mesh contact, appearance and cinematic quality require separate gait diagnostics and complete browser visual review. Performance profiles must run after offline generation stops; recordings made while CPU generation runs are visual evidence rather than clean performance benchmarks.

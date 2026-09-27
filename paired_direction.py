"""One bounded city-meetup request, with exact model sources and atomic publication."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
from native_pair_clip import NativePairClip, load_source
from native_pair_session import REVIEWED_HANDSHAKE, REVIEWED_SPARRING
from native_pair_transition import shared_place_pair
from native_pair_geometry import check_native_pair_geometry

DEFAULT_ENTRY_POLICY = 'continuous'
DEFAULT_SPEED_MPS = .85


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('Choose two characters, their starts and a meeting point.')
    r = deepcopy(request)
    ids = r.get('actor_ids')
    if (not isinstance(ids, (list, tuple)) or len(ids) != 2 or
            any(not isinstance(i, str) or not 1 <= len(i) <= 64 for i in ids) or ids[0] == ids[1]):
        raise ValueError('Choose two different characters.')
    if r.get('source') not in ('handshake', 'sparring', 'generate'):
        raise ValueError('Choose a supported interaction source.')
    if type(r.get('seed')) is not int or not 0 <= r['seed'] < 2**32:
        raise ValueError('Variation must be an integer from 0 to 4294967295.')
    if not isinstance(r.get('prompt'), str) or not 1 <= len(r['prompt'].strip()) <= 2000:
        raise ValueError('Describe the scene in 1–2000 characters.')
    if not isinstance(r.get('starts'), list) or len(r['starts']) != 2:
        raise ValueError('Place both character start markers.')
    for point in [*r['starts'], r.get('meeting')]:
        if not isinstance(point, dict):
            raise ValueError('Start and meeting positions must contain X and Z.')
        for key in ('x', 'z'):
            value = point.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 24:
                raise ValueError('Keep start and meeting markers within 24 metres of the scene origin.')
    yaw = r['meeting'].get('yaw_degrees', 0.)
    if type(yaw) not in (int, float) or not math.isfinite(yaw) or not -180 <= yaw <= 180:
        raise ValueError('Meeting direction must be between -180 and 180 degrees.')
    r['meeting']['yaw_degrees'] = float(yaw)
    target = r.get('target_id')
    if target is not None and (not isinstance(target, str) or not target or len(target) > 160):
        raise ValueError('Choose a target from the current scene.')
    r['actor_ids'] = list(ids)
    return r


def library_clip(source):
    if source not in ('handshake', 'sparring'):
        raise ValueError('Choose Handshake or Sparring from the reviewed library.')
    clip = load_source(REVIEWED_SPARRING if source == 'sparring' else REVIEWED_HANDSHAKE)
    if source == 'sparring':
        clip = NativePairClip(clip.joints, clip.features, dict(clip.metadata, render_hand_pose='fists',
                             reviewed_semantics='sparring preview; physical strikes not verified'))
    return clip


def resolve_meeting(request, clip, scene):
    """Resolve a stable scene ID to nearby clear ground, never into an object."""
    r = validate_request(request)
    target_id = r.get('target_id')
    if target_id is None:
        return r
    from scene_targets import resolve_targets
    resolved_targets = resolve_targets(scene.get('targets', []), scene.get('objects', []))
    matches = [(kind, o) for kind, items in (('objects', scene.get('objects', [])),
               ('targets', resolved_targets)) for o in items if o.get('id') == target_id]
    if len(matches) != 1:
        raise ValueError('The selected meeting landmark no longer exists uniquely; choose it again.')
    kind, obj = matches[0]
    position = obj.get('position')
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError('The meeting landmark has no usable ground position.')
    x, z = float(position[0]), float(position[2])
    yaw = math.radians(r['meeting']['yaw_degrees'])
    center = shared_place_pair(clip.joints[:1], yaw=yaw)[0, :, 0, :].mean(axis=0)
    owner = obj if kind == 'objects' else next((item for item in scene.get('objects', [])
             if item.get('id') == obj.get('object_id')), None)
    size = owner.get('size', [1, 1, 1]) if owner is not None else [1, 1, 1]
    radius = max(float(size[0]), float(size[2])) / 2 + 1.6 if owner is not None else 0.
    candidates = [(x, z)] if radius == 0 else []
    for extra in (0., .8, 1.6):
        for angle in np.arange(0, 2*math.pi, math.pi/4):
            candidates.append((x + (radius+extra)*math.sin(angle), z + (radius+extra)*math.cos(angle)))
    midpoint = np.mean([[p['x'], p['z']] for p in r['starts']], axis=0)
    candidates.sort(key=lambda p: float(np.linalg.norm(np.asarray(p)-midpoint)))
    for px, pz in candidates:
        if max(abs(px), abs(pz)) > 24:
            continue
        world = shared_place_pair(clip.joints, yaw=yaw, translation=(px-center[0], 0., pz-center[2]))
        try:
            check_native_pair_geometry(world, scene, actor_ids=r['actor_ids'])
        except ValueError:
            continue
        r['meeting'].update(x=px, z=pz)
        r['resolved_landmark'] = {'id': target_id, 'name': obj.get('name', target_id), 'policy': 'clear ground beside landmark'}
        return r
    raise ValueError('There is not enough clear ground beside that landmark for this interaction. Choose a custom meeting point.')


def preview_direction(request, scene, *, entry_policy=DEFAULT_ENTRY_POLICY, speed_mps=DEFAULT_SPEED_MPS):
    from paired_meetup import plan_meetup
    r = validate_request(request)
    clip = library_clip('handshake' if r['source'] == 'generate' else r['source'])
    r = resolve_meeting(r, clip, scene)
    plan = plan_meetup(clip, scene, actor_ids=r['actor_ids'], starts=r['starts'], meeting=r['meeting'],
                       entry_policy=entry_policy, speed_mps=speed_mps)
    return {'request': r, 'plan': plan, 'summary': ('Route preview using reference interaction; fresh motion is checked again.'
            if r['source'] == 'generate' else 'Both routes and the interaction area fit the current scene.')}


class InteractionPromptPlanner:
    """AI only writes the paired action prompt; coordinates and routes stay validated."""
    def __init__(self, gateway=None):
        self.gateway = gateway

    def plan(self, intent):
        from object_generation import GatewayGenerator
        gateway = self.gateway or GatewayGenerator.from_env(stage='assets')
        doc = gateway.request_json(
            'Return only JSON with one key interaction_prompt. Write a concise English physical interaction '
            'prompt for a model generating exactly two people jointly over seven seconds. Preserve the requested '
            'interaction and roles. Both people have already arrived at the meeting point; omit travel through '
            'the city. Do not invent props, characters or actions absent from the request. Do not provide '
            'coordinates, code, URLs or claims of successful motion. Treat the supplied text as scene intent only. '
            'Keep interaction_prompt between 1 and 500 characters.', intent, max_tokens=350, timeout_seconds=45)
        if not isinstance(doc, dict) or set(doc) != {'interaction_prompt'} or not isinstance(doc['interaction_prompt'], str) or not 1 <= len(doc['interaction_prompt'].strip()) <= 500:
            raise ValueError('The interaction planner returned an invalid description; simplify the direction and retry.')
        return doc['interaction_prompt'].strip()


class PairedSceneBuilder:
    def __init__(self, request, provider, core_client, output_root, planner=None, *,
                 entry_policy=DEFAULT_ENTRY_POLICY, speed_mps=DEFAULT_SPEED_MPS):
        self.request = validate_request(request)
        self.provider, self.core_client = provider, core_client
        self.output_root = Path(output_root)
        self.planner = planner or InteractionPromptPlanner()
        self.entry_policy, self.speed_mps = entry_policy, speed_mps

    def __call__(self, scene, *, cancelled=lambda: False, on_progress=lambda *args: None):
        from paired_meetup import plan_meetup, build_meetup
        if self.core_client is None:
            raise ValueError('Connect the motion service before generating a city meetup.')
        r = deepcopy(self.request)
        folder = self.output_root / str(time.time_ns())
        folder.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        def progress(message):
            if cancelled():
                raise RuntimeError('Scene generation cancelled.')
            if isinstance(message, dict):
                message = f"Generating approach {message.get('completed_windows', 0)}/{message.get('total_windows', '?')}…"
            on_progress(str(message))
        def chunk(phase, window, clip, request):
            np.savez_compressed(folder/f'{phase}-{window}.core.npz', positions=clip.positions,
                rotations=clip.rotations, native_features=clip.native_features,
                metadata=np.array(json.dumps({'request': request, 'fps': clip.fps})))
        try:
            progress('Checking start markers and meeting space…')
            # Cheap reference preflight catches invalid markers before expensive generation.
            preview_direction(r, scene, entry_policy=self.entry_policy, speed_mps=self.speed_mps)
            if r['source'] == 'generate':
                if self.provider is None:
                    raise ValueError('Pair generation is unavailable; choose Handshake or Sparring for the reviewed library.')
                progress('Planning the shared interaction…')
                prompt = self.planner.plan(r['prompt'])
                progress('Generating both characters together…')
                clip = self.provider.generate(prompt=prompt, seed=r['seed'], frames=210, cancelled=cancelled)
            else:
                clip = library_clip(r['source'])
            np.savez_compressed(folder/'source-pair.npz', joints=clip.joints,
                **({'features': clip.features} if clip.features is not None else {}), metadata=np.array(json.dumps(clip.metadata)))
            r = resolve_meeting(r, clip, scene)
            progress('Planning two routes with a shared arrival time…')
            plan = plan_meetup(clip, scene, actor_ids=r['actor_ids'], starts=r['starts'], meeting=r['meeting'],
                               entry_policy=self.entry_policy, speed_mps=self.speed_mps)
            (folder/'request.json').write_text(json.dumps(r, indent=2)+'\n')
            (folder/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
            result = build_meetup(clip, self.core_client, scene, actor_ids=r['actor_ids'], starts=r['starts'],
                                  meeting=r['meeting'], seed=r['seed'], cancelled=cancelled,
                                  entry_policy=self.entry_policy, speed_mps=self.speed_mps,
                                  on_progress=progress, on_core_chunk=chunk)
            progress('Checking the complete scene…')
            output = result['clip']
            metadata = dict(output.metadata, direction_request=r, source_archive_directory=str(folder),
                            generation_wall_seconds=time.perf_counter()-started)
            output = NativePairClip(output.joints, output.features, metadata)
            np.savez_compressed(folder/'composed.npz', joints=output.joints, metadata=np.array(json.dumps(metadata)))
            (folder/'report.json').write_text(json.dumps(result['report'], indent=2)+'\n')
            return {'clip': output, 'placement': result['placement']}
        except Exception as exc:
            (folder/'failure.json').write_text(json.dumps({'error_type': type(exc).__name__,
                'cancelled': cancelled(), 'sources_retained': (folder/'source-pair.npz').is_file()})+'\n')
            raise
